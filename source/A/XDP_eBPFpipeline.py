#usage
#sudo apt-get update
#sudo apt-get install bpfcc-tools linux-headers-$(uname -r)
#sudo apt-get install python3-bcc
#!pip install bcc

# ====== Configuration and Imports ======
INTERFACE = "lo"            # Network interface to attach XDP (e.g., "lo" for loopback or "eth0")
BATCH_SIZE = 100            # Number of packets to process in one GPU batch
RUNTIME_SEC = 10            # Duration to run the capture loop (in seconds)
CAP_LEN = 64               # Number of bytes of each packet to capture for GPU processing

from bcc import BPF
import numpy as np
import cupy as cp
import pycuda.autoprimaryctx  # Use primary CUDA context to avoid conflicts with CuPy
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time

# Optional: NVML for GPU utilization measurement
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlShutdown
    nvml_available = True
except ImportError:
    nvml_available = False

# ====== eBPF XDP Program Setup (using BCC) ======
bpf_program = r"""
#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/udp.h>
#define CAP_LEN %d

struct pkt_event {
    u64 ts;              // Timestamp (nanoseconds)
    u32 len;             // Captured payload length
    u8  data[CAP_LEN];   // Captured packet data (first CAP_LEN bytes)
};

// Define a ring buffer map to output events to user-space
BPF_RINGBUF_OUTPUT(events, 256);

SEC("xdp")
int xdp_prog(struct xdp_md *ctx) {
    void *data     = (void*)(long)ctx->data;
    void *data_end = (void*)(long)ctx->data_end;
    u64 pkt_len = data_end - data;
    if (pkt_len <= 0) {
        return XDP_PASS;
    }
    // Reserve space in ring buffer for an event
    struct pkt_event *evt = events.ringbuf_reserve(sizeof(struct pkt_event));
    if (!evt) {
        // Ring buffer full, drop packet to avoid backlog
        return XDP_DROP;
    }
    evt->ts = bpf_ktime_get_ns();
    u32 copy_len = pkt_len > CAP_LEN ? CAP_LEN : pkt_len;
    evt->len = copy_len;
    // Copy packet data to event (up to CAP_LEN bytes) with bounds checking
    #pragma unroll
    for (int i = 0; i < CAP_LEN; i++) {
        if (i >= copy_len) break;
        if ((char*)data + i + 1 > (char*)data_end) break;
        evt->data[i] = *((char*)data + i);
    }
    // Zero out any unused bytes (for safety/consistency)
    #pragma unroll
    for (int j = 0; j < CAP_LEN; j++) {
        if (j >= copy_len) {
            evt->data[j] = 0;
        }
    }
    events.ringbuf_submit(evt, 0);
    return XDP_PASS;
}
char _license[] SEC("license") = "GPL";
""" % CAP_LEN

# Compile and load the BPF program
bpf = BPF(text=bpf_program)
fn = bpf.load_func("xdp_prog", BPF.XDP)
# Attach XDP program to the specified network interface
flags = 0
try:
    bpf.attach_xdp(dev=INTERFACE, fn=fn, flags=flags)
except Exception:
    # If native XDP not supported, attach in SKB (generic) mode for portability
    flags = BPF.XDP_FLAGS_SKB_MODE
    bpf.attach_xdp(dev=INTERFACE, fn=fn, flags=flags)

# ====== GPU (PyCUDA + CuPy) Setup ======
# Initialize NVML for GPU utilization if available
if nvml_available:
    nvmlInit()
    nvml_handle = nvmlDeviceGetHandleByIndex(0)
# Prepare CUDA kernel (compiled with PyCUDA) for packet processing
kernel_code = r"""
extern "C"
__global__ void process_data(const unsigned char *in, unsigned char *out, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    unsigned char val = in[idx];
    unsigned int temp = 0;
    // Simulate some heavy computation per byte
    for (int j = 0; j < 100; ++j) {
        temp += (unsigned int)((val * j + idx) & 0xFF);
    }
    out[idx] = (unsigned char)(temp & 0xFF);
}
"""
cuda_mod   = SourceModule(kernel_code)
cuda_func  = cuda_mod.get_function("process_data")

# ====== Data Structures for Processing and Metrics ======
events_queue = []               # Queue for incoming packet events from BPF
total_count = 0                 # Total packets processed
total_latency_ns = 0            # Cumulative latency in nanoseconds
max_latency_ns = 0              # Maximum observed latency
gpu_util_samples = []           # Samples of GPU utilization (%)
# Define ring buffer callback to collect events
def handle_event(ctx, data, size):
    # Parse raw event data into Python object
    event = bpf["events"].event(data)
    events_queue.append(event)

# Open the BPF ring buffer and set up the callback
rb = bpf["events"]
rb.open_ring_buffer(handle_event)

# ====== Packet Processing Loop & Benchmarking ======
end_time = time.time() + RUNTIME_SEC
last_util_check = time.time()
try:
    start_time = time.time()
    # Process events until timeout or interruption
    try:
        while time.time() < end_time:
            # Poll BPF ring buffer (timeout 100 ms to allow periodic checks)
            bpf.ring_buffer_poll(timeout=100)
            # If enough events accumulated, process them in batches on the GPU
            while len(events_queue) >= BATCH_SIZE:
                batch = events_queue[:BATCH_SIZE]
                del events_queue[:BATCH_SIZE]
                # Prepare batch data for GPU
                n = len(batch)
                data_np = np.zeros((n, CAP_LEN), dtype=np.uint8)  # host buffer
                ts_list = []
                for i, evt in enumerate(batch):
                    length = int(evt.len)
                    # Copy packet bytes to numpy array
                    data_np[i, :length] = evt.data[:length]
                    # (evt.data beyond length is already zero from BPF)
                    ts_list.append(evt.ts)
                # Transfer data to GPU (using CuPy) and allocate output buffer
                d_in  = cp.asarray(data_np)             # GPU input array
                d_out = cp.empty_like(d_in)             # GPU output array
                total_elems = d_in.size                 # total number of bytes
                # Launch CUDA kernel (one thread per byte)
                block_size = 256
                grid_size = (int((total_elems + block_size - 1) / block_size), 1, 1)
                cuda_func(cuda.In(np.intp(d_in.data.ptr)),
                          cuda.InOut(np.intp(d_out.data.ptr)),
                          np.int32(total_elems),
                          block=(block_size, 1, 1), grid=grid_size)
                # Wait for kernel to finish
                cuda.Context.synchronize()
                # Record latency for each packet in this batch
                finish_ts = time.monotonic_ns()
                for ts in ts_list:
                    # Latency = time from packet capture to processing completion (ns)
                    lat_ns = finish_ts - ts
                    total_latency_ns += lat_ns
                    if lat_ns > max_latency_ns:
                        max_latency_ns = lat_ns
                total_count += n
            # Periodically sample GPU utilization (once per second)
            if nvml_available and (time.time() - last_util_check >= 1.0):
                util = nvmlDeviceGetUtilizationRates(nvml_handle).gpu
                gpu_util_samples.append(util)
                last_util_check = time.time()
    except KeyboardInterrupt:
        print("Interrupted by user, stopping capture.")
    # Process any remaining events in the queue after loop ends
    if events_queue:
        # Process final partial batch
        batch = events_queue
        events_queue = []
        n = len(batch)
        data_np = np.zeros((n, CAP_LEN), dtype=np.uint8)
        ts_list = []
        for i, evt in enumerate(batch):
            length = int(evt.len)
            data_np[i, :length] = evt.data[:length]
            ts_list.append(evt.ts)
        d_in  = cp.asarray(data_np)
        d_out = cp.empty_like(d_in)
        total_elems = d_in.size
        block_size = 256
        grid_size = (int((total_elems + block_size - 1) / block_size), 1, 1)
        cuda_func(cuda.In(np.intp(d_in.data.ptr)),
                  cuda.InOut(np.intp(d_out.data.ptr)),
                  np.int32(total_elems),
                  block=(block_size, 1, 1), grid=grid_size)
        cuda.Context.synchronize()
        finish_ts = time.monotonic_ns()
        for ts in ts_list:
            lat_ns = finish_ts - ts
            total_latency_ns += lat_ns
            if lat_ns > max_latency_ns:
                max_latency_ns = lat_ns
        total_count += n
finally:
    # ====== Cleanup: Detach XDP and Shutdown NVML ======
    bpf.remove_xdp(INTERFACE, flags)
    if nvml_available:
        nvmlShutdown()

# ====== Results Summary ======
elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
if total_count > 0 and elapsed_time > 0:
    throughput = total_count / elapsed_time
    avg_latency_ms = (total_latency_ns / total_count) / 1e6
    max_latency_ms = max_latency_ns / 1e6
    avg_gpu_util = (sum(gpu_util_samples) / len(gpu_util_samples)) if gpu_util_samples else 0.0
    print(f"Processed {total_count} packets in {elapsed_time:.2f} seconds")
    print(f"Throughput: {throughput:,.0f} packets/sec")
    print(f"Average latency: {avg_latency_ms:.3f} ms   (max {max_latency_ms:.3f} ms)")
    if gpu_util_samples:
        print(f"Average GPU utilization: {avg_gpu_util:.1f}%")
else:
    print("No packets were processed. Ensure traffic is flowing on the interface and try again.")
