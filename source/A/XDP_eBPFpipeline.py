

"""
XDP/eBPF Packet Processing Pipeline with GPU Acceleration (PyCUDA/CuPy)

#usage
sudo apt-get update
sudo apt-get install bpfcc-tools linux-headers-$(uname -r)
sudo apt-get install python3-bcc
!pip install bcc

This script demonstrates a pipeline where network packets are captured at high speed
using eBPF/XDP attached to a network interface, sent to user space via a ring buffer,
batched, and then processed in parallel on a GPU using CUDA.

Potential Improvements & Considerations:
-----------------------------------------
1.  GPU Memory Management Efficiency:
    - Currently, GPU memory (`d_pkt`, `d_len`, `d_out`) is allocated and freed
      within `process_batch` for every batch.
    - Suggestion: Allocate these GPU buffers once at the start based on BATCH_SIZE
      and MAX_PKT_SIZE, and reuse them in `process_batch` by overwriting with
      `cuda.memcpy_htod`. Free the buffers only at script exit (in the `finally`
      block). This reduces memory allocation overhead.

2.  Overlap Data Transfer and Kernel Execution (Asynchronicity):
    - The current flow is synchronous: copy HtoD -> kernel launch -> synchronize.
    - Suggestion (Advanced): Use CUDA Streams (`pycuda.driver.Stream`) to overlap
      data transfers (Host-to-Device), kernel execution, and potentially
      Device-to-Host transfers. This can significantly improve throughput by keeping
      both the CPU (preparing next batch) and GPU busy concurrently, but increases
      code complexity.

3.  Batch Data Preparation Efficiency:
    - The loop in `process_batch` copying packet `bytes` into the NumPy `pkt_batch`
      array involves Python-level iteration.
    - Consideration: For extreme packet rates, this Python loop could become a
      bottleneck. Investigate if data can be copied more directly into a pinned
      (page-locked) host buffer within the `handle_event` callback, potentially
      reducing overhead in `process_batch`. This might be complex to achieve with
      bcc's ring buffer callback mechanism. The current NumPy approach is practical.

4.  eBPF `bpf_probe_read_kernel` Logic:
    - The `else if` block after `bpf_probe_read_kernel` in `xdp_prog` seems potentially
      redundant or unnecessary, as `copy_len` is already clamped based on `pkt_len`
      (calculated from `data_end - data`) and `MAX_PKT_SIZE`.
    - Suggestion: Verify and potentially simplify the copy logic to just:
      `if (copy_len > 0) { bpf_probe_read_kernel(evt->data, copy_len, data); }`

5.  Configuration Flexibility:
    - The network interface (`INTERFACE`) and GPU device index (implicitly 0 via
      `pycuda.autoinit` and `pynvml.nvmlDeviceGetHandleByIndex(0)`) are hardcoded.
    - Suggestion: Make these configurable via command-line arguments (`argparse`)
      or a configuration file for greater flexibility.

6.  Logging Verbosity:
    - Frequent `logging.info` messages (e.g., per batch completion) can be noisy.
    - Suggestion: Consider lowering the level for some messages (e.g., to `logging.debug`)
      or implementing aggregated statistics reporting (e.g., packets/sec, avg batch size)
      at intervals instead of logging every single batch event.

Prerequisites:
--------------
- Linux kernel with eBPF support (XDP, ring buffer).
- bcc tools and python3-bcc installed (`apt-get install bpfcc-tools linux-headers-$(uname -r) python3-bcc`).
- Python packages: `pycuda`, `cupy`, `numpy`, `pynvml` (optional for metrics).
- NVIDIA GPU with CUDA toolkit installed.
- Run this script with root privileges (`sudo python3 script.py`) or necessary capabilities (CAP_BPF, CAP_NET_ADMIN).
"""


#!/usr/bin/env python3
import sys
import ctypes as ct
import logging

# Configure logging for informative output
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------- Configuration ----------------------------
INTERFACE = "lo"  # Network interface to attach XDP (use "eth0" or real NIC for real traffic tests)
BATCH_SIZE = 256   # Number of packets to process in one GPU batch (tunable based on workload)
MAX_PKT_SIZE = 2048  # Maximum packet bytes to capture from each packet (adjust as needed)

# Note: Using loopback ("lo") for testing will capture locally-generated traffic (e.g., ping 127.0.0.1). 
# Results on loopback may not reflect real NIC performance. For realistic tests, attach to a physical interface like "eth0".
# Also ensure to run as root (or with CAP_BPF) to load and attach the eBPF XDP program.

# ---------------------------- eBPF Program (XDP) ----------------------------
bpf_program = r"""
#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/udp.h>

// Define maximum packet size to capture
#define MAX_PKT_SIZE """ + str(MAX_PKT_SIZE) + r"""

// Structure of events to send to user-space via ring buffer
struct event {
    u64 timestamp;                // Timestamp (nanoseconds) when packet was processed
    u32 len;                     // Length of captured packet data
    u8  data[MAX_PKT_SIZE];      // Packet data (zero-padded to MAX_PKT_SIZE)
};

// Ring buffer map for sending packet events to user-space (size: 16 pages for efficiency)
BPF_RINGBUF_OUTPUT(events, 16);

// BPF array map (single slot) to count ring buffer reserve failures (when buffer is full)
BPF_ARRAY(ringbuf_fail_cnt, u64, 1);

// XDP program function - runs for every incoming packet on the interface
int xdp_prog(struct xdp_md *ctx) {
    // Calculate packet length and clamp to MAX_PKT_SIZE
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;
    u32 pkt_len = data_end > data ? (u32)(data_end - data) : 0;
    u32 copy_len = pkt_len;
    if (copy_len > MAX_PKT_SIZE) {
        copy_len = MAX_PKT_SIZE;
    }

    // Reserve space in ring buffer for the event
    struct event *evt = events.ringbuf_reserve(sizeof(struct event));
    if (!evt) {
        // Ring buffer is full - increment failure counter and drop the packet
        u32 idx = 0;
        u64 *fail_cnt = bpf_map_lookup_elem(&ringbuf_fail_cnt, &idx);
        if (fail_cnt) {
            __sync_fetch_and_add(fail_cnt, 1);  // atomic increment of drop counter
        }
        return XDP_DROP;  // Drop packet since we couldn't queue it (prevent backlog)
    }

    // Fill event data
    evt->timestamp = bpf_ktime_get_ns();  // capture current kernel timestamp
    evt->len = copy_len;
    // Copy packet bytes to event->data safely
    if (copy_len > 0 && data + copy_len <= data_end) {
        bpf_probe_read_kernel(evt->data, copy_len, data);
    } else if (data < data_end) {
        // If packet is smaller than indicated length (unlikely), adjust copy_len
        copy_len = (u32)(data_end - data);
        evt->len = copy_len;
        bpf_probe_read_kernel(evt->data, copy_len, data);
    }
    // Zero-pad any remaining bytes to avoid leaking uninitialized kernel memory
    if (copy_len < MAX_PKT_SIZE) {
        __builtin_memset(evt->data + copy_len, 0, MAX_PKT_SIZE - copy_len);
    }

    // Submit the event to the ring buffer for user-space consumption
    events.ringbuf_submit(evt, 0);

    // XDP action: drop the packet after capturing it to avoid duplicate processing in the OS stack.
    // (Use XDP_PASS instead if you want the packet to continue through normal networking stack.)
    return XDP_DROP;
}
"""
# Append license to satisfy eBPF requirements (using GPL to allow helper usage)
bpf_program += "\nBPF_LICENSE(\"GPL\");\n"

# ---------------------------- Load and Attach eBPF ----------------------------
from bcc import BPF
bpf = None
xdp_fn = None
attached = False
try:
    bpf = BPF(text=bpf_program)  # compile and load eBPF program
    xdp_fn = bpf.load_func("xdp_prog", BPF.XDP)  # get XDP program handle
    bpf.attach_xdp(dev=INTERFACE, fn=xdp_fn, flags=0)
    attached = True
    logging.info(f"eBPF XDP program successfully attached to interface '{INTERFACE}'.")
except Exception as e:
    logging.error(f"Failed to load or attach XDP program: {e}")
    if bpf:
        # If attach failed, attempt to unload program (if loaded) and exit
        try:
            bpf.remove_xdp(INTERFACE, 0)
        except Exception:
            pass
    sys.exit(1)

# Prepare Python ctypes structure matching the eBPF event struct for parsing data in callback
class Event(ct.Structure):
    _fields_ = [
        ("timestamp", ct.c_ulonglong),
        ("len", ct.c_uint32),
        ("data", ct.c_ubyte * MAX_PKT_SIZE)
    ]

# Lists to accumulate packets and lengths for batch processing
packets = []
lengths = []

def handle_event(ctx, data, size):
    """Callback function for ring buffer; called for each event from eBPF."""
    event = ct.cast(data, ct.POINTER(Event)).contents  # parse raw data as Event struct
    # Copy out the packet data (up to event.len) to user-space buffer
    pkt_len = event.len
    pkt_bytes = bytes(event.data[:pkt_len])  # only copy valid bytes
    packets.append(pkt_bytes)
    lengths.append(pkt_len)
    # (We do minimal processing in the callback to avoid slowing down event consumption)

# Set up the ring buffer consumer with our callback
try:
    bpf["events"].open_ring_buffer(handle_event)
except Exception as e:
    logging.error(f"Failed to open ring buffer: {e}")
    # Detach XDP before exiting
    if attached:
        try: 
            bpf.remove_xdp(INTERFACE, 0)
        except Exception:
            pass
    sys.exit(1)

# ---------------------------- Initialize GPU (PyCUDA & CuPy) ----------------------------
# Try to initialize NVML (NVIDIA Management Library) for utilization metrics
nvml_handle = None
try:
    import pynvml
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(nvml_handle).decode('utf-8')
    logging.info(f"Using GPU: {gpu_name}")
except ImportError:
    logging.warning("pynvml not installed, GPU utilization metrics will be unavailable.")
except Exception as e:
    logging.error(f"NVML initialization failed: {e}")
    nvml_handle = None

# Initialize CUDA driver and create a context on GPU 0.
try:
    import pycuda.autoinit  # initializes CUDA context automatically
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    import numpy as np
    import cupy as cp
except Exception as e:
    logging.error(f"Failed to initialize CUDA/PyCUDA/CuPy: {e}")
    # Detach XDP program before exit
    if attached:
        try:
            bpf.remove_xdp(INTERFACE, 0)
        except Exception:
            pass
    sys.exit(1)

# Define the CUDA kernel code for packet processing (as a string for SourceModule)
kernel_code = r"""
extern "C" __global__ 
void process_packets(const unsigned char *pkt_data, const int *pkt_len, unsigned char *out_data,
                     int max_pkt_size, int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    // Each thread handles one packet
    int len = pkt_len[idx];
    const unsigned char *packet = pkt_data + idx * max_pkt_size;
    unsigned char *output = out_data + idx * max_pkt_size;
    // *** Stage 1: Header processing (e.g., parse or copy packet headers) ***
    int header_bytes = len < 64 ? len : 64;  // example header length to process
    for (int i = 0; i < header_bytes; ++i) {
        // Copy header bytes to output as-is (placeholder for actual header parsing logic)
        output[i] = packet[i];
    }
    // *** Stage 2: Payload processing (e.g., content inspection or transformation) ***
    for (int j = header_bytes; j < len; ++j) {
        // Example payload transformation: invert bits of each byte (dummy logic)
        output[j] = packet[j] ^ 0xFF;
    }
    // If packet is smaller than max size, pad the remaining output bytes with 0 (optional safety)
    for (int j = len; j < max_pkt_size; ++j) {
        output[j] = 0;
    }
    // *** Stage 3: Result writing (e.g., set flags or computed values) ***
    // (No additional result in this demo; all processed data is in out_data array)
}
"""
try:
    module = SourceModule(kernel_code)
    gpu_kernel = module.get_function("process_packets")
    logging.info("CUDA kernel compiled and loaded successfully.")
except Exception as e:
    logging.error(f"CUDA kernel compilation failed: {e}")
    # Detach XDP and exit
    if attached:
        try:
            bpf.remove_xdp(INTERFACE, 0)
        except Exception:
            pass
    sys.exit(1)

# ---------------------------- GPU Batch Processing Function ----------------------------
def process_batch():
    """Transfer accumulated packets to GPU, run processing kernel, and retrieve metrics."""
    global packets, lengths
    batch_size = len(packets)
    if batch_size == 0:
        return  # nothing to do

    # Prepare host arrays for packet data and lengths
    # Using NumPy for host-side buffers
    pkt_batch = np.zeros((batch_size, MAX_PKT_SIZE), dtype=np.uint8)
    len_batch = np.array(lengths[:batch_size], dtype=np.int32)
    for i, pkt in enumerate(packets[:batch_size]):
        pkt_len = len_batch[i]
        # Copy each packet's bytes into the NumPy array (only up to its length)
        pkt_array = np.frombuffer(pkt, dtype=np.uint8)  # create view of bytes
        pkt_batch[i, :pkt_len] = pkt_array[:pkt_len]    # copy data into row (remaining bytes already zero)

    # Allocate GPU memory and copy input data (packets and lengths) to device
    try:
        d_pkt = cuda.mem_alloc(pkt_batch.nbytes)
        d_len = cuda.mem_alloc(len_batch.nbytes)
        cuda.memcpy_htod(d_pkt, pkt_batch)   # copy packet data to GPU
        cuda.memcpy_htod(d_len, len_batch)   # copy lengths array to GPU
        # Allocate output buffer on GPU for processed data
        d_out = cuda.mem_alloc(pkt_batch.nbytes)
    except cuda.Error as ce:
        logging.error(f"CUDA memory allocation failed: {ce}")
        return
    except Exception as e:
        logging.error(f"Unexpected error during GPU memory setup: {e}")
        return

    # Determine CUDA launch configuration
    THREADS_PER_BLOCK = 256
    grid_size = (batch_size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    # Launch the GPU kernel to process the batch
    try:
        gpu_kernel(d_pkt, d_len, d_out, np.int32(MAX_PKT_SIZE), np.int32(batch_size),
                   block=(THREADS_PER_BLOCK, 1, 1), grid=(grid_size, 1, 1))
        # Synchronize to ensure GPU processing is complete before proceeding
        cuda.Context.synchronize()
        # (Batch-level synchronization ensures all packet processing is done.
        # In a more complex async design, one could use CUDA streams and events 
        # to overlap data transfers and kernel execution, synchronizing per stream as needed.)
    except Exception as e:
        logging.error(f"CUDA kernel execution failed: {e}")
        # Free GPU memory on failure
        d_pkt.free(); d_len.free(); d_out.free()
        return

    # (Optional) Retrieve or utilize results from d_out if needed.
    # For example, one could copy some results back to host to verify processing or gather statistics.
    # In this demonstration, we skip copying the full output back for performance, 
    # but we could do so if needed using cuda.memcpy_dtoh().
    # e.g., cuda.memcpy_dtoh(pkt_batch, d_out) to get processed data back in pkt_batch.

    # If NVML is available, fetch GPU utilization metrics for this batch (for monitoring/debugging)
    if nvml_handle:
        util = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
        # GPU utilization is the percentage of time the GPU was busy (compute cores).
        # Memory utilization is the percentage of GPU memory bandwidth used.
        logging.info(f"GPU Utilization: {util.gpu}%, Memory Utilization: {util.memory}%")

    # Free the GPU memory used for this batch (to avoid memory leak; alternatively, reuse buffers across batches)
    d_pkt.free()
    d_len.free()
    d_out.free()
    # Clear the batch lists now that processing is done
    packets = []
    lengths = []
    # (For high-frequency batches, consider reusing allocated GPU buffers to reduce allocation overhead.)
    logging.info(f"Processed a batch of {batch_size} packets on GPU.")


# Main loop to consume events and trigger GPU processing
try:
    logging.info("Starting to poll ring buffer for packet events. Press Ctrl+C to stop.")
    while True:
        # Poll ring buffer for events (with a timeout to allow periodic checks)
        # This will invoke handle_event for each packet event available.
        bpf.ring_buffer_poll(timeout=100)  # timeout in milliseconds
        # If we have accumulated a full batch, process it on the GPU
        if len(packets) >= BATCH_SIZE:
            process_batch()
        # (Loop continues; any leftover packets after processing a batch will remain in the list for the next iteration.)
except KeyboardInterrupt:
    logging.info("Stopping packet capture and processing.")
    # Process any remaining packets in a final batch before exit
    if packets:
        logging.info(f"Processing final batch of {len(packets)} packets before exit.")
        process_batch()
finally:
    # Detach the eBPF program from the interface to restore normal operation
    if attached:
        try:
            bpf.remove_xdp(INTERFACE, 0)
            logging.info(f"Detached XDP program from interface '{INTERFACE}'.")
        except Exception as e:
            logging.error(f"Error detaching XDP program: {e}")
    # Report ring buffer drop count from eBPF (if any)
    try:
        fail_map = bpf.get_table("ringbuf_fail_cnt")
        if fail_map:
            # The map has one entry (key 0)
            fail_count = list(fail_map.items())[0][1].value
            if fail_count:
                logging.info(f"Ring buffer reserve failures (events lost due to full buffer): {fail_count}")
    except Exception:
        pass
    # Cleanup NVML
    if nvml_handle:
        pynvml.nvmlShutdown()
