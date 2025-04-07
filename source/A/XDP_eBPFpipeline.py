# usage
# sudo apt-get update
# sudo apt-get install -y bpfcc-tools linux-headers-$(uname -r) python3-bpfcc python3-pycuda python3-pynvml pip
# sudo pip install numpy # Or use apt's python3-numpy if preferred
# sudo python3 XDP_eBPFpipeline_fixed.py -i <interface_name> -b <batch_size>

#!/usr/bin/env python3
import sys
import ctypes as ct
import logging
import argparse
import os
import time

# Configure logging for informative output
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Check for root privileges (required for BPF operations)
if os.geteuid() != 0:
    logging.error("This script requires root privileges to load and attach BPF programs.")
    sys.exit(1)

# ---------------------------- Argument Parsing ----------------------------
parser = argparse.ArgumentParser(description="XDP->GPU Packet Processing Pipeline")
parser.add_argument("-i", "--interface", type=str, default="lo",
                    help="Network interface to attach XDP (default: lo)")
parser.add_argument("-b", "--batch_size", type=int, default=256,
                    help="Number of packets to process in one GPU batch (default: 256)")
parser.add_argument("-m", "--max_pkt_size", type=int, default=2048,
                    help="Maximum packet bytes to capture from each packet (default: 2048)")
args = parser.parse_args()

INTERFACE = args.interface
BATCH_SIZE = args.batch_size
MAX_PKT_SIZE = args.max_pkt_size

logging.info(f"Configuration: Interface='{INTERFACE}', BatchSize={BATCH_SIZE}, MaxPktSize={MAX_PKT_SIZE}")
if INTERFACE == "lo":
    logging.warning("Using loopback ('lo') interface. This captures local traffic only "
                    "and performance may not reflect real NICs.")

# ---------------------------- eBPF Program (XDP) ----------------------------
# Using an f-string requires escaping curly braces for C code: {{ }} -> { }
bpf_program = fr"""
#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <string.h> // For __builtin_memset

// Define maximum packet size to capture
#define MAX_PKT_SIZE {MAX_PKT_SIZE}

// Structure of events to send to user-space via ring buffer
struct event {{
    u64 timestamp;                // Timestamp (nanoseconds) when packet was processed
    u32 len;                     // Length of captured packet data
    u8  data[MAX_PKT_SIZE];      // Packet data (zero-padded to MAX_PKT_SIZE)
}};

// Ring buffer map for sending packet events to user-space (size: 16 pages for efficiency)
BPF_RINGBUF_OUTPUT(events, 16);

// BPF array map (single slot) to count ring buffer reserve failures (when buffer is full)
BPF_ARRAY(ringbuf_fail_cnt, u64, 1);

// XDP program function - runs for every incoming packet on the interface
int xdp_prog(struct xdp_md *ctx) {{
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;
    u32 pkt_len = data_end > data ? (u32)(data_end - data) : 0;
    u32 copy_len = pkt_len;

    // Clamp copy length to MAX_PKT_SIZE
    if (copy_len > MAX_PKT_SIZE) {{
        copy_len = MAX_PKT_SIZE;
    }}

    // Reserve space in ring buffer for the event
    struct event *evt = events.ringbuf_reserve(sizeof(struct event));
    if (!evt) {{
        // Ring buffer is full - increment failure counter and drop the packet
        u32 idx = 0;
        u64 *fail_cnt = ringbuf_fail_cnt.lookup(&idx); // Use helper instead of map dereference
        if (fail_cnt) {{
            __sync_fetch_and_add(fail_cnt, 1);  // atomic increment of drop counter
        }}
        return XDP_DROP;  // Drop packet since we couldn't queue it
    }}

    // Fill event data
    evt->timestamp = bpf_ktime_get_ns();  // capture current kernel timestamp
    evt->len = copy_len;

    // Safely copy packet data using bpf_probe_read_kernel
    // Ensure the requested copy length doesn't exceed actual packet boundaries
    // The verifier needs this check even if copy_len was calculated from data_end
    if (data + copy_len <= data_end) {{
        // bpf_probe_read_kernel returns 0 on success
        if (bpf_probe_read_kernel(evt->data, copy_len, data) == 0) {{
            // Zero-pad remaining bytes if needed
            if (copy_len < MAX_PKT_SIZE) {{
                // Using __builtin_memset is efficient and common in BPF
                 __builtin_memset(evt->data + copy_len, 0, MAX_PKT_SIZE - copy_len);
            }}
        }} else {{
            // Read failed, maybe packet disappeared? Drop event.
            events.ringbuf_discard(evt, 0);
            return XDP_DROP;
        }}
    }} else {{
         // This case implies an issue or a race condition if copy_len calculation was correct.
         // Handle gracefully: maybe copy less or discard. Here, we discard.
        events.ringbuf_discard(evt, 0);
        return XDP_DROP;
    }}

    // Submit the event to the ring buffer for user-space consumption
    events.ringbuf_submit(evt, 0);

    // XDP action: drop the packet after capturing it to avoid duplicate processing in the OS stack.
    // (Use XDP_PASS instead if you want the packet to continue through normal networking stack.)
    return XDP_DROP;
}}
"""
# Append license AFTER defining the program string
bpf_program += "\nBPF_LICENSE(\"GPL\");\n"

# ---------------------------- Load and Attach eBPF ----------------------------
from bcc import BPF
bpf = None
xdp_fn = None
attached = False
try:
    logging.info("Compiling and loading eBPF program...")
    bpf = BPF(text=bpf_program)  # compile and load eBPF program
    xdp_fn = bpf.load_func("xdp_prog", BPF.XDP)  # get XDP program handle
    logging.info(f"Attaching XDP program to interface '{INTERFACE}'...")
    bpf.attach_xdp(dev=INTERFACE, fn=xdp_fn, flags=0) # Use generic XDP mode (flags=0)
    attached = True
    logging.info(f"eBPF XDP program successfully attached.")
except Exception as e:
    logging.error(f"Failed to load or attach XDP program: {e}")
    if bpf and attached: # Check if attached before attempting removal
        try:
            logging.info(f"Attempting to detach XDP from {INTERFACE} due to error...")
            bpf.remove_xdp(INTERFACE, 0)
        except Exception as remove_e:
            logging.error(f"Error during XDP detachment on failure: {remove_e}")
    sys.exit(1)

# Prepare Python ctypes structure matching the eBPF event struct
class Event(ct.Structure):
    _fields_ = [
        ("timestamp", ct.c_ulonglong),
        ("len", ct.c_uint32),
        ("data", ct.c_ubyte * MAX_PKT_SIZE)
    ]

# Lists to accumulate packets and lengths for batch processing
packets = []
lengths = []

# Callback function for ring buffer events
def handle_event(ctx, data, size):
    """Callback function for ring buffer; called for each event from eBPF."""
    if size < ct.sizeof(Event):
        logging.warning(f"Received undersized event ({size} bytes), expected at least {ct.sizeof(Event)}. Skipping.")
        return
        
    event = ct.cast(data, ct.POINTER(Event)).contents  # parse raw data as Event struct
    
    # Ensure event.len doesn't exceed buffer size (sanity check)
    pkt_len = min(event.len, MAX_PKT_SIZE)
    
    # Copy out the packet data (up to pkt_len) to user-space buffer
    # bytes() creates an immutable copy efficiently
    pkt_bytes = bytes(event.data[:pkt_len])
    
    packets.append(pkt_bytes)
    lengths.append(pkt_len)
    # Minimal processing in callback: just copy data and append to lists

# Set up the ring buffer consumer
try:
    bpf["events"].open_ring_buffer(handle_event)
except Exception as e:
    logging.error(f"Failed to open ring buffer: {e}")
    if attached:
        try:
            bpf.remove_xdp(INTERFACE, 0)
        except Exception: pass # Best effort cleanup
    sys.exit(1)

# ---------------------------- Initialize GPU (PyCUDA & optional NVML) ----------------------------
nvml_handle = None
gpu_initialized = False
try:
    import pynvml
    try:
        pynvml.nvmlInit()
        nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Assuming GPU 0
        gpu_name = pynvml.nvmlDeviceGetName(nvml_handle).decode('utf-8')
        logging.info(f"Using GPU: {gpu_name} (via NVML)")
    except ImportError:
        logging.warning("pynvml library found but failed to initialize. GPU utilization metrics unavailable.")
        nvml_handle = None
    except Exception as e:
        logging.warning(f"NVML initialization failed: {e}. GPU utilization metrics unavailable.")
        nvml_handle = None
except ImportError:
    logging.warning("pynvml not installed, GPU utilization metrics will be unavailable.")

# Initialize CUDA driver and create a context
d_pkt, d_len, d_out = None, None, None # Pre-declare GPU buffer handles
try:
    import pycuda.driver as cuda
    import pycuda.autoinit # Initializes CUDA context automatically on import
    from pycuda.compiler import SourceModule
    import numpy as np
    gpu_initialized = True
    logging.info(f"PyCUDA initialized successfully on device: {pycuda.autoinit.device.name()}")
except Exception as e:
    logging.error(f"Failed to initialize CUDA/PyCUDA: {e}")
    if attached:
        try: bpf.remove_xdp(INTERFACE, 0)
        except Exception: pass
    sys.exit(1)

# Define the CUDA kernel code
kernel_code = fr"""
extern "C" __global__
// Input: pkt_data (flat array of all packet bytes), pkt_len (array of lengths)
// Output: out_data (flat array for processed packets)
// Params: max_pkt_size (stride between packets), actual_batch_size (number of packets in this run)
void process_packets(const unsigned char *pkt_data, const int *pkt_len, unsigned char *out_data,
                     int max_pkt_size, int actual_batch_size)
{{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index

    // Boundary check: only process threads corresponding to a valid packet index
    if (idx >= actual_batch_size) return;

    // Calculate pointers to the start of the current packet's data and output area
    const unsigned char *packet = pkt_data + (size_t)idx * max_pkt_size;
    unsigned char *output = out_data + (size_t)idx * max_pkt_size;
    int len = pkt_len[idx]; // Get the actual length of this specific packet

    // Ensure length does not exceed max_pkt_size (safety check)
    if (len > max_pkt_size) {{ len = max_pkt_size; }}

    // *** Stage 1: Header processing (Example: Copy first N bytes) ***
    int header_bytes_to_copy = len < 64 ? len : 64; // Example: copy up to 64 header bytes
    for (int i = 0; i < header_bytes_to_copy; ++i) {{
        output[i] = packet[i]; // Simple copy
    }}

    // *** Stage 2: Payload processing (Example: Invert bits) ***
    for (int j = header_bytes_to_copy; j < len; ++j) {{
        output[j] = packet[j] ^ 0xFF; // Example: XOR transformation
    }}

    // *** Stage 3: Padding (Optional but recommended) ***
    // Zero out the rest of the buffer for this packet slot if pkt len < max_pkt_size
    for (int k = len; k < max_pkt_size; ++k) {{
        output[k] = 0;
    }}

    // *** Stage 4: Result writing (if needed beyond modifying out_data) ***
    // e.g., write classification results to another output array
}}
"""

# Compile CUDA kernel
gpu_kernel = None
try:
    module = SourceModule(kernel_code)
    gpu_kernel = module.get_function("process_packets")
    logging.info("CUDA kernel compiled and loaded successfully.")
except Exception as e:
    logging.error(f"CUDA kernel compilation failed: {e}")
    if attached:
        try: bpf.remove_xdp(INTERFACE, 0)
        except Exception: pass
    if nvml_handle: pynvml.nvmlShutdown()
    sys.exit(1)

# ------------------ Pre-allocate GPU Buffers for Performance ------------------
# Allocate GPU memory ONCE, sized for the maximum BATCH_SIZE. Reuse these buffers.
try:
    logging.info(f"Allocating GPU buffers for batch size {BATCH_SIZE} and max packet size {MAX_PKT_SIZE}...")
    d_pkt = cuda.mem_alloc(BATCH_SIZE * MAX_PKT_SIZE) # Input packet data
    d_len = cuda.mem_alloc(BATCH_SIZE * np.dtype(np.int32).itemsize) # Input packet lengths
    d_out = cuda.mem_alloc(BATCH_SIZE * MAX_PKT_SIZE) # Output processed data
    logging.info("GPU buffers allocated.")
except cuda.Error as ce:
    logging.error(f"Failed to allocate required GPU memory (BatchSize={BATCH_SIZE}, MaxPktSize={MAX_PKT_SIZE}): {ce}")
    if attached:
        try: bpf.remove_xdp(INTERFACE, 0)
        except Exception: pass
    if nvml_handle: pynvml.nvmlShutdown()
    # Attempt to free any partially allocated buffers
    if d_pkt: d_pkt.free()
    if d_len: d_len.free()
    if d_out: d_out.free()
    sys.exit(1)
except Exception as e:
    logging.error(f"Unexpected error during GPU buffer allocation: {e}")
    if attached:
        try: bpf.remove_xdp(INTERFACE, 0)
        except Exception: pass
    if nvml_handle: pynvml.nvmlShutdown()
    if d_pkt: d_pkt.free()
    if d_len: d_len.free()
    if d_out: d_out.free()
    sys.exit(1)

# ---------------------------- GPU Batch Processing Function ----------------------------
def process_batch():
    """Transfer accumulated packets to GPU, run processing kernel, and retrieve metrics."""
    global packets, lengths
    current_batch_size = len(packets)
    if current_batch_size == 0:
        return # Nothing to process

    # 1. Prepare Host Data (NumPy arrays for efficient transfer)
    # Create host arrays sized exactly for the *current* batch size
    host_pkt_batch = np.zeros((current_batch_size, MAX_PKT_SIZE), dtype=np.uint8)
    host_len_batch = np.array(lengths, dtype=np.int32) # lengths list already matches current batch

    for i, pkt_bytes in enumerate(packets):
        pkt_len = host_len_batch[i]
        # Ensure we don't copy more than MAX_PKT_SIZE bytes
        # Create a view and copy into the numpy array row
        pkt_array_view = np.frombuffer(pkt_bytes, dtype=np.uint8)
        host_pkt_batch[i, :pkt_len] = pkt_array_view[:pkt_len]
        # Remaining bytes in host_pkt_batch[i] are already zero

    # 2. Transfer Data to *Pre-allocated* GPU Buffers
    try:
        # Copy only the data for the current batch into the start of the GPU buffers
        cuda.memcpy_htod(d_pkt, host_pkt_batch) # Copies host_pkt_batch.nbytes
        cuda.memcpy_htod(d_len, host_len_batch) # Copies host_len_batch.nbytes
    except cuda.Error as ce:
        logging.error(f"CUDA memory copy Host-to-Device failed: {ce}")
        # Clear lists to prevent reprocessing the same failed batch
        packets.clear()
        lengths.clear()
        return # Skip kernel launch if copy fails
    except Exception as e:
        logging.error(f"Unexpected error during HtoD memory copy: {e}")
        packets.clear()
        lengths.clear()
        return

    # 3. Launch GPU Kernel
    # Determine CUDA launch configuration based on *current* batch size
    THREADS_PER_BLOCK = 256
    grid_size = (current_batch_size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    try:
        gpu_kernel(d_pkt, d_len, d_out, np.int32(MAX_PKT_SIZE), np.int32(current_batch_size),
                   block=(THREADS_PER_BLOCK, 1, 1), grid=(grid_size, 1))
        # Synchronize context to ensure kernel completion before proceeding
        # (Crucial for measuring time or ensuring data is ready for next step/copy-back)
        cuda.Context.synchronize()
    except Exception as e:
        logging.error(f"CUDA kernel execution failed: {e}")
        # Clear lists even if kernel fails, data consistency might be compromised
        packets.clear()
        lengths.clear()
        return # Don't proceed if kernel fails

    # 4. (Optional) Retrieve Results from GPU
    # Example: Copy processed data back to host (if needed for verification/output)
    # host_output_batch = np.empty_like(host_pkt_batch)
    # try:
    #     cuda.memcpy_dtoh(host_output_batch, d_out)
    # except cuda.Error as ce:
    #     logging.error(f"CUDA memory copy Device-to-Host failed: {ce}")
    # Process host_output_batch here...

    # 5. Log Performance Metrics (Optional)
    if nvml_handle:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
            logging.info(f"Batch ({current_batch_size} pkts) processed. GPU Util: {util.gpu}%, Mem Util: {util.memory}%")
        except pynvml.NVMLError as ne:
             logging.warning(f"Could not get GPU utilization: {ne}")
    else:
        logging.info(f"Processed a batch of {current_batch_size} packets on GPU.")

    # 6. Clear Python lists for the next batch
    packets.clear()
    lengths.clear()

# ---------------------------- Main Loop ----------------------------
packet_count = 0
start_time = time.time()

try:
    logging.info("Starting packet capture and processing loop. Press Ctrl+C to stop.")
    while True:
        # Poll the ring buffer for new packet events (non-blocking with timeout)
        # This calls handle_event for each received packet.
        bpf.ring_buffer_poll(timeout=100) # Poll for 100ms

        # Check if a full batch has accumulated
        if len(packets) >= BATCH_SIZE:
            batch_start_time = time.time()
            process_batch()
            packet_count += BATCH_SIZE # Assuming process_batch clears the list of BATCH_SIZE packets
            batch_end_time = time.time()
            logging.debug(f"Batch processing time: {batch_end_time - batch_start_time:.6f} seconds")

        # Optional: Process smaller batches if no new packets arrive for a while
        # (Could add logic here based on time since last packet/batch)

except KeyboardInterrupt:
    logging.info("Ctrl+C detected. Stopping...")
    # Process any remaining packets collected before exiting
    if packets:
        logging.info(f"Processing final batch of {len(packets)} packets...")
        final_batch_size = len(packets)
        process_batch()
        packet_count += final_batch_size
    end_time = time.time()
    logging.info(f"Processed {packet_count} packets in {end_time - start_time:.2f} seconds.")

finally:
    # ---------------------------- Cleanup ----------------------------
    logging.info("Cleaning up resources...")
    # Detach the eBPF program
    if attached:
        try:
            bpf.remove_xdp(INTERFACE, 0)
            logging.info(f"XDP program detached from interface '{INTERFACE}'.")
        except Exception as e:
            logging.error(f"Error detaching XDP program: {e}")

    # Report ring buffer drops
    try:
        fail_map = bpf.get_table("ringbuf_fail_cnt")
        # Access value at index 0
        fail_count = fail_map[ct.c_int(0)].value
        if fail_count > 0:
            logging.warning(f"Total ring buffer reserve failures (packets lost): {fail_count}")
        else:
             logging.info("No ring buffer reserve failures detected.")
    except Exception as e:
        logging.warning(f"Could not read ring buffer failure count: {e}")

    # Free pre-allocated GPU memory
    logging.info("Freeing GPU buffers...")
    if d_pkt:
        try: d_pkt.free()
        except Exception as e: logging.error(f"Error freeing d_pkt: {e}")
    if d_len:
        try: d_len.free()
        except Exception as e: logging.error(f"Error freeing d_len: {e}")
    if d_out:
        try: d_out.free()
        except Exception as e: logging.error(f"Error freeing d_out: {e}")
    logging.info("GPU buffers freed.")


    # Shutdown NVML
    if nvml_handle:
        try:
            pynvml.nvmlShutdown()
            logging.info("NVML shut down.")
        except Exception as e:
            logging.error(f"Error shutting down NVML: {e}")

    # Note: PyCUDA context is usually cleaned up automatically by autoinit or program exit
    logging.info("Cleanup complete. Exiting.")
