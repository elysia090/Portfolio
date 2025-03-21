import cupy as cp
import math

def generate_twiddle_constants(N, inverse=False):
    """
    Generate a list of twiddle factors for each FFT stage.
    For stage s (1..log2(N)), for j in range(0, 2^(s-1)):
        w = exp( (inverse? +1 : -1) * 2*pi*i * j / (2^s) )
    All stage factors are concatenated into one list.
    """
    stages = int(math.log2(N))
    twiddles = []
    for s in range(1, stages+1):
        m = 1 << s          # m = 2^s
        half_m = m >> 1     # m/2
        for j in range(half_m):
            theta = 2.0 * math.pi * j / m
            if inverse:
                # Inverse FFT uses a + sign in the exponent.
                r = math.cos(theta)
                i = math.sin(theta)
            else:
                # Forward FFT uses a - sign in the exponent.
                r = math.cos(theta)
                i = -math.sin(theta)
            twiddles.append((r, i))
    return twiddles

def format_constant_array(twiddles, name):
    """
    Format the list of twiddles into a C++ initializer for a constant memory array.
    Each element is a cuDoubleComplex with .x (real) and .y (imaginary) fields.
    """
    lines = []
    for (r, i) in twiddles:
        lines.append(f"{{{r:.17g}, {i:.17g}}}")
    joined = ",\n    ".join(lines)
    return f"__constant__ cuDoubleComplex {name}[{len(twiddles)}] = {{\n    {joined}\n}};"

# FFT size
N = 1024

# Precompute twiddle factors for forward and inverse FFT
twiddles_forward = generate_twiddle_constants(N, inverse=False)
twiddles_inverse = generate_twiddle_constants(N, inverse=True)
const_forward_str = format_constant_array(twiddles_forward, "const_twiddles_forward")
const_inverse_str = format_constant_array(twiddles_inverse, "const_twiddles_inverse")

# CUDA kernel code with integrated bit reversal
kernel_code = r'''
#include <cuComplex.h>
#include <math.h>

// Precomputed constant twiddle factors.
''' + const_forward_str + "\n" + const_inverse_str + r'''

extern "C"
__global__ void fft_shared(const cuDoubleComplex * __restrict__ in,
                           cuDoubleComplex * __restrict__ out,
                           int N, int inverse)
{
    // Allocate dynamic shared memory for FFT data.
    extern __shared__ cuDoubleComplex s_data[];
    int tid = threadIdx.x;

    // Calculate the number of FFT stages: stages = log2(N)
    int stages = 0, temp = N;
    while (temp > 1) {
        stages++;
        temp >>= 1;
    }
    
    // Integrated Bit Reversal: load input using bit-reversed index.
    unsigned int rev = __brev(tid) >> (32 - stages);
    if (tid < N)
        s_data[rev] = in[tid];
    __syncthreads();

    // Main FFT loop with unrolled stage loop.
#pragma unroll
    for (int s = 1; s <= stages; s++) {
        int m = 1 << s;         // Sub-FFT size: m = 2^s.
        int half_m = m >> 1;      // Half the sub-FFT size.
        // Offset into the twiddle factor table: for stage s, offset = 2^(s-1) - 1.
        int offset = (1 << (s - 1)) - 1;
        int j = tid % m;        // Position within the sub-FFT block.
        int base = tid - j;     // Base index of the current sub-FFT block.
        if (j < half_m) {
            cuDoubleComplex u = s_data[base + j];
            cuDoubleComplex v = s_data[base + j + half_m];
            // Fetch the precomputed twiddle factor.
            cuDoubleComplex w;
            if (inverse)
                w = const_twiddles_inverse[offset + j];
            else
                w = const_twiddles_forward[offset + j];
            
            cuDoubleComplex t_val;
            // Butterfly computation using fused multiply-add.
            t_val.x = fma(v.x, w.x, -v.y * w.y);
            t_val.y = fma(v.x, w.y,  v.y * w.x);
            
            s_data[base + j].x = u.x + t_val.x;
            s_data[base + j].y = u.y + t_val.y;
            s_data[base + j + half_m].x = u.x - t_val.x;
            s_data[base + j + half_m].y = u.y - t_val.y;
        }
        __syncthreads();
    }

    // Scale the inverse FFT result by 1/N.
    if (inverse && tid < N) {
        double scale = 1.0 / N;
        s_data[tid].x *= scale;
        s_data[tid].y *= scale;
    }
    __syncthreads();

    // Write the FFT result from shared memory back to global memory.
    if (tid < N)
        out[tid] = s_data[tid];
}
'''

# Compile the CUDA kernel using CuPy's RawModule with fast math optimizations.
module = cp.RawModule(code=kernel_code, options=("-use_fast_math",), backend="nvcc")
fft_shared = module.get_function("fft_shared")

class HPDFT_CuPy_ConstantFFT:
    """
    Optimized FFT using precomputed constant twiddle factors and integrated bit reversal.
    This implementation is specialized for a fixed FFT size of 1024.
    """
    def __init__(self):
        self.N = 1024
        self.fft_shared = fft_shared

    def transform(self, x, inverse=False):
        """
        Compute the FFT or inverse FFT using the constant-memory kernel.
        
        Args:
            x: Input CuPy array of type cp.complex128 with length 1024.
            inverse: If True, compute the inverse FFT.
        
        Returns:
            A CuPy array of type cp.complex128 containing the FFT result.
        """
        if not isinstance(x, cp.ndarray):
            x = cp.asarray(x)
        if x.size != self.N:
            raise ValueError(f"Input size must be {self.N}.")
        result = cp.empty_like(x)
        # Allocate shared memory for N complex numbers.
        shared_mem_size = x.nbytes

        self.fft_shared(
            (1,),              # Single block launch.
            (self.N,),         # N threads.
            (x.data.ptr, result.data.ptr, cp.int32(self.N), cp.int32(1 if inverse else 0)),
            shared_mem=shared_mem_size
        )
        return result

if __name__ == "__main__":
    # Instantiate the optimized FFT class.
    fft_cupy = HPDFT_CuPy_ConstantFFT()
    
    # Generate a random complex input array on the GPU.
    x = cp.random.random(1024, dtype=cp.float64) + 1j * cp.random.random(1024, dtype=cp.float64)
    
    # Time the forward FFT using CuPy CUDA events.
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    X = fft_cupy.transform(x, inverse=False)
    end.record()
    end.synchronize()
    elapsed_fwd = cp.cuda.get_elapsed_time(start, end)
    print(f"Optimized constant-memory FFT forward (N = 1024) took {elapsed_fwd:.3f} ms.")
    
    # Time the inverse FFT.
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    x_rec = fft_cupy.transform(X, inverse=True)
    end.record()
    end.synchronize()
    elapsed_inv = cp.cuda.get_elapsed_time(start, end)
    print(f"Optimized constant-memory FFT inverse (N = 1024) took {elapsed_inv:.3f} ms.")
    
    # Check the maximum reconstruction error.
    error = cp.max(cp.abs(x - x_rec))
    print(f"Max reconstruction error: {error:.3e}")
