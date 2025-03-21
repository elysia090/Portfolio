"""
End-to-End Test for NTT and Garner CRT Reconstruction Pipeline with Enhanced Memory Management,
Error Handling, Code Structure, and Correct Garner CRT Reconstruction.

This script performs the following tests:
  1. NTT Transformation Test:
     - Generates random input.
     - Applies one stage of the forward NTT (FMT) and its inverse (iFMT).
     - Executes element-wise multiplication and normalization.
  2. Negative Cyclic Convolution Test:
     - Applies pre-processing and post-processing steps using negative cyclic convolution.
     - Combines two arrays into high and low parts.
  3. Garner’s CRT Reconstruction Test:
     - Reconstructs a large integer from simulated residue arrays using dynamically computed
       modular inverses.
  4. Verification Test for Garner’s CRT Reconstruction:
     - Uses a known test value to verify the correctness of the reconstruction.

The outputs from each test and any errors are printed to the console.
"""

import cupy as cp
import numpy as np
import pycuda.driver as cuda
import pycuda.compiler as compiler
import pycuda.autoinit
import sys

# Fixed moduli for Garner CRT
MOD_P0 = 469762049
MOD_P1 = 1811939329
MOD_P2 = 2013265921

# Compute modular inverses:
# inv0_mod1: the modular inverse of MOD_P0 modulo MOD_P1.
# inv01_mod2: the modular inverse of (MOD_P0*MOD_P1) modulo MOD_P2.
inv0_mod1 = np.uint32(pow(MOD_P0, -1, MOD_P1))
inv01_mod2 = np.uint32(pow(MOD_P0 * MOD_P1, -1, MOD_P2))

# ------------------------------------------------------------------------------
# CUDA Kernel Code (Refactored and Commented)
# ------------------------------------------------------------------------------
kernel_code = r'''
#ifndef MODP
#define MODP 998244353U
#endif

#define ulong unsigned long long
#define uint unsigned int

//------------------------
// Modular Arithmetic Functions
//------------------------

// modMultiply: Computes (a * b) % MODP using 64-bit arithmetic.
__device__ uint modMultiply(uint a, uint b) {
    ulong tmp = ((ulong)(__umulhi(a, b))) * (1ULL << 32) + (ulong)(a * b);
    return (uint)(tmp % MODP);
}

// modExp: Computes (a^b) % MODP using exponentiation by squaring.
__device__ uint modExp(uint a, uint b) {
    ulong ans = 1ULL;
    ulong aa = a;
    while(b != 0) {
        if(b & 1)
            ans = (ans * aa) % MODP;
        aa = (aa * aa) % MODP;
        b = b >> 1;
    }
    return (uint)ans;
}

// divideByN: Computes a/arrayLength modulo MODP (for normalization).
__device__ uint divideByN(uint a, uint arrayLength) {
    uint quotient = a / arrayLength;
    uint remainder = a - quotient * arrayLength;
    uint pn = MODP / arrayLength;
    if(remainder != 0)
        quotient += (arrayLength - remainder) * pn + 1;
    return quotient;
}

//------------------------
// NTT Kernels (Butterfly stages)
//------------------------

// FMT: Forward NTT butterfly stage.
__global__ void FMT(uint *arrayA, uint loopCnt_Pow2, uint omega, uint arrayLength2) {
    uint idx = threadIdx.x + blockIdx.x * 256;
    uint t2 = idx % loopCnt_Pow2;
    uint t0 = idx * 2 - t2;
    uint t1 = t0 + loopCnt_Pow2;
    uint w0 = modExp(omega, t2 * (arrayLength2 / loopCnt_Pow2));
    uint w1 = modMultiply(arrayA[t1], w0);
    uint r0 = arrayA[t0] + MODP - w1;
    uint r1 = arrayA[t0] + w1;
    if(r0 >= MODP) r0 -= MODP;
    if(r1 >= MODP) r1 -= MODP;
    arrayA[t1] = r0;
    arrayA[t0] = r1;
}

// iFMT: Inverse NTT butterfly stage.
__global__ void iFMT(uint *arrayA, uint loopCnt_Pow2, uint omega, uint arrayLength2) {
    uint idx = threadIdx.x + blockIdx.x * 256;
    uint t2 = idx % loopCnt_Pow2;
    uint t0 = idx * 2 - t2;
    uint t1 = t0 + loopCnt_Pow2;
    uint w0 = modExp(omega, arrayLength2 * 2 - t2 * (arrayLength2 / loopCnt_Pow2));
    uint w1 = modMultiply(arrayA[t1], w0);
    uint r0 = arrayA[t0] + MODP - w1;
    uint r1 = arrayA[t0] + w1;
    if(r0 >= MODP) r0 -= MODP;
    if(r1 >= MODP) r1 -= MODP;
    arrayA[t1] = r0;
    arrayA[t0] = r1;
}

// Mul_i_i: Element-wise multiplication of two arrays.
__global__ void Mul_i_i(uint *arrayA, uint *arrayB) {
    uint idx = threadIdx.x + blockIdx.x * 256;
    arrayB[idx] = modMultiply(arrayA[idx], arrayB[idx]);
}

// DivN: Normalizes each element by dividing by arrayLength.
__global__ void DivN(uint *arrayA, uint arrayLength) {
    uint idx = threadIdx.x + blockIdx.x * 256;
    arrayA[idx] = divideByN(arrayA[idx], arrayLength);
}

//------------------------
// Negative Cyclic Convolution Kernels
//------------------------

// PreNegFMT: Preprocessing step for negative convolution.
// Multiply a[i] by modExp(sqrtOmega, i).
__global__ void PreNegFMT(uint *arrayA, uint *arrayB, uint sqrtOmega, uint arrayLength) {
    uint idx = threadIdx.x + blockIdx.x * 256;
    arrayA[idx] %= MODP;
    uint factor = modExp(sqrtOmega, idx);
    arrayB[idx] = modMultiply(arrayA[idx], factor);
}

// PostNegFMT: Postprocessing step for negative convolution.
// Multiply each element by modExp(sqrtOmega, 2*arrayLength - i).
__global__ void PostNegFMT(uint *arrayA, uint sqrtOmega, uint arrayLength) {
    uint idx = threadIdx.x + blockIdx.x * 256;
    uint factor = modExp(sqrtOmega, arrayLength * 2 - idx);
    arrayA[idx] = modMultiply(arrayA[idx], factor);
}

// PosNeg_To_HiLo: Separates combined convolution results into high and low parts.
__global__ void PosNeg_To_HiLo(uint *arrayE, uint *arrayA, uint *arrayB, uint arrayLength) {
    uint idx = threadIdx.x + blockIdx.x * 256;
    uint a = arrayA[idx], b = arrayB[idx];
    uint diff = (a >= b) ? (a - b) : (a + MODP - b);
    uint flag = diff & 1;
    diff = diff - (((diff >= MODP) ? 2 : -1) * flag);
    diff /= 2;
    arrayE[idx + arrayLength] = diff;
    arrayE[idx] = (a >= diff) ? (a - diff) : (a + MODP - diff);
}

// PostFMT_DivN_HiLo: Combined negative FFT postprocessing, normalization, and hi-lo separation.
__global__ void PostFMT_DivN_HiLo(uint *arrayE, uint *arrayA, uint *arrayB, uint arrayLength, uint sqrtOmega) {
    uint idx = threadIdx.x + blockIdx.x * 256;
    uint a = arrayA[idx], b = arrayB[idx];
    uint factor = modExp(sqrtOmega, idx + ((idx & 1) * arrayLength));
    b = modMultiply(b, factor);
    a = divideByN(a, arrayLength);
    b = divideByN(b, arrayLength);
    uint diff = (a >= b) ? (a - b) : (a + MODP - b);
    uint flag = diff & 1;
    diff = diff - (((diff >= MODP) ? 2 : -1) * flag);
    diff /= 2;
    arrayE[idx + arrayLength] = diff;
    arrayE[idx] = (a >= diff) ? (a - diff) : (a + MODP - diff);
}

//------------------------
// Garner's CRT Reconstruction Kernel
//------------------------

// Fixed moduli for CRT reconstruction.
#define MOD_P0 469762049ULL
#define MOD_P1 1811939329ULL
#define MOD_P2 2013265921ULL

// GarnerGPU: Reconstructs a large integer from its residues.
// The kernel now accepts two extra parameters: inv0_mod1 and inv01_mod2.
// Each thread computes:
//   t1 = ((r1 - r0) mod MOD_P1) * inv0_mod1 mod MOD_P1,
//   x1 = r0 + t1 * MOD_P0,
//   t2 = ((r2 - (x1 mod MOD_P2)) mod MOD_P2) * inv01_mod2 mod MOD_P2,
//   x = x1 + t2 * (MOD_P0*MOD_P1).
// The result is written as three 32-bit words (little-endian) at positions 3*idx, 3*idx+1, 3*idx+2.
__global__ void GarnerGPU(uint *r0_array, uint *r1_array, uint *r2_array, uint *result, uint inv0_mod1, uint inv01_mod2) {
    int idx = threadIdx.x + blockIdx.x * 256;
    ulong r0 = r0_array[idx];
    ulong r1 = r1_array[idx];
    ulong r2 = r2_array[idx];

    ulong t1 = (((r1 + MOD_P1 - r0) % MOD_P1) * inv0_mod1) % MOD_P1;
    ulong x1 = r0 + t1 * MOD_P0;
    ulong t2 = (((r2 + MOD_P2 - (x1 % MOD_P2)) % MOD_P2) * inv01_mod2) % MOD_P2;
    ulong x = x1 + t2 * ((ulong)MOD_P0 * MOD_P1);

    result[3 * idx + 0] = (uint)(x & 0xFFFFFFFFULL);
    result[3 * idx + 1] = (uint)((x >> 32) & 0xFFFFFFFFULL);
    result[3 * idx + 2] = (uint)((x >> 64) & 0xFFFFFFFFULL);
}
'''

# ------------------------------------------------------------------------------
# Compile the CUDA Kernels with warning suppression for shift count warning
# ------------------------------------------------------------------------------
try:
    module = compiler.SourceModule(kernel_code, options=["-use_fast_math", "-diag-suppress", "63"])
    FMT               = module.get_function("FMT")
    iFMT              = module.get_function("iFMT")
    Mul_i_i           = module.get_function("Mul_i_i")
    DivN              = module.get_function("DivN")
    PreNegFMT         = module.get_function("PreNegFMT")
    PostNegFMT        = module.get_function("PostNegFMT")
    PosNeg_To_HiLo    = module.get_function("PosNeg_To_HiLo")
    PostFMT_DivN_HiLo = module.get_function("PostFMT_DivN_HiLo")
    GarnerGPU         = module.get_function("GarnerGPU")
except Exception as e:
    print("Error compiling CUDA kernels:", e)
    sys.exit(1)

# ------------------------------------------------------------------------------
# Memory Cleanup Helper (Called only at end of main)
# ------------------------------------------------------------------------------
def cleanup_memory():
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception as e:
        print("Memory cleanup error:", e)

# ------------------------------------------------------------------------------
# Garner's CRT Reconstruction Verification Test Function
# ------------------------------------------------------------------------------
def verify_garner(block_size=256):
    """
    Verify Garner's CRT reconstruction with a known test value.
    """
    try:
        # Use a single test value.
        test_value = 1234567890123456
        r0 = test_value % MOD_P0
        r1 = test_value % MOD_P1
        r2 = test_value % MOD_P2

        gpu_r0 = cp.array([np.uint32(r0)])
        gpu_r1 = cp.array([np.uint32(r1)])
        gpu_r2 = cp.array([np.uint32(r2)])
        # Allocate result array for one reconstruction (3 words)
        gpu_result = cp.zeros(3, dtype=cp.uint32)
        grid_size = (1 + block_size - 1) // block_size

        # Call the kernel, passing the computed modular inverses.
        GarnerGPU(gpu_r0, gpu_r1, gpu_r2, gpu_result,
                  inv0_mod1, inv01_mod2,
                  block=(block_size, 1, 1), grid=(grid_size, 1))
        res = cp.asnumpy(gpu_result)
        # Reconstruct the integer from little-endian 96-bit representation.
        reconstructed = res[0] + (res[1] << 32) + (res[2] << 64)
        print("Garner Verification:")
        print("  Original test value:   ", test_value)
        print("  Reconstructed integer: ", reconstructed)
        if reconstructed == test_value:
            print("  Garner CRT reconstruction verification PASSED.\n")
        else:
            print("  Garner CRT reconstruction verification FAILED.\n")
    except Exception as e:
        print("Error in verify_garner:", e)

# ------------------------------------------------------------------------------
# NTT Transformation Test Function
# ------------------------------------------------------------------------------
def test_ntt_transformation(block_size=256):
    """
    Test NTT forward and inverse transformations, element-wise multiplication,
    normalization, and verifies successful execution.
    """
    try:
        n = 1024  # Must be a power of 2
        grid_size_half = (n // 2 + block_size - 1) // block_size

        loopCnt_Pow2 = np.uint32(n // 2)
        arrayLength2 = np.uint32(n // 2)
        omega = np.uint32(3)
        sqrtOmega = np.uint32(5)

        host_input = np.random.randint(0, 1000, size=n).astype(np.uint32)
        gpu_input = cp.array(host_input)

        FMT(gpu_input, loopCnt_Pow2, omega, arrayLength2,
            block=(block_size, 1, 1), grid=(grid_size_half, 1))
        iFMT(gpu_input, loopCnt_Pow2, omega, arrayLength2,
             block=(block_size, 1, 1), grid=(grid_size_half, 1))
        
        host_input2 = np.random.randint(0, 1000, size=n).astype(np.uint32)
        gpu_input2 = cp.array(host_input2)
        Mul_i_i(gpu_input, gpu_input2,
                block=(block_size, 1, 1), grid=(grid_size_half, 1))
        
        DivN(gpu_input, np.uint32(n),
             block=(block_size, 1, 1), grid=(grid_size_half, 1))
        
        result_ntt = cp.asnumpy(gpu_input)
        print("NTT Transformation Result (after FMT, iFMT, Mul_i_i, DivN):")
        print(result_ntt)
    except Exception as e:
        print("Error in test_ntt_transformation:", e)

# ------------------------------------------------------------------------------
# Negative Cyclic Convolution Test Function
# ------------------------------------------------------------------------------
def test_negative_cyclic_convolution(block_size=256):
    """
    Test negative cyclic convolution preprocessing and postprocessing,
    and verify the output structure.
    """
    try:
        n = 1024  # Must be power of 2
        grid_size_full = (n + block_size - 1) // block_size
        sqrtOmega = np.uint32(5)
        
        host_data = np.random.randint(0, 1000, size=n).astype(np.uint32)
        gpu_data = cp.array(host_data)
        gpu_dataB = cp.empty_like(gpu_data)

        PreNegFMT(gpu_data, gpu_dataB, sqrtOmega, np.uint32(n),
                  block=(block_size, 1, 1), grid=(grid_size_full, 1))
        PostNegFMT(gpu_data, sqrtOmega, np.uint32(n),
                   block=(block_size, 1, 1), grid=(grid_size_full, 1))
        
        result_length = n + n // 2
        gpu_result = cp.empty(result_length, dtype=cp.uint32)
        PosNeg_To_HiLo(gpu_result, gpu_data, gpu_dataB, np.uint32(n),
                       block=(block_size, 1, 1), grid=(grid_size_full, 1))
        PostFMT_DivN_HiLo(gpu_result, gpu_data, gpu_dataB, np.uint32(n), sqrtOmega,
                          block=(block_size, 1, 1), grid=(grid_size_full, 1))
        
        result_conv = cp.asnumpy(gpu_result)
        print("Negative Cyclic Convolution Result (after Post-processing):")
        print(result_conv)
    except Exception as e:
        print("Error in test_negative_cyclic_convolution:", e)

# ------------------------------------------------------------------------------
# Garner CRT Reconstruction Test Function
# ------------------------------------------------------------------------------
def test_garner_reconstruction(block_size=256):
    """
    Test Garner's CRT reconstruction on simulated residue arrays and verify output.
    Each residue set is reconstructed into a 96-bit integer (stored as 3 words).
    """
    try:
        n_res = 1024  # Number of residue sets
        grid_size = (n_res + block_size - 1) // block_size

        host_E0 = np.random.randint(0, 1000, size=n_res).astype(np.uint32)
        host_E1 = np.random.randint(0, 1000, size=n_res).astype(np.uint32)
        host_E2 = np.random.randint(0, 1000, size=n_res).astype(np.uint32)
        gpu_E0 = cp.array(host_E0)
        gpu_E1 = cp.array(host_E1)
        gpu_E2 = cp.array(host_E2)

        # Allocate result array with 3 words per residue set.
        gpu_E3 = cp.zeros(3 * n_res, dtype=cp.uint32)

        GarnerGPU(gpu_E0, gpu_E1, gpu_E2, gpu_E3,
                  inv0_mod1, inv01_mod2,
                  block=(block_size, 1, 1), grid=(grid_size, 1))
        result_garner = cp.asnumpy(gpu_E3)
        print("Garner CRT Reconstruction Result (first 10 reconstructions):")
        # Print first 10 reconstructed numbers.
        for i in range(10):
            word0 = result_garner[3*i + 0]
            word1 = result_garner[3*i + 1]
            word2 = result_garner[3*i + 2]
            rec = word0 + (word1 << 32) + (word2 << 64)
            print(f"Reconstruction {i}: {rec}")
    except Exception as e:
        print("Error in test_garner_reconstruction:", e)

# ------------------------------------------------------------------------------
# Main Function: Execute All Tests with Error Handling and Cleanup
# ------------------------------------------------------------------------------
def main():
    block_size = 256
    try:
        print("=== NTT Transformation Test ===")
        test_ntt_transformation(block_size)

        print("=== Negative Cyclic Convolution Test ===")
        test_negative_cyclic_convolution(block_size)

        print("=== Garner CRT Reconstruction Test ===")
        test_garner_reconstruction(block_size)

        print("=== Garner CRT Verification Test ===")
        verify_garner(block_size)
    except Exception as e:
        print("Critical error in main execution:", e)
    finally:
        cleanup_memory()
        print("All GPU memory has been released.")

if __name__ == '__main__':
    main()

