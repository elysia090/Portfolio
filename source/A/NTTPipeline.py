"""
End-to-End Test for NTT and Garner CRT Reconstruction Pipeline

This test runs the entire workflow:
  1. It generates a random input and applies one stage of forward NTT (FMT) and its inverse (iFMT),
     followed by pointwise multiplication and normalization.
  2. It demonstrates negative cyclic convolution processing.
  3. It executes Garner’s CRT reconstruction on simulated residue arrays.
  4. It performs a verification test for Garner’s CRT reconstruction using a known test value.

The output from the test is printed to the console.
"""

import cupy as cp
import numpy as np
import pycuda.driver as cuda
import pycuda.compiler as compiler
import pycuda.autoinit

# ------------------------------------------------------------------------------
# CUDA Kernel Code (Faithfully translated and refactored)
# ------------------------------------------------------------------------------
kernel_code = r'''
#ifndef MODP
#define MODP 998244353U
#endif

#define ulong unsigned long long
#define uint unsigned int

//------------------------
// Modular Arithmetic
//------------------------

// modMultiply: (a * b) % MODP using 64-bit arithmetic.
__device__ uint modMultiply(uint a, uint b) {
    ulong tmp = ((ulong)(__umulhi(a, b))) * (1ULL << 32) + (ulong)(a * b);
    return (uint)(tmp % MODP);
}

// modExp: (a^b) % MODP.
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

// divideByN: Computes a/arrayLength modulo MODP.
__device__ uint divideByN(uint a, uint arrayLength) {
    uint quotient = a / arrayLength;
    uint remainder = a - quotient * arrayLength;
    uint pn = MODP / arrayLength;
    if(remainder != 0)
        quotient += (arrayLength - remainder) * pn + 1;
    return quotient;
}

//------------------------
// NTT Kernels
//------------------------

// FMT: Forward NTT (butterfly stage).
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

// iFMT: Inverse NTT (butterfly stage).
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

// Mul_i_i: Element-wise multiplication.
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

// PreNegFMT: Multiply a[i] by modExp(sqrtOmega, i).
__global__ void PreNegFMT(uint *arrayA, uint *arrayB, uint sqrtOmega, uint arrayLength) {
    uint idx = threadIdx.x + blockIdx.x * 256;
    arrayA[idx] %= MODP;
    uint factor = modExp(sqrtOmega, idx);
    arrayB[idx] = modMultiply(arrayA[idx], factor);
}

// PostNegFMT: Multiply each element by modExp(sqrtOmega, 2*arrayLength - i).
__global__ void PostNegFMT(uint *arrayA, uint sqrtOmega, uint arrayLength) {
    uint idx = threadIdx.x + blockIdx.x * 256;
    uint factor = modExp(sqrtOmega, arrayLength * 2 - idx);
    arrayA[idx] = modMultiply(arrayA[idx], factor);
}

// PosNeg_To_HiLo: Combine two arrays into high and low parts.
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

// Fixed moduli.
#define MOD_P0 469762049ULL
#define MOD_P1 1811939329ULL
#define MOD_P2 2013265921ULL

// GarnerGPU: Reconstructs a large integer from residues.
__global__ void GarnerGPU(uint *arrayE0, uint *arrayE1, uint *arrayE2, uint *arrayE3, uint arrayLength2) {
    int idx = threadIdx.x + blockIdx.x * 256;
    ulong r0 = arrayE0[idx];
    ulong r1 = arrayE1[idx];
    ulong r2 = arrayE2[idx];
    ulong x = r0;
    ulong diff1 = (r1 >= x) ? (r1 - x) : (r1 + MOD_P1 - x);
    x = x + ((diff1 * 1540148431ULL) % MOD_P1) * MOD_P0;
    ulong diff2 = (r2 + MOD_P2 >= (x % MOD_P2)) ? (r2 + MOD_P2 - (x % MOD_P2)) : (r2 - (x % MOD_P2));
    if(diff2 >= MOD_P2) diff2 -= MOD_P2;
    ulong w1 = (diff2 * 1050399624ULL) % MOD_P2;
    ulong w2 = MOD_P0 * MOD_P1;
    ulong low = w1 * w2;
    ulong high = __umul64hi(w1, w2);
    if(low > (low + x))
        high++;
    low += x;
    uint word0 = (uint)(low & 0xFFFFFFFFULL);
    uint word1 = (uint)((low >> 32) & 0xFFFFFFFFULL);
    uint word2 = (uint)(high & 0xFFFFFFFFULL);
    uint last0 = atomicAdd(&arrayE3[idx + 0], word0);
    if((last0 + word0) < last0) {
        word1++;
        if(word1 == 0)
            word2++;
    }
    if(word1 != 0) {
        uint last1 = atomicAdd(&arrayE3[idx + 1], word1);
        if((last1 + word1) < last1)
            word2++;
    }
    if(word2 != 0) {
        uint last2 = atomicAdd(&arrayE3[idx + 2], word2);
        uint carry = 0;
        if((last2 + word2) < last2)
            carry = 1;
        for (int i = idx + 3; i < arrayLength2 && carry; i++) {
            uint lastVal = atomicAdd(&arrayE3[i], carry);
            carry = (lastVal + carry) < lastVal ? 1 : 0;
        }
    }
}
'''

# ------------------------------------------------------------------------------
# Compile Kernels
# ------------------------------------------------------------------------------
module = compiler.SourceModule(kernel_code, options=["-use_fast_math"])
FMT               = module.get_function("FMT")
iFMT              = module.get_function("iFMT")
Mul_i_i           = module.get_function("Mul_i_i")
DivN              = module.get_function("DivN")
PreNegFMT         = module.get_function("PreNegFMT")
PostNegFMT        = module.get_function("PostNegFMT")
PosNeg_To_HiLo    = module.get_function("PosNeg_To_HiLo")
PostFMT_DivN_HiLo = module.get_function("PostFMT_DivN_HiLo")
GarnerGPU         = module.get_function("GarnerGPU")

# ------------------------------------------------------------------------------
# Verification for Garner's CRT Reconstruction
# ------------------------------------------------------------------------------
def verify_garner(block_size=256):
    MOD_P0 = 469762049
    MOD_P1 = 1811939329
    MOD_P2 = 2013265921
    test_value = 1234567890123456
    r0 = test_value % MOD_P0
    r1 = test_value % MOD_P1
    r2 = test_value % MOD_P2

    gpu_r0 = cp.array([np.uint32(r0)])
    gpu_r1 = cp.array([np.uint32(r1)])
    gpu_r2 = cp.array([np.uint32(r2)])
    gpu_result = cp.zeros(1 + 10, dtype=cp.uint32)
    grid_size = (1 + block_size - 1) // block_size
    GarnerGPU(gpu_r0, gpu_r1, gpu_r2, gpu_result, np.uint32(1 + 10),
              block=(block_size, 1, 1), grid=(grid_size, 1))
    res = cp.asnumpy(gpu_result)
    reconstructed = res[0] + (res[1] << 32) + (res[2] << 64)
    print("Garner Verification:")
    print("Original test value:    ", test_value)
    print("Reconstructed integer:  ", reconstructed)
    if reconstructed == test_value:
        print("Garner CRT reconstruction verification PASSED.")
    else:
        print("Garner CRT reconstruction verification FAILED.")

# ------------------------------------------------------------------------------
# End-to-End Test
# ------------------------------------------------------------------------------
def main():
    block_size = 256

    # ----- NTT Transformation Test -----
    n = 1024  # Must be power of 2
    grid_size_n2 = (n // 2 + block_size - 1) // block_size  # For kernels on n/2 elements
    grid_size_n = (n + block_size - 1) // block_size         # For kernels on n elements

    loopCnt_Pow2 = np.uint32(n // 2)
    arrayLength2 = np.uint32(n // 2)
    omega = np.uint32(3)
    sqrtOmega = np.uint32(5)

    host_array = np.random.randint(0, 1000, size=n).astype(np.uint32)
    gpu_array = cp.array(host_array)

    # Forward and inverse NTT stage.
    FMT(gpu_array, loopCnt_Pow2, omega, arrayLength2,
        block=(block_size, 1, 1), grid=(grid_size_n2, 1))
    iFMT(gpu_array, loopCnt_Pow2, omega, arrayLength2,
         block=(block_size, 1, 1), grid=(grid_size_n2, 1))
    
    # Element-wise multiplication test.
    host_array2 = np.random.randint(0, 1000, size=n).astype(np.uint32)
    gpu_array2 = cp.array(host_array2)
    Mul_i_i(gpu_array, gpu_array2,
            block=(block_size, 1, 1), grid=(grid_size_n2, 1))
    
    # Normalization.
    DivN(gpu_array, np.uint32(n),
         block=(block_size, 1, 1), grid=(grid_size_n2, 1))
    
    result_ntt = cp.asnumpy(gpu_array)
    print("NTT Transformation Result (after FMT, iFMT, Mul_i_i, DivN):")
    print(result_ntt)

    # ----- Negative Cyclic Convolution Test -----
    gpu_arrayB = cp.empty_like(gpu_array)
    PreNegFMT(gpu_array, gpu_arrayB, sqrtOmega, np.uint32(n),
              block=(block_size, 1, 1), grid=(grid_size_n, 1))
    PostNegFMT(gpu_array, sqrtOmega, np.uint32(n),
               block=(block_size, 1, 1), grid=(grid_size_n, 1))
    
    gpu_result = cp.empty(n + n // 2, dtype=cp.uint32)
    PosNeg_To_HiLo(gpu_result, gpu_array, gpu_arrayB, np.uint32(n),
                   block=(block_size, 1, 1), grid=(grid_size_n, 1))
    PostFMT_DivN_HiLo(gpu_result, gpu_array, gpu_arrayB, np.uint32(n), sqrtOmega,
                      block=(block_size, 1, 1), grid=(grid_size_n, 1))
    
    # ----- Garner CRT Reconstruction Test -----
    n_res = 1024
    host_E0 = np.random.randint(0, 1000, size=n_res).astype(np.uint32)
    host_E1 = np.random.randint(0, 1000, size=n_res).astype(np.uint32)
    host_E2 = np.random.randint(0, 1000, size=n_res).astype(np.uint32)
    gpu_E0 = cp.array(host_E0)
    gpu_E1 = cp.array(host_E1)
    gpu_E2 = cp.array(host_E2)
    result_length = n_res + 10
    gpu_E3 = cp.zeros(result_length, dtype=cp.uint32)
    grid_size_recon = (n_res + block_size - 1) // block_size
    GarnerGPU(gpu_E0, gpu_E1, gpu_E2, gpu_E3, np.uint32(result_length),
              block=(block_size, 1, 1), grid=(grid_size_recon, 1))
    result_garner = cp.asnumpy(gpu_E3)
    print("Garner CRT Reconstruction Result:")
    print(result_garner)
    
    # ----- Verification -----
    verify_garner(block_size)

if __name__ == '__main__':
    main()
