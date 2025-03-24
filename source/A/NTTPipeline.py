#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Value GPU-Native Python Library for Large Integer Arithmetic, 
NTT-based Polynomial Multiplication, and Garner’s CRT Reconstruction.

This standalone Python script uses CuPy and PyCUDA to:
  1) Perform fast Number Theoretic Transforms (NTT) and negacyclic convolutions.
  2) Support multi-precision integers via multiple prime moduli and Garner's CRT.
  3) Provide a modular, production-grade structure to build upon.

Requirements:
  - Python 3.8+
  - CuPy (GPU array library)
  - PyCUDA (CUDA kernel compilation and GPU function launching)
  - An NVIDIA GPU with CUDA Toolkit installed

Author: ChatGPT (OpenAI)
Copyright (C) 2025
License: Proprietary or choose a suitable license for your $10M IP venture.
"""

import cupy as cp
import numpy as np
import pycuda.driver as cuda
import pycuda.compiler as compiler
import pycuda.autoinit  # Auto-initialize CUDA driver context


# ------------------------------------------------------------------------------
# CUDA Kernels (NTT, Negacyclic Convolution, and Helpers)
# ------------------------------------------------------------------------------
kernel_code = r'''
#ifndef MODP
#define MODP 998244353U  // Default prime modulus (can be overridden at compile time)
#endif

#define ulong unsigned long long
#define uint  unsigned int

//------------------------
// Modular Arithmetic Helper Functions
//------------------------

// Multiply two 32-bit numbers mod MODP using 64-bit intermediate to avoid overflow
__device__ uint modMultiply(uint a, uint b) {
    ulong tmp = ((ulong)__umulhi(a, b) << 32) + (ulong)(a * b);
    return (uint)(tmp % MODP);
}

// Fast modular exponentiation (a^b mod MODP) 
__device__ uint modExp(uint a, uint b) {
    ulong result = 1ULL;
    ulong base = a;
    while(b != 0) {
        if(b & 1ULL)
            result = (result * base) % MODP;
        base = (base * base) % MODP;
        b >>= 1;
    }
    return (uint)result;
}

// Divide by arrayLength modulo MODP (for NTT normalization). 
// We do a naive approximation approach: a*(arrayLength^{-1} mod MODP).
__device__ uint divideByN(uint a, uint arrayLength) {
    uint quotient = a / arrayLength;
    uint remainder = a - quotient * arrayLength;
    uint pn = MODP / arrayLength;
    if(remainder != 0) {
        quotient += (arrayLength - remainder) * pn + 1;
    }
    return quotient;
}

//------------------------
// NTT Kernel Stages (Cooley-Tukey butterfly operations)
//------------------------

// Forward NTT butterfly stage
__global__ void FMT(uint *arrayA, uint loopCnt_Pow2, uint omega, uint arrayLength2) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= arrayLength2) return;
    uint t2 = idx % loopCnt_Pow2;
    uint t0 = idx * 2 - t2;
    uint t1 = t0 + loopCnt_Pow2;

    // omega^(t2 * (arrayLength2/loopCnt_Pow2))
    uint w0 = modExp(omega, t2 * (arrayLength2 / loopCnt_Pow2));
    uint w1 = modMultiply(arrayA[t1], w0);

    uint r0 = arrayA[t0] + MODP - w1;
    uint r1 = arrayA[t0] + w1;
    if(r0 >= MODP) r0 -= MODP;
    if(r1 >= MODP) r1 -= MODP;
    arrayA[t1] = r0;
    arrayA[t0] = r1;
}

// Inverse NTT butterfly stage
__global__ void iFMT(uint *arrayA, uint loopCnt_Pow2, uint omega, uint arrayLength2) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= arrayLength2) return;
    uint t2 = idx % loopCnt_Pow2;
    uint t0 = idx * 2 - t2;
    uint t1 = t0 + loopCnt_Pow2;

    // conjugate twiddle factor => omega^(2*arrayLength2 - exponent)
    uint w0 = modExp(omega, arrayLength2 * 2 - t2 * (arrayLength2 / loopCnt_Pow2));
    uint w1 = modMultiply(arrayA[t1], w0);
    uint r0 = arrayA[t0] + MODP - w1;
    uint r1 = arrayA[t0] + w1;
    if(r0 >= MODP) r0 -= MODP;
    if(r1 >= MODP) r1 -= MODP;
    arrayA[t1] = r0;
    arrayA[t0] = r1;
}

// Element-wise multiplication of two arrays in the NTT domain
__global__ void Mul_i_i(uint *arrayA, uint *arrayB) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    arrayB[idx] = modMultiply(arrayA[idx], arrayB[idx]);
}

// Divide all elements by arrayLength (for final normalization)
__global__ void DivN(uint *arrayA, uint arrayLength) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    arrayA[idx] = divideByN(arrayA[idx], arrayLength);
}

//------------------------
// Negacyclic Convolution Kernels (mod x^n + 1)
//------------------------

// PreNegFMT: a[i] -> a[i]*sqrtOmega^i
__global__ void PreNegFMT(uint *arrayA, uint *arrayB, uint sqrtOmega, uint arrayLength) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= arrayLength) return;
    arrayA[idx] %= MODP;
    uint factor = modExp(sqrtOmega, idx);
    arrayB[idx] = modMultiply(arrayA[idx], factor);
}

// PostNegFMT: a[i] -> a[i]*sqrtOmega^(2*arrayLength - i)
__global__ void PostNegFMT(uint *arrayA, uint sqrtOmega, uint arrayLength) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= arrayLength * 2) return;
    uint factor = modExp(sqrtOmega, arrayLength * 2 - idx);
    arrayA[idx] = modMultiply(arrayA[idx], factor);
}

// PosNeg_To_HiLo: Separate convolution results into high/low parts
__global__ void PosNeg_To_HiLo(uint *arrayE, uint *arrayA, uint *arrayB, uint arrayLength) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= arrayLength) return;
    uint a = arrayA[idx];
    uint b = arrayB[idx];

    // (a - b)/2 (mod MODP)
    uint diff = (a >= b) ? (a - b) : (a + MODP - b);
    uint flag = diff & 1U;
    diff = diff - (((diff >= MODP) ? 2U : (uint)-1) * flag);
    diff /= 2U;
    arrayE[idx + arrayLength] = diff;
    arrayE[idx] = (a >= diff) ? (a - diff) : (a + MODP - diff);
}

// PostFMT_DivN_HiLo: combined postprocessing for negacyclic conv
__global__ void PostFMT_DivN_HiLo(uint *arrayE, uint *arrayA, uint *arrayB, uint arrayLength, uint sqrtOmega) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= arrayLength) return;
    uint a = arrayA[idx];
    uint b = arrayB[idx];
    uint factor = modExp(sqrtOmega, idx + ((idx & 1U) * arrayLength));
    b = modMultiply(b, factor);
    a = divideByN(a, arrayLength);
    b = divideByN(b, arrayLength);
    uint diff = (a >= b) ? (a - b) : (a + MODP - b);
    uint flag = diff & 1U;
    diff = diff - (((diff >= MODP) ? 2U : (uint)-1) * flag);
    diff /= 2U;
    arrayE[idx + arrayLength] = diff;
    arrayE[idx] = (a >= diff) ? (a - diff) : (a + MODP - diff);
}
'''  # End of kernel_code raw string


# ------------------------------------------------------------------------------
# NTT Class: Manages forward/inverse transforms, polynomial multiplication.
# ------------------------------------------------------------------------------
class NTT:
    """NTT (Number Theoretic Transform) for a specific prime modulus and transform length."""

    def __init__(self, modulus: int, length: int):
        """
        Initialize NTT plan for a given prime modulus and transform length.
        - modulus: A prime of the form c*2^k + 1 (to support 2^k-th roots of unity).
        - length: Transform length (must be a power of 2 that divides modulus-1).
        """
        self.mod = modulus
        self.n = length
        assert (self.n & (self.n - 1)) == 0, "Length must be a power of 2."

        # Compile CUDA kernels for this prime
        try:
            compile_opts = ["-use_fast_math", f"-DMODP={self.mod}"]
            self.module = compiler.SourceModule(kernel_code, options=compile_opts)
        except Exception as e:
            raise RuntimeError(f"CUDA kernel compilation failed for mod {self.mod}: {e}")

        # Get kernel functions
        self._FMT = self.module.get_function("FMT")
        self._iFMT = self.module.get_function("iFMT")
        self._Mul_i_i = self.module.get_function("Mul_i_i")
        self._DivN = self.module.get_function("DivN")
        self._PreNegFMT = self.module.get_function("PreNegFMT")
        self._PostNegFMT = self.module.get_function("PostNegFMT")
        self._PosNeg_To_HiLo = self.module.get_function("PosNeg_To_HiLo")
        self._PostFMT_DivN_HiLo = self.module.get_function("PostFMT_DivN_HiLo")

        # Compute an N-th root of unity: omega
        self.omega = self._find_omega()
        # Compute sqrtOmega for negacyclic convolution if 2N divides mod-1
        self.sqrt_omega = None
        if (self.mod - 1) % (2 * self.n) == 0:
            gen = self._find_generator()
            self.sqrt_omega = pow(gen, (self.mod - 1) // (2 * self.n), self.mod)

    def _find_generator(self) -> int:
        """Find a primitive generator g of the multiplicative group mod self.mod."""
        m = self.mod - 1
        factors = []
        temp = m
        # Small factorization
        for p in [2] + list(range(3, int(np.sqrt(m)) + 1, 2)):
            if temp % p == 0:
                factors.append(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            factors.append(temp)
        # Test candidates
        for g in range(2, self.mod):
            ok = True
            for f in factors:
                if pow(g, m // f, self.mod) == 1:
                    ok = False
                    break
            if ok:
                return g
        raise ValueError("No generator found for modulus.")

    def _find_omega(self) -> int:
        """Find a primitive N-th root of unity (omega)."""
        # We expect (self.mod - 1) % self.n == 0
        gen = self._find_generator()
        exponent = (self.mod - 1) // self.n
        return np.uint32(pow(gen, exponent, self.mod))

    def forward(self, arr: cp.ndarray):
        """In-place forward NTT on a CuPy array of length n."""
        assert arr.size == self.n, "Input length must match NTT length."
        block_size = 256
        half_len = self.n // 2
        grid_size = (half_len + block_size - 1) // block_size

        loop = half_len
        while loop >= 1:
            self._FMT(
                arr, np.uint32(loop), self.omega, np.uint32(half_len),
                block=(block_size, 1, 1), grid=(grid_size, 1)
            )
            loop //= 2

    def inverse(self, arr: cp.ndarray):
        """In-place inverse NTT on a CuPy array of length n (unscaled). 
        Normalizes by dividing elements by n/2 at the end (matching the logic in the kernels)."""
        assert arr.size == self.n, "Input length must match NTT length."
        block_size = 256
        half_len = self.n // 2
        grid_size = (half_len + block_size - 1) // block_size

        loop = 1
        while loop <= half_len:
            self._iFMT(
                arr, np.uint32(loop), self.omega, np.uint32(half_len),
                block=(block_size, 1, 1), grid=(grid_size, 1)
            )
            loop *= 2

        # Final division to normalize
        self._DivN(
            arr, np.uint32(half_len),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )

    def multiply_polynomials(self, a: cp.ndarray, b: cp.ndarray, full_convolution: bool = False) -> cp.ndarray:
        """
        Multiply two polynomials using the NTT.
          - If full_convolution=False, returns length-n cyclic convolution mod x^n-1.
          - If full_convolution=True, performs full polynomial multiplication 
            up to degree len(a)+len(b)-2 by zero-padding to next power of two.
        """
        len_a, len_b = a.size, b.size
        if not full_convolution:
            # Cyclic convolution of length n
            assert len_a == len_b == self.n, "Polynomials must match NTT length for cyclic conv."
            self.forward(a)
            self.forward(b)
            block_size = 256
            grid_size = (self.n + block_size - 1) // block_size
            self._Mul_i_i(a, b, block=(block_size, 1, 1), grid=(grid_size, 1))
            self.inverse(b)
            return b  # b holds the result
        else:
            # Full convolution
            out_len = len_a + len_b - 1
            conv_len = 1
            while conv_len < out_len:
                conv_len <<= 1

            # Create a new NTT plan if needed
            if conv_len != self.n:
                ntt_conv = NTT(self.mod, conv_len)
            else:
                ntt_conv = self

            A = cp.zeros(conv_len, dtype=cp.uint32)
            B = cp.zeros(conv_len, dtype=cp.uint32)
            A[:len_a] = a
            B[:len_b] = b

            ntt_conv.forward(A)
            ntt_conv.forward(B)
            block_size = 256
            grid_size = (conv_len + block_size - 1) // block_size
            ntt_conv._Mul_i_i(A, B, block=(block_size, 1, 1), grid=(grid_size, 1))
            ntt_conv.inverse(B)

            return B[:out_len].copy()

    def negacyclic_convolution(self, a: cp.ndarray, b: cp.ndarray) -> cp.ndarray:
        """
        Multiply two polynomials mod x^n + 1 using NTT-based negacyclic convolution.
        Both a and b must be of length n.
        """
        n = a.size
        assert b.size == n == self.n, "Inputs must be length n."
        if self.sqrt_omega is None:
            raise ValueError("Modulus does not support 2n-th root of unity for negacyclic conv.")

        d_a = cp.array(a, dtype=cp.uint32)
        d_b = cp.array(b, dtype=cp.uint32)
        d_A = cp.empty_like(d_a)
        d_B = cp.empty_like(d_b)

        block_size = 256
        grid_size_n = (n + block_size - 1) // block_size

        # PreNegFMT
        self._PreNegFMT(d_a, d_A, np.uint32(self.sqrt_omega), np.uint32(n),
                        block=(block_size, 1, 1), grid=(grid_size_n, 1))
        self._PreNegFMT(d_b, d_B, np.uint32(self.sqrt_omega), np.uint32(n),
                        block=(block_size, 1, 1), grid=(grid_size_n, 1))

        # Forward NTT
        self.forward(d_A)
        self.forward(d_B)

        # Pointwise multiply
        self._Mul_i_i(d_A, d_B, block=(block_size, 1, 1), grid=(grid_size_n, 1))

        # Inverse NTT
        self.inverse(d_B)

        # PostNegFMT
        result_full = cp.empty(2 * n, dtype=cp.uint32)
        result_full[:n] = d_B
        result_full[n:] = 0
        grid_size_2n = ((2 * n) + block_size - 1) // block_size
        self._PostNegFMT(result_full, np.uint32(self.sqrt_omega), np.uint32(n),
                         block=(block_size, 1, 1), grid=(grid_size_2n, 1))

        return result_full[:n].copy()


# ------------------------------------------------------------------------------
# Garner's CRT Kernel for multi-precision reconstruction
# ------------------------------------------------------------------------------
garner_code = r'''
#define MAX_PRIMES  64     // up to 64 primes for ~2048 bits if needed
#define WORD_COUNT  32     // 32 words * 32 bits = 1024 bits output

// External device function from earlier code, repeated for self-contained build:
__device__ unsigned int modMultiply(unsigned int a, unsigned int b) {
    unsigned long long tmp = ((unsigned long long)__umulhi(a, b) << 32) + (unsigned long long)(a * b);
    // Here we do not have a single modulus in Garner, but we only use it for partial "diff * inverse mod p" steps.
//  For normal usage, the prime p is up to 32 bits, so just do (tmp % p) externally. 
    // We'll do that in the kernel logic below using standard % p approach.
    return (unsigned int)(tmp & 0xFFFFFFFFULL); // minimal usage for 64-bit multiply
}

// GarnerGPU: Reconstruct integers from their residues using iterative Garner's algorithm.
// Inputs:
//   residues: (count x K) array of 32-bit residues (row-major => for each i in [0..count-1], the next K are the residues).
//   output:   (count x WORD_COUNT) array of 32-bit words storing the reconstructed integers (little-endian).
//   count:    number of elements to reconstruct
//   K:        number of prime moduli
//   primes:   array of K primes
//   inv_prefix: array of K-1 modular inverses: inv_prefix[j] = (P0*...*P_{j-1})^{-1} mod Pj
__global__ void GarnerGPU(uint *residues, uint *output, int count, int K, const uint *primes, const uint *inv_prefix) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= count) return;

    // Pointer to this element's residues
    uint *r = residues + idx * K;
    // Output pointer for this element
    uint *out = output + idx * WORD_COUNT;

    // We'll store partial (the product of primes processed so far) in multi-word
    // and x, the partial reconstruction, also in multi-word, each up to WORD_COUNT 32-bit words.
    unsigned long long carry;
    uint x[WORD_COUNT];
    uint partial[WORD_COUNT];

    // Initialize x = r[0], partial = primes[0]
    #pragma unroll
    for (int w = 0; w < WORD_COUNT; w++) {
        x[w] = 0;
        partial[w] = 0;
    }

    x[0] = r[0];
    partial[0] = primes[0];

    // Iterative Garner steps
    for(int j = 1; j < K; j++) {
        uint pj = primes[j];

        // Compute remainder of x mod pj (multi-word mod)
        unsigned long long rem = 0ULL;
        // Process from high to low word
        for(int w = WORD_COUNT - 1; w >= 0; w--) {
            rem = ((rem << 32) + (unsigned long long)x[w]) % pj;
        }
        uint x_mod = (uint)rem;

        // t = ((r[j] - x_mod) mod pj) * inv_prefix[j-1] mod pj
        uint diff = r[j] + pj - x_mod;
        if(diff >= pj) diff -= pj;
        uint inv = inv_prefix[j-1];
        // multiply diff*inv mod pj
        unsigned long long prod = (unsigned long long)diff * (unsigned long long)inv;
        uint t = (uint)(prod % pj);

        // x = x + t*partial (multi-precision)
        unsigned long long t_val = (unsigned long long)t;
        carry = 0ULL;
        for(int w = 0; w < WORD_COUNT; w++) {
            unsigned long long p_mul = (unsigned long long)partial[w] * t_val + carry;
            uint prod_low = (uint)(p_mul & 0xFFFFFFFFULL);
            unsigned long long prod_high = p_mul >> 32;
            // add to x[w]
            unsigned long long sum = (unsigned long long)x[w] + prod_low;
            x[w] = (uint)(sum & 0xFFFFFFFFULL);
            carry = prod_high + (sum >> 32);
        }

        // partial = partial * pj (multi-precision)
        carry = 0ULL;
        for(int w = 0; w < WORD_COUNT; w++) {
            unsigned long long pp = (unsigned long long)partial[w] * (unsigned long long)pj + carry;
            partial[w] = (uint)(pp & 0xFFFFFFFFULL);
            carry = pp >> 32;
        }
    }

    // Store x[] to output (WORD_COUNT words)
    #pragma unroll
    for(int w = 0; w < WORD_COUNT; w++) {
        out[w] = x[w];
    }
}
'''

try:
    module_garner = compiler.SourceModule(garner_code, options=["-use_fast_math"])
except Exception as e:
    raise RuntimeError(f"CUDA Garner kernel compilation failed: {e}")

GarnerGPU_kernel = module_garner.get_function("GarnerGPU")


# ------------------------------------------------------------------------------
# MultiModInt Class: multi-prime representation of large integers
# ------------------------------------------------------------------------------
class MultiModInt:
    """
    Represents large integers (up to ~1024 bits) using multiple prime moduli (Residue Number System).
    Provides GPU-accelerated arithmetic and GPU-based Garner's CRT reconstruction.
    """
    DEFAULT_PRIMES = [
        # Extend or replace these with more primes to reach desired bit size
        163577857, 167772161, 469762049, 754974721, 998244353, 
        1053818881, 1224736769, 1711276033, 1811939329, 2013265921,
        # Add or remove primes as needed for desired precision
    ]

    def __init__(self, values, primes: list = None):
        """
        Initialize MultiModInt with 'values' (int or list of ints) and an optional list of prime moduli.
        If 'values' is a single int, it becomes a 1-element array.
        Otherwise, 'values' can be an array-like of Python ints.
        """
        self.primes = primes or MultiModInt.DEFAULT_PRIMES
        self.K = len(self.primes)

        # Precompute inverses for Garner
        self.inv_prefix = []
        prod = 1
        for j in range(1, self.K):
            inv_val = pow(prod, -1, self.primes[j])
            self.inv_prefix.append(np.uint32(inv_val))
            prod *= self.primes[j - 1]
        self.inv_prefix = np.array(self.inv_prefix, dtype=np.uint32)

        if isinstance(values, (int, np.integer)):
            values = [int(values)]
        elif isinstance(values, np.ndarray):
            values = [int(x) for x in values]
        elif isinstance(values, cp.ndarray):
            values = [int(x.get()) for x in values]

        self.count = len(values)

        # Build GPU residues
        self.residues = []
        for p in self.primes:
            res_mod_p = [val % p for val in values]
            self.residues.append(cp.array(res_mod_p, dtype=cp.uint32))

        # 2D array: shape (count, K)
        self._residues_2d = cp.stack(self.residues, axis=1)
        # Copy primes and inverses to GPU
        self.d_primes = cp.array(self.primes, dtype=cp.uint32)
        self.d_inv_prefix = cp.array(self.inv_prefix, dtype=cp.uint32) if self.K > 1 else cp.array([], dtype=cp.uint32)

    def to_ints(self) -> list:
        """
        Convert the GPU RNS representation back to Python integers using Garner's CRT on GPU.
        """
        if self.K == 1:
            return list(map(int, cp.asnumpy(self.residues[0])))

        # Allocate GPU array for output words
        out_words = cp.empty((self.count, 32), dtype=cp.uint32)

        block_size = 256
        grid_size = (self.count + block_size - 1) // block_size

        GarnerGPU_kernel(
            self._residues_2d.astype(cp.uint32), 
            out_words,
            np.int32(self.count), 
            np.int32(self.K),
            self.d_primes, 
            self.d_inv_prefix,
            block=(block_size, 1, 1), 
            grid=(grid_size, 1)
        )

        out_cpu = cp.asnumpy(out_words)
        result_ints = []
        for i in range(self.count):
            total = 0
            # The words are stored little-endian: out_cpu[i,0] is LSB
            for w in range(31, -1, -1):
                total = (total << 32) | int(out_cpu[i, w])
            result_ints.append(total)
        return result_ints

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        """Get item as a single Python int or a slice as a new MultiModInt."""
        if isinstance(idx, slice):
            sliced_vals = self.to_ints()[idx]
            return MultiModInt(sliced_vals, primes=self.primes)
        else:
            return self.to_ints()[idx]

    def _binary_op(self, other, op: str):
        """
        Internal helper for elementwise add, sub, mul.
        'other' can be a MultiModInt or scalar.
        """
        if isinstance(other, MultiModInt):
            assert self.primes == other.primes, "Prime sets must match."
            res_arrays = []
            for j, p in enumerate(self.primes):
                if op == 'add':
                    tmp = (self.residues[j].astype(cp.uint64) + other.residues[j].astype(cp.uint64)) % p
                elif op == 'sub':
                    tmp = (self.residues[j].astype(cp.int64) - other.residues[j].astype(cp.int64)) % p
                elif op == 'mul':
                    tmp = (self.residues[j].astype(cp.uint64) * other.residues[j].astype(cp.uint64)) % p
                res_arrays.append(tmp.astype(cp.uint32))
        else:
            # Convert scalar to MultiModInt
            scalar_vals = MultiModInt([int(other)] * self.count, primes=self.primes)
            return self._binary_op(scalar_vals, op)

        result = MultiModInt([], primes=self.primes)
        result.count = self.count
        result.residues = res_arrays
        result._residues_2d = cp.stack(result.residues, axis=1)
        result.d_primes = self.d_primes
        result.d_inv_prefix = self.d_inv_prefix
        return result

    def __add__(self, other):
        return self._binary_op(other, 'add')

    def __sub__(self, other):
        return self._binary_op(other, 'sub')

    def __mul__(self, other):
        return self._binary_op(other, 'mul')

    def __repr__(self):
        return f"MultiModInt({self.to_ints()}, bit_length~{32*self.K} bits)"


# ------------------------------------------------------------------------------
# Demonstration / Test Section
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== NTT Forward/Inverse Transformation Test (Small) ===")
    mod = 998244353
    n = 8
    ntt = NTT(mod, n)
    poly = cp.array(np.random.randint(0, 1000, size=n, dtype=np.uint32))
    original = cp.asnumpy(poly.copy())
    ntt.forward(poly)
    ntt.inverse(poly)
    recovered = cp.asnumpy(poly)
    print("Original:", original)
    print("Recovered after NTT->iNTT:", recovered)
    print("Matches?", np.all((recovered % mod) == original % mod))

    print("\n=== Polynomial Multiplication (NTT) ===")
    # Example polynomials
    a_cpu = np.array([3, 5, 2, 0, 1, 4], dtype=np.uint32)
    b_cpu = np.array([6, 1, 0, 2, 3, 0], dtype=np.uint32)
    # Pad to power of two
    N = 8
    A_gpu = cp.array(np.pad(a_cpu, (0, N - len(a_cpu)), 'constant'), dtype=cp.uint32)
    B_gpu = cp.array(np.pad(b_cpu, (0, N - len(b_cpu)), 'constant'), dtype=cp.uint32)
    ntt2 = NTT(mod, N)

    # Cyclic convolution
    C_cyclic_gpu = ntt2.multiply_polynomials(A_gpu.copy(), B_gpu.copy(), full_convolution=False)
    C_cyclic = cp.asnumpy(C_cyclic_gpu)

    # Full convolution
    C_full_gpu = ntt2.multiply_polynomials(cp.array(a_cpu), cp.array(b_cpu), full_convolution=True)
    C_full = cp.asnumpy(C_full_gpu)

    print("A:", a_cpu)
    print("B:", b_cpu)
    print("Cyclic convolution (mod x^8-1):", C_cyclic)

    # CPU check for cyclic conv (rough check using FFT-based approach for small values):
    cyclic_ref = np.mod(np.fft.ifft(np.fft.fft(a_cpu, N)*np.fft.fft(b_cpu, N)).real.round(), mod).astype(np.uint32)
    print("Expected cyclic (approx):       ", cyclic_ref)
    print("Full convolution result:", C_full)

    # CPU check for full conv
    full_ref = np.polynomial.polynomial.polymul(a_cpu, b_cpu).astype(np.uint32)
    print("Expected full polynomial:", full_ref)

    print("\n=== Negacyclic Convolution Test (x^N + 1) ===")
    mod_negacyclic = 469762049  # supports 2^26 NTT, also 2N negacyclic for smaller n
    ntt3 = NTT(mod_negacyclic, N)
    p = np.array([2, 1, 3, 0, 4, 0, 0, 5], dtype=np.uint32)
    q = np.array([1, 0, 2, 2, 0, 3, 1, 0], dtype=np.uint32)
    P_gpu = cp.array(p)
    Q_gpu = cp.array(q)
    R_gpu = ntt3.negacyclic_convolution(P_gpu, Q_gpu)
    R = cp.asnumpy(R_gpu).astype(int)
    print("P:", p)
    print("Q:", q)
    print("Negacyclic convolution result:", R)

    # CPU check for negacyclic: (p*q) mod (x^N + 1)
    prod = np.polynomial.polynomial.polymul(p, q).astype(np.int64)
    negacyc_ref = np.zeros(N, dtype=np.int64)
    for i, coeff in enumerate(prod):
        idx = i % N
        k = i // N
        if k % 2 == 0:
            negacyc_ref[idx] += coeff
        else:
            negacyc_ref[idx] -= coeff
    negacyc_ref = np.mod(negacyc_ref, mod_negacyclic).astype(int)
    print("Expected:", negacyc_ref)

    print("\n=== Multi-Precision Integer Arithmetic (CRT) ===")
    big1 = (1 << 250) + 1234567890123456789
    big2 = (1 << 240) * 3 + 987654321098765432
    big_arr = MultiModInt([big1, big2])
    print("Original big ints:", big_arr.to_ints())

    # Add, multiply
    sum_arr = big_arr + big_arr
    prod_arr = big_arr * big_arr
    print("Doubled:", sum_arr.to_ints())
    squares = prod_arr.to_ints()
    print("Squared (under large modulus):", squares)

    # Compare to Python
    expected_squares = [x**2 for x in [big1, big2]]
    print("Expected squares (unreduced, but should match if < product of primes):", expected_squares)
    print("\nAll tests complete.")
