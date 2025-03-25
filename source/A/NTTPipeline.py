#!/usr/bin/env python3
"""
Enhanced GPU-Accelerated Multi-Precision Arithmetic (up to ~2^300)
using Batched NTT + Chinese Remainder Theorem (Garner's algorithm),
plus a benchmark test.

Overview
--------
- BigInt class implements addition, subtraction, and GPU-accelerated
  multiplication via NTT + CRT.
- We use ~10 prime moduli so that their product exceeds 2^300, allowing
  us to handle large (~300-bit) multiplications.
- A benchmark function tests multiplication speed for randomly generated
  BigInts of various bit sizes.
"""

import cupy as cp
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from typing import List, Dict
import time
import random

# -----------------------------------------
# Global Constants
# -----------------------------------------
BASE: int = 1 << 16
BASE_MASK: int = BASE - 1

# Larger set of prime moduli, so their product > 2^300
MODS: List[int] = [
    167772161,
    469762049,
    754974721,
    998244353,
    1004535809,
    1107296257,
    2013265921,
    2281701377,
    3221225473,
    3489660929
]
# We assume '3' or small integers are valid NTT roots for each prime in this demo
ROOTS: List[int] = [3, 3, 11, 3, 3, 3, 3, 3, 3, 3]

_bitrev_cache: Dict[int, cp.ndarray] = {}

# -----------------------------------------
# Bit Reversal and Batched NTT
# -----------------------------------------
def get_bit_reversal_indices(n: int) -> cp.ndarray:
    """
    Compute and cache bit-reversal permutation indices for length n.
    """
    if n in _bitrev_cache:
        return _bitrev_cache[n]
    indices_np = np.arange(n, dtype=np.int32)
    rev_np = np.zeros(n, dtype=np.int32)
    log_n = n.bit_length() - 1
    for i in range(n):
        j = 0
        x = i
        for _ in range(log_n):
            j = (j << 1) | (x & 1)
            x >>= 1
        rev_np[i] = j
    rev_cp = cp.asarray(rev_np)
    _bitrev_cache[n] = rev_cp
    return rev_cp


def batched_ntt(polys: cp.ndarray, mods: List[int], roots: List[int]) -> cp.ndarray:
    """
    Batched forward NTT for a 2D array of shape (k, n).
    Each row i is processed with modulus mods[i] and root roots[i].
    """
    k, n = polys.shape
    rev_indices = get_bit_reversal_indices(n)

    # Apply bit-reversal permutation
    for row_i in range(k):
        polys[row_i] = polys[row_i][rev_indices]

    m = 2
    while m <= n:
        half_m = m >> 1
        for row_i in range(k):
            mod = mods[row_i]
            root = roots[row_i]
            data = polys[row_i]

            # w_m = root^((mod-1)//m) mod mod
            w_m = pow(root, (mod - 1) // m, mod)
            exps = cp.arange(half_m, dtype=cp.int64)
            stage_twiddles = cp.power(w_m, exps, dtype=cp.int64) % mod

            data = data.reshape(-1, m)
            left = data[:, :half_m]
            right = data[:, half_m:]
            right = (right * stage_twiddles) % mod
            data[:, :half_m] = (left + right) % mod
            data[:, half_m:] = (left - right) % mod
            polys[row_i] = data.reshape(-1)
        m <<= 1
    return polys


def batched_intt(polys: cp.ndarray, mods: List[int], roots: List[int]) -> cp.ndarray:
    """
    Batched inverse NTT. Each row uses the inverse of its root and
    multiplies by n^{-1} mod to complete the inverse transform.
    """
    k, n = polys.shape
    inv_roots = [pow(r, m - 2, m) for r, m in zip(roots, mods)]
    polys = batched_ntt(polys, mods, inv_roots)
    for i in range(k):
        mod = mods[i]
        inv_n = pow(n, mod - 2, mod)
        polys[i] = (polys[i] * inv_n) % mod
    return polys


def batched_convolution(seq_a: np.ndarray,
                        seq_b: np.ndarray,
                        mods: List[int],
                        roots: List[int]) -> np.ndarray:
    """
    Perform polynomial convolution for seq_a and seq_b under each modulus,
    returning an array of shape (k, len(seq_a)+len(seq_b)-1).
    """
    conv_len = len(seq_a) + len(seq_b) - 1
    n = 1
    while n < conv_len:
        n <<= 1

    k = len(mods)
    A_batch = cp.zeros((k, n), dtype=cp.int64)
    B_batch = cp.zeros((k, n), dtype=cp.int64)

    for i, m in enumerate(mods):
        seq_a_mod = np.remainder(seq_a, m)
        seq_b_mod = np.remainder(seq_b, m)
        A_batch[i, :len(seq_a)] = cp.asarray(seq_a_mod, dtype=cp.int64)
        B_batch[i, :len(seq_b)] = cp.asarray(seq_b_mod, dtype=cp.int64)

    A_ntt = batched_ntt(A_batch, mods, roots)
    B_ntt = batched_ntt(B_batch, mods, roots)

    for i in range(k):
        A_ntt[i] = (A_ntt[i] * B_ntt[i]) % mods[i]

    C_batch = batched_intt(A_ntt, mods, roots)
    return C_batch[:, :conv_len].get()


def crt_combine(residues: np.ndarray, mods: List[int]) -> np.ndarray:
    """
    Combine residues from shape (k, length) using CRT, returning an array
    of length 'length' with Python int elements (dtype=object).
    """
    k = len(mods)
    length = residues.shape[1]

    M = 1
    for m in mods:
        M *= m

    out = np.zeros(length, dtype=object)
    M_list = []
    inv_list = []
    for i in range(k):
        Mi = M // mods[i]
        M_list.append(Mi)
        inv_list.append(pow(Mi, -1, mods[i]))

    for idx in range(length):
        val = 0
        for i in range(k):
            ri = int(residues[i, idx])
            val = (val + ri * M_list[i] * inv_list[i]) % M
        out[idx] = val

    return out  # keep dtype=object to hold Python ints > 64 bits


# -----------------------------------------
# BigInt Class
# -----------------------------------------
class BigInt:
    """
    Multi-precision integer with base 2^16 digits in little-endian order.
    Supports addition, subtraction, and GPU-accelerated multiplication
    via batched NTT + CRT.
    """
    __slots__ = ['value']

    def __init__(self, val: int) -> None:
        self.value = []
        if val == 0:
            self.value = [0]
        else:
            tmp = val
            while tmp > 0:
                self.value.append(tmp & BASE_MASK)
                tmp >>= 16

    def to_int(self) -> int:
        out = 0
        for digit in reversed(self.value):
            out = (out << 16) + digit
        return out

    def __add__(self, other: 'BigInt') -> 'BigInt':
        length = max(len(self.value), len(other.value))
        carry = 0
        result_digits = []
        for i in range(length):
            a_dig = self.value[i] if i < len(self.value) else 0
            b_dig = other.value[i] if i < len(other.value) else 0
            s = a_dig + b_dig + carry
            result_digits.append(s & BASE_MASK)
            carry = s >> 16
        if carry:
            result_digits.append(carry)
        res = BigInt(0)
        res.value = result_digits
        return res

    def __sub__(self, other: 'BigInt') -> 'BigInt':
        length = max(len(self.value), len(other.value))
        borrow = 0
        result_digits = []
        for i in range(length):
            a_dig = self.value[i] if i < len(self.value) else 0
            b_dig = other.value[i] if i < len(other.value) else 0
            s = a_dig - b_dig - borrow
            if s < 0:
                s += BASE
                borrow = 1
            else:
                borrow = 0
            result_digits.append(s)
        while len(result_digits) > 1 and result_digits[-1] == 0:
            result_digits.pop()
        res = BigInt(0)
        res.value = result_digits
        return res

    def __mul__(self, other: 'BigInt') -> 'BigInt':
        # Convert digits to np arrays
        a_arr = np.array(self.value, dtype=np.int64)
        b_arr = np.array(other.value, dtype=np.int64)

        # 1) Convolve under each modulus
        poly_residues = batched_convolution(a_arr, b_arr, MODS, ROOTS)

        # 2) CRT combine to get big coefficients as Python ints
        combined = crt_combine(poly_residues, MODS)

        # 3) Extract base-2^16 digits
        result_digits = []
        carry = 0
        for coeff_obj in combined:
            s = coeff_obj + carry
            digit = s & BASE_MASK
            carry = s >> 16
            result_digits.append(digit)
        while carry:
            result_digits.append(carry & BASE_MASK)
            carry >>= 16

        prod = BigInt(0)
        prod.value = result_digits
        return prod

    def __str__(self) -> str:
        return str(self.to_int())


# -------------------------------------------------
# Benchmark Function
# -------------------------------------------------
def benchmark(iterations: int = 3) -> None:
    """
    Benchmark BigInt multiplication at several bit sizes.
    We randomly generate two BigInts of a given bit length,
    multiply them, and measure the elapsed GPU-accelerated time.

    :param iterations: How many times to repeat each test to get an average
    """
    bit_sizes = [256, 512, 1024, 2048]
    print("Benchmarking BigInt multiplication with GPU-based NTT...\n")
    for bits in bit_sizes:
        elapsed_times = []
        for _ in range(iterations):
            # Generate two random integers, each ~ 'bits' wide
            a_val = random.getrandbits(bits)
            b_val = random.getrandbits(bits)
            A = BigInt(a_val)
            B = BigInt(b_val)

            # Measure multiplication time
            start = time.time()
            _ = A * B  # just discard the product
            end = time.time()
            elapsed_times.append(end - start)

        avg_time = sum(elapsed_times) / iterations
        print(f"  Bit size ~ {bits:4d} bits:  {avg_time:.6f} seconds (avg of {iterations} runs)")


# -------------------------------------------------
# Example Usage
# -------------------------------------------------
if __name__ == "__main__":
    # Demo with ~2^300 range
    a_val = (1 << 300) + 12345678901234567890
    b_val = (1 << 299) + 98765432109876543210

    a = BigInt(a_val)
    b = BigInt(b_val)

    print("Demo with ~2^300 scale numbers:")
    print("a =", a)
    print("b =", b)

    sum_ab = a + b
    print("a + b =", sum_ab)

    if a.to_int() >= b.to_int():
        diff_ab = a - b
    else:
        diff_ab = b - a
    print("a - b =", diff_ab)

    prod_ab = a * b
    print("a * b =", prod_ab)

    # Run benchmark
    print("\nRunning benchmark tests...")
    benchmark(iterations=3)
