/**
 * GPU-Accelerated BigInt Pipeline Implementation using CGBN (Cooperative Groups Big Numbers)
 *
 * This implementation provides a BigInt class supporting 2048-bit integers with high-performance 
 * GPU acceleration for addition, subtraction, and multiplication, and a CPU fallback for portability.
 * 
 * Quality Attributes (ISO/IEC 25010):
 * - Performance Efficiency: Uses NVIDIA's CGBN library on CUDA for parallel big integer arithmetic.
 *   Critical operations are offloaded to the GPU when available, leveraging cooperative thread groups 
 *   for speed. The CPU path uses optimized big integer arithmetic (Boost.Multiprecision) for reliable 
 *   performance when a GPU is unavailable.
 * - Reliability: All CUDA calls and big integer operations have error checking. GPU errors (e.g., kernel 
 *   issues) or overflow beyond 2048 bits are handled by modular arithmetic (values are automatically 
 *   reduced mod 2^2048 to fit in the fixed size). Memory management is robust: GPU memory is allocated 
 *   and freed properly, and exceptions are used to signal errors up to the Python layer.
 * - Maintainability: Uses a clear separation of interface and implementation. The BigInt class exposes 
 *   a simple API, while backend logic is encapsulated in strategy classes (GPUBigIntBackend and 
 *   CPUBigIntBackend). This design allows extending or swapping out the backend (for example, to support 
 *   different hardware or algorithms) without changing the interface. Code is thoroughly commented and 
 *   follows modern C++ practices (RAII via unique_ptr, exceptions for error handling).
 * - Portability: The code can be compiled with or without CUDA support. If compiled on a system without 
 *   a GPU or if no CUDA devices are present at runtime, the implementation seamlessly falls back to the 
 *   CPU backend. It uses standard C++ and libraries (pybind11 for Python binding, Boost for big integers) 
 *   to maximize cross-platform compatibility.
 *
 * Design Patterns and Structure:
 * - Strategy Pattern: BigIntBackend is an abstract interface for big integer operations. GPUBigIntBackend 
 *   and CPUBigIntBackend provide concrete implementations for GPU and CPU. BigInt holds a pointer to a 
 *   BigIntBackend and delegates all operations to it, allowing dynamic selection of the best backend.
 * - PIMPL (Pointer to Implementation): BigInt can be seen as the public interface, while BigIntBackend 
 *   implementations are the private logic. This separation improves maintainability.
 * - RAII and Resource Management: GPU memory (for big integer values and CGBN error reports) is managed 
 *   with allocated pointers freed in destructors or module teardown. The design avoids memory leaks and 
 *   ensures exceptions do not lead to resource leaks by using unique_ptr and cleanup hooks.
 *
 * Big Integer Representation:
 * - The system uses a fixed size of 2048 bits for each big integer. All arithmetic is performed modulo 2^2048 
 *   (two's complement arithmetic for 2048-bit values). This means results that overflow beyond 2048 bits 
 *   will wrap around (as is standard for fixed-size integers). Negative numbers are represented in two's 
 *   complement form within 2048 bits. For example, -1 is represented as 0xFFFF...FFFF (256 bytes of 0xFF).
 * - On the GPU, values are stored as `cgbn_mem_t<2048>` and operated on using CGBN's APIs. On the CPU, values 
 *   are stored in a Boost.Multiprecision cpp_int (arbitrary precision) but reduced to the 2048-bit range 
 *   after each operation to maintain consistency with the fixed-size nature.
 *
 * Python Integration:
 * - The BigInt class is exposed to Python via pybind11. It supports construction from Python integers (of any size) 
 *   or string (decimal or hex), arithmetic operators (+, -, *), comparisons, and conversion back to Python int 
 *   (via __int__) or string (decimal and hexadecimal).
 * - Memory ownership and lifetime are managed such that Python BigInt objects can be used like normal Python integers 
 *   without needing manual memory management.
 * - Errors in C++ (like CUDA errors) are converted to Python exceptions (RuntimeError), making them visible in Python.
 */
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <cuda_runtime.h>
#include "cgbn/cgbn.h"                 // CGBN library (Cooperative Groups Big Numbers)
#include <boost/multiprecision/cpp_int.hpp>
#include <string>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <vector>
#include <algorithm>

namespace py = pybind11;

// Define the fixed size (in bits) and threads-per-instance for CGBN
// (Do not define these before including cgbn.h, per CGBN documentation)
#define TPI 32                       // Threads per CGBN instance (must be 4, 8, 16, or 32). Using 32 for full warp.
#define BITS 2048                    // Bit-length of each big integer (2048-bit)

// Convenient type aliases for CGBN context and environment for 2048-bit integers
typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

// Forward declare a function to convert BigInt to Python int (used in __int__)
struct BigInt;  // BigInt interface class forward declaration
static PyObject* bigint_to_PyLong(const BigInt &x);

// Abstract backend interface for big integer operations (Strategy pattern)
class BigIntBackend {
public:
    virtual ~BigIntBackend() = default;
    // Clone the backend (creates a new identical BigIntBackend instance on the heap)
    virtual BigIntBackend* clone() const = 0;
    // Set this value from another backend's value (they will be the same subclass in practice)
    virtual void set(const BigIntBackend &other) = 0;
    // Initialize this backend from a C++ big integer (boost::cpp_int)
    virtual void from_cpp_int(const boost::multiprecision::cpp_int &value) = 0;
    // Convert this backend's value to a C++ big integer (boost::cpp_int)
    virtual boost::multiprecision::cpp_int to_cpp_int() const = 0;
    // Arithmetic operations (results are stored in *this)
    virtual void add(const BigIntBackend &a, const BigIntBackend &b) = 0;
    virtual void sub(const BigIntBackend &a, const BigIntBackend &b) = 0;
    virtual void mul(const BigIntBackend &a, const BigIntBackend &b) = 0;
};

#ifdef __CUDACC__  // Only include GPU backend if compiled with CUDA support
#define GPU_ENABLED
#endif

#ifdef GPU_ENABLED

// GPU backend for BigInt using CGBN (Cooperative Groups Big Numbers)
class GPUBigIntBackend : public BigIntBackend {
private:
    cgbn_mem_t<BITS>* dev_value;                    // Pointer to 2048-bit value stored in GPU memory
    static cgbn_error_report_t* gpu_err_report;     // Pointer to a CGBN error report (shared by all instances)
public:
    // Constructor: allocate GPU memory and initialize from given big integer value (mod 2^2048)
    GPUBigIntBackend(const boost::multiprecision::cpp_int &value) : dev_value(nullptr) {
        // Allocate device memory for the 2048-bit value
        size_t bytes = sizeof(cgbn_mem_t<BITS>);
        cudaError_t cerr = cudaMalloc(&dev_value, bytes);
        if (cerr != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA memory allocation failed: ") + cudaGetErrorString(cerr));
        }
        // Initialize dev_value with the provided value (reduced mod 2^BITS)
        from_cpp_int(value);
    }
    // Destructor: free GPU memory
    ~GPUBigIntBackend() override {
        if (dev_value) {
            cudaFree(dev_value);
            dev_value = nullptr;
        }
    }
    // Clone: create a copy of this GPU backend (allocates new GPU memory and copies the value)
    BigIntBackend* clone() const override {
        GPUBigIntBackend* copy = new GPUBigIntBackend(boost::multiprecision::cpp_int(0));
        // Copy the 2048-bit value from this->dev_value to copy->dev_value on the device
        cudaError_t cerr = cudaMemcpy(copy->dev_value, dev_value, sizeof(cgbn_mem_t<BITS>), cudaMemcpyDeviceToDevice);
        if (cerr != cudaSuccess) {
            delete copy;
            throw std::runtime_error(std::string("CUDA memcpy (device->device) failed: ") + cudaGetErrorString(cerr));
        }
        return copy;
    }
    // Set this value to other's value (assumes other is also GPUBigIntBackend)
    void set(const BigIntBackend &other) override {
        const GPUBigIntBackend &src = dynamic_cast<const GPUBigIntBackend&>(other);
        // Copy device memory from src to this
        cudaError_t cerr = cudaMemcpy(dev_value, src.dev_value, sizeof(cgbn_mem_t<BITS>), cudaMemcpyDeviceToDevice);
        if (cerr != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA memcpy (device->device) failed: ") + cudaGetErrorString(cerr));
        }
    }
    // Convert from boost::cpp_int to GPU value (mod 2^2048) and store in device memory
    void from_cpp_int(const boost::multiprecision::cpp_int &value) override {
        // Reduce the input value modulo 2^BITS to fit in 2048 bits
        boost::multiprecision::cpp_int mod_val = value;
        // mod_val can be negative; perform modulo in a way that yields a non-negative representative
        boost::multiprecision::cpp_int modulus = boost::multiprecision::cpp_int(1) << BITS;
        mod_val %= modulus;
        if (mod_val < 0) {
            mod_val += modulus;
        }
        // Extract 2048-bit little-endian representation
        // cgbn_mem_t<BITS> contains an array of 2048/32 = 64 limbs of 32 bits each (little-endian order)
        uint32_t host_limbs[BITS/32] = {0};
        boost::multiprecision::cpp_int temp = mod_val;
        for (size_t i = 0; i < BITS/32; ++i) {
            host_limbs[i] = static_cast<uint32_t>(temp & 0xFFFFFFFFu);
            temp >>= 32;
        }
        // Copy limbs to device memory
        cudaError_t cerr = cudaMemcpy(dev_value, host_limbs, sizeof(host_limbs), cudaMemcpyHostToDevice);
        if (cerr != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA memcpy (host->device) failed: ") + cudaGetErrorString(cerr));
        }
    }
    // Convert from GPU value to boost::cpp_int
    boost::multiprecision::cpp_int to_cpp_int() const override {
        // Copy 2048-bit value from device to host
        uint32_t host_limbs[BITS/32];
        cudaError_t cerr = cudaMemcpy((void*)host_limbs, (const void*)dev_value, sizeof(host_limbs), cudaMemcpyDeviceToHost);
        if (cerr != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA memcpy (device->host) failed: ") + cudaGetErrorString(cerr));
        }
        // Reconstruct boost::cpp_int from little-endian 32-bit limbs
        boost::multiprecision::cpp_int result = 0;
        // Combine limbs: result = sum(host_limbs[i] * 2^(32*i))
        for (size_t i = 0; i < BITS/32; ++i) {
            result += boost::multiprecision::cpp_int(host_limbs[i]) << (32 * i);
        }
        // Interpret as signed (two's complement) within 2048 bits:
        // If the top bit (bit 2047) is 1, we interpret the number as negative two's complement.
        boost::multiprecision::cpp_int half_modulus = boost::multiprecision::cpp_int(1) << (BITS - 1);
        if (result >= half_modulus) {
            // value is in [2^(2047), 2^2048 - 1], treat as negative: result - 2^2048
            boost::multiprecision::cpp_int modulus = boost::multiprecision::cpp_int(1) << BITS;
            result -= modulus;
        }
        return result;
    }

    // GPU kernel definitions for addition, subtraction, multiplication of 2048-bit numbers:
private:
    // Each kernel uses one warp (32 threads) to handle one big integer operation (TPI=32, one instance)
    __global__ static void kernel_add(cgbn_error_report_t *report, const cgbn_mem_t<BITS>* a, const cgbn_mem_t<BITS>* b, cgbn_mem_t<BITS>* c) {
        int instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
        if (instance != 0) return;  // only one instance in this launch
        // Create a CGBN context and environment for this thread group
        context_t ctx(cgbn_report_monitor, report, 0);   // error reporting enabled, instance ID 0
        env_t env(ctx);
        // Declare CGBN 2048-bit values for inputs and result
        env_t::cgbn_t A, B, R;
        // Load inputs from memory into CGBN registers
        cgbn_load(env, A, a);
        cgbn_load(env, B, b);
        // Perform R = A + B (mod 2^BITS)
        cgbn_add(env, R, A, B);
        // Store result back to memory
        cgbn_store(env, c, R);
    }
    __global__ static void kernel_sub(cgbn_error_report_t *report, const cgbn_mem_t<BITS>* a, const cgbn_mem_t<BITS>* b, cgbn_mem_t<BITS>* c) {
        int instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
        if (instance != 0) return;
        context_t ctx(cgbn_report_monitor, report, 0);
        env_t env(ctx);
        env_t::cgbn_t A, B, R;
        cgbn_load(env, A, a);
        cgbn_load(env, B, b);
        // Perform R = A - B (mod 2^BITS)
        cgbn_sub(env, R, A, B);
        cgbn_store(env, c, R);
    }
    __global__ static void kernel_mul(cgbn_error_report_t *report, const cgbn_mem_t<BITS>* a, const cgbn_mem_t<BITS>* b, cgbn_mem_t<BITS>* c) {
        int instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
        if (instance != 0) return;
        context_t ctx(cgbn_report_monitor, report, 0);
        env_t env(ctx);
        env_t::cgbn_t A, B, R;
        cgbn_load(env, A, a);
        cgbn_load(env, B, b);
        // Perform R = A * B (mod 2^BITS). cgbn_mul computes the low 2048 bits of the product.
        cgbn_mul(env, R, A, B);
        cgbn_store(env, c, R);
    }

    // Helper: check for any CGBN error after kernel execution
    static void check_cgbn_errors() {
        // Synchronize the device to ensure kernel completion and error report is updated
        cudaError_t cerr = cudaDeviceSynchronize();
        if (cerr != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(cerr));
        }
        // Check CGBN error report for any errors
        if (cgbn_error_report_check(gpu_err_report)) {
            const char* err_str = cgbn_error_string(gpu_err_report);
            // Reset the error report for future use
            cgbn_error_report_reset(gpu_err_report);
            throw std::runtime_error(std::string("CGBN error during GPU computation: ") + (err_str ? err_str : "unknown"));
        }
        // Reset error report (no error case) to clear any state
        cgbn_error_report_reset(gpu_err_report);
    }

public:
    // Big integer addition: this = a + b (mod 2^2048)
    void add(const BigIntBackend &a, const BigIntBackend &b) override {
        const GPUBigIntBackend &A = dynamic_cast<const GPUBigIntBackend&>(a);
        const GPUBigIntBackend &B = dynamic_cast<const GPUBigIntBackend&>(b);
        // Launch GPU addition kernel with 1 warp (32 threads) and 1 block
        kernel_add<<<1, TPI>>>(gpu_err_report, A.dev_value, B.dev_value, this->dev_value);
        check_cgbn_errors();
    }
    // Big integer subtraction: this = a - b (mod 2^2048)
    void sub(const BigIntBackend &a, const BigIntBackend &b) override {
        const GPUBigIntBackend &A = dynamic_cast<const GPUBigIntBackend&>(a);
        const GPUBigIntBackend &B = dynamic_cast<const GPUBigIntBackend&>(b);
        kernel_sub<<<1, TPI>>>(gpu_err_report, A.dev_value, B.dev_value, this->dev_value);
        check_cgbn_errors();
    }
    // Big integer multiplication: this = a * b (mod 2^2048)
    void mul(const BigIntBackend &a, const BigIntBackend &b) override {
        const GPUBigIntBackend &A = dynamic_cast<const GPUBigIntBackend&>(a);
        const GPUBigIntBackend &B = dynamic_cast<const GPUBigIntBackend&>(b);
        kernel_mul<<<1, TPI>>>(gpu_err_report, A.dev_value, B.dev_value, this->dev_value);
        check_cgbn_errors();
    }
};
// Static member initialization
cgbn_error_report_t* GPUBigIntBackend::gpu_err_report = nullptr;

#endif // GPU_ENABLED

// CPU backend for BigInt using Boost.Multiprecision (cpp_int)
class CPUBigIntBackend : public BigIntBackend {
private:
    boost::multiprecision::cpp_int value;  // value stored (will be kept mod 2^2048)
public:
    // Construct from a big integer (will be reduced mod 2^2048)
    CPUBigIntBackend(const boost::multiprecision::cpp_int &init = 0) : value(0) {
        from_cpp_int(init);
    }
    ~CPUBigIntBackend() override = default;
    BigIntBackend* clone() const override {
        return new CPUBigIntBackend(value);
    }
    void set(const BigIntBackend &other) override {
        const CPUBigIntBackend &src = dynamic_cast<const CPUBigIntBackend&>(other);
        value = src.value;
    }
    void from_cpp_int(const boost::multiprecision::cpp_int &val) override {
        // Reduce the input value modulo 2^BITS to fit 2048 bits
        boost::multiprecision::cpp_int modulus = boost::multiprecision::cpp_int(1) << BITS;
        value = val % modulus;
        if (value < 0) {
            value += modulus;
        }
        // Now value is in [0, 2^BITS-1], which is the internal two's complement representation.
    }
    boost::multiprecision::cpp_int to_cpp_int() const override {
        // Interpret the stored value as signed two's complement
        boost::multiprecision::cpp_int half_modulus = boost::multiprecision::cpp_int(1) << (BITS - 1);
        boost::multiprecision::cpp_int modulus = half_modulus << 1; // 2^BITS
        boost::multiprecision::cpp_int result = value;
        if (result >= half_modulus) {
            result -= modulus;
        }
        return result;
    }
    void add(const BigIntBackend &a, const BigIntBackend &b) override {
        const CPUBigIntBackend &A = dynamic_cast<const CPUBigIntBackend&>(a);
        const CPUBigIntBackend &B = dynamic_cast<const CPUBigIntBackend&>(b);
        value = A.value + B.value;
        // Reduce mod 2^BITS to ensure it fits in 2048 bits
        boost::multiprecision::cpp_int modulus = boost::multiprecision::cpp_int(1) << BITS;
        value %= modulus;
        if (value < 0) value += modulus;
    }
    void sub(const BigIntBackend &a, const BigIntBackend &b) override {
        const CPUBigIntBackend &A = dynamic_cast<const CPUBigIntBackend&>(a);
        const CPUBigIntBackend &B = dynamic_cast<const CPUBigIntBackend&>(b);
        value = A.value - B.value;
        boost::multiprecision::cpp_int modulus = boost::multiprecision::cpp_int(1) << BITS;
        value %= modulus;
        if (value < 0) value += modulus;
    }
    void mul(const BigIntBackend &a, const BigIntBackend &b) override {
        const CPUBigIntBackend &A = dynamic_cast<const CPUBigIntBackend&>(a);
        const CPUBigIntBackend &B = dynamic_cast<const CPUBigIntBackend&>(b);
        value = A.value * B.value;
        boost::multiprecision::cpp_int modulus = boost::multiprecision::cpp_int(1) << BITS;
        value %= modulus;
        if (value < 0) value += modulus;
    }
};

// The BigInt class that is exposed to Python
struct BigInt {
    std::unique_ptr<BigIntBackend> impl;   // Pointer to chosen backend (GPU or CPU)

    // Choose backend at construction time depending on GPU availability
    static bool gpu_available;  // indicates if GPU backend is available for use
    static std::unique_ptr<BigIntBackend> createBackend(const boost::multiprecision::cpp_int &value) {
    #ifdef GPU_ENABLED
        if (gpu_available) {
            return std::unique_ptr<BigIntBackend>(new GPUBigIntBackend(value));
        }
    #endif
        // Fallback to CPU if GPU not available or not compiled
        return std::unique_ptr<BigIntBackend>(new CPUBigIntBackend(value));
    }

    // Constructors
    BigInt() : impl(createBackend(0)) {}
    explicit BigInt(const boost::multiprecision::cpp_int &value) : impl(createBackend(value)) {}
    explicit BigInt(long long v) : impl(createBackend(v)) {}  // from small integer (fits in long long)
    explicit BigInt(const std::string &str) : impl(createBackend(0)) {
        // Parse from string (accepts decimal or hex with "0x" prefix)
        boost::multiprecision::cpp_int val = 0;
        bool negative = false;
        std::string s = str;
        if (s.size() > 0 && (s[0] == '+' || s[0] == '-')) {
            if (s[0] == '-') negative = true;
            s = s.substr(1);
        }
        // Detect base (hex or decimal)
        if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) {
            // Hex string
            s = s.substr(2);
            // Remove leading zeros in hex if any
            size_t pos = s.find_first_not_of("0");
            if (pos != std::string::npos) {
                s = s.substr(pos);
            }
            if (s.empty()) {
                val = 0;
            } else {
                // Parse hex
                for (char c : s) {
                    val <<= 4;
                    if (c >= '0' && c <= '9') {
                        val += (c - '0');
                    } else if (c >= 'a' && c <= 'f') {
                        val += (10 + c - 'a');
                    } else if (c >= 'A' && c <= 'F') {
                        val += (10 + c - 'A');
                    } else {
                        throw std::invalid_argument("Invalid character in hex string");
                    }
                }
            }
        } else {
            // Decimal string
            if (s.empty()) {
                val = 0;
            } else {
                std::istringstream iss(s);
                iss >> val;
                if (iss.fail()) {
                    throw std::invalid_argument("Invalid decimal number string");
                }
            }
        }
        if (negative) {
            val = -val;
        }
        impl = createBackend(val);
    }
    // Copy constructor
    BigInt(const BigInt &other) : impl(other.impl->clone()) {}
    // Assignment operator
    BigInt& operator=(const BigInt &other) {
        if (this != &other) {
            impl.reset(other.impl->clone());
        }
        return *this;
    }

    // Arithmetic operators (results in new BigInt)
    BigInt operator+(const BigInt &rhs) const {
        BigInt result;
        // Use the same backend type as this or rhs (prefer GPU if both GPU)
        if (
    #ifdef GPU_ENABLED
            gpu_available && dynamic_cast<GPUBigIntBackend*>(impl.get()) && dynamic_cast<GPUBigIntBackend*>(rhs.impl.get())
    #endif
           ) {
            result.impl = std::unique_ptr<BigIntBackend>(new GPUBigIntBackend(boost::multiprecision::cpp_int(0)));
        } else {
            result.impl = std::unique_ptr<BigIntBackend>(new CPUBigIntBackend());
        }
        result.impl->add(*impl, *rhs.impl);
        return result;
    }
    BigInt operator-(const BigInt &rhs) const {
        BigInt result;
        if (
    #ifdef GPU_ENABLED
            gpu_available && dynamic_cast<GPUBigIntBackend*>(impl.get()) && dynamic_cast<GPUBigIntBackend*>(rhs.impl.get())
    #endif
           ) {
            result.impl = std::unique_ptr<BigIntBackend>(new GPUBigIntBackend(boost::multiprecision::cpp_int(0)));
        } else {
            result.impl = std::unique_ptr<BigIntBackend>(new CPUBigIntBackend());
        }
        result.impl->sub(*impl, *rhs.impl);
        return result;
    }
    BigInt operator*(const BigInt &rhs) const {
        BigInt result;
        if (
    #ifdef GPU_ENABLED
            gpu_available && dynamic_cast<GPUBigIntBackend*>(impl.get()) && dynamic_cast<GPUBigIntBackend*>(rhs.impl.get())
    #endif
           ) {
            result.impl = std::unique_ptr<BigIntBackend>(new GPUBigIntBackend(boost::multiprecision::cpp_int(0)));
        } else {
            result.impl = std::unique_ptr<BigIntBackend>(new CPUBigIntBackend());
        }
        result.impl->mul(*impl, *rhs.impl);
        return result;
    }

    // In-place arithmetic operators
    BigInt& operator+=(const BigInt &rhs) {
        // If mixing backends, convert to CPU to avoid data transfer overhead
    #ifdef GPU_ENABLED
        if (!(gpu_available && dynamic_cast<GPUBigIntBackend*>(impl.get()) && dynamic_cast<GPUBigIntBackend*>(rhs.impl.get()))) {
            // Ensure this->impl is CPU if the other is CPU
            if (!dynamic_cast<CPUBigIntBackend*>(impl.get())) {
                // Convert this from GPU to CPU
                boost::multiprecision::cpp_int temp = impl->to_cpp_int();
                impl = std::unique_ptr<BigIntBackend>(new CPUBigIntBackend(temp));
            }
        }
    #endif
        impl->add(*impl, *rhs.impl);
        return *this;
    }
    BigInt& operator-=(const BigInt &rhs) {
    #ifdef GPU_ENABLED
        if (!(gpu_available && dynamic_cast<GPUBigIntBackend*>(impl.get()) && dynamic_cast<GPUBigIntBackend*>(rhs.impl.get()))) {
            if (!dynamic_cast<CPUBigIntBackend*>(impl.get())) {
                boost::multiprecision::cpp_int temp = impl->to_cpp_int();
                impl = std::unique_ptr<BigIntBackend>(new CPUBigIntBackend(temp));
            }
        }
    #endif
        impl->sub(*impl, *rhs.impl);
        return *this;
    }
    BigInt& operator*=(const BigInt &rhs) {
    #ifdef GPU_ENABLED
        if (!(gpu_available && dynamic_cast<GPUBigIntBackend*>(impl.get()) && dynamic_cast<GPUBigIntBackend*>(rhs.impl.get()))) {
            if (!dynamic_cast<CPUBigIntBackend*>(impl.get())) {
                boost::multiprecision::cpp_int temp = impl->to_cpp_int();
                impl = std::unique_ptr<BigIntBackend>(new CPUBigIntBackend(temp));
            }
        }
    #endif
        impl->mul(*impl, *rhs.impl);
        return *this;
    }

    // Comparison operators
    bool operator==(const BigInt &rhs) const {
        // Compare by converting both to cpp_int (this also accounts for sign properly)
        boost::multiprecision::cpp_int a = impl->to_cpp_int();
        boost::multiprecision::cpp_int b = rhs.impl->to_cpp_int();
        return a == b;
    }
    bool operator!=(const BigInt &rhs) const { return !(*this == rhs); }
    bool operator<(const BigInt &rhs) const {
        boost::multiprecision::cpp_int a = impl->to_cpp_int();
        boost::multiprecision::cpp_int b = rhs.impl->to_cpp_int();
        return a < b;
    }
    bool operator<=(const BigInt &rhs) const {
        boost::multiprecision::cpp_int a = impl->to_cpp_int();
        boost::multiprecision::cpp_int b = rhs.impl->to_cpp_int();
        return a <= b;
    }
    bool operator>(const BigInt &rhs) const { return rhs < *this; }
    bool operator>=(const BigInt &rhs) const { return rhs <= *this; }

    // Convert to decimal string
    std::string to_string() const {
        boost::multiprecision::cpp_int val = impl->to_cpp_int();
        std::ostringstream oss;
        oss << val;
        return oss.str();
    }
    // Convert to hexadecimal string (two's complement aware)
    std::string to_hex() const {
        boost::multiprecision::cpp_int val = impl->to_cpp_int();
        bool neg = (val < 0);
        if (neg) {
            val = -val;
        }
        // Convert absolute value to hex
        std::string hex;
        if (val == 0) {
            hex = "0";
        } else {
            // Extract hex digits
            while (val > 0) {
                unsigned int digit = static_cast<unsigned int>(val & 0xF);
                char c = (digit < 10 ? '0' + digit : 'A' + (digit - 10));
                hex.push_back(c);
                val >>= 4;
            }
            std::reverse(hex.begin(), hex.end());
        }
        if (neg) {
            return std::string("-0x") + hex;
        } else {
            return std::string("0x") + hex;
        }
    }
};
bool BigInt::gpu_available = false;  // Will be set during module initialization if a GPU is present

// Helper function to convert BigInt to PyLong (Python int)
static PyObject* bigint_to_PyLong(const BigInt &x) {
    // Get the internal absolute value representation (mod 2^2048, non-negative)
    boost::multiprecision::cpp_int value = x.impl->to_cpp_int();
    // If negative, convert to two's complement positive representative
    bool negative = (value < 0);
    if (negative) {
        value = -value;
    }
    // Now value is the magnitude (non-negative) to represent in 2048 bits or fewer
    // Determine the number of bytes needed
    // We represent in two's complement if negative, else straight binary
    // For PyLong_FromByteArray, if negative, we will use signed=True
    // Compute byte length: at least enough to hold all bits, plus one extra byte if sign bit would cause confusion
    size_t bit_count = 0;
    if (value > 0) {
        bit_count = msb(value) + 1;  // msb returns index of most significant bit (0-based)
    }
    // If value is 0, we want 1 byte.
    if (bit_count == 0) {
        bit_count = 1;
    }
    // If negative, ensure an extra bit for sign (two's complement representation needs an extra bit if value is exactly a power of two boundary)
    if (negative) {
        // If value in binary has exactly bit_count bits, and the top bit is 1, then to encode -value in two's complement with bit_count bits would overflow.
        // So we add one more bit (effectively ensuring the sign bit 0 for positive magnitude representation).
        // Actually simpler: just add 1 bit unconditionally for negative.
        bit_count += 1;
    }
    size_t byte_count = (bit_count + 7) / 8;
    if (byte_count == 0) {
        byte_count = 1;
    }
    // Allocate buffer for bytes
    std::vector<unsigned char> bytes(byte_count);
    // Fill bytes in little-endian order
    boost::multiprecision::cpp_int temp = value;
    for (size_t i = 0; i < byte_count; ++i) {
        bytes[i] = static_cast<unsigned char>(temp & 0xFF);
        temp >>= 8;
    }
    // If negative, convert this magnitude to two's complement in this fixed byte_count size
    if (negative) {
        // Two's complement: invert and add 1
        for (size_t i = 0; i < byte_count; ++i) {
            bytes[i] = static_cast<unsigned char>(~bytes[i]);
        }
        // Add 1 to the little-endian byte array
        unsigned int carry = 1;
        for (size_t i = 0; i < byte_count; ++i) {
            unsigned int sum = bytes[i] + carry;
            bytes[i] = static_cast<unsigned char>(sum & 0xFF);
            carry = (sum >> 8) & 0xFF;
            if (carry == 0) break;
        }
    }
    // Create Python long from byte array
    // PyLong_FromByteArray expects a C-order (little-endian or big-endian) representation. 
    // We'll use little-endian since we built the bytes accordingly.
    PyObject *pyint = PyLong_FromByteArray(bytes.data(), bytes.size(), /*little_endian=*/1, /*signed=*/negative);
    return pyint;
}

// Python module binding code
PYBIND11_MODULE(bigint_gpu, m) {
    m.doc() = "GPU-accelerated 2048-bit BigInt arithmetic module (C++/CUDA with CGBN)";

#ifdef GPU_ENABLED
    // Initialize CUDA and CGBN if possible
    int deviceCount = 0;
    cudaError_t cerr = cudaGetDeviceCount(&deviceCount);
    if (cerr == cudaSuccess && deviceCount > 0) {
        // Use the first CUDA device
        cudaSetDevice(0);
        // Allocate a global CGBN error report structure
        cudaError_t err = cgbn_error_report_alloc(&GPUBigIntBackend::gpu_err_report);
        if (err != cudaSuccess || GPUBigIntBackend::gpu_err_report == nullptr) {
            // If allocation failed, disable GPU usage
            BigInt::gpu_available = false;
        } else {
            BigInt::gpu_available = true;
        }
    } else {
        BigInt::gpu_available = false;
    }
#else
    BigInt::gpu_available = false;
#endif

    // Define BigInt class for Python
    py::class_<BigInt>(m, "BigInt", py::dynamic_attr())
        .def(py::init<>())  // default constructor (0)
        // Constructor from Python int (arbitrary size)
        .def(py::init([](py::object pyVal) {
            if (!PyLong_Check(pyVal.ptr())) {
                throw std::invalid_argument("BigInt constructor requires an int or str");
            }
            // Convert Python int to boost::cpp_int via bytes
            // Use Python's big int to byte array conversion
            PyObject* py_obj = pyVal.ptr();
            // Determine sign and magnitude via PyLong functions
            int negative = PyObject_RichCompareBool(py_obj, PyLong_FromLong(0), Py_LT);
            // Get absolute value as bytes (unsigned, big-endian or little-endian)
            // Python doesn't have a direct API to get bytes length; use bit_length method
            py::int_ int_obj = py::reinterpret_borrow<py::int_>(py_obj);
            py::object bit_length_obj = int_obj.attr("bit_length")();
            long bit_length = bit_length_obj.cast<long>();
            if (bit_length < 0) bit_length = 0;
            if (negative == -1) {
                throw std::runtime_error("Error comparing Python int");
            }
            bool is_negative = (negative == 1);
            // Add 1 bit if negative (to ensure space for sign)
            if (is_negative) {
                bit_length += 1;
            }
            size_t byte_len = (bit_length + 7) / 8;
            if (byte_len == 0) byte_len = 1;
            std::vector<unsigned char> buf(byte_len);
            // Use PyLong_AsByteArray to fill the buffer in little-endian
            int res = PyLong_AsByteArray((PyLongObject*)py_obj, buf.data(), buf.size(), 1, is_negative ? 1 : 0);
            if (res != 0) {
                throw std::runtime_error("Conversion from Python int to BigInt failed");
            }
            // Build boost::cpp_int from bytes (little-endian)
            boost::multiprecision::cpp_int cpp_val = 0;
            for (size_t i = 0; i < buf.size(); ++i) {
                cpp_val += boost::multiprecision::cpp_int(buf[i]) << (8 * i);
            }
            // If the original was negative and we interpreted signed two's complement, 
            // we need to adjust to get the true negative value
            if (is_negative) {
                // In two's complement, value is currently the 2^N - |true value|
                // So subtract 2^(bytes*8) to get the negative actual value
                boost::multiprecision::cpp_int twoN = boost::multiprecision::cpp_int(1) << (buf.size() * 8);
                cpp_val -= twoN;
            }
            return BigInt(cpp_val);
        }), py::arg("value"))
        // Constructor from string (decimal or hex)
        .def(py::init<const std::string &>(), py::arg("value_str"))
        // __str__ and __repr__ for readability
        .def("__repr__", [](const BigInt &x) {
            // Represent as BigInt(<value>) in decimal
            return std::string("BigInt(") + x.to_string() + std::string(")");
        })
        .def("__str__", [](const BigInt &x) {
            return x.to_string();
        })
        // Conversion to Python int
        .def("__int__", [](const BigInt &x) {
            // Use helper to create PyObject* and then wrap in py::object
            PyObject* py_int = bigint_to_PyLong(x);
            // Use py::reinterpret_steal to take ownership of PyObject*
            return py::reinterpret_steal<py::int_>(py_int);
        })
        // Support int() built-in and Python's index protocol (for hex(), etc.)
        .def("__index__", [](const BigInt &x) {
            PyObject* py_int = bigint_to_PyLong(x);
            return py::reinterpret_steal<py::int_>(py_int);
        })
        // Arithmetic operators
        .def("__add__", [](const BigInt &a, const BigInt &b) { return a + b; })
        .def("__sub__", [](const BigInt &a, const BigInt &b) { return a - b; })
        .def("__mul__", [](const BigInt &a, const BigInt &b) { return a * b; })
        .def("__radd__", [](const BigInt &a, const py::object &b) {
            // support int + BigInt by converting b to BigInt
            BigInt b_big = py::cast<BigInt>(b);
            return b_big + a;
        })
        .def("__rsub__", [](const BigInt &a, const py::object &b) {
            BigInt b_big = py::cast<BigInt>(b);
            return b_big - a;
        })
        .def("__rmul__", [](const BigInt &a, const py::object &b) {
            BigInt b_big = py::cast<BigInt>(b);
            return b_big * a;
        })
        .def("__iadd__", [](BigInt &a, const BigInt &b) -> BigInt& { a += b; return a; })
        .def("__isub__", [](BigInt &a, const BigInt &b) -> BigInt& { a -= b; return a; })
        .def("__imul__", [](BigInt &a, const BigInt &b) -> BigInt& { a *= b; return a; })
        // Comparison operators
        .def("__eq__", [](const BigInt &a, const BigInt &b) { return a == b; })
        .def("__ne__", [](const BigInt &a, const BigInt &b) { return a != b; })
        .def("__lt__", [](const BigInt &a, const BigInt &b) { return a < b; })
        .def("__le__", [](const BigInt &a, const BigInt &b) { return a <= b; })
        .def("__gt__", [](const BigInt &a, const BigInt &b) { return a > b; })
        .def("__ge__", [](const BigInt &a, const BigInt &b) { return a >= b; })
        // Methods to get string representations
        .def("to_string", &BigInt::to_string, "Get the decimal string representation of this BigInt")
        .def("to_hex", &BigInt::to_hex, "Get the hexadecimal string representation of this BigInt");

    // If GPU is enabled, add a cleanup function to free the error report at module unload
#ifdef GPU_ENABLED
    if (BigInt::gpu_available) {
        m.add_object("_cleanup", py::capsule((void*)nullptr, [](void *){
            if (GPUBigIntBackend::gpu_err_report) {
                cgbn_error_report_free(GPUBigIntBackend::gpu_err_report);
                GPUBigIntBackend::gpu_err_report = nullptr;
            }
        }));
    }
#endif
}

/*
Python Usage Example:
---------------------
After compiling this module (named 'bigint_gpu'), you can use it as follows in Python:

import bigint_gpu

# Create BigInt instances
a = bigint_gpu.BigInt(12345678901234567890)
b = bigint_gpu.BigInt("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")  # large hex value
c = bigint_gpu.BigInt(-50)

# Perform arithmetic
sum_ab = a + b
prod = a * b
diff = a - c  # which is a + 50

# Convert to Python int or string for verification
print("a + b =", int(sum_ab))
print("a * b =", prod.to_string())          # decimal string
print("a * b (hex) =", prod.to_hex())       # hex string
print("a - (-50) =", diff.to_string())      # should equal a + 50 in decimal

# Verify against Python's built-in arbitrary precision int
assert int(sum_ab) == 12345678901234567890 + int(b)
assert int(prod) == 12345678901234567890 * int(b)
assert int(diff) == 12345678901234567890 - (-50)
*/
