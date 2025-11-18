#include "fp.h"
#include "device.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Example CUDA kernel for parallel Fp operations
// This demonstrates how to use the Fp arithmetic in CUDA kernels


// Kernel: Add two arrays of Fp elements
// a[i] + b[i] -> c[i] for all i
__global__ void kernel_fp_add_array(
    Fp* c,
    const Fp* a,
    const Fp* b,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fp_add(c[idx], a[idx], b[idx]);
    }
}

// Kernel: Multiply two arrays of Fp elements
// a[i] * b[i] -> c[i] for all i
__global__ void kernel_fp_mul_array(
    Fp* c,
    const Fp* a,
    const Fp* b,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fp_mul(c[idx], a[idx], b[idx]);
    }
}

// Kernel: Scalar multiplication (all elements multiplied by same scalar)
// a[i] * scalar -> c[i] for all i
__global__ void kernel_fp_mul_scalar(
    Fp* c,
    const Fp* a,
    const Fp* scalar,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fp_mul(c[idx], a[idx], *scalar);
    }
}

// Test kernels for individual operations - these run single operations on GPU
// Used to verify arithmetic operations work correctly on device

// Test kernel: single addition on GPU
__global__ void kernel_test_fp_add(Fp* result, const Fp* a, const Fp* b) {
    fp_add(*result, *a, *b);
}

// Test kernel: single subtraction on GPU
__global__ void kernel_test_fp_sub(Fp* result, const Fp* a, const Fp* b) {
    fp_sub(*result, *a, *b);
}

// Test kernel: single multiplication on GPU
__global__ void kernel_test_fp_mul(Fp* result, const Fp* a, const Fp* b) {
    fp_mul(*result, *a, *b);
}

// Test kernel: single negation on GPU
__global__ void kernel_test_fp_neg(Fp* result, const Fp* a) {
    fp_neg(*result, *a);
}

// Test kernel: single inversion on GPU
__global__ void kernel_test_fp_inv(Fp* result, const Fp* a) {
    fp_inv(*result, *a);
}

// Test kernel: single division on GPU
__global__ void kernel_test_fp_div(Fp* result, const Fp* a, const Fp* b) {
    fp_div(*result, *a, *b);
}

// Test kernel: Montgomery conversion on GPU
__global__ void kernel_test_fp_to_montgomery(Fp* result, const Fp* a) {
    fp_to_montgomery(*result, *a);
}

__global__ void kernel_test_fp_from_montgomery(Fp* result, const Fp* a) {
    fp_from_montgomery(*result, *a);
}

// Test kernel: Montgomery multiplication on GPU
__global__ void kernel_test_fp_mont_mul(Fp* result, const Fp* a, const Fp* b) {
    fp_mont_mul(*result, *a, *b);
}

// Test kernel: comparison on GPU
__global__ void kernel_test_fp_cmp(int* result, const Fp* a, const Fp* b) {
    *result = fp_cmp(*a, *b);
}

// Test kernel: check if zero on GPU
__global__ void kernel_test_fp_is_zero(bool* result, const Fp* a) {
    *result = fp_is_zero(*a);
}

// Test kernel: check if one on GPU
__global__ void kernel_test_fp_is_one(bool* result, const Fp* a) {
    *result = fp_is_one(*a);
}

// Test kernel: copy on GPU
__global__ void kernel_test_fp_copy(Fp* result, const Fp* a) {
    fp_copy(*result, *a);
}

// Test kernel: conditional move on GPU
__global__ void kernel_test_fp_cmov(Fp* result, const Fp* src, uint64_t condition) {
    fp_cmov(*result, *src, condition);
}

// Test kernel: square root on GPU
__global__ void kernel_test_fp_sqrt(bool* has_sqrt, Fp* result, const Fp* a) {
    *has_sqrt = fp_sqrt(*result, *a);
}

// Test kernel: check if quadratic residue on GPU
__global__ void kernel_test_fp_is_quadratic_residue(bool* result, const Fp* a) {
    *result = fp_is_quadratic_residue(*a);
}

// Test kernel: exponentiation with uint64_t exponent on GPU
__global__ void kernel_test_fp_pow_u64(Fp* result, const Fp* base, uint64_t exp) {
    fp_pow_u64(*result, *base, exp);
}

// Test kernel: set to zero on GPU
__global__ void kernel_test_fp_zero(Fp* result) {
    fp_zero(*result);
}

// Test kernel: set to one on GPU
__global__ void kernel_test_fp_one(Fp* result) {
    fp_one(*result);
}

// Host wrapper functions
void fp_add_array_host(cudaStream_t stream, uint32_t gpu_index, Fp* c, const Fp* a, const Fp* b, int n) {
    // Validate inputs
    PANIC_IF_FALSE(n >= 0, "fp_add_array_host: invalid size n=%d", n);
    if (n == 0) {
        return;  // Nothing to do
    }
    PANIC_IF_FALSE(c != nullptr && a != nullptr && b != nullptr, "fp_add_array_host: null pointer argument");
    
    // Set the device context
    cuda_set_device(gpu_index);
    
    // Declare all variables at the top to avoid goto issues
    Fp *d_c = nullptr, *d_a = nullptr, *d_b = nullptr;
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Allocate device memory (asynchronous with stream)
    d_c = (Fp*)cuda_malloc_async(n * sizeof(Fp), stream, gpu_index);
    d_a = (Fp*)cuda_malloc_async(n * sizeof(Fp), stream, gpu_index);
    d_b = (Fp*)cuda_malloc_async(n * sizeof(Fp), stream, gpu_index);
    
    // Copy to device (asynchronous with stream)
    cuda_memcpy_async_to_gpu(d_a, a, n * sizeof(Fp), stream, gpu_index);
    cuda_memcpy_async_to_gpu(d_b, b, n * sizeof(Fp), stream, gpu_index);
    
    // Launch kernel (with stream)
    kernel_fp_add_array<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_c, d_a, d_b, n);
    
    // Check for kernel launch errors
    check_cuda_error(cudaGetLastError());
    
    // Synchronize stream to ensure kernel completes before copying back
    cuda_synchronize_stream(stream, gpu_index);
    
    // Copy back (synchronous after stream sync)
    cuda_memcpy_async_to_cpu(c, d_c, n * sizeof(Fp), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    
    // Free device memory (asynchronous with stream)
    if (d_c != nullptr) {
        cuda_drop_async(d_c, stream, gpu_index);
    }
    if (d_a != nullptr) {
        cuda_drop_async(d_a, stream, gpu_index);
    }
    if (d_b != nullptr) {
        cuda_drop_async(d_b, stream, gpu_index);
    }
}

// Device-resident API: assumes all pointers are already on device
void fp_add_array_device(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a, const Fp* d_b, int n) {
    // Validate inputs
    PANIC_IF_FALSE(n >= 0, "fp_add_array_device: invalid size n=%d", n);
    if (n == 0) {
        return;  // Nothing to do
    }
    PANIC_IF_FALSE(d_c != nullptr && d_a != nullptr && d_b != nullptr, "fp_add_array_device: null pointer argument");
    
    // Set the device context
    cuda_set_device(gpu_index);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel (with stream)
    kernel_fp_add_array<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_c, d_a, d_b, n);
    
    // Check for kernel launch errors
    check_cuda_error(cudaGetLastError());
}

void fp_mul_array_device(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a, const Fp* d_b, int n) {
    // Validate inputs
    PANIC_IF_FALSE(n >= 0, "fp_mul_array_device: invalid size n=%d", n);
    if (n == 0) {
        return;  // Nothing to do
    }
    PANIC_IF_FALSE(d_c != nullptr && d_a != nullptr && d_b != nullptr, "fp_mul_array_device: null pointer argument");
    
    // Set the device context
    cuda_set_device(gpu_index);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel (with stream)
    kernel_fp_mul_array<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_c, d_a, d_b, n);
    
    // Check for kernel launch errors
    check_cuda_error(cudaGetLastError());
}

void fp_mul_array_host(cudaStream_t stream, uint32_t gpu_index, Fp* c, const Fp* a, const Fp* b, int n) {
    // Validate inputs
    PANIC_IF_FALSE(n >= 0, "fp_mul_array_host: invalid size n=%d", n);
    if (n == 0) {
        return;  // Nothing to do
    }
    PANIC_IF_FALSE(c != nullptr && a != nullptr && b != nullptr, "fp_mul_array_host: null pointer argument");
    
    // Set the device context
    cuda_set_device(gpu_index);
    
    // Declare all variables at the top to avoid goto issues
    Fp *d_c = nullptr, *d_a = nullptr, *d_b = nullptr;
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Allocate device memory (asynchronous with stream)
    d_c = (Fp*)cuda_malloc_async(n * sizeof(Fp), stream, gpu_index);
    d_a = (Fp*)cuda_malloc_async(n * sizeof(Fp), stream, gpu_index);
    d_b = (Fp*)cuda_malloc_async(n * sizeof(Fp), stream, gpu_index);
    
    // Copy to device (asynchronous with stream)
    cuda_memcpy_async_to_gpu(d_a, a, n * sizeof(Fp), stream, gpu_index);
    cuda_memcpy_async_to_gpu(d_b, b, n * sizeof(Fp), stream, gpu_index);
    
    // Launch kernel (with stream)
    kernel_fp_mul_array<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_c, d_a, d_b, n);
    
    // Check for kernel launch errors
    check_cuda_error(cudaGetLastError());
    
    // Synchronize stream to ensure kernel completes before copying back
    cuda_synchronize_stream(stream, gpu_index);
    
    // Copy back (synchronous after stream sync)
    cuda_memcpy_async_to_cpu(c, d_c, n * sizeof(Fp), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    
    // Free device memory (asynchronous with stream)
    if (d_c != nullptr) {
        cuda_drop_async(d_c, stream, gpu_index);
    }
    if (d_a != nullptr) {
        cuda_drop_async(d_a, stream, gpu_index);
    }
    if (d_b != nullptr) {
        cuda_drop_async(d_b, stream, gpu_index);
    }
}

// Host wrapper functions for testing individual operations on GPU
// These functions launch single-operation kernels to verify arithmetic works on device

void fp_add_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a, const Fp* b) {
    // Set the device context
    cuda_set_device(gpu_index);
    
    Fp *d_result = nullptr, *d_a = nullptr, *d_b = nullptr;
    
    d_result = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    d_a = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    d_b = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    
    cuda_memcpy_async_to_gpu(d_a, a, sizeof(Fp), stream, gpu_index);
    cuda_memcpy_async_to_gpu(d_b, b, sizeof(Fp), stream, gpu_index);
    
    kernel_test_fp_add<<<1, 1, 0, stream>>>(d_result, d_a, d_b);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_stream(stream, gpu_index);
    
    cuda_memcpy_async_to_cpu(result, d_result, sizeof(Fp), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    
    if (d_result != nullptr) cuda_drop_async(d_result, stream, gpu_index);
    if (d_a != nullptr) cuda_drop_async(d_a, stream, gpu_index);
    if (d_b != nullptr) cuda_drop_async(d_b, stream, gpu_index);
}

void fp_sub_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a, const Fp* b) {
    // Set the device context
    cuda_set_device(gpu_index);
    
    Fp *d_result = nullptr, *d_a = nullptr, *d_b = nullptr;
    
    d_result = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    d_a = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    d_b = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    
    cuda_memcpy_async_to_gpu(d_a, a, sizeof(Fp), stream, gpu_index);
    
    cuda_memcpy_async_to_gpu(d_b, b, sizeof(Fp), stream, gpu_index);
    
    kernel_test_fp_sub<<<1, 1, 0, stream>>>(d_result, d_a, d_b);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_stream(stream, gpu_index);
    
    cuda_memcpy_async_to_cpu(result, d_result, sizeof(Fp), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    
    if (d_result != nullptr) cuda_drop_async(d_result, stream, gpu_index);
    if (d_a != nullptr) cuda_drop_async(d_a, stream, gpu_index);
    if (d_b != nullptr) cuda_drop_async(d_b, stream, gpu_index);
}

void fp_mul_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a, const Fp* b) {
    // Set the device context
    cuda_set_device(gpu_index);
    
    Fp *d_result = nullptr, *d_a = nullptr, *d_b = nullptr;
    
    d_result = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    d_a = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    d_b = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    
    cuda_memcpy_async_to_gpu(d_a, a, sizeof(Fp), stream, gpu_index);
    
    cuda_memcpy_async_to_gpu(d_b, b, sizeof(Fp), stream, gpu_index);
    
    kernel_test_fp_mul<<<1, 1, 0, stream>>>(d_result, d_a, d_b);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_stream(stream, gpu_index);
    
    cuda_memcpy_async_to_cpu(result, d_result, sizeof(Fp), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    
    if (d_result != nullptr) cuda_drop_async(d_result, stream, gpu_index);
    if (d_a != nullptr) cuda_drop_async(d_a, stream, gpu_index);
    if (d_b != nullptr) cuda_drop_async(d_b, stream, gpu_index);
}

void fp_neg_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a) {
    // Set the device context
    cuda_set_device(gpu_index);
    
    Fp *d_result = nullptr, *d_a = nullptr;
    
    d_result = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    d_a = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    
    cuda_memcpy_async_to_gpu(d_a, a, sizeof(Fp), stream, gpu_index);
    
    kernel_test_fp_neg<<<1, 1, 0, stream>>>(d_result, d_a);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_stream(stream, gpu_index);
    
    cuda_memcpy_async_to_cpu(result, d_result, sizeof(Fp), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    
    if (d_result != nullptr) cuda_drop_async(d_result, stream, gpu_index);
    if (d_a != nullptr) cuda_drop_async(d_a, stream, gpu_index);
}

void fp_inv_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a) {
    // Set the device context
    cuda_set_device(gpu_index);
    
    Fp *d_result = nullptr, *d_a = nullptr;
    
    d_result = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    d_a = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    
    cuda_memcpy_async_to_gpu(d_a, a, sizeof(Fp), stream, gpu_index);
    
    kernel_test_fp_inv<<<1, 1, 0, stream>>>(d_result, d_a);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_stream(stream, gpu_index);
    
    cuda_memcpy_async_to_cpu(result, d_result, sizeof(Fp), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    
    if (d_result != nullptr) cuda_drop_async(d_result, stream, gpu_index);
    if (d_a != nullptr) cuda_drop_async(d_a, stream, gpu_index);
}

void fp_div_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a, const Fp* b) {
    // Set the device context
    cuda_set_device(gpu_index);
    
    Fp *d_result = nullptr, *d_a = nullptr, *d_b = nullptr;
    
    d_result = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    d_a = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    d_b = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    
    cuda_memcpy_async_to_gpu(d_a, a, sizeof(Fp), stream, gpu_index);
    
    cuda_memcpy_async_to_gpu(d_b, b, sizeof(Fp), stream, gpu_index);
    
    kernel_test_fp_div<<<1, 1, 0, stream>>>(d_result, d_a, d_b);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_stream(stream, gpu_index);
    
    cuda_memcpy_async_to_cpu(result, d_result, sizeof(Fp), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    
    if (d_result != nullptr) cuda_drop_async(d_result, stream, gpu_index);
    if (d_a != nullptr) cuda_drop_async(d_a, stream, gpu_index);
    if (d_b != nullptr) cuda_drop_async(d_b, stream, gpu_index);
}

int fp_cmp_gpu(cudaStream_t stream, uint32_t gpu_index, const Fp* a, const Fp* b) {
    // Set the device context
    cuda_set_device(gpu_index);
    
    int *d_result = nullptr, *h_result = nullptr;
    Fp *d_a = nullptr, *d_b = nullptr;
    int result = 0;
    
    h_result = new int;
    d_result = (int*)cuda_malloc_async(sizeof(int), stream, gpu_index);
    d_a = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    d_b = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    
    cuda_memcpy_async_to_gpu(d_a, a, sizeof(Fp), stream, gpu_index);
    
    cuda_memcpy_async_to_gpu(d_b, b, sizeof(Fp), stream, gpu_index);
    
    kernel_test_fp_cmp<<<1, 1, 0, stream>>>(d_result, d_a, d_b);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_stream(stream, gpu_index);
    
    cuda_memcpy_async_to_cpu(h_result, d_result, sizeof(int), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    result = *h_result;
    
    if (d_result != nullptr) cuda_drop_async(d_result, stream, gpu_index);
    if (d_a != nullptr) cuda_drop_async(d_a, stream, gpu_index);
    if (d_b != nullptr) cuda_drop_async(d_b, stream, gpu_index);
    if (h_result != nullptr) delete h_result;
    return result;
}

bool fp_is_zero_gpu(cudaStream_t stream, uint32_t gpu_index, const Fp* a) {
    // Set the device context
    cuda_set_device(gpu_index);
    
    bool *d_result = nullptr, *h_result = nullptr;
    Fp *d_a = nullptr;
    bool result = false;
    
    h_result = new bool;
    d_result = (bool*)cuda_malloc_async(sizeof(bool), stream, gpu_index);
    d_a = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    
    cuda_memcpy_async_to_gpu(d_a, a, sizeof(Fp), stream, gpu_index);
    
    kernel_test_fp_is_zero<<<1, 1, 0, stream>>>(d_result, d_a);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_stream(stream, gpu_index);
    
    cuda_memcpy_async_to_cpu(h_result, d_result, sizeof(bool), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    result = *h_result;
    
    if (d_result != nullptr) cuda_drop_async(d_result, stream, gpu_index);
    if (d_a != nullptr) cuda_drop_async(d_a, stream, gpu_index);
    if (h_result != nullptr) delete h_result;
    return result;
}

bool fp_is_one_gpu(cudaStream_t stream, uint32_t gpu_index, const Fp* a) {
    // Set the device context
    cuda_set_device(gpu_index);
    
    bool *d_result = nullptr, *h_result = nullptr;
    Fp *d_a = nullptr;
    bool result = false;
    
    h_result = new bool;
    d_result = (bool*)cuda_malloc_async(sizeof(bool), stream, gpu_index);
    d_a = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    
    cuda_memcpy_async_to_gpu(d_a, a, sizeof(Fp), stream, gpu_index);
    
    kernel_test_fp_is_one<<<1, 1, 0, stream>>>(d_result, d_a);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_stream(stream, gpu_index);
    
    cuda_memcpy_async_to_cpu(h_result, d_result, sizeof(bool), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    result = *h_result;
    
    if (d_result != nullptr) cuda_drop_async(d_result, stream, gpu_index);
    if (d_a != nullptr) cuda_drop_async(d_a, stream, gpu_index);
    if (h_result != nullptr) delete h_result;
    return result;
}


void fp_to_montgomery_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a) {
    cuda_set_device(gpu_index);
    
    Fp *d_result = nullptr, *d_a = nullptr;
    
    d_result = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    d_a = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    
    cuda_memcpy_async_to_gpu(d_a, a, sizeof(Fp), stream, gpu_index);
    
    kernel_test_fp_to_montgomery<<<1, 1, 0, stream>>>(d_result, d_a);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_stream(stream, gpu_index);
    
    cuda_memcpy_async_to_cpu(result, d_result, sizeof(Fp), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    
    if (d_result != nullptr) cuda_drop_async(d_result, stream, gpu_index);
    if (d_a != nullptr) cuda_drop_async(d_a, stream, gpu_index);
}

void fp_from_montgomery_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a) {
    cuda_set_device(gpu_index);
    
    Fp *d_result = nullptr, *d_a = nullptr;
    
    d_result = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    d_a = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    
    cuda_memcpy_async_to_gpu(d_a, a, sizeof(Fp), stream, gpu_index);
    
    kernel_test_fp_from_montgomery<<<1, 1, 0, stream>>>(d_result, d_a);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_stream(stream, gpu_index);
    
    cuda_memcpy_async_to_cpu(result, d_result, sizeof(Fp), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    
    if (d_result != nullptr) cuda_drop_async(d_result, stream, gpu_index);
    if (d_a != nullptr) cuda_drop_async(d_a, stream, gpu_index);
}

void fp_mont_mul_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a, const Fp* b) {
    cuda_set_device(gpu_index);
    
    Fp *d_result = nullptr, *d_a = nullptr, *d_b = nullptr;
    
    d_result = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    d_a = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    d_b = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    
    cuda_memcpy_async_to_gpu(d_a, a, sizeof(Fp), stream, gpu_index);
    
    cuda_memcpy_async_to_gpu(d_b, b, sizeof(Fp), stream, gpu_index);
    
    kernel_test_fp_mont_mul<<<1, 1, 0, stream>>>(d_result, d_a, d_b);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_stream(stream, gpu_index);
    
    cuda_memcpy_async_to_cpu(result, d_result, sizeof(Fp), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    
    if (d_result != nullptr) cuda_drop_async(d_result, stream, gpu_index);
    if (d_a != nullptr) cuda_drop_async(d_a, stream, gpu_index);
    if (d_b != nullptr) cuda_drop_async(d_b, stream, gpu_index);
}

void fp_copy_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a) {
    cuda_set_device(gpu_index);
    
    Fp *d_result = nullptr, *d_a = nullptr;
    
    d_result = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    d_a = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    
    cuda_memcpy_async_to_gpu(d_a, a, sizeof(Fp), stream, gpu_index);
    
    kernel_test_fp_copy<<<1, 1, 0, stream>>>(d_result, d_a);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_stream(stream, gpu_index);
    
    cuda_memcpy_async_to_cpu(result, d_result, sizeof(Fp), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    
    if (d_result != nullptr) cuda_drop_async(d_result, stream, gpu_index);
    if (d_a != nullptr) cuda_drop_async(d_a, stream, gpu_index);
}

void fp_cmov_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* src, uint64_t condition) {
    cuda_set_device(gpu_index);
    
    Fp *d_result = nullptr, *d_src = nullptr;
    uint64_t *d_condition = nullptr;
    
    d_result = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    d_src = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    d_condition = (uint64_t*)cuda_malloc_async(sizeof(uint64_t), stream, gpu_index);
    
    // Copy result first (it's the destination that may be modified)
    cuda_memcpy_async_to_gpu(d_result, result, sizeof(Fp), stream, gpu_index);
    
    cuda_memcpy_async_to_gpu(d_src, src, sizeof(Fp), stream, gpu_index);
    
    cuda_memcpy_async_to_gpu(d_condition, &condition, sizeof(uint64_t), stream, gpu_index);
    
    kernel_test_fp_cmov<<<1, 1, 0, stream>>>(d_result, d_src, condition);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_stream(stream, gpu_index);
    
    cuda_memcpy_async_to_cpu(result, d_result, sizeof(Fp), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    
    if (d_result != nullptr) cuda_drop_async(d_result, stream, gpu_index);
    if (d_src != nullptr) cuda_drop_async(d_src, stream, gpu_index);
    if (d_condition != nullptr) cuda_drop_async(d_condition, stream, gpu_index);
}

bool fp_sqrt_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a) {
    cuda_set_device(gpu_index);
    
    bool *d_has_sqrt = nullptr, *h_has_sqrt = nullptr;
    Fp *d_result = nullptr, *d_a = nullptr;
    bool has_sqrt = false;
    
    h_has_sqrt = new bool;
    d_has_sqrt = (bool*)cuda_malloc_async(sizeof(bool), stream, gpu_index);
    
    d_result = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    d_a = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    
    cuda_memcpy_async_to_gpu(d_a, a, sizeof(Fp), stream, gpu_index);
    
    kernel_test_fp_sqrt<<<1, 1, 0, stream>>>(d_has_sqrt, d_result, d_a);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_stream(stream, gpu_index);
    
    cuda_memcpy_async_to_cpu(h_has_sqrt, d_has_sqrt, sizeof(bool), stream, gpu_index);
    cuda_memcpy_async_to_cpu(result, d_result, sizeof(Fp), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    has_sqrt = *h_has_sqrt;
    
    if (d_has_sqrt != nullptr) cuda_drop_async(d_has_sqrt, stream, gpu_index);
    if (d_result != nullptr) cuda_drop_async(d_result, stream, gpu_index);
    if (d_a != nullptr) cuda_drop_async(d_a, stream, gpu_index);
    if (h_has_sqrt != nullptr) delete h_has_sqrt;
    return has_sqrt;
}

bool fp_is_quadratic_residue_gpu(cudaStream_t stream, uint32_t gpu_index, const Fp* a) {
    cuda_set_device(gpu_index);
    
    bool *d_result = nullptr, *h_result = nullptr;
    Fp *d_a = nullptr;
    bool result = false;
    
    h_result = new bool;
    d_result = (bool*)cuda_malloc_async(sizeof(bool), stream, gpu_index);
    d_a = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    
    cuda_memcpy_async_to_gpu(d_a, a, sizeof(Fp), stream, gpu_index);
    
    kernel_test_fp_is_quadratic_residue<<<1, 1, 0, stream>>>(d_result, d_a);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_stream(stream, gpu_index);
    
    cuda_memcpy_async_to_cpu(h_result, d_result, sizeof(bool), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    result = *h_result;
    
    if (d_result != nullptr) cuda_drop_async(d_result, stream, gpu_index);
    if (d_a != nullptr) cuda_drop_async(d_a, stream, gpu_index);
    if (h_result != nullptr) delete h_result;
    return result;
}

void fp_pow_u64_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* base, uint64_t exp) {
    cuda_set_device(gpu_index);
    
    Fp *d_result = nullptr, *d_base = nullptr;
    
    d_result = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    d_base = (Fp*)cuda_malloc_async(sizeof(Fp), stream, gpu_index);
    
    cuda_memcpy_async_to_gpu(d_base, base, sizeof(Fp), stream, gpu_index);
    
    // exp is a simple value, can be passed directly to kernel
    kernel_test_fp_pow_u64<<<1, 1, 0, stream>>>(d_result, d_base, exp);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_stream(stream, gpu_index);
    
    cuda_memcpy_async_to_cpu(result, d_result, sizeof(Fp), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    
    if (d_result != nullptr) cuda_drop_async(d_result, stream, gpu_index);
    if (d_base != nullptr) cuda_drop_async(d_base, stream, gpu_index);
}

// ============================================================================
// Async/Sync API implementations
// All pointers are device pointers (already allocated)
// ============================================================================

// Addition: d_c = d_a + d_b mod p
void fp_add_async(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a, const Fp* d_b) {
    PANIC_IF_FALSE(d_c != nullptr && d_a != nullptr && d_b != nullptr, "fp_add_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_test_fp_add<<<1, 1, 0, stream>>>(d_c, d_a, d_b);
    check_cuda_error(cudaGetLastError());
}

void fp_add(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a, const Fp* d_b) {
    fp_add_async(stream, gpu_index, d_c, d_a, d_b);
    cuda_synchronize_stream(stream, gpu_index);
}

// Subtraction: d_c = d_a - d_b mod p
void fp_sub_async(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a, const Fp* d_b) {
    PANIC_IF_FALSE(d_c != nullptr && d_a != nullptr && d_b != nullptr, "fp_sub_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_test_fp_sub<<<1, 1, 0, stream>>>(d_c, d_a, d_b);
    check_cuda_error(cudaGetLastError());
}

void fp_sub(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a, const Fp* d_b) {
    fp_sub_async(stream, gpu_index, d_c, d_a, d_b);
    cuda_synchronize_stream(stream, gpu_index);
}

// Multiplication: d_c = d_a * d_b mod p
void fp_mul_async(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a, const Fp* d_b) {
    PANIC_IF_FALSE(d_c != nullptr && d_a != nullptr && d_b != nullptr, "fp_mul_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_test_fp_mul<<<1, 1, 0, stream>>>(d_c, d_a, d_b);
    check_cuda_error(cudaGetLastError());
}

void fp_mul(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a, const Fp* d_b) {
    fp_mul_async(stream, gpu_index, d_c, d_a, d_b);
    cuda_synchronize_stream(stream, gpu_index);
}

// Negation: d_c = -d_a mod p
void fp_neg_async(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a) {
    PANIC_IF_FALSE(d_c != nullptr && d_a != nullptr, "fp_neg_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_test_fp_neg<<<1, 1, 0, stream>>>(d_c, d_a);
    check_cuda_error(cudaGetLastError());
}

void fp_neg(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a) {
    fp_neg_async(stream, gpu_index, d_c, d_a);
    cuda_synchronize_stream(stream, gpu_index);
}

// Inversion: d_c = d_a^(-1) mod p
void fp_inv_async(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a) {
    PANIC_IF_FALSE(d_c != nullptr && d_a != nullptr, "fp_inv_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_test_fp_inv<<<1, 1, 0, stream>>>(d_c, d_a);
    check_cuda_error(cudaGetLastError());
}

void fp_inv(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a) {
    fp_inv_async(stream, gpu_index, d_c, d_a);
    cuda_synchronize_stream(stream, gpu_index);
}

// Division: d_c = d_a / d_b mod p
void fp_div_async(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a, const Fp* d_b) {
    PANIC_IF_FALSE(d_c != nullptr && d_a != nullptr && d_b != nullptr, "fp_div_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_test_fp_div<<<1, 1, 0, stream>>>(d_c, d_a, d_b);
    check_cuda_error(cudaGetLastError());
}

void fp_div(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a, const Fp* d_b) {
    fp_div_async(stream, gpu_index, d_c, d_a, d_b);
    cuda_synchronize_stream(stream, gpu_index);
}

// Copy: d_dst = d_src
void fp_copy_async(cudaStream_t stream, uint32_t gpu_index, Fp* d_dst, const Fp* d_src) {
    PANIC_IF_FALSE(d_dst != nullptr && d_src != nullptr, "fp_copy_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_test_fp_copy<<<1, 1, 0, stream>>>(d_dst, d_src);
    check_cuda_error(cudaGetLastError());
}

void fp_copy(cudaStream_t stream, uint32_t gpu_index, Fp* d_dst, const Fp* d_src) {
    fp_copy_async(stream, gpu_index, d_dst, d_src);
    cuda_synchronize_stream(stream, gpu_index);
}

// Set to zero: d_a = 0
void fp_zero_async(cudaStream_t stream, uint32_t gpu_index, Fp* d_a) {
    PANIC_IF_FALSE(d_a != nullptr, "fp_zero_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_test_fp_zero<<<1, 1, 0, stream>>>(d_a);
    check_cuda_error(cudaGetLastError());
}

void fp_zero(cudaStream_t stream, uint32_t gpu_index, Fp* d_a) {
    fp_zero_async(stream, gpu_index, d_a);
    cuda_synchronize_stream(stream, gpu_index);
}

// Set to one: d_a = 1
void fp_one_async(cudaStream_t stream, uint32_t gpu_index, Fp* d_a) {
    PANIC_IF_FALSE(d_a != nullptr, "fp_one_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_test_fp_one<<<1, 1, 0, stream>>>(d_a);
    check_cuda_error(cudaGetLastError());
}

void fp_one(cudaStream_t stream, uint32_t gpu_index, Fp* d_a) {
    fp_one_async(stream, gpu_index, d_a);
    cuda_synchronize_stream(stream, gpu_index);
}

// Convert to Montgomery form: d_c = (d_a * R) mod p
void fp_to_montgomery_async(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a) {
    PANIC_IF_FALSE(d_c != nullptr && d_a != nullptr, "fp_to_montgomery_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_test_fp_to_montgomery<<<1, 1, 0, stream>>>(d_c, d_a);
    check_cuda_error(cudaGetLastError());
}

void fp_to_montgomery(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a) {
    fp_to_montgomery_async(stream, gpu_index, d_c, d_a);
    cuda_synchronize_stream(stream, gpu_index);
}

// Convert from Montgomery form: d_c = (d_a * R_INV) mod p
void fp_from_montgomery_async(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a) {
    PANIC_IF_FALSE(d_c != nullptr && d_a != nullptr, "fp_from_montgomery_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_test_fp_from_montgomery<<<1, 1, 0, stream>>>(d_c, d_a);
    check_cuda_error(cudaGetLastError());
}

void fp_from_montgomery(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a) {
    fp_from_montgomery_async(stream, gpu_index, d_c, d_a);
    cuda_synchronize_stream(stream, gpu_index);
}

// Montgomery multiplication: d_c = (d_a * d_b * R_INV) mod p
void fp_mont_mul_async(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a, const Fp* d_b) {
    PANIC_IF_FALSE(d_c != nullptr && d_a != nullptr && d_b != nullptr, "fp_mont_mul_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_test_fp_mont_mul<<<1, 1, 0, stream>>>(d_c, d_a, d_b);
    check_cuda_error(cudaGetLastError());
}

void fp_mont_mul(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a, const Fp* d_b) {
    fp_mont_mul_async(stream, gpu_index, d_c, d_a, d_b);
    cuda_synchronize_stream(stream, gpu_index);
}
