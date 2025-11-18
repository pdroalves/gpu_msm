#include "fp2.h"
#include "fp2_kernels.h"
#include "device.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Kernel: Add two arrays of Fp2 elements
// a[i] + b[i] -> c[i] for all i
__global__ void kernel_fp2_add_array(
    Fp2* c,
    const Fp2* a,
    const Fp2* b,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fp2_add(c[idx], a[idx], b[idx]);
    }
}

// Kernel: Multiply two arrays of Fp2 elements
// a[i] * b[i] -> c[i] for all i
__global__ void kernel_fp2_mul_array(
    Fp2* c,
    const Fp2* a,
    const Fp2* b,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fp2_mul(c[idx], a[idx], b[idx]);
    }
}

// Test kernels for individual operations - these run single operations on GPU
// Used to verify arithmetic operations work correctly on device

// Test kernel: single addition on GPU
__global__ void kernel_test_fp2_add(Fp2* result, const Fp2* a, const Fp2* b) {
    fp2_add(*result, *a, *b);
}

// Test kernel: single subtraction on GPU
__global__ void kernel_test_fp2_sub(Fp2* result, const Fp2* a, const Fp2* b) {
    fp2_sub(*result, *a, *b);
}

// Test kernel: single multiplication on GPU
__global__ void kernel_test_fp2_mul(Fp2* result, const Fp2* a, const Fp2* b) {
    fp2_mul(*result, *a, *b);
}

// Test kernel: single negation on GPU
__global__ void kernel_test_fp2_neg(Fp2* result, const Fp2* a) {
    fp2_neg(*result, *a);
}

// Test kernel: single conjugation on GPU
__global__ void kernel_test_fp2_conjugate(Fp2* result, const Fp2* a) {
    fp2_conjugate(*result, *a);
}

// Test kernel: single squaring on GPU
__global__ void kernel_test_fp2_square(Fp2* result, const Fp2* a) {
    fp2_square(*result, *a);
}

// Test kernel: single inversion on GPU
__global__ void kernel_test_fp2_inv(Fp2* result, const Fp2* a) {
    fp2_inv(*result, *a);
}

// Test kernel: single division on GPU
__global__ void kernel_test_fp2_div(Fp2* result, const Fp2* a, const Fp2* b) {
    fp2_div(*result, *a, *b);
}

// Test kernel: multiply by i on GPU
__global__ void kernel_test_fp2_mul_by_i(Fp2* result, const Fp2* a) {
    fp2_mul_by_i(*result, *a);
}

// Test kernel: Frobenius map on GPU
__global__ void kernel_test_fp2_frobenius(Fp2* result, const Fp2* a) {
    fp2_frobenius(*result, *a);
}

// Test kernel: comparison on GPU
__global__ void kernel_test_fp2_cmp(int* result, const Fp2* a, const Fp2* b) {
    *result = fp2_cmp(*a, *b);
}

// Test kernel: check if zero on GPU
__global__ void kernel_test_fp2_is_zero(bool* result, const Fp2* a) {
    *result = fp2_is_zero(*a);
}

// Test kernel: check if one on GPU
__global__ void kernel_test_fp2_is_one(bool* result, const Fp2* a) {
    *result = fp2_is_one(*a);
}

// Test kernel: copy on GPU
__global__ void kernel_test_fp2_copy(Fp2* result, const Fp2* a) {
    fp2_copy(*result, *a);
}

// Test kernel: conditional move on GPU
__global__ void kernel_test_fp2_cmov(Fp2* result, const Fp2* src, uint64_t condition) {
    fp2_cmov(*result, *src, condition);
}

// Host wrapper functions
void fp2_add_array_host(cudaStream_t stream, uint32_t gpu_index, Fp2* c, const Fp2* a, const Fp2* b, int n) {
    // Validate inputs
    PANIC_IF_FALSE(n >= 0, "fp2_add_array_host: invalid size n=%d", n);
    if (n == 0) {
        return;  // Nothing to do
    }
    PANIC_IF_FALSE(c != nullptr && a != nullptr && b != nullptr, "fp2_add_array_host: null pointer argument");
    
    // Set the device context
    cuda_set_device(gpu_index);
    
    // Declare all variables at the top to avoid goto issues
    Fp2 *d_c = nullptr, *d_a = nullptr, *d_b = nullptr;
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Allocate device memory (asynchronous with stream)
    d_c = (Fp2*)cuda_malloc_async(n * sizeof(Fp2), stream, gpu_index);
    d_a = (Fp2*)cuda_malloc_async(n * sizeof(Fp2), stream, gpu_index);
    d_b = (Fp2*)cuda_malloc_async(n * sizeof(Fp2), stream, gpu_index);
    
    // Copy to device (asynchronous with stream)
    cuda_memcpy_async_to_gpu(d_a, a, n * sizeof(Fp2), stream, gpu_index);
    cuda_memcpy_async_to_gpu(d_b, b, n * sizeof(Fp2), stream, gpu_index);
    
    // Launch kernel (with stream)
    kernel_fp2_add_array<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_c, d_a, d_b, n);
    
    // Check for kernel launch errors
    check_cuda_error(cudaGetLastError());
    
    // Synchronize stream to ensure kernel completes before copying back
    cuda_synchronize_stream(stream, gpu_index);
    
    // Copy back (synchronous after stream sync)
    cuda_memcpy_async_to_cpu(c, d_c, n * sizeof(Fp2), stream, gpu_index);
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

void fp2_mul_array_host(cudaStream_t stream, uint32_t gpu_index, Fp2* c, const Fp2* a, const Fp2* b, int n) {
    // Validate inputs
    PANIC_IF_FALSE(n >= 0, "fp2_mul_array_host: invalid size n=%d", n);
    if (n == 0) {
        return;  // Nothing to do
    }
    PANIC_IF_FALSE(c != nullptr && a != nullptr && b != nullptr, "fp2_mul_array_host: null pointer argument");
    
    // Set the device context
    cuda_set_device(gpu_index);
    
    // Declare all variables at the top to avoid goto issues
    Fp2 *d_c = nullptr, *d_a = nullptr, *d_b = nullptr;
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Allocate device memory (asynchronous with stream)
    d_c = (Fp2*)cuda_malloc_async(n * sizeof(Fp2), stream, gpu_index);
    d_a = (Fp2*)cuda_malloc_async(n * sizeof(Fp2), stream, gpu_index);
    d_b = (Fp2*)cuda_malloc_async(n * sizeof(Fp2), stream, gpu_index);
    
    // Copy to device (asynchronous with stream)
    cuda_memcpy_async_to_gpu(d_a, a, n * sizeof(Fp2), stream, gpu_index);
    cuda_memcpy_async_to_gpu(d_b, b, n * sizeof(Fp2), stream, gpu_index);
    
    // Launch kernel (with stream)
    kernel_fp2_mul_array<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_c, d_a, d_b, n);
    
    // Check for kernel launch errors
    check_cuda_error(cudaGetLastError());
    
    // Synchronize stream to ensure kernel completes before copying back
    cuda_synchronize_stream(stream, gpu_index);
    
    // Copy back (synchronous after stream sync)
    cuda_memcpy_async_to_cpu(c, d_c, n * sizeof(Fp2), stream, gpu_index);
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

void fp2_add_gpu(Fp2* result, const Fp2* a, const Fp2* b) {
    uint32_t gpu_index = cuda_get_device();
    Fp2 *d_result = nullptr, *d_a = nullptr, *d_b = nullptr;
    
    d_result = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    d_a = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    d_b = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    
    check_cuda_error(cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice));
    check_cuda_error(cudaMemcpy(d_b, b, sizeof(Fp2), cudaMemcpyHostToDevice));
    
    kernel_test_fp2_add<<<1, 1>>>(d_result, d_a, d_b);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_device(gpu_index);
    
    check_cuda_error(cudaMemcpy(result, d_result, sizeof(Fp2), cudaMemcpyDeviceToHost));
    
    if (d_result != nullptr) cuda_drop(d_result, gpu_index);
    if (d_a != nullptr) cuda_drop(d_a, gpu_index);
    if (d_b != nullptr) cuda_drop(d_b, gpu_index);
}

void fp2_sub_gpu(Fp2* result, const Fp2* a, const Fp2* b) {
    uint32_t gpu_index = cuda_get_device();
    Fp2 *d_result = nullptr, *d_a = nullptr, *d_b = nullptr;
    
    d_result = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    d_a = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    d_b = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    
    check_cuda_error(cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice));
    check_cuda_error(cudaMemcpy(d_b, b, sizeof(Fp2), cudaMemcpyHostToDevice));
    
    kernel_test_fp2_sub<<<1, 1>>>(d_result, d_a, d_b);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_device(gpu_index);
    
    check_cuda_error(cudaMemcpy(result, d_result, sizeof(Fp2), cudaMemcpyDeviceToHost));
    
    if (d_result != nullptr) cuda_drop(d_result, gpu_index);
    if (d_a != nullptr) cuda_drop(d_a, gpu_index);
    if (d_b != nullptr) cuda_drop(d_b, gpu_index);
}

void fp2_mul_gpu(Fp2* result, const Fp2* a, const Fp2* b) {
    uint32_t gpu_index = cuda_get_device();
    Fp2 *d_result = nullptr, *d_a = nullptr, *d_b = nullptr;
    
    d_result = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    d_a = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    d_b = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    
    check_cuda_error(cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice));
    check_cuda_error(cudaMemcpy(d_b, b, sizeof(Fp2), cudaMemcpyHostToDevice));
    
    kernel_test_fp2_mul<<<1, 1>>>(d_result, d_a, d_b);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_device(gpu_index);
    
    check_cuda_error(cudaMemcpy(result, d_result, sizeof(Fp2), cudaMemcpyDeviceToHost));
    
    if (d_result != nullptr) cuda_drop(d_result, gpu_index);
    if (d_a != nullptr) cuda_drop(d_a, gpu_index);
    if (d_b != nullptr) cuda_drop(d_b, gpu_index);
}

void fp2_neg_gpu(Fp2* result, const Fp2* a) {
    uint32_t gpu_index = cuda_get_device();
    Fp2 *d_result = nullptr, *d_a = nullptr;
    
    d_result = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    d_a = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    
    check_cuda_error(cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice));
    
    kernel_test_fp2_neg<<<1, 1>>>(d_result, d_a);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_device(gpu_index);
    
    check_cuda_error(cudaMemcpy(result, d_result, sizeof(Fp2), cudaMemcpyDeviceToHost));
    
    if (d_result != nullptr) cuda_drop(d_result, gpu_index);
    if (d_a != nullptr) cuda_drop(d_a, gpu_index);
}

void fp2_conjugate_gpu(Fp2* result, const Fp2* a) {
    uint32_t gpu_index = cuda_get_device();
    Fp2 *d_result = nullptr, *d_a = nullptr;
    
    d_result = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    d_a = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    
    check_cuda_error(cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice));
    
    kernel_test_fp2_conjugate<<<1, 1>>>(d_result, d_a);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_device(gpu_index);
    
    check_cuda_error(cudaMemcpy(result, d_result, sizeof(Fp2), cudaMemcpyDeviceToHost));
    
    if (d_result != nullptr) cuda_drop(d_result, gpu_index);
    if (d_a != nullptr) cuda_drop(d_a, gpu_index);
}

void fp2_square_gpu(Fp2* result, const Fp2* a) {
    uint32_t gpu_index = cuda_get_device();
    Fp2 *d_result = nullptr, *d_a = nullptr;
    
    d_result = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    d_a = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    
    check_cuda_error(cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice));
    
    kernel_test_fp2_square<<<1, 1>>>(d_result, d_a);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_device(gpu_index);
    
    check_cuda_error(cudaMemcpy(result, d_result, sizeof(Fp2), cudaMemcpyDeviceToHost));
    
    if (d_result != nullptr) cuda_drop(d_result, gpu_index);
    if (d_a != nullptr) cuda_drop(d_a, gpu_index);
}

void fp2_inv_gpu(Fp2* result, const Fp2* a) {
    uint32_t gpu_index = cuda_get_device();
    Fp2 *d_result = nullptr, *d_a = nullptr;
    
    d_result = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    d_a = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    
    check_cuda_error(cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice));
    
    kernel_test_fp2_inv<<<1, 1>>>(d_result, d_a);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_device(gpu_index);
    
    check_cuda_error(cudaMemcpy(result, d_result, sizeof(Fp2), cudaMemcpyDeviceToHost));
    
    if (d_result != nullptr) cuda_drop(d_result, gpu_index);
    if (d_a != nullptr) cuda_drop(d_a, gpu_index);
}

void fp2_div_gpu(Fp2* result, const Fp2* a, const Fp2* b) {
    uint32_t gpu_index = cuda_get_device();
    Fp2 *d_result = nullptr, *d_a = nullptr, *d_b = nullptr;
    
    d_result = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    d_a = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    d_b = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    
    check_cuda_error(cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice));
    check_cuda_error(cudaMemcpy(d_b, b, sizeof(Fp2), cudaMemcpyHostToDevice));
    
    kernel_test_fp2_div<<<1, 1>>>(d_result, d_a, d_b);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_device(gpu_index);
    
    check_cuda_error(cudaMemcpy(result, d_result, sizeof(Fp2), cudaMemcpyDeviceToHost));
    
    if (d_result != nullptr) cuda_drop(d_result, gpu_index);
    if (d_a != nullptr) cuda_drop(d_a, gpu_index);
    if (d_b != nullptr) cuda_drop(d_b, gpu_index);
}

void fp2_mul_by_i_gpu(Fp2* result, const Fp2* a) {
    uint32_t gpu_index = cuda_get_device();
    Fp2 *d_result = nullptr, *d_a = nullptr;
    
    d_result = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    d_a = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    
    check_cuda_error(cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice));
    
    kernel_test_fp2_mul_by_i<<<1, 1>>>(d_result, d_a);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_device(gpu_index);
    
    check_cuda_error(cudaMemcpy(result, d_result, sizeof(Fp2), cudaMemcpyDeviceToHost));
    
    if (d_result != nullptr) cuda_drop(d_result, gpu_index);
    if (d_a != nullptr) cuda_drop(d_a, gpu_index);
}

void fp2_frobenius_gpu(Fp2* result, const Fp2* a) {
    uint32_t gpu_index = cuda_get_device();
    Fp2 *d_result = nullptr, *d_a = nullptr;
    
    d_result = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    d_a = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    
    check_cuda_error(cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice));
    
    kernel_test_fp2_frobenius<<<1, 1>>>(d_result, d_a);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_device(gpu_index);
    
    check_cuda_error(cudaMemcpy(result, d_result, sizeof(Fp2), cudaMemcpyDeviceToHost));
    
    if (d_result != nullptr) cuda_drop(d_result, gpu_index);
    if (d_a != nullptr) cuda_drop(d_a, gpu_index);
}

int fp2_cmp_gpu(const Fp2* a, const Fp2* b) {
    uint32_t gpu_index = cuda_get_device();
    int *d_result = nullptr, *h_result = nullptr;
    Fp2 *d_a = nullptr, *d_b = nullptr;
    int result = 0;
    
    h_result = new int;
    d_result = (int*)cuda_malloc(sizeof(int), gpu_index);
    d_a = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    d_b = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    
    check_cuda_error(cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice));
    check_cuda_error(cudaMemcpy(d_b, b, sizeof(Fp2), cudaMemcpyHostToDevice));
    
    kernel_test_fp2_cmp<<<1, 1>>>(d_result, d_a, d_b);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_device(gpu_index);
    
    check_cuda_error(cudaMemcpy(h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    result = *h_result;
    
    if (d_result != nullptr) cuda_drop(d_result, gpu_index);
    if (d_a != nullptr) cuda_drop(d_a, gpu_index);
    if (d_b != nullptr) cuda_drop(d_b, gpu_index);
    if (h_result != nullptr) delete h_result;
    return result;
}

bool fp2_is_zero_gpu(const Fp2* a) {
    uint32_t gpu_index = cuda_get_device();
    bool *d_result = nullptr, *h_result = nullptr;
    Fp2 *d_a = nullptr;
    bool result = false;
    
    h_result = new bool;
    d_result = (bool*)cuda_malloc(sizeof(bool), gpu_index);
    d_a = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    
    check_cuda_error(cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice));
    
    kernel_test_fp2_is_zero<<<1, 1>>>(d_result, d_a);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_device(gpu_index);
    
    check_cuda_error(cudaMemcpy(h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost));
    result = *h_result;
    
    if (d_result != nullptr) cuda_drop(d_result, gpu_index);
    if (d_a != nullptr) cuda_drop(d_a, gpu_index);
    if (h_result != nullptr) delete h_result;
    return result;
}

bool fp2_is_one_gpu(const Fp2* a) {
    uint32_t gpu_index = cuda_get_device();
    bool *d_result = nullptr, *h_result = nullptr;
    Fp2 *d_a = nullptr;
    bool result = false;
    
    h_result = new bool;
    d_result = (bool*)cuda_malloc(sizeof(bool), gpu_index);
    d_a = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    
    check_cuda_error(cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice));
    
    kernel_test_fp2_is_one<<<1, 1>>>(d_result, d_a);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_device(gpu_index);
    
    check_cuda_error(cudaMemcpy(h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost));
    result = *h_result;
    
    if (d_result != nullptr) cuda_drop(d_result, gpu_index);
    if (d_a != nullptr) cuda_drop(d_a, gpu_index);
    if (h_result != nullptr) delete h_result;
    return result;
}


void fp2_copy_gpu(Fp2* result, const Fp2* a) {
    uint32_t gpu_index = cuda_get_device();
    Fp2 *d_result = nullptr, *d_a = nullptr;
    
    d_result = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    d_a = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    
    check_cuda_error(cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice));
    
    kernel_test_fp2_copy<<<1, 1>>>(d_result, d_a);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_device(gpu_index);
    
    check_cuda_error(cudaMemcpy(result, d_result, sizeof(Fp2), cudaMemcpyDeviceToHost));
    
    if (d_result != nullptr) cuda_drop(d_result, gpu_index);
    if (d_a != nullptr) cuda_drop(d_a, gpu_index);
}

void fp2_cmov_gpu(Fp2* result, const Fp2* src, uint64_t condition) {
    uint32_t gpu_index = cuda_get_device();
    Fp2 *d_result = nullptr, *d_src = nullptr;
    uint64_t *d_condition = nullptr;
    
    d_result = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    d_src = (Fp2*)cuda_malloc(sizeof(Fp2), gpu_index);
    d_condition = (uint64_t*)cuda_malloc(sizeof(uint64_t), gpu_index);
    
    // Copy result first (it's the destination that may be modified)
    check_cuda_error(cudaMemcpy(d_result, result, sizeof(Fp2), cudaMemcpyHostToDevice));
    check_cuda_error(cudaMemcpy(d_src, src, sizeof(Fp2), cudaMemcpyHostToDevice));
    check_cuda_error(cudaMemcpy(d_condition, &condition, sizeof(uint64_t), cudaMemcpyHostToDevice));
    
    kernel_test_fp2_cmov<<<1, 1>>>(d_result, d_src, condition);
    check_cuda_error(cudaGetLastError());
    
    cuda_synchronize_device(gpu_index);
    
    check_cuda_error(cudaMemcpy(result, d_result, sizeof(Fp2), cudaMemcpyDeviceToHost));
    
    if (d_result != nullptr) cuda_drop(d_result, gpu_index);
    if (d_src != nullptr) cuda_drop(d_src, gpu_index);
    if (d_condition != nullptr) cuda_drop(d_condition, gpu_index);
}

// ============================================================================
// Async/Sync API implementations
// All pointers are device pointers (already allocated)
// ============================================================================

// Addition: d_c = d_a + d_b
void fp2_add_async(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a, const Fp2* d_b) {
    PANIC_IF_FALSE(d_c != nullptr && d_a != nullptr && d_b != nullptr, "fp2_add_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_test_fp2_add<<<1, 1, 0, stream>>>(d_c, d_a, d_b);
    check_cuda_error(cudaGetLastError());
}

void fp2_add(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a, const Fp2* d_b) {
    fp2_add_async(stream, gpu_index, d_c, d_a, d_b);
    cuda_synchronize_stream(stream, gpu_index);
}

// Subtraction: d_c = d_a - d_b
void fp2_sub_async(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a, const Fp2* d_b) {
    PANIC_IF_FALSE(d_c != nullptr && d_a != nullptr && d_b != nullptr, "fp2_sub_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_test_fp2_sub<<<1, 1, 0, stream>>>(d_c, d_a, d_b);
    check_cuda_error(cudaGetLastError());
}

void fp2_sub(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a, const Fp2* d_b) {
    fp2_sub_async(stream, gpu_index, d_c, d_a, d_b);
    cuda_synchronize_stream(stream, gpu_index);
}

// Multiplication: d_c = d_a * d_b
void fp2_mul_async(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a, const Fp2* d_b) {
    PANIC_IF_FALSE(d_c != nullptr && d_a != nullptr && d_b != nullptr, "fp2_mul_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_test_fp2_mul<<<1, 1, 0, stream>>>(d_c, d_a, d_b);
    check_cuda_error(cudaGetLastError());
}

void fp2_mul(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a, const Fp2* d_b) {
    fp2_mul_async(stream, gpu_index, d_c, d_a, d_b);
    cuda_synchronize_stream(stream, gpu_index);
}

// Squaring: d_c = d_a^2
void fp2_square_async(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a) {
    PANIC_IF_FALSE(d_c != nullptr && d_a != nullptr, "fp2_square_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_test_fp2_square<<<1, 1, 0, stream>>>(d_c, d_a);
    check_cuda_error(cudaGetLastError());
}

void fp2_square(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a) {
    fp2_square_async(stream, gpu_index, d_c, d_a);
    cuda_synchronize_stream(stream, gpu_index);
}

// Negation: d_c = -d_a
void fp2_neg_async(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a) {
    PANIC_IF_FALSE(d_c != nullptr && d_a != nullptr, "fp2_neg_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_test_fp2_neg<<<1, 1, 0, stream>>>(d_c, d_a);
    check_cuda_error(cudaGetLastError());
}

void fp2_neg(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a) {
    fp2_neg_async(stream, gpu_index, d_c, d_a);
    cuda_synchronize_stream(stream, gpu_index);
}

// Conjugation: d_c = d_a.conjugate()
void fp2_conjugate_async(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a) {
    PANIC_IF_FALSE(d_c != nullptr && d_a != nullptr, "fp2_conjugate_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_test_fp2_conjugate<<<1, 1, 0, stream>>>(d_c, d_a);
    check_cuda_error(cudaGetLastError());
}

void fp2_conjugate(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a) {
    fp2_conjugate_async(stream, gpu_index, d_c, d_a);
    cuda_synchronize_stream(stream, gpu_index);
}

// Inversion: d_c = d_a^(-1)
void fp2_inv_async(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a) {
    PANIC_IF_FALSE(d_c != nullptr && d_a != nullptr, "fp2_inv_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_test_fp2_inv<<<1, 1, 0, stream>>>(d_c, d_a);
    check_cuda_error(cudaGetLastError());
}

void fp2_inv(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a) {
    fp2_inv_async(stream, gpu_index, d_c, d_a);
    cuda_synchronize_stream(stream, gpu_index);
}

// Division: d_c = d_a / d_b
void fp2_div_async(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a, const Fp2* d_b) {
    PANIC_IF_FALSE(d_c != nullptr && d_a != nullptr && d_b != nullptr, "fp2_div_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_test_fp2_div<<<1, 1, 0, stream>>>(d_c, d_a, d_b);
    check_cuda_error(cudaGetLastError());
}

void fp2_div(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a, const Fp2* d_b) {
    fp2_div_async(stream, gpu_index, d_c, d_a, d_b);
    cuda_synchronize_stream(stream, gpu_index);
}

// Copy: d_dst = d_src
void fp2_copy_async(cudaStream_t stream, uint32_t gpu_index, Fp2* d_dst, const Fp2* d_src) {
    PANIC_IF_FALSE(d_dst != nullptr && d_src != nullptr, "fp2_copy_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_test_fp2_copy<<<1, 1, 0, stream>>>(d_dst, d_src);
    check_cuda_error(cudaGetLastError());
}

void fp2_copy(cudaStream_t stream, uint32_t gpu_index, Fp2* d_dst, const Fp2* d_src) {
    fp2_copy_async(stream, gpu_index, d_dst, d_src);
    cuda_synchronize_stream(stream, gpu_index);
}
