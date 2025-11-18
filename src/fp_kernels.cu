#include "fp.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Example CUDA kernel for parallel Fp operations
// This demonstrates how to use the Fp arithmetic in CUDA kernels

// Helper function to check CUDA errors
static void checkCudaError(cudaError_t err, const char* file, int line, const char* func) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d in %s: %s\n", 
                file, line, func, cudaGetErrorString(err));
        fflush(stderr);
        // Don't abort in library code - let caller handle it
    }
}

// Macro to check CUDA errors with context
#define CHECK_CUDA(err) checkCudaError(err, __FILE__, __LINE__, __FUNCTION__)

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

// Host wrapper functions
void fp_add_array_host(cudaStream_t stream, uint32_t gpu_index, Fp* c, const Fp* a, const Fp* b, int n) {
    // Validate inputs
    if (n < 0) {
        fprintf(stderr, "fp_add_array_host: invalid size n=%d\n", n);
        return;
    }
    if (n == 0) {
        return;  // Nothing to do
    }
    if (c == nullptr || a == nullptr || b == nullptr) {
        fprintf(stderr, "fp_add_array_host: null pointer argument\n");
        return;
    }
    
    // Set the device context
    cudaError_t err = cudaSetDevice(gpu_index);
    CHECK_CUDA(err);
    if (err != cudaSuccess) return;
    
    // Declare all variables at the top to avoid goto issues
    Fp *d_c = nullptr, *d_a = nullptr, *d_b = nullptr;
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Allocate device memory (asynchronous with stream)
    err = cudaMallocAsync(&d_c, n * sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_a, n * sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_b, n * sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // Copy to device (asynchronous with stream)
    err = cudaMemcpyAsync(d_a, a, n * sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_b, b, n * sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // Launch kernel (with stream)
    kernel_fp_add_array<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_c, d_a, d_b, n);
    
    // Check for kernel launch errors
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // Synchronize stream to ensure kernel completes before copying back
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // Copy back (synchronous after stream sync)
    err = cudaMemcpy(c, d_c, n * sizeof(Fp), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
cleanup:
    // Free device memory (asynchronous with stream)
    if (d_c != nullptr) {
        cudaFreeAsync(d_c, stream);
    }
    if (d_a != nullptr) {
        cudaFreeAsync(d_a, stream);
    }
    if (d_b != nullptr) {
        cudaFreeAsync(d_b, stream);
    }
}

// Device-resident API: assumes all pointers are already on device
void fp_add_array_device(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a, const Fp* d_b, int n) {
    // Validate inputs
    if (n < 0) {
        fprintf(stderr, "fp_add_array_device: invalid size n=%d\n", n);
        return;
    }
    if (n == 0) {
        return;  // Nothing to do
    }
    if (d_c == nullptr || d_a == nullptr || d_b == nullptr) {
        fprintf(stderr, "fp_add_array_device: null pointer argument\n");
        return;
    }
    
    // Set the device context
    cudaError_t err = cudaSetDevice(gpu_index);
    CHECK_CUDA(err);
    if (err != cudaSuccess) return;
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel (with stream)
    kernel_fp_add_array<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_c, d_a, d_b, n);
    
    // Check for kernel launch errors
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
}

void fp_mul_array_device(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a, const Fp* d_b, int n) {
    // Validate inputs
    if (n < 0) {
        fprintf(stderr, "fp_mul_array_device: invalid size n=%d\n", n);
        return;
    }
    if (n == 0) {
        return;  // Nothing to do
    }
    if (d_c == nullptr || d_a == nullptr || d_b == nullptr) {
        fprintf(stderr, "fp_mul_array_device: null pointer argument\n");
        return;
    }
    
    // Set the device context
    cudaError_t err = cudaSetDevice(gpu_index);
    CHECK_CUDA(err);
    if (err != cudaSuccess) return;
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel (with stream)
    kernel_fp_mul_array<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_c, d_a, d_b, n);
    
    // Check for kernel launch errors
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
}

void fp_mul_array_host(cudaStream_t stream, uint32_t gpu_index, Fp* c, const Fp* a, const Fp* b, int n) {
    // Validate inputs
    if (n < 0) {
        fprintf(stderr, "fp_mul_array_host: invalid size n=%d\n", n);
        return;
    }
    if (n == 0) {
        return;  // Nothing to do
    }
    if (c == nullptr || a == nullptr || b == nullptr) {
        fprintf(stderr, "fp_mul_array_host: null pointer argument\n");
        return;
    }
    
    // Set the device context
    cudaError_t err = cudaSetDevice(gpu_index);
    CHECK_CUDA(err);
    if (err != cudaSuccess) return;
    
    // Declare all variables at the top to avoid goto issues
    Fp *d_c = nullptr, *d_a = nullptr, *d_b = nullptr;
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Allocate device memory (asynchronous with stream)
    err = cudaMallocAsync(&d_c, n * sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_a, n * sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_b, n * sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // Copy to device (asynchronous with stream)
    err = cudaMemcpyAsync(d_a, a, n * sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_b, b, n * sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // Launch kernel (with stream)
    kernel_fp_mul_array<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_c, d_a, d_b, n);
    
    // Check for kernel launch errors
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // Synchronize stream to ensure kernel completes before copying back
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // Copy back (synchronous after stream sync)
    err = cudaMemcpy(c, d_c, n * sizeof(Fp), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
cleanup:
    // Free device memory (asynchronous with stream)
    if (d_c != nullptr) {
        cudaFreeAsync(d_c, stream);
    }
    if (d_a != nullptr) {
        cudaFreeAsync(d_a, stream);
    }
    if (d_b != nullptr) {
        cudaFreeAsync(d_b, stream);
    }
}

// Host wrapper functions for testing individual operations on GPU
// These functions launch single-operation kernels to verify arithmetic works on device

void fp_add_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a, const Fp* b) {
    // Set the device context
    cudaError_t err = cudaSetDevice(gpu_index);
    CHECK_CUDA(err);
    if (err != cudaSuccess) return;
    
    Fp *d_result = nullptr, *d_a = nullptr, *d_b = nullptr;
    
    err = cudaMallocAsync(&d_result, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_a, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_b, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_a, a, sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_b, b, sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp_add<<<1, 1, 0, stream>>>(d_result, d_a, d_b);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(result, d_result, sizeof(Fp), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    
cleanup:
    if (d_result != nullptr) cudaFreeAsync(d_result, stream);
    if (d_a != nullptr) cudaFreeAsync(d_a, stream);
    if (d_b != nullptr) cudaFreeAsync(d_b, stream);
}

void fp_sub_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a, const Fp* b) {
    // Set the device context
    cudaError_t err = cudaSetDevice(gpu_index);
    CHECK_CUDA(err);
    if (err != cudaSuccess) return;
    
    Fp *d_result = nullptr, *d_a = nullptr, *d_b = nullptr;
    
    err = cudaMallocAsync(&d_result, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_a, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_b, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_a, a, sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_b, b, sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp_sub<<<1, 1, 0, stream>>>(d_result, d_a, d_b);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(result, d_result, sizeof(Fp), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    
cleanup:
    if (d_result != nullptr) cudaFreeAsync(d_result, stream);
    if (d_a != nullptr) cudaFreeAsync(d_a, stream);
    if (d_b != nullptr) cudaFreeAsync(d_b, stream);
}

void fp_mul_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a, const Fp* b) {
    // Set the device context
    cudaError_t err = cudaSetDevice(gpu_index);
    CHECK_CUDA(err);
    if (err != cudaSuccess) return;
    
    Fp *d_result = nullptr, *d_a = nullptr, *d_b = nullptr;
    
    err = cudaMallocAsync(&d_result, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_a, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_b, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_a, a, sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_b, b, sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp_mul<<<1, 1, 0, stream>>>(d_result, d_a, d_b);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(result, d_result, sizeof(Fp), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    
cleanup:
    if (d_result != nullptr) cudaFreeAsync(d_result, stream);
    if (d_a != nullptr) cudaFreeAsync(d_a, stream);
    if (d_b != nullptr) cudaFreeAsync(d_b, stream);
}

void fp_neg_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a) {
    // Set the device context
    cudaError_t err = cudaSetDevice(gpu_index);
    CHECK_CUDA(err);
    if (err != cudaSuccess) return;
    
    Fp *d_result = nullptr, *d_a = nullptr;
    
    err = cudaMallocAsync(&d_result, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_a, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_a, a, sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp_neg<<<1, 1, 0, stream>>>(d_result, d_a);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(result, d_result, sizeof(Fp), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    
cleanup:
    if (d_result != nullptr) cudaFreeAsync(d_result, stream);
    if (d_a != nullptr) cudaFreeAsync(d_a, stream);
}

void fp_inv_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a) {
    // Set the device context
    cudaError_t err = cudaSetDevice(gpu_index);
    CHECK_CUDA(err);
    if (err != cudaSuccess) return;
    
    Fp *d_result = nullptr, *d_a = nullptr;
    
    err = cudaMallocAsync(&d_result, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_a, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_a, a, sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp_inv<<<1, 1, 0, stream>>>(d_result, d_a);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(result, d_result, sizeof(Fp), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    
cleanup:
    if (d_result != nullptr) cudaFreeAsync(d_result, stream);
    if (d_a != nullptr) cudaFreeAsync(d_a, stream);
}

void fp_div_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a, const Fp* b) {
    // Set the device context
    cudaError_t err = cudaSetDevice(gpu_index);
    CHECK_CUDA(err);
    if (err != cudaSuccess) return;
    
    Fp *d_result = nullptr, *d_a = nullptr, *d_b = nullptr;
    
    err = cudaMallocAsync(&d_result, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_a, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_b, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_a, a, sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_b, b, sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp_div<<<1, 1, 0, stream>>>(d_result, d_a, d_b);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(result, d_result, sizeof(Fp), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    
cleanup:
    if (d_result != nullptr) cudaFreeAsync(d_result, stream);
    if (d_a != nullptr) cudaFreeAsync(d_a, stream);
    if (d_b != nullptr) cudaFreeAsync(d_b, stream);
}

int fp_cmp_gpu(cudaStream_t stream, uint32_t gpu_index, const Fp* a, const Fp* b) {
    // Set the device context
    cudaError_t err = cudaSetDevice(gpu_index);
    CHECK_CUDA(err);
    if (err != cudaSuccess) return 0;
    
    int *d_result = nullptr, *h_result = nullptr;
    Fp *d_a = nullptr, *d_b = nullptr;
    int result = 0;
    
    h_result = new int;
    err = cudaMallocAsync(&d_result, sizeof(int), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_a, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_b, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_a, a, sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_b, b, sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp_cmp<<<1, 1, 0, stream>>>(d_result, d_a, d_b);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    if (err == cudaSuccess) {
        result = *h_result;
    }
    
cleanup:
    if (d_result != nullptr) cudaFreeAsync(d_result, stream);
    if (d_a != nullptr) cudaFreeAsync(d_a, stream);
    if (d_b != nullptr) cudaFreeAsync(d_b, stream);
    if (h_result != nullptr) delete h_result;
    return result;
}

bool fp_is_zero_gpu(cudaStream_t stream, uint32_t gpu_index, const Fp* a) {
    // Set the device context
    cudaError_t err = cudaSetDevice(gpu_index);
    CHECK_CUDA(err);
    if (err != cudaSuccess) return false;
    
    bool *d_result = nullptr, *h_result = nullptr;
    Fp *d_a = nullptr;
    bool result = false;
    
    h_result = new bool;
    err = cudaMallocAsync(&d_result, sizeof(bool), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_a, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_a, a, sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp_is_zero<<<1, 1, 0, stream>>>(d_result, d_a);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    if (err == cudaSuccess) {
        result = *h_result;
    }
    
cleanup:
    if (d_result != nullptr) cudaFreeAsync(d_result, stream);
    if (d_a != nullptr) cudaFreeAsync(d_a, stream);
    if (h_result != nullptr) delete h_result;
    return result;
}

bool fp_is_one_gpu(cudaStream_t stream, uint32_t gpu_index, const Fp* a) {
    // Set the device context
    cudaError_t err = cudaSetDevice(gpu_index);
    CHECK_CUDA(err);
    if (err != cudaSuccess) return false;
    
    bool *d_result = nullptr, *h_result = nullptr;
    Fp *d_a = nullptr;
    bool result = false;
    
    h_result = new bool;
    err = cudaMallocAsync(&d_result, sizeof(bool), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_a, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_a, a, sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp_is_one<<<1, 1, 0, stream>>>(d_result, d_a);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    if (err == cudaSuccess) {
        result = *h_result;
    }
    
cleanup:
    if (d_result != nullptr) cudaFreeAsync(d_result, stream);
    if (d_a != nullptr) cudaFreeAsync(d_a, stream);
    if (h_result != nullptr) delete h_result;
    return result;
}


void fp_to_montgomery_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a) {
    cudaError_t err = cudaSetDevice(gpu_index);
    CHECK_CUDA(err);
    if (err != cudaSuccess) return;
    
    Fp *d_result = nullptr, *d_a = nullptr;
    
    err = cudaMallocAsync(&d_result, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_a, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_a, a, sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp_to_montgomery<<<1, 1, 0, stream>>>(d_result, d_a);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(result, d_result, sizeof(Fp), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    
cleanup:
    if (d_result != nullptr) cudaFreeAsync(d_result, stream);
    if (d_a != nullptr) cudaFreeAsync(d_a, stream);
}

void fp_from_montgomery_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a) {
    cudaError_t err = cudaSetDevice(gpu_index);
    CHECK_CUDA(err);
    if (err != cudaSuccess) return;
    
    Fp *d_result = nullptr, *d_a = nullptr;
    
    err = cudaMallocAsync(&d_result, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_a, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_a, a, sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp_from_montgomery<<<1, 1, 0, stream>>>(d_result, d_a);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(result, d_result, sizeof(Fp), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    
cleanup:
    if (d_result != nullptr) cudaFreeAsync(d_result, stream);
    if (d_a != nullptr) cudaFreeAsync(d_a, stream);
}

void fp_mont_mul_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a, const Fp* b) {
    cudaError_t err = cudaSetDevice(gpu_index);
    CHECK_CUDA(err);
    if (err != cudaSuccess) return;
    
    Fp *d_result = nullptr, *d_a = nullptr, *d_b = nullptr;
    
    err = cudaMallocAsync(&d_result, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_a, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_b, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_a, a, sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_b, b, sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp_mont_mul<<<1, 1, 0, stream>>>(d_result, d_a, d_b);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(result, d_result, sizeof(Fp), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    
cleanup:
    if (d_result != nullptr) cudaFreeAsync(d_result, stream);
    if (d_a != nullptr) cudaFreeAsync(d_a, stream);
    if (d_b != nullptr) cudaFreeAsync(d_b, stream);
}

void fp_copy_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a) {
    cudaError_t err = cudaSetDevice(gpu_index);
    CHECK_CUDA(err);
    if (err != cudaSuccess) return;
    
    Fp *d_result = nullptr, *d_a = nullptr;
    
    err = cudaMallocAsync(&d_result, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_a, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_a, a, sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp_copy<<<1, 1, 0, stream>>>(d_result, d_a);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(result, d_result, sizeof(Fp), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    
cleanup:
    if (d_result != nullptr) cudaFreeAsync(d_result, stream);
    if (d_a != nullptr) cudaFreeAsync(d_a, stream);
}

void fp_cmov_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* src, uint64_t condition) {
    cudaError_t err = cudaSetDevice(gpu_index);
    CHECK_CUDA(err);
    if (err != cudaSuccess) return;
    
    Fp *d_result = nullptr, *d_src = nullptr;
    uint64_t *d_condition = nullptr;
    
    err = cudaMallocAsync(&d_result, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_src, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_condition, sizeof(uint64_t), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // Copy result first (it's the destination that may be modified)
    err = cudaMemcpyAsync(d_result, result, sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_src, src, sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_condition, &condition, sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp_cmov<<<1, 1, 0, stream>>>(d_result, d_src, condition);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(result, d_result, sizeof(Fp), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    
cleanup:
    if (d_result != nullptr) cudaFreeAsync(d_result, stream);
    if (d_src != nullptr) cudaFreeAsync(d_src, stream);
    if (d_condition != nullptr) cudaFreeAsync(d_condition, stream);
}

bool fp_sqrt_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a) {
    cudaError_t err = cudaSetDevice(gpu_index);
    CHECK_CUDA(err);
    if (err != cudaSuccess) return false;
    
    bool *d_has_sqrt = nullptr, *h_has_sqrt = nullptr;
    Fp *d_result = nullptr, *d_a = nullptr;
    bool has_sqrt = false;
    
    h_has_sqrt = new bool;
    err = cudaMallocAsync(&d_has_sqrt, sizeof(bool), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_result, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_a, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_a, a, sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp_sqrt<<<1, 1, 0, stream>>>(d_has_sqrt, d_result, d_a);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(h_has_sqrt, d_has_sqrt, sizeof(bool), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    if (err == cudaSuccess) {
        has_sqrt = *h_has_sqrt;
    }
    
    err = cudaMemcpy(result, d_result, sizeof(Fp), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    
cleanup:
    if (d_has_sqrt != nullptr) cudaFreeAsync(d_has_sqrt, stream);
    if (d_result != nullptr) cudaFreeAsync(d_result, stream);
    if (d_a != nullptr) cudaFreeAsync(d_a, stream);
    if (h_has_sqrt != nullptr) delete h_has_sqrt;
    return has_sqrt;
}

bool fp_is_quadratic_residue_gpu(cudaStream_t stream, uint32_t gpu_index, const Fp* a) {
    cudaError_t err = cudaSetDevice(gpu_index);
    CHECK_CUDA(err);
    if (err != cudaSuccess) return false;
    
    bool *d_result = nullptr, *h_result = nullptr;
    Fp *d_a = nullptr;
    bool result = false;
    
    h_result = new bool;
    err = cudaMallocAsync(&d_result, sizeof(bool), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_a, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_a, a, sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp_is_quadratic_residue<<<1, 1, 0, stream>>>(d_result, d_a);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    if (err == cudaSuccess) {
        result = *h_result;
    }
    
cleanup:
    if (d_result != nullptr) cudaFreeAsync(d_result, stream);
    if (d_a != nullptr) cudaFreeAsync(d_a, stream);
    if (h_result != nullptr) delete h_result;
    return result;
}

void fp_pow_u64_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* base, uint64_t exp) {
    cudaError_t err = cudaSetDevice(gpu_index);
    CHECK_CUDA(err);
    if (err != cudaSuccess) return;
    
    Fp *d_result = nullptr, *d_base = nullptr;
    
    err = cudaMallocAsync(&d_result, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_base, sizeof(Fp), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_base, base, sizeof(Fp), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // exp is a simple value, can be passed directly to kernel
    kernel_test_fp_pow_u64<<<1, 1, 0, stream>>>(d_result, d_base, exp);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(result, d_result, sizeof(Fp), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    
cleanup:
    if (d_result != nullptr) cudaFreeAsync(d_result, stream);
    if (d_base != nullptr) cudaFreeAsync(d_base, stream);
}
