#include "fp2.h"
#include "fp2_kernels.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

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
    if (n < 0) {
        fprintf(stderr, "fp2_add_array_host: invalid size n=%d\n", n);
        return;
    }
    if (n == 0) {
        return;  // Nothing to do
    }
    if (c == nullptr || a == nullptr || b == nullptr) {
        fprintf(stderr, "fp2_add_array_host: null pointer argument\n");
        return;
    }
    
    // Set the device context
    cudaError_t err = cudaSetDevice(gpu_index);
    CHECK_CUDA(err);
    if (err != cudaSuccess) return;
    
    // Declare all variables at the top to avoid goto issues
    Fp2 *d_c = nullptr, *d_a = nullptr, *d_b = nullptr;
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Allocate device memory (asynchronous with stream)
    err = cudaMallocAsync(&d_c, n * sizeof(Fp2), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_a, n * sizeof(Fp2), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_b, n * sizeof(Fp2), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // Copy to device (asynchronous with stream)
    err = cudaMemcpyAsync(d_a, a, n * sizeof(Fp2), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_b, b, n * sizeof(Fp2), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // Launch kernel (with stream)
    kernel_fp2_add_array<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_c, d_a, d_b, n);
    
    // Check for kernel launch errors
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // Synchronize stream to ensure kernel completes before copying back
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // Copy back (synchronous after stream sync)
    err = cudaMemcpy(c, d_c, n * sizeof(Fp2), cudaMemcpyDeviceToHost);
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

void fp2_mul_array_host(cudaStream_t stream, uint32_t gpu_index, Fp2* c, const Fp2* a, const Fp2* b, int n) {
    // Validate inputs
    if (n < 0) {
        fprintf(stderr, "fp2_mul_array_host: invalid size n=%d\n", n);
        return;
    }
    if (n == 0) {
        return;  // Nothing to do
    }
    if (c == nullptr || a == nullptr || b == nullptr) {
        fprintf(stderr, "fp2_mul_array_host: null pointer argument\n");
        return;
    }
    
    // Set the device context
    cudaError_t err = cudaSetDevice(gpu_index);
    CHECK_CUDA(err);
    if (err != cudaSuccess) return;
    
    // Declare all variables at the top to avoid goto issues
    Fp2 *d_c = nullptr, *d_a = nullptr, *d_b = nullptr;
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Allocate device memory (asynchronous with stream)
    err = cudaMallocAsync(&d_c, n * sizeof(Fp2), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_a, n * sizeof(Fp2), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMallocAsync(&d_b, n * sizeof(Fp2), stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // Copy to device (asynchronous with stream)
    err = cudaMemcpyAsync(d_a, a, n * sizeof(Fp2), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpyAsync(d_b, b, n * sizeof(Fp2), cudaMemcpyHostToDevice, stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // Launch kernel (with stream)
    kernel_fp2_mul_array<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_c, d_a, d_b, n);
    
    // Check for kernel launch errors
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // Synchronize stream to ensure kernel completes before copying back
    err = cudaStreamSynchronize(stream);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // Copy back (synchronous after stream sync)
    err = cudaMemcpy(c, d_c, n * sizeof(Fp2), cudaMemcpyDeviceToHost);
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

void fp2_add_gpu(Fp2* result, const Fp2* a, const Fp2* b) {
    Fp2 *d_result = nullptr, *d_a = nullptr, *d_b = nullptr;
    cudaError_t err;
    
    err = cudaMalloc(&d_result, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_a, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_b, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_b, b, sizeof(Fp2), cudaMemcpyHostToDevice);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp2_add<<<1, 1>>>(d_result, d_a, d_b);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaDeviceSynchronize();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(result, d_result, sizeof(Fp2), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    
cleanup:
    if (d_result != nullptr) cudaFree(d_result);
    if (d_a != nullptr) cudaFree(d_a);
    if (d_b != nullptr) cudaFree(d_b);
}

void fp2_sub_gpu(Fp2* result, const Fp2* a, const Fp2* b) {
    Fp2 *d_result = nullptr, *d_a = nullptr, *d_b = nullptr;
    cudaError_t err;
    
    err = cudaMalloc(&d_result, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_a, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_b, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_b, b, sizeof(Fp2), cudaMemcpyHostToDevice);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp2_sub<<<1, 1>>>(d_result, d_a, d_b);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaDeviceSynchronize();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(result, d_result, sizeof(Fp2), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    
cleanup:
    if (d_result != nullptr) cudaFree(d_result);
    if (d_a != nullptr) cudaFree(d_a);
    if (d_b != nullptr) cudaFree(d_b);
}

void fp2_mul_gpu(Fp2* result, const Fp2* a, const Fp2* b) {
    Fp2 *d_result = nullptr, *d_a = nullptr, *d_b = nullptr;
    cudaError_t err;
    
    err = cudaMalloc(&d_result, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_a, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_b, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_b, b, sizeof(Fp2), cudaMemcpyHostToDevice);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp2_mul<<<1, 1>>>(d_result, d_a, d_b);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaDeviceSynchronize();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(result, d_result, sizeof(Fp2), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    
cleanup:
    if (d_result != nullptr) cudaFree(d_result);
    if (d_a != nullptr) cudaFree(d_a);
    if (d_b != nullptr) cudaFree(d_b);
}

void fp2_neg_gpu(Fp2* result, const Fp2* a) {
    Fp2 *d_result = nullptr, *d_a = nullptr;
    cudaError_t err;
    
    err = cudaMalloc(&d_result, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_a, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp2_neg<<<1, 1>>>(d_result, d_a);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaDeviceSynchronize();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(result, d_result, sizeof(Fp2), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    
cleanup:
    if (d_result != nullptr) cudaFree(d_result);
    if (d_a != nullptr) cudaFree(d_a);
}

void fp2_conjugate_gpu(Fp2* result, const Fp2* a) {
    Fp2 *d_result = nullptr, *d_a = nullptr;
    cudaError_t err;
    
    err = cudaMalloc(&d_result, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_a, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp2_conjugate<<<1, 1>>>(d_result, d_a);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaDeviceSynchronize();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(result, d_result, sizeof(Fp2), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    
cleanup:
    if (d_result != nullptr) cudaFree(d_result);
    if (d_a != nullptr) cudaFree(d_a);
}

void fp2_square_gpu(Fp2* result, const Fp2* a) {
    Fp2 *d_result = nullptr, *d_a = nullptr;
    cudaError_t err;
    
    err = cudaMalloc(&d_result, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_a, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp2_square<<<1, 1>>>(d_result, d_a);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaDeviceSynchronize();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(result, d_result, sizeof(Fp2), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    
cleanup:
    if (d_result != nullptr) cudaFree(d_result);
    if (d_a != nullptr) cudaFree(d_a);
}

void fp2_inv_gpu(Fp2* result, const Fp2* a) {
    Fp2 *d_result = nullptr, *d_a = nullptr;
    cudaError_t err;
    
    err = cudaMalloc(&d_result, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_a, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp2_inv<<<1, 1>>>(d_result, d_a);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaDeviceSynchronize();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(result, d_result, sizeof(Fp2), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    
cleanup:
    if (d_result != nullptr) cudaFree(d_result);
    if (d_a != nullptr) cudaFree(d_a);
}

void fp2_div_gpu(Fp2* result, const Fp2* a, const Fp2* b) {
    Fp2 *d_result = nullptr, *d_a = nullptr, *d_b = nullptr;
    cudaError_t err;
    
    err = cudaMalloc(&d_result, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_a, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_b, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_b, b, sizeof(Fp2), cudaMemcpyHostToDevice);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp2_div<<<1, 1>>>(d_result, d_a, d_b);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaDeviceSynchronize();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(result, d_result, sizeof(Fp2), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    
cleanup:
    if (d_result != nullptr) cudaFree(d_result);
    if (d_a != nullptr) cudaFree(d_a);
    if (d_b != nullptr) cudaFree(d_b);
}

void fp2_mul_by_i_gpu(Fp2* result, const Fp2* a) {
    Fp2 *d_result = nullptr, *d_a = nullptr;
    cudaError_t err;
    
    err = cudaMalloc(&d_result, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_a, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp2_mul_by_i<<<1, 1>>>(d_result, d_a);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaDeviceSynchronize();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(result, d_result, sizeof(Fp2), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    
cleanup:
    if (d_result != nullptr) cudaFree(d_result);
    if (d_a != nullptr) cudaFree(d_a);
}

void fp2_frobenius_gpu(Fp2* result, const Fp2* a) {
    Fp2 *d_result = nullptr, *d_a = nullptr;
    cudaError_t err;
    
    err = cudaMalloc(&d_result, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_a, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp2_frobenius<<<1, 1>>>(d_result, d_a);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaDeviceSynchronize();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(result, d_result, sizeof(Fp2), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    
cleanup:
    if (d_result != nullptr) cudaFree(d_result);
    if (d_a != nullptr) cudaFree(d_a);
}

int fp2_cmp_gpu(const Fp2* a, const Fp2* b) {
    int *d_result = nullptr, *h_result = nullptr;
    Fp2 *d_a = nullptr, *d_b = nullptr;
    cudaError_t err;
    int result = 0;
    
    h_result = new int;
    err = cudaMalloc(&d_result, sizeof(int));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_a, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_b, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_b, b, sizeof(Fp2), cudaMemcpyHostToDevice);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp2_cmp<<<1, 1>>>(d_result, d_a, d_b);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaDeviceSynchronize();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    if (err == cudaSuccess) {
        result = *h_result;
    }
    
cleanup:
    if (d_result != nullptr) cudaFree(d_result);
    if (d_a != nullptr) cudaFree(d_a);
    if (d_b != nullptr) cudaFree(d_b);
    if (h_result != nullptr) delete h_result;
    return result;
}

bool fp2_is_zero_gpu(const Fp2* a) {
    bool *d_result = nullptr, *h_result = nullptr;
    Fp2 *d_a = nullptr;
    cudaError_t err;
    bool result = false;
    
    h_result = new bool;
    err = cudaMalloc(&d_result, sizeof(bool));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_a, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp2_is_zero<<<1, 1>>>(d_result, d_a);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaDeviceSynchronize();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    if (err == cudaSuccess) {
        result = *h_result;
    }
    
cleanup:
    if (d_result != nullptr) cudaFree(d_result);
    if (d_a != nullptr) cudaFree(d_a);
    if (h_result != nullptr) delete h_result;
    return result;
}

bool fp2_is_one_gpu(const Fp2* a) {
    bool *d_result = nullptr, *h_result = nullptr;
    Fp2 *d_a = nullptr;
    cudaError_t err;
    bool result = false;
    
    h_result = new bool;
    err = cudaMalloc(&d_result, sizeof(bool));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_a, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp2_is_one<<<1, 1>>>(d_result, d_a);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaDeviceSynchronize();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    if (err == cudaSuccess) {
        result = *h_result;
    }
    
cleanup:
    if (d_result != nullptr) cudaFree(d_result);
    if (d_a != nullptr) cudaFree(d_a);
    if (h_result != nullptr) delete h_result;
    return result;
}


void fp2_copy_gpu(Fp2* result, const Fp2* a) {
    Fp2 *d_result = nullptr, *d_a = nullptr;
    cudaError_t err;
    
    err = cudaMalloc(&d_result, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_a, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_a, a, sizeof(Fp2), cudaMemcpyHostToDevice);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp2_copy<<<1, 1>>>(d_result, d_a);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaDeviceSynchronize();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(result, d_result, sizeof(Fp2), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    
cleanup:
    if (d_result != nullptr) cudaFree(d_result);
    if (d_a != nullptr) cudaFree(d_a);
}

void fp2_cmov_gpu(Fp2* result, const Fp2* src, uint64_t condition) {
    Fp2 *d_result = nullptr, *d_src = nullptr;
    uint64_t *d_condition = nullptr;
    cudaError_t err;
    
    err = cudaMalloc(&d_result, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_src, sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_condition, sizeof(uint64_t));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // Copy result first (it's the destination that may be modified)
    err = cudaMemcpy(d_result, result, sizeof(Fp2), cudaMemcpyHostToDevice);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_src, src, sizeof(Fp2), cudaMemcpyHostToDevice);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_condition, &condition, sizeof(uint64_t), cudaMemcpyHostToDevice);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    kernel_test_fp2_cmov<<<1, 1>>>(d_result, d_src, condition);
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaDeviceSynchronize();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(result, d_result, sizeof(Fp2), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    
cleanup:
    if (d_result != nullptr) cudaFree(d_result);
    if (d_src != nullptr) cudaFree(d_src);
    if (d_condition != nullptr) cudaFree(d_condition);
}
