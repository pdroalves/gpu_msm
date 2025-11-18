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

// Host wrapper functions
void fp2_add_array_host(Fp2* c, const Fp2* a, const Fp2* b, int n) {
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
    
    // Declare all variables at the top to avoid goto issues
    Fp2 *d_c = nullptr, *d_a = nullptr, *d_b = nullptr;
    cudaError_t err;
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Allocate device memory
    err = cudaMalloc(&d_c, n * sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_a, n * sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_b, n * sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // Copy to device
    err = cudaMemcpy(d_a, a, n * sizeof(Fp2), cudaMemcpyHostToDevice);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_b, b, n * sizeof(Fp2), cudaMemcpyHostToDevice);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // Launch kernel
    kernel_fp2_add_array<<<blocksPerGrid, threadsPerBlock>>>(d_c, d_a, d_b, n);
    
    // Check for kernel launch errors
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // Synchronize to ensure kernel completes
    err = cudaDeviceSynchronize();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // Copy back
    err = cudaMemcpy(c, d_c, n * sizeof(Fp2), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
cleanup:
    // Free device memory (safe to call even if pointers are null)
    if (d_c != nullptr) {
        cudaFree(d_c);
    }
    if (d_a != nullptr) {
        cudaFree(d_a);
    }
    if (d_b != nullptr) {
        cudaFree(d_b);
    }
}

void fp2_mul_array_host(Fp2* c, const Fp2* a, const Fp2* b, int n) {
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
    
    // Declare all variables at the top to avoid goto issues
    Fp2 *d_c = nullptr, *d_a = nullptr, *d_b = nullptr;
    cudaError_t err;
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Allocate device memory
    err = cudaMalloc(&d_c, n * sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_a, n * sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_b, n * sizeof(Fp2));
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // Copy to device
    err = cudaMemcpy(d_a, a, n * sizeof(Fp2), cudaMemcpyHostToDevice);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_b, b, n * sizeof(Fp2), cudaMemcpyHostToDevice);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // Launch kernel
    kernel_fp2_mul_array<<<blocksPerGrid, threadsPerBlock>>>(d_c, d_a, d_b, n);
    
    // Check for kernel launch errors
    err = cudaPeekAtLastError();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // Synchronize to ensure kernel completes
    err = cudaDeviceSynchronize();
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
    // Copy back
    err = cudaMemcpy(c, d_c, n * sizeof(Fp2), cudaMemcpyDeviceToHost);
    CHECK_CUDA(err);
    if (err != cudaSuccess) goto cleanup;
    
cleanup:
    // Free device memory (safe to call even if pointers are null)
    if (d_c != nullptr) {
        cudaFree(d_c);
    }
    if (d_a != nullptr) {
        cudaFree(d_a);
    }
    if (d_b != nullptr) {
        cudaFree(d_b);
    }
}

