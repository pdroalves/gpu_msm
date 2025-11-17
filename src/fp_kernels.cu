#include "fp.h"
#include <cuda_runtime.h>

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

// Host wrapper functions
void fp_add_array_host(Fp* c, const Fp* a, const Fp* b, int n) {
    Fp *d_c, *d_a, *d_b;
    
    // Allocate device memory
    cudaMalloc(&d_c, n * sizeof(Fp));
    cudaMalloc(&d_a, n * sizeof(Fp));
    cudaMalloc(&d_b, n * sizeof(Fp));
    
    // Copy to device
    cudaMemcpy(d_a, a, n * sizeof(Fp), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(Fp), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    kernel_fp_add_array<<<blocksPerGrid, threadsPerBlock>>>(d_c, d_a, d_b, n);
    
    // Copy back
    cudaMemcpy(c, d_c, n * sizeof(Fp), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_c);
    cudaFree(d_a);
    cudaFree(d_b);
}

void fp_mul_array_host(Fp* c, const Fp* a, const Fp* b, int n) {
    Fp *d_c, *d_a, *d_b;
    
    // Allocate device memory
    cudaMalloc(&d_c, n * sizeof(Fp));
    cudaMalloc(&d_a, n * sizeof(Fp));
    cudaMalloc(&d_b, n * sizeof(Fp));
    
    // Copy to device
    cudaMemcpy(d_a, a, n * sizeof(Fp), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(Fp), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    kernel_fp_mul_array<<<blocksPerGrid, threadsPerBlock>>>(d_c, d_a, d_b, n);
    
    // Copy back
    cudaMemcpy(c, d_c, n * sizeof(Fp), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_c);
    cudaFree(d_a);
    cudaFree(d_b);
}

