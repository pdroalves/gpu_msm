#pragma once

#include "fp.h"
#include <cstdint>
#include <cuda_runtime.h>

// CUDA kernel declarations and host wrappers

// Host wrapper: Add two arrays element-wise
// c[i] = a[i] + b[i] mod p
// Allocates/frees device memory on each call (inefficient for repeated use)
// For better performance, use fp_add_array_device with pre-allocated device memory
// stream: CUDA stream to use for all operations
// gpu_index: GPU device index to use
void fp_add_array_host(cudaStream_t stream, uint32_t gpu_index, Fp* c, const Fp* a, const Fp* b, int n);

// Host wrapper: Multiply two arrays element-wise
// c[i] = a[i] * b[i] mod p
// Allocates/frees device memory on each call (inefficient for repeated use)
// For better performance, use fp_mul_array_device with pre-allocated device memory
// stream: CUDA stream to use for all operations
// gpu_index: GPU device index to use
void fp_mul_array_host(cudaStream_t stream, uint32_t gpu_index, Fp* c, const Fp* a, const Fp* b, int n);

// Device-resident API: Work with device memory directly (more efficient for batch operations)
// These functions assume all pointers point to device memory that is already allocated.
// The caller is responsible for memory management.

// Add two arrays on device: d_c[i] = d_a[i] + d_b[i] mod p
// All pointers must be valid device memory
// stream: CUDA stream to use for kernel launch
// gpu_index: GPU device index to use
void fp_add_array_device(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a, const Fp* d_b, int n);

// Multiply two arrays on device: d_c[i] = d_a[i] * d_b[i] mod p
// All pointers must be valid device memory
// stream: CUDA stream to use for kernel launch
// gpu_index: GPU device index to use
void fp_mul_array_device(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a, const Fp* d_b, int n);

// GPU test wrappers: Run individual arithmetic operations on GPU
// These functions launch single-operation kernels to verify arithmetic works on device
// Useful for testing that arithmetic operations work correctly when called from device code
// stream: CUDA stream to use for all operations
// gpu_index: GPU device index to use

void fp_add_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a, const Fp* b);
void fp_sub_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a, const Fp* b);
void fp_mul_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a, const Fp* b);
void fp_neg_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a);
void fp_inv_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a);
void fp_div_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a, const Fp* b);
int fp_cmp_gpu(cudaStream_t stream, uint32_t gpu_index, const Fp* a, const Fp* b);
bool fp_is_zero_gpu(cudaStream_t stream, uint32_t gpu_index, const Fp* a);
bool fp_is_one_gpu(cudaStream_t stream, uint32_t gpu_index, const Fp* a);
void fp_to_montgomery_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a);
void fp_from_montgomery_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a);
void fp_mont_mul_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a, const Fp* b);
void fp_copy_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a);
void fp_cmov_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* src, uint64_t condition);
bool fp_sqrt_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* a);
bool fp_is_quadratic_residue_gpu(cudaStream_t stream, uint32_t gpu_index, const Fp* a);
void fp_pow_u64_gpu(cudaStream_t stream, uint32_t gpu_index, Fp* result, const Fp* base, uint64_t exp);

