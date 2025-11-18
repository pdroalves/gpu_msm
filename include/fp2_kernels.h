#pragma once

#include "fp2.h"
#include <cstdint>
#include <cuda_runtime.h>

// CUDA kernel declarations and host wrappers for Fp2

// Host wrapper: Add two arrays element-wise
// c[i] = a[i] + b[i]
// stream: CUDA stream to use for all operations
// gpu_index: GPU device index to use
void fp2_add_array_host(cudaStream_t stream, uint32_t gpu_index, Fp2* c, const Fp2* a, const Fp2* b, int n);

// Host wrapper: Multiply two arrays element-wise
// c[i] = a[i] * b[i]
// stream: CUDA stream to use for all operations
// gpu_index: GPU device index to use
void fp2_mul_array_host(cudaStream_t stream, uint32_t gpu_index, Fp2* c, const Fp2* a, const Fp2* b, int n);

// GPU test wrappers: Run individual arithmetic operations on GPU
// These functions launch single-operation kernels to verify arithmetic works on device
// Useful for testing that arithmetic operations work correctly when called from device code

void fp2_add_gpu(Fp2* result, const Fp2* a, const Fp2* b);
void fp2_sub_gpu(Fp2* result, const Fp2* a, const Fp2* b);
void fp2_mul_gpu(Fp2* result, const Fp2* a, const Fp2* b);
void fp2_neg_gpu(Fp2* result, const Fp2* a);
void fp2_conjugate_gpu(Fp2* result, const Fp2* a);
void fp2_square_gpu(Fp2* result, const Fp2* a);
void fp2_inv_gpu(Fp2* result, const Fp2* a);
void fp2_div_gpu(Fp2* result, const Fp2* a, const Fp2* b);
void fp2_mul_by_i_gpu(Fp2* result, const Fp2* a);
void fp2_frobenius_gpu(Fp2* result, const Fp2* a);
int fp2_cmp_gpu(const Fp2* a, const Fp2* b);
bool fp2_is_zero_gpu(const Fp2* a);
bool fp2_is_one_gpu(const Fp2* a);
void fp2_copy_gpu(Fp2* result, const Fp2* a);
void fp2_cmov_gpu(Fp2* result, const Fp2* src, uint64_t condition);

