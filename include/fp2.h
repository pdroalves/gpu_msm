#pragma once

#include "fp.h"
#include <cstdint>
#include <cuda_runtime.h>

// Fp2: Quadratic extension field Fp[i] where i^2 = -1
// An Fp2 element is represented as a + b*i where a, b ∈ Fp
// This is a POD type for CUDA compatibility
struct Fp2 {
    Fp c0;  // Real part (coefficient of 1)
    Fp c1;  // Imaginary part (coefficient of i)
};

// Multi-precision arithmetic operations for Fp2
// All operations are CUDA-compatible (can be called from host or device)

// Comparison: returns -1 if a < b, 0 if a == b, 1 if a > b
// Comparison is lexicographic: compare c0 first, then c1
__host__ __device__ int fp2_cmp(const Fp2& a, const Fp2& b);

// Check if a == 0
__host__ __device__ bool fp2_is_zero(const Fp2& a);

// Check if a == 1
__host__ __device__ bool fp2_is_one(const Fp2& a);

// Set to zero
__host__ __device__ void fp2_zero(Fp2& a);

// Set to one (1 + 0*i)
__host__ __device__ void fp2_one(Fp2& a);

// Copy: dst = src
__host__ __device__ void fp2_copy(Fp2& dst, const Fp2& src);

// Addition: c = a + b
__host__ __device__ void fp2_add(Fp2& c, const Fp2& a, const Fp2& b);

// Subtraction: c = a - b
__host__ __device__ void fp2_sub(Fp2& c, const Fp2& a, const Fp2& b);

// Multiplication: c = a * b
// (a0 + a1*i) * (b0 + b1*i) = (a0*b0 - a1*b1) + (a0*b1 + a1*b0)*i
__host__ __device__ void fp2_mul(Fp2& c, const Fp2& a, const Fp2& b);

// Squaring: c = a^2
// (a0 + a1*i)^2 = (a0^2 - a1^2) + 2*a0*a1*i
// Optimized version that uses fewer multiplications
__host__ __device__ void fp2_square(Fp2& c, const Fp2& a);

// Negation: c = -a
__host__ __device__ void fp2_neg(Fp2& c, const Fp2& a);

// Conjugation: c = a.conjugate() = a0 - a1*i
__host__ __device__ void fp2_conjugate(Fp2& c, const Fp2& a);

// Inversion: c = a^(-1)
// Uses the formula: (a0 + a1*i)^(-1) = (a0 - a1*i) / (a0^2 + a1^2)
__host__ __device__ void fp2_inv(Fp2& c, const Fp2& a);

// Division: c = a / b = a * b^(-1)
__host__ __device__ void fp2_div(Fp2& c, const Fp2& a, const Fp2& b);

// Conditional assignment: if condition, dst = src, else dst unchanged
__host__ __device__ void fp2_cmov(Fp2& dst, const Fp2& src, uint64_t condition);

// Frobenius map: c = a^p
// For Fp2, the Frobenius map is: (a0 + a1*i)^p = a0 - a1*i = conjugate
// This is because i^p = i^(p mod 4) = i^(-1) = -i (since p ≡ 3 mod 4 for BLS12 curves)
__host__ __device__ void fp2_frobenius(Fp2& c, const Fp2& a);

// Multiply by i: c = a * i
// (a0 + a1*i) * i = -a1 + a0*i
__host__ __device__ void fp2_mul_by_i(Fp2& c, const Fp2& a);

// Multiply by non-residue: For BLS12, non-residue is typically i
// This is the same as mul_by_i
__host__ __device__ void fp2_mul_by_non_residue(Fp2& c, const Fp2& a);

// ============================================================================
// Async/Sync API for device memory operations
// ============================================================================
// All pointers in these functions point to device memory (already allocated)
// _async versions: Launch kernels asynchronously, return immediately (no sync)
//  versions: Call _async then synchronize stream
// ============================================================================

// Addition: d_c = d_a + d_b (all device pointers)
void fp2_add_async(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a, const Fp2* d_b);
void fp2_add(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a, const Fp2* d_b);

// Subtraction: d_c = d_a - d_b (all device pointers)
void fp2_sub_async(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a, const Fp2* d_b);
void fp2_sub(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a, const Fp2* d_b);

// Multiplication: d_c = d_a * d_b (all device pointers)
void fp2_mul_async(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a, const Fp2* d_b);
void fp2_mul(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a, const Fp2* d_b);

// Squaring: d_c = d_a^2 (all device pointers)
void fp2_square_async(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a);
void fp2_square(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a);

// Negation: d_c = -d_a (all device pointers)
void fp2_neg_async(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a);
void fp2_neg(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a);

// Conjugation: d_c = d_a.conjugate() = d_a0 - d_a1*i (all device pointers)
void fp2_conjugate_async(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a);
void fp2_conjugate(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a);

// Inversion: d_c = d_a^(-1) (all device pointers)
void fp2_inv_async(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a);
void fp2_inv(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a);

// Division: d_c = d_a / d_b (all device pointers)
void fp2_div_async(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a, const Fp2* d_b);
void fp2_div(cudaStream_t stream, uint32_t gpu_index, Fp2* d_c, const Fp2* d_a, const Fp2* d_b);

// Copy: d_dst = d_src (all device pointers)
void fp2_copy_async(cudaStream_t stream, uint32_t gpu_index, Fp2* d_dst, const Fp2* d_src);
void fp2_copy(cudaStream_t stream, uint32_t gpu_index, Fp2* d_dst, const Fp2* d_src);

