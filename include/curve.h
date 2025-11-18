#pragma once

#include "fp.h"
#include "fp2.h"
#include <cstdint>
#include <cuda_runtime.h>

// Initialize device curve constants
// Must be called once per device before using curve functions on device
// stream: CUDA stream to use (can be nullptr for default stream synchronization)
// gpu_index: GPU device index to use
void init_device_curve(cudaStream_t stream, uint32_t gpu_index);

// Elliptic curve point structures for BLS12-446

// G1 point: (x, y) coordinates in Fp
// Curve equation: y^2 = x^3 + b (short Weierstrass form with a = 0)
struct G1Point {
    Fp x;
    Fp y;
    bool infinity;  // true if point at infinity (identity element)
};

// G2 point: (x, y) coordinates in Fp2
// Curve equation: y^2 = x^3 + b' (twisted curve over Fp2)
struct G2Point {
    Fp2 x;
    Fp2 y;
    bool infinity;  // true if point at infinity (identity element)
};

// Curve parameters for BLS12-446
// These are constants that define the curve equation

// Get the curve coefficient b for G1 (y^2 = x^3 + b)
// Returns b as an Fp element
__host__ __device__ const Fp& curve_b_g1();

// Get the curve coefficient b' for G2 (y^2 = x^3 + b')
// Returns b' as an Fp2 element
__host__ __device__ const Fp2& curve_b_g2();

// Check if a G1 point is on the curve
// Returns true if the point satisfies y^2 = x^3 + b (or is point at infinity)
__host__ __device__ bool is_on_curve_g1(const G1Point& point);

// Check if a G2 point is on the curve
// Returns true if the point satisfies y^2 = x^3 + b' (or is point at infinity)
__host__ __device__ bool is_on_curve_g2(const G2Point& point);

// Helper functions for point creation

// Create G1 point at infinity (identity element)
__host__ __device__ void g1_point_at_infinity(G1Point& point);

// Create G2 point at infinity (identity element)
__host__ __device__ void g2_point_at_infinity(G2Point& point);

// Check if G1 point is at infinity
__host__ __device__ bool g1_is_infinity(const G1Point& point);

// Check if G2 point is at infinity
__host__ __device__ bool g2_is_infinity(const G2Point& point);

// Point operations for G1

// Point addition: result = p1 + p2
__host__ __device__ void g1_add(G1Point& result, const G1Point& p1, const G1Point& p2);

// Point doubling: result = 2 * p
__host__ __device__ void g1_double(G1Point& result, const G1Point& p);

// Point negation: result = -p
__host__ __device__ void g1_neg(G1Point& result, const G1Point& p);

// Scalar multiplication: result = scalar * point
// scalar is represented as little-endian limbs (uint64_t array)
// scalar_limbs is the number of limbs (at most FP_LIMBS)
__host__ __device__ void g1_scalar_mul(G1Point& result, const G1Point& point, const uint64_t* scalar, int scalar_limbs);

// Scalar multiplication with 64-bit scalar: result = scalar * point
__host__ __device__ void g1_scalar_mul_u64(G1Point& result, const G1Point& point, uint64_t scalar);

// Point operations for G2

// Point addition: result = p1 + p2
__host__ __device__ void g2_add(G2Point& result, const G2Point& p1, const G2Point& p2);

// Point doubling: result = 2 * p
__host__ __device__ void g2_double(G2Point& result, const G2Point& p);

// Point negation: result = -p
__host__ __device__ void g2_neg(G2Point& result, const G2Point& p);

// Scalar multiplication: result = scalar * point
// scalar is represented as little-endian limbs (uint64_t array)
// scalar_limbs is the number of limbs (at most FP_LIMBS)
__host__ __device__ void g2_scalar_mul(G2Point& result, const G2Point& point, const uint64_t* scalar, int scalar_limbs);

// Scalar multiplication with 64-bit scalar: result = scalar * point
__host__ __device__ void g2_scalar_mul_u64(G2Point& result, const G2Point& point, uint64_t scalar);

// Generator points (to be set from tfhe-rs)
// These will be initialized by init_device_generators()
extern __constant__ G1Point DEVICE_G1_GENERATOR;
extern __constant__ G2Point DEVICE_G2_GENERATOR;

// Initialize device generator points
// Must be called once per device after init_device_curve()
void init_device_generators(cudaStream_t stream, uint32_t gpu_index);

// Get G1 generator point
__host__ __device__ const G1Point& g1_generator();

// Get G2 generator point
__host__ __device__ const G2Point& g2_generator();

// Multi-Scalar Multiplication (MSM)
// Computes: result = sum(scalars[i] * points[i]) for i = 0 to n-1
// Uses Pippenger's algorithm (bucket method) for efficiency
// The algorithm splits scalars into windows and uses buckets to accumulate points,
// significantly reducing the number of point operations compared to naive methods

// Pippenger algorithm constants
#define MSM_WINDOW_SIZE 4  // 4-bit windows
#define MSM_BUCKET_COUNT 16  // 2^MSM_WINDOW_SIZE buckets (0-15)

// MSM for G1
// stream: CUDA stream
// ============================================================================
// Async/Sync API for curve operations
// ============================================================================
// All pointers point to device memory (already allocated)
// _async versions: Launch kernels asynchronously, return immediately (no sync)
//  versions: Call _async then synchronize stream
// ============================================================================

// G1 Point operations (all device pointers)

// Point addition: d_result = d_p1 + d_p2
void g1_add_async(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_p1, const G1Point* d_p2);
void g1_add(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_p1, const G1Point* d_p2);

// Point doubling: d_result = 2 * d_p
void g1_double_async(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_p);
void g1_double(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_p);

// Point negation: d_result = -d_p
void g1_neg_async(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_p);
void g1_neg(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_p);

// Scalar multiplication: d_result = scalar * d_point (64-bit scalar)
void g1_scalar_mul_u64_async(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_point, uint64_t scalar);
void g1_scalar_mul_u64(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_point, uint64_t scalar);

// Scalar multiplication: d_result = scalar * d_point (multi-limb scalar, device pointer)
void g1_scalar_mul_async(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_point, const uint64_t* d_scalar, int scalar_limbs);
void g1_scalar_mul(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_point, const uint64_t* d_scalar, int scalar_limbs);

// Point at infinity: d_result = O (identity element)
void g1_point_at_infinity_async(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result);
void g1_point_at_infinity(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result);

// Convert G1 point to Montgomery form: d_result = to_montgomery(d_point)
// NOTE: All point operations assume points are in Montgomery form for performance
void g1_to_montgomery_async(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_point);
void g1_to_montgomery(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_point);

// Convert G1 point from Montgomery form: d_result = from_montgomery(d_point)
void g1_from_montgomery_async(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_point);
void g1_from_montgomery(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_point);

// G2 Point operations (all device pointers)

// Point addition: d_result = d_p1 + d_p2
void g2_add_async(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_p1, const G2Point* d_p2);
void g2_add(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_p1, const G2Point* d_p2);

// Point doubling: d_result = 2 * d_p
void g2_double_async(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_p);
void g2_double(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_p);

// Point negation: d_result = -d_p
void g2_neg_async(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_p);
void g2_neg(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_p);

// Scalar multiplication: d_result = scalar * d_point (64-bit scalar)
void g2_scalar_mul_u64_async(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_point, uint64_t scalar);
void g2_scalar_mul_u64(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_point, uint64_t scalar);

// Scalar multiplication: d_result = scalar * d_point (multi-limb scalar, device pointer)
void g2_scalar_mul_async(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_point, const uint64_t* d_scalar, int scalar_limbs);
void g2_scalar_mul(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_point, const uint64_t* d_scalar, int scalar_limbs);

// Point at infinity: d_result = O (identity element)
void g2_point_at_infinity_async(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result);
void g2_point_at_infinity(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result);

// Convert G2 point to Montgomery form: d_result = to_montgomery(d_point)
// NOTE: All point operations assume points are in Montgomery form for performance
void g2_to_montgomery_async(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_point);
void g2_to_montgomery(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_point);

// Convert G2 point from Montgomery form: d_result = from_montgomery(d_point)
void g2_from_montgomery_async(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_point);
void g2_from_montgomery(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_point);

// ============================================================================
// Refactored MSM API (device pointers only, no allocations/copies/frees)
// ============================================================================
// All pointers are device pointers (already allocated by caller)
// Temporary buffer must be provided by caller:
//   - d_scratch: buffer of size (num_blocks + 1) * MSM_BUCKET_COUNT * sizeof(G1Point/G2Point)
//     where num_blocks = (n + threadsPerBlock - 1) / threadsPerBlock (typically 256 threads per block)
//     This provides space for:
//       * num_blocks * MSM_BUCKET_COUNT points for per-block bucket accumulations
//       * MSM_BUCKET_COUNT points for final buckets
//     MSM_BUCKET_COUNT is typically 16 (for 4-bit windows)
// Uses Pippenger algorithm (bucket method) with sppark-style single-pass accumulation

// MSM for G1 with 64-bit scalars
void g1_msm_u64_async(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_points, const uint64_t* d_scalars, G1Point* d_scratch, int n);
void g1_msm_u64(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_points, const uint64_t* d_scalars, G1Point* d_scratch, int n);

// MSM for G1 with multi-limb scalars
void g1_msm_async(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_points, const uint64_t* d_scalars, int scalar_limbs, G1Point* d_scratch, int n);
void g1_msm(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_points, const uint64_t* d_scalars, int scalar_limbs, G1Point* d_scratch, int n);

// MSM for G2 with 64-bit scalars
void g2_msm_u64_async(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_points, const uint64_t* d_scalars, G2Point* d_scratch, int n);
void g2_msm_u64(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_points, const uint64_t* d_scalars, G2Point* d_scratch, int n);

// MSM for G2 with multi-limb scalars
void g2_msm_async(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_points, const uint64_t* d_scalars, int scalar_limbs, G2Point* d_scratch, int n);
void g2_msm(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_points, const uint64_t* d_scalars, int scalar_limbs, G2Point* d_scratch, int n);

