#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// BLS12-446: 446-bit prime field
// Using 7 limbs of 64 bits each (448 bits total, 2 bits headroom)
#define FP_LIMBS 7
#define FP_BITS 446

// Little-endian limbs: limb[0] is least significant word
// Note: This is a POD (Plain Old Data) type for CUDA __constant__ compatibility
struct Fp {
    uint64_t limb[FP_LIMBS];
};

// Prime modulus p for BLS12-446
// Device-side constant (hardcoded in fp.cu at compile time)
extern __constant__ const Fp DEVICE_MODULUS;

// Montgomery constants
// R = 2^448 (for 7 limbs of 64 bits)
// R^2 mod p, R_INV mod p, and p' = -p^(-1) mod 2^64
// All hardcoded at compile time
extern __constant__ const Fp DEVICE_R2;
extern __constant__ const Fp DEVICE_R_INV;
extern __constant__ const uint64_t DEVICE_P_PRIME;

// Host-side initialization function
// Call this once per device before using device code
// stream: CUDA stream to use (can be nullptr for default stream synchronization)
// gpu_index: GPU device index to use
void init_device_modulus(cudaStream_t stream, uint32_t gpu_index);

// Multi-precision arithmetic operations
// All operations are CUDA-compatible (can be called from host or device)

// Comparison: returns -1 if a < b, 0 if a == b, 1 if a > b
__host__ __device__ int fp_cmp(const Fp& a, const Fp& b);

// Check if a == 0
__host__ __device__ bool fp_is_zero(const Fp& a);

// Check if a == 1
__host__ __device__ bool fp_is_one(const Fp& a);

// Set to zero
__host__ __device__ void fp_zero(Fp& a);

// Set to one (normal form)
__host__ __device__ void fp_one(Fp& a);

// Set to one in Montgomery form (R mod p)
__host__ __device__ void fp_one_montgomery(Fp& a);

// Copy: dst = src
__host__ __device__ void fp_copy(Fp& dst, const Fp& src);

// Addition: c = a + b (without reduction)
// Returns carry out
__host__ __device__ uint64_t fp_add_raw(Fp& c, const Fp& a, const Fp& b);

// Subtraction: c = a - b (without reduction)
// Returns borrow (1 if a < b, 0 otherwise)
__host__ __device__ uint64_t fp_sub_raw(Fp& c, const Fp& a, const Fp& b);

// Addition with modular reduction: c = (a + b) mod p
__host__ __device__ void fp_add(Fp& c, const Fp& a, const Fp& b);

// Subtraction with modular reduction: c = (a - b) mod p
__host__ __device__ void fp_sub(Fp& c, const Fp& a, const Fp& b);

// Multiplication: c = a * b (without reduction)
// Result stored in double-width (14 limbs)
__host__ __device__ void fp_mul_raw(uint64_t* c, const Fp& a, const Fp& b);

// Modular reduction: c = a mod p (where a is 2*FP_LIMBS limbs)
__host__ __device__ void fp_reduce(Fp& c, const uint64_t* a);

// Montgomery reduction: c = (a * R_INV) mod p
// Input a is 2*FP_LIMBS limbs (result of multiplication)
// Output c is FP_LIMBS limbs in Montgomery form
__host__ __device__ void fp_mont_reduce(Fp& c, const uint64_t* a);

// Montgomery multiplication: c = (a * b * R_INV) mod p
// Both a and b are in Montgomery form, result is in Montgomery form
__host__ __device__ void fp_mont_mul(Fp& c, const Fp& a, const Fp& b);

// Convert to Montgomery form: c = (a * R) mod p
__host__ __device__ void fp_to_montgomery(Fp& c, const Fp& a);

// Convert from Montgomery form: c = (a * R_INV) mod p
__host__ __device__ void fp_from_montgomery(Fp& c, const Fp& a);

// Batch conversion to Montgomery form: dst[i] = (src[i] * R) mod p
// More efficient than calling fp_to_montgomery in a loop
// For CUDA: can be called from host or device, but arrays must be in same memory space
__host__ __device__ void fp_to_montgomery_batch(Fp* dst, const Fp* src, int n);

// Batch conversion from Montgomery form: dst[i] = (src[i] * R_INV) mod p
// More efficient than calling fp_from_montgomery in a loop
// For CUDA: can be called from host or device, but arrays must be in same memory space
__host__ __device__ void fp_from_montgomery_batch(Fp* dst, const Fp* src, int n);

// Multiplication with modular reduction: c = (a * b) mod p
// NOTE: This now uses Montgomery form internally
// For external API, use fp_mont_mul if values are already in Montgomery form
__host__ __device__ void fp_mul(Fp& c, const Fp& a, const Fp& b);

// Negation: c = -a mod p
__host__ __device__ void fp_neg(Fp& c, const Fp& a);

// Inversion: c = a^(-1) mod p
// Uses Fermat's little theorem: a^(p-2) = a^(-1) mod p
// Returns c = 0 if a = 0 (division by zero)
// NOTE: Assumes input is in normal form and converts to/from Montgomery
__host__ __device__ void fp_inv(Fp& c, const Fp& a);

// Montgomery inversion: c = a^(-1) mod p (all in Montgomery form)
// NOTE: Input and output are in Montgomery form (no conversions)
__host__ __device__ void fp_mont_inv(Fp& c, const Fp& a);

// Division: c = a / b mod p = a * b^(-1) mod p
// Returns c = 0 if b = 0 (division by zero)
__host__ __device__ void fp_div(Fp& c, const Fp& a, const Fp& b);

// Exponentiation: c = a^e mod p
// e is represented as a big integer in little-endian format (limb[0] is LSB)
// e_limbs is the number of limbs in the exponent (at most FP_LIMBS)
// For exponents larger than 448 bits, only the lower 448 bits are used
__host__ __device__ void fp_pow(Fp& c, const Fp& a, const uint64_t* e, int e_limbs);

// Exponentiation with 64-bit exponent: c = a^e mod p
__host__ __device__ void fp_pow_u64(Fp& c, const Fp& a, uint64_t e);

// Square root: c = sqrt(a) mod p if a is a quadratic residue
// Returns true if a is a quadratic residue (square root exists), false otherwise
// If false, c is set to 0
// For primes p ≡ 3 (mod 4): sqrt(a) = a^((p+1)/4) mod p
// For other primes, uses Tonelli-Shanks algorithm
__host__ __device__ bool fp_sqrt(Fp& c, const Fp& a);

// Check if a is a quadratic residue (has a square root)
// Returns true if a is a quadratic residue, false otherwise
// Uses Euler's criterion: a is a quadratic residue if a^((p-1)/2) = 1 mod p
__host__ __device__ bool fp_is_quadratic_residue(const Fp& a);

// Conditional assignment: if condition, dst = src, else dst unchanged
__host__ __device__ void fp_cmov(Fp& dst, const Fp& src, uint64_t condition);

// Helper functions to access constants
// Get modulus reference (device: from constant memory, host: static copy)
__host__ __device__ const Fp& fp_modulus();

// ============================================================================
// Async/Sync API for device memory operations
// ============================================================================
// All pointers in these functions point to device memory (already allocated)
// _async versions: Launch kernels asynchronously, return immediately (no sync)
// Default versions (no suffix): Call _async then synchronize stream
// ============================================================================

// Addition: d_c = d_a + d_b mod p (all device pointers)
void fp_add_async(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a, const Fp* d_b);
void fp_add(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a, const Fp* d_b);

// Subtraction: d_c = d_a - d_b mod p (all device pointers)
void fp_sub_async(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a, const Fp* d_b);
void fp_sub(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a, const Fp* d_b);

// Multiplication: d_c = d_a * d_b mod p (all device pointers)
void fp_mul_async(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a, const Fp* d_b);
void fp_mul(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a, const Fp* d_b);

// Negation: d_c = -d_a mod p (all device pointers)
void fp_neg_async(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a);
void fp_neg(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a);

// Inversion: d_c = d_a^(-1) mod p (all device pointers)
void fp_inv_async(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a);
void fp_inv(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a);

// Division: d_c = d_a / d_b mod p (all device pointers)
void fp_div_async(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a, const Fp* d_b);
void fp_div(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a, const Fp* d_b);

// Copy: d_dst = d_src (all device pointers)
void fp_copy_async(cudaStream_t stream, uint32_t gpu_index, Fp* d_dst, const Fp* d_src);
void fp_copy(cudaStream_t stream, uint32_t gpu_index, Fp* d_dst, const Fp* d_src);

// Set to zero: d_a = 0 (device pointer)
void fp_zero_async(cudaStream_t stream, uint32_t gpu_index, Fp* d_a);
void fp_zero(cudaStream_t stream, uint32_t gpu_index, Fp* d_a);

// Set to one: d_a = 1 (device pointer, normal form)
void fp_one_async(cudaStream_t stream, uint32_t gpu_index, Fp* d_a);
void fp_one(cudaStream_t stream, uint32_t gpu_index, Fp* d_a);

// Convert to Montgomery form: d_c = (d_a * R) mod p (all device pointers)
void fp_to_montgomery_async(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a);
void fp_to_montgomery(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a);

// Convert from Montgomery form: d_c = (d_a * R_INV) mod p (all device pointers)
void fp_from_montgomery_async(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a);
void fp_from_montgomery(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a);

// Montgomery multiplication: d_c = (d_a * d_b * R_INV) mod p (all device pointers)
void fp_mont_mul_async(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a, const Fp* d_b);
void fp_mont_mul(cudaStream_t stream, uint32_t gpu_index, Fp* d_c, const Fp* d_a, const Fp* d_b);

