#include "curve.h"
#include "fp.h"
#include "fp2.h"
#include "device.h"
#include <cstring>
#include <cstdio>

// ============================================================================
// Template Traits System for Point Operations
// ============================================================================
// This traits system allows us to write generic point operations that work
// for both G1 (Fp) and G2 (Fp2) points using the same algorithm.

template<typename PointType>
struct PointTraits;

// Specialization for G1Point (uses Fp)
template<>
struct PointTraits<G1Point> {
    using FieldType = Fp;
    
    __host__ __device__ static void field_zero(FieldType& a) { fp_zero(a); }
    __host__ __device__ static void field_copy(FieldType& dst, const FieldType& src) { fp_copy(dst, src); }
    __host__ __device__ static void field_neg(FieldType& c, const FieldType& a) { fp_neg(c, a); }
    __host__ __device__ static void field_add(FieldType& c, const FieldType& a, const FieldType& b) { fp_add(c, a, b); }
    __host__ __device__ static void field_sub(FieldType& c, const FieldType& a, const FieldType& b) { fp_sub(c, a, b); }
    __host__ __device__ static void field_mul(FieldType& c, const FieldType& a, const FieldType& b) { fp_mont_mul(c, a, b); }
    __host__ __device__ static void field_inv(FieldType& c, const FieldType& a) { fp_mont_inv(c, a); }
    __host__ __device__ static int field_cmp(const FieldType& a, const FieldType& b) { return fp_cmp(a, b); }
    __host__ __device__ static bool field_is_zero(const FieldType& a) { return fp_is_zero(a); }
    __host__ __device__ static void field_to_montgomery(FieldType& c, const FieldType& a) { fp_to_montgomery(c, a); }
    __host__ __device__ static void field_from_montgomery(FieldType& c, const FieldType& a) { fp_from_montgomery(c, a); }
    
    __host__ __device__ static void point_at_infinity(G1Point& point) { g1_point_at_infinity(point); }
    __host__ __device__ static bool is_infinity(const G1Point& point) { return g1_is_infinity(point); }
    __host__ __device__ static const FieldType& curve_b() { return curve_b_g1(); }
};

// Specialization for G2Point (uses Fp2)
template<>
struct PointTraits<G2Point> {
    using FieldType = Fp2;
    
    __host__ __device__ static void field_zero(FieldType& a) { fp2_zero(a); }
    __host__ __device__ static void field_copy(FieldType& dst, const FieldType& src) { fp2_copy(dst, src); }
    __host__ __device__ static void field_neg(FieldType& c, const FieldType& a) { fp2_neg(c, a); }
    __host__ __device__ static void field_add(FieldType& c, const FieldType& a, const FieldType& b) { fp2_add(c, a, b); }
    __host__ __device__ static void field_sub(FieldType& c, const FieldType& a, const FieldType& b) { fp2_sub(c, a, b); }
    __host__ __device__ static void field_mul(FieldType& c, const FieldType& a, const FieldType& b) { fp2_mont_mul(c, a, b); }
    __host__ __device__ static void field_inv(FieldType& c, const FieldType& a) { fp2_mont_inv(c, a); }
    __host__ __device__ static int field_cmp(const FieldType& a, const FieldType& b) { return fp2_cmp(a, b); }
    __host__ __device__ static bool field_is_zero(const FieldType& a) { return fp2_is_zero(a); }
    __host__ __device__ static void field_to_montgomery(FieldType& c, const FieldType& a) {
        fp_to_montgomery(c.c0, a.c0);
        fp_to_montgomery(c.c1, a.c1);
    }
    __host__ __device__ static void field_from_montgomery(FieldType& c, const FieldType& a) {
        fp_from_montgomery(c.c0, a.c0);
        fp_from_montgomery(c.c1, a.c1);
    }
    
    __host__ __device__ static void point_at_infinity(G2Point& point) { g2_point_at_infinity(point); }
    __host__ __device__ static bool is_infinity(const G2Point& point) { return g2_is_infinity(point); }
    __host__ __device__ static const FieldType& curve_b() { return curve_b_g2(); }
};

// ============================================================================
// Template Point Operations
// ============================================================================

// Generic point negation: result = -p = (x, -y)
template<typename PointType>
__host__ __device__ void point_neg(PointType& result, const PointType& p) {
    using Traits = PointTraits<PointType>;
    if (Traits::is_infinity(p)) {
        Traits::point_at_infinity(result);
        return;
    }
    Traits::field_copy(result.x, p.x);
    Traits::field_neg(result.y, p.y);
    result.infinity = false;
}

// Generic point doubling: result = 2 * p
template<typename PointType>
__host__ __device__ void point_double(PointType& result, const PointType& p) {
    using Traits = PointTraits<PointType>;
    using FieldType = typename Traits::FieldType;
    
    if (Traits::is_infinity(p) || Traits::field_is_zero(p.y)) {
        Traits::point_at_infinity(result);
        return;
    }
    
    // Compute lambda = 3*x^2 / (2*y)
    FieldType x_squared, three_x_squared, two_y, lambda;
    Traits::field_mul(x_squared, p.x, p.x);
    Traits::field_add(three_x_squared, x_squared, x_squared);  // 2*x^2
    Traits::field_add(three_x_squared, three_x_squared, x_squared);  // 3*x^2
    
    Traits::field_add(two_y, p.y, p.y);  // 2*y
    Traits::field_inv(lambda, two_y);  // 1/(2*y)
    Traits::field_mul(lambda, lambda, three_x_squared);  // 3*x^2 / (2*y)
    
    // x_result = lambda^2 - 2*x
    FieldType lambda_squared, two_x, x_result;
    Traits::field_mul(lambda_squared, lambda, lambda);
    Traits::field_add(two_x, p.x, p.x);  // 2*x
    Traits::field_sub(x_result, lambda_squared, two_x);
    
    // y_result = lambda*(x - x_result) - y
    FieldType x_minus_xr, y_result;
    Traits::field_sub(x_minus_xr, p.x, x_result);
    Traits::field_mul(y_result, lambda, x_minus_xr);
    Traits::field_sub(y_result, y_result, p.y);
    
    Traits::field_copy(result.x, x_result);
    Traits::field_copy(result.y, y_result);
    result.infinity = false;
}

// Generic point addition: result = p1 + p2
template<typename PointType>
__host__ __device__ void point_add(PointType& result, const PointType& p1, const PointType& p2) {
    using Traits = PointTraits<PointType>;
    using FieldType = typename Traits::FieldType;
    
    // Handle infinity cases
    if (Traits::is_infinity(p1)) {
        Traits::field_copy(result.x, p2.x);
        Traits::field_copy(result.y, p2.y);
        result.infinity = p2.infinity;
        return;
    }
    if (Traits::is_infinity(p2)) {
        Traits::field_copy(result.x, p1.x);
        Traits::field_copy(result.y, p1.y);
        result.infinity = p1.infinity;
        return;
    }
    
    // Check if p1 == -p2 (same x, opposite y)
    FieldType neg_y2;
    Traits::field_neg(neg_y2, p2.y);
    if (Traits::field_cmp(p1.x, p2.x) == 0 && Traits::field_cmp(p1.y, neg_y2) == 0) {
        Traits::point_at_infinity(result);
        return;
    }
    
    // Check if p1 == p2 (use doubling)
    if (Traits::field_cmp(p1.x, p2.x) == 0 && Traits::field_cmp(p1.y, p2.y) == 0) {
        point_double(result, p1);
        return;
    }
    
    // Standard addition: lambda = (y2 - y1) / (x2 - x1)
    FieldType dx, dy, lambda, lambda_squared, x_result;
    Traits::field_sub(dx, p2.x, p1.x);
    Traits::field_sub(dy, p2.y, p1.y);
    Traits::field_inv(lambda, dx);  // 1 / (x2 - x1)
    Traits::field_mul(lambda, lambda, dy);  // (y2 - y1) / (x2 - x1)
    
    // x_result = lambda^2 - x1 - x2
    Traits::field_mul(lambda_squared, lambda, lambda);
    Traits::field_sub(x_result, lambda_squared, p1.x);
    Traits::field_sub(x_result, x_result, p2.x);
    
    // y_result = lambda * (x1 - x_result) - y1
    FieldType x1_minus_xr, y_result;
    Traits::field_sub(x1_minus_xr, p1.x, x_result);
    Traits::field_mul(y_result, lambda, x1_minus_xr);
    Traits::field_sub(y_result, y_result, p1.y);
    
    Traits::field_copy(result.x, x_result);
    Traits::field_copy(result.y, y_result);
    result.infinity = false;
}

// Generic scalar multiplication: result = scalar * point
template<typename PointType>
__host__ __device__ void point_scalar_mul(PointType& result, const PointType& point, const uint64_t* scalar, int scalar_limbs) {
    using Traits = PointTraits<PointType>;
    
    // Start with point at infinity
    Traits::point_at_infinity(result);
    
    if (Traits::is_infinity(point)) {
        return;
    }
    
    // Check if scalar is zero
    bool all_zero = true;
    for (int i = 0; i < scalar_limbs; i++) {
        if (scalar[i] != 0) {
            all_zero = false;
            break;
        }
    }
    if (all_zero) {
        return;
    }
    
    PointType current;
    Traits::field_copy(current.x, point.x);
    Traits::field_copy(current.y, point.y);
    current.infinity = point.infinity;
    
    // Process bits from MSB to LSB
    for (int limb = scalar_limbs - 1; limb >= 0; limb--) {
        for (int bit = 63; bit >= 0; bit--) {
            point_double(result, result);
            
            if ((scalar[limb] >> bit) & 1) {
                point_add(result, result, current);
            }
        }
    }
}

// Curve parameters for BLS12-446
// G1 curve: y^2 = x^3 + b
// G2 curve: y^2 = x^3 + b' where b' = b * ξ (ξ is a non-residue in Fp2)

// For BLS12 curves, typically b = 4
// This can be updated with the actual BLS12-446 parameter if different
static Fp get_host_curve_b_g1() {
    Fp b;
    fp_zero(b);
    b.limb[0] = 4;  // b = 4 (typical for BLS12 curves)
    return b;
}

// G2 curve coefficient: b' = b * ξ where ξ is a non-residue
// For Fp2, typically ξ = i (the imaginary unit)
// So b' = (0 + 4*i) in Fp2
static Fp2 get_host_curve_b_g2() {
    Fp2 b_prime;
    fp2_zero(b_prime);
    fp_zero(b_prime.c0);  // Real part = 0
    fp_zero(b_prime.c1);
    b_prime.c1.limb[0] = 4;  // Imaginary part = 4 (b * i)
    return b_prime;
}

// Device constants for curve parameters
__constant__ Fp DEVICE_CURVE_B_G1;
__constant__ Fp2 DEVICE_CURVE_B_G2;

// Initialize device curve constants
// Must be called once per device before using device code
// stream: CUDA stream to use (can be nullptr for default stream synchronization)
// gpu_index: GPU device index to use
void init_device_curve(cudaStream_t stream, uint32_t gpu_index) {
    // Set the device context
    cuda_set_device(gpu_index);
    
    Fp host_b_g1 = get_host_curve_b_g1();
    Fp2 host_b_g2 = get_host_curve_b_g2();
    
    // Note: cudaMemcpyToSymbolAsync doesn't exist for __constant__ memory
    // We must use synchronous cudaMemcpyToSymbol, but we ensure the device is set correctly
    check_cuda_error(cudaMemcpyToSymbol(DEVICE_CURVE_B_G1, &host_b_g1, sizeof(Fp)));
    check_cuda_error(cudaMemcpyToSymbol(DEVICE_CURVE_B_G2, &host_b_g2, sizeof(Fp2)));
    
    // Synchronize stream to ensure completion
    if (stream != nullptr) {
        cuda_synchronize_stream(stream, gpu_index);
    }
}

// Get curve coefficient b for G1
__host__ __device__ const Fp& curve_b_g1() {
#ifdef __CUDA_ARCH__
    return DEVICE_CURVE_B_G1;
#else
    static const Fp host_b = get_host_curve_b_g1();
    return host_b;
#endif
}

// Get curve coefficient b' for G2
__host__ __device__ const Fp2& curve_b_g2() {
#ifdef __CUDA_ARCH__
    return DEVICE_CURVE_B_G2;
#else
    static const Fp2 host_b = get_host_curve_b_g2();
    return host_b;
#endif
}

// Check if a G1 point is on the curve: y^2 = x^3 + b
__host__ __device__ bool is_on_curve_g1(const G1Point& point) {
    // Point at infinity is always on the curve
    if (point.infinity) {
        return true;
    }
    
    // Compute y^2
    Fp y_squared;
    fp_mul(y_squared, point.y, point.y);
    
    // Compute x^3
    Fp x_squared, x_cubed;
    fp_mul(x_squared, point.x, point.x);
    fp_mul(x_cubed, x_squared, point.x);
    
    // Compute x^3 + b
    Fp rhs;
    const Fp& b = curve_b_g1();
    fp_add(rhs, x_cubed, b);
    
    // Check if y^2 == x^3 + b
    return fp_cmp(y_squared, rhs) == 0;
}

// Check if a G2 point is on the curve: y^2 = x^3 + b'
__host__ __device__ bool is_on_curve_g2(const G2Point& point) {
    // Point at infinity is always on the curve
    if (point.infinity) {
        return true;
    }
    
    // Compute y^2
    Fp2 y_squared;
    fp2_mul(y_squared, point.y, point.y);
    
    // Compute x^3
    Fp2 x_squared, x_cubed;
    fp2_mul(x_squared, point.x, point.x);
    fp2_mul(x_cubed, x_squared, point.x);
    
    // Compute x^3 + b'
    Fp2 rhs;
    const Fp2& b_prime = curve_b_g2();
    fp2_add(rhs, x_cubed, b_prime);
    
    // Check if y^2 == x^3 + b'
    return fp2_cmp(y_squared, rhs) == 0;
}

// Create G1 point at infinity
__host__ __device__ void g1_point_at_infinity(G1Point& point) {
    fp_zero(point.x);
    fp_zero(point.y);
    point.infinity = true;
}

// Create G2 point at infinity
__host__ __device__ void g2_point_at_infinity(G2Point& point) {
    fp2_zero(point.x);
    fp2_zero(point.y);
    point.infinity = true;
}

// Check if G1 point is at infinity
__host__ __device__ bool g1_is_infinity(const G1Point& point) {
    return point.infinity;
}

// Check if G2 point is at infinity
__host__ __device__ bool g2_is_infinity(const G2Point& point) {
    return point.infinity;
}


// ============================================================================
// Generator Points
// ============================================================================

__constant__ G1Point DEVICE_G1_GENERATOR;
__constant__ G2Point DEVICE_G2_GENERATOR;


// Host-side generator initialization
// Generator points from tfhe-rs: https://github.com/zama-ai/tfhe-rs/blob/main/tfhe-zk-pok/src/curve_446/mod.rs
// G1_GENERATOR_X = 143189966182216199425404656824735381247272236095050141599848381692039676741476615087722874458136990266833440576646963466074693171606778
// G1_GENERATOR_Y = 75202396197342917254523279069469674666303680671605970245803554133573745859131002231546341942288521574682619325841484506619191207488304
static G1Point get_host_g1_generator() {
    G1Point gen;
    gen.infinity = false;
    
    // Set G1_GENERATOR_X from hex limbs (little-endian)
    gen.x.limb[0] = 0x3bf9166c8236f4faULL;
    gen.x.limb[1] = 0x8bc02b7cbe6a9e8dULL;
    gen.x.limb[2] = 0x11c1e56b3e4bc80bULL;
    gen.x.limb[3] = 0x6b20d782901a6f62ULL;
    gen.x.limb[4] = 0x2ce8c34265bf3841ULL;
    gen.x.limb[5] = 0x11b73d3d76ae9851ULL;
    gen.x.limb[6] = 0x326ed6bd777fc6a3ULL;
    
    // Set G1_GENERATOR_Y from hex limbs (little-endian)
    gen.y.limb[0] = 0xfe6f792612016b30ULL;
    gen.y.limb[1] = 0x22db0ce6034a9db9ULL;
    gen.y.limb[2] = 0xb9093f32002756daULL;
    gen.y.limb[3] = 0x39d7f424b6660204ULL;
    gen.y.limb[4] = 0xf843c947aa57f571ULL;
    gen.y.limb[5] = 0xd6d62d244e413636ULL;
    gen.y.limb[6] = 0x1a7caf4a4d3887a6ULL;
    
    // Convert to Montgomery form
    fp_to_montgomery(gen.x, gen.x);
    fp_to_montgomery(gen.y, gen.y);
    
    return gen;
}

static G2Point get_host_g2_generator() {
    G2Point gen;
    gen.infinity = false;
    
    // G2_GENERATOR_X_C0 from hex limbs (little-endian)
    gen.x.c0.limb[0] = 0xe529ee4dce9991dULL;
    gen.x.c0.limb[1] = 0xd6ebaf149094f1ccULL;
    gen.x.c0.limb[2] = 0x43c6bf16312d638ULL;
    gen.x.c0.limb[3] = 0x62b61439640e885ULL;
    gen.x.c0.limb[4] = 0x18dad8ed784dd225ULL;
    gen.x.c0.limb[5] = 0xa57c0038441f7d15ULL;
    gen.x.c0.limb[6] = 0x21f8d4a76f74541aULL;
    
    // G2_GENERATOR_X_C1 from hex limbs (little-endian)
    gen.x.c1.limb[0] = 0xcaf5185423a7d23aULL;
    gen.x.c1.limb[1] = 0x7cef6acb145b6413ULL;
    gen.x.c1.limb[2] = 0x2879dd439b019b8bULL;
    gen.x.c1.limb[3] = 0x71449cdeca4f0007ULL;
    gen.x.c1.limb[4] = 0xdebaf4a2c5534527ULL;
    gen.x.c1.limb[5] = 0xa1b4e791d1b86560ULL;
    gen.x.c1.limb[6] = 0x1e0f563c601bb8dcULL;
    
    // G2_GENERATOR_Y_C0 from hex limbs (little-endian)
    gen.y.c0.limb[0] = 0x274315837455b919ULL;
    gen.y.c0.limb[1] = 0x82039e4221ff3507ULL;
    gen.y.c0.limb[2] = 0x346cebad16a036ULL;
    gen.y.c0.limb[3] = 0x177bfd6654e681eULL;
    gen.y.c0.limb[4] = 0xddff621b5db3f897ULL;
    gen.y.c0.limb[5] = 0xcc61570301497a7ULL;
    gen.y.c0.limb[6] = 0x115ea2305a78f646ULL;
    
    // G2_GENERATOR_Y_C1 from hex limbs (little-endian)
    gen.y.c1.limb[0] = 0x392236e9cf2976c2ULL;
    gen.y.c1.limb[1] = 0xd8ab17c84b9f03cdULL;
    gen.y.c1.limb[2] = 0x8a8e6755f9d82fd1ULL;
    gen.y.c1.limb[3] = 0x7532834528cd5a64ULL;
    gen.y.c1.limb[4] = 0xb0bcc3fb6f2161cULL;
    gen.y.c1.limb[5] = 0x76a2ffcb7d47679dULL;
    gen.y.c1.limb[6] = 0x25ed2192b203c1feULL;
    
    // Convert to Montgomery form
    fp_to_montgomery(gen.x.c0, gen.x.c0);
    fp_to_montgomery(gen.x.c1, gen.x.c1);
    fp_to_montgomery(gen.y.c0, gen.y.c0);
    fp_to_montgomery(gen.y.c1, gen.y.c1);
    
    return gen;
}

void init_device_generators(cudaStream_t stream, uint32_t gpu_index) {
    cuda_set_device(gpu_index);
    
    G1Point host_g1_gen = get_host_g1_generator();
    G2Point host_g2_gen = get_host_g2_generator();
    
    check_cuda_error(cudaMemcpyToSymbol(DEVICE_G1_GENERATOR, &host_g1_gen, sizeof(G1Point)));
    check_cuda_error(cudaMemcpyToSymbol(DEVICE_G2_GENERATOR, &host_g2_gen, sizeof(G2Point)));
    
    if (stream != nullptr) {
        cuda_synchronize_stream(stream, gpu_index);
    }
}

__host__ __device__ const G1Point& g1_generator() {
#ifdef __CUDA_ARCH__
    return DEVICE_G1_GENERATOR;
#else
    static const G1Point host_gen = get_host_g1_generator();
    return host_gen;
#endif
}

__host__ __device__ const G2Point& g2_generator() {
#ifdef __CUDA_ARCH__
    return DEVICE_G2_GENERATOR;
#else
    static const G2Point host_gen = get_host_g2_generator();
    return host_gen;
#endif
}

// ============================================================================
// Multi-Scalar Multiplication (MSM) - Pippenger Algorithm
// ============================================================================

// Pippenger algorithm parameters
// MSM_WINDOW_SIZE and MSM_BUCKET_COUNT are defined in curve.h

// Helper function to extract a window from a scalar
__device__ __forceinline__ int extract_window_u64(uint64_t scalar, int window_idx, int window_size) {
    int bit_offset = window_idx * window_size;
    if (bit_offset >= 64) return 0;
    int bits_remaining = 64 - bit_offset;
    int actual_size = (bits_remaining < window_size) ? bits_remaining : window_size;
    return (scalar >> bit_offset) & ((1ULL << actual_size) - 1);
}

// Helper function to extract a window from a multi-limb scalar
__device__ __forceinline__ int extract_window_multi(const uint64_t* scalar, int scalar_limbs, int window_idx, int window_size) {
    int total_bits = scalar_limbs * 64;
    int bit_offset = window_idx * window_size;
    if (bit_offset >= total_bits) return 0;
    
    int limb_idx = bit_offset / 64;
    int bit_in_limb = bit_offset % 64;
    
    if (limb_idx >= scalar_limbs) return 0;
    
    uint64_t mask = (1ULL << window_size) - 1;
    uint64_t window = (scalar[limb_idx] >> bit_in_limb) & mask;
    
    // If window spans two limbs, combine them
    if (bit_in_limb + window_size > 64 && limb_idx + 1 < scalar_limbs) {
        int bits_from_next = (bit_in_limb + window_size) - 64;
        uint64_t next_bits = scalar[limb_idx + 1] & ((1ULL << bits_from_next) - 1);
        window |= (next_bits << (window_size - bits_from_next));
    }
    
    return (int)window;
}

// Pippenger kernel: Clear buckets
template<typename PointType>
__global__ void kernel_clear_buckets(
    PointType* buckets,
    int num_buckets
) {
    using Traits = PointTraits<PointType>;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_buckets) {
        Traits::point_at_infinity(buckets[idx]);
    }
}

// Pippenger kernel: Final reduction of bucket contributions from multiple blocks
// This kernel combines per-block bucket accumulations into final buckets
template<typename PointType>
__global__ void kernel_reduce_buckets(
    PointType* final_buckets,
    const PointType* block_buckets,
    int num_blocks,
    int num_buckets
) {
    using Traits = PointTraits<PointType>;
    
    // Each thread handles one bucket across all blocks
    int bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_idx == 0 || bucket_idx >= num_buckets) return;
    
    PointType bucket_sum;
    Traits::point_at_infinity(bucket_sum);
    
    // Sum contributions from all blocks for this bucket
    for (int block = 0; block < num_blocks; block++) {
        int idx = block * num_buckets + bucket_idx;
        const PointType& block_contrib = block_buckets[idx];
        if (!Traits::is_infinity(block_contrib)) {
            if (Traits::is_infinity(bucket_sum)) {
                Traits::field_copy(bucket_sum.x, block_contrib.x);
                Traits::field_copy(bucket_sum.y, block_contrib.y);
                bucket_sum.infinity = block_contrib.infinity;
            } else {
                PointType temp;
                point_add(temp, bucket_sum, block_contrib);
                Traits::field_copy(bucket_sum.x, temp.x);
                Traits::field_copy(bucket_sum.y, temp.y);
                bucket_sum.infinity = temp.infinity;
            }
        }
    }
    
    // Write final result
    Traits::field_copy(final_buckets[bucket_idx].x, bucket_sum.x);
    Traits::field_copy(final_buckets[bucket_idx].y, bucket_sum.y);
    final_buckets[bucket_idx].infinity = bucket_sum.infinity;
}

// Pippenger kernel: Accumulate points into buckets for a specific window (64-bit scalars)
// Sppark-style single-pass approach: process all points in parallel, one thread per point
// Each block accumulates per-bucket contributions in shared memory, writes to block_buckets
template<typename PointType>
__global__ void kernel_accumulate_buckets_u64(
    PointType* block_buckets,  // Output: num_blocks * MSM_BUCKET_COUNT (per-block bucket accumulations)
    const PointType* points,
    const uint64_t* scalars,
    int n,
    int window_idx,
    int window_size
) {
    using Traits = PointTraits<PointType>;
    
    // Shared memory layout:
    // - shared_buckets[MSM_BUCKET_COUNT]: per-bucket accumulations
    // - thread_points[blockDim.x]: points processed by each thread
    // - thread_buckets[blockDim.x]: bucket index for each thread's point
    extern __shared__ char shared_mem[];
    PointType* shared_buckets = (PointType*)shared_mem;
    PointType* thread_points = (PointType*)(shared_mem + MSM_BUCKET_COUNT * sizeof(PointType));
    int* thread_buckets = (int*)(shared_mem + MSM_BUCKET_COUNT * sizeof(PointType) + blockDim.x * sizeof(PointType));
    
    // Initialize shared memory buckets to infinity
    for (int i = threadIdx.x; i < MSM_BUCKET_COUNT; i += blockDim.x) {
        Traits::point_at_infinity(shared_buckets[i]);
    }
    __syncthreads();
    
    // Phase 1: Each thread processes one point and stores it in shared memory
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx < n) {
        int bucket_idx = extract_window_u64(scalars[point_idx], window_idx, window_size);
        thread_points[threadIdx.x] = points[point_idx];
        thread_buckets[threadIdx.x] = bucket_idx;
    } else {
        Traits::point_at_infinity(thread_points[threadIdx.x]);
        thread_buckets[threadIdx.x] = 0;
    }
    __syncthreads();
    
    // Phase 2: Parallel reduction per bucket
    // Optimized: Each bucket thread sums contributions using optimized loop
    // Cache bucket_idx and reduce redundant checks
    if (threadIdx.x < MSM_BUCKET_COUNT) {
        int bucket_idx = threadIdx.x;
        PointType bucket_sum;
        Traits::point_at_infinity(bucket_sum);
        
        // Optimized: Sum contributions from threads that belong to this bucket
        // Pre-check global bounds once, then iterate with minimal checks
        int block_start = blockIdx.x * blockDim.x;
        int block_end = min(block_start + blockDim.x, n);
        
        for (int local_t = 0; local_t < blockDim.x; local_t++) {
            int global_t_idx = block_start + local_t;
            if (global_t_idx < block_end && thread_buckets[local_t] == bucket_idx) {
                // Only process if point is not infinity (already checked bucket match)
                const PointType& pt = thread_points[local_t];
                if (!Traits::is_infinity(pt)) {
                    if (Traits::is_infinity(bucket_sum)) {
                        Traits::field_copy(bucket_sum.x, pt.x);
                        Traits::field_copy(bucket_sum.y, pt.y);
                        bucket_sum.infinity = pt.infinity;
                    } else {
                        PointType temp;
                        point_add(temp, bucket_sum, pt);
                        Traits::field_copy(bucket_sum.x, temp.x);
                        Traits::field_copy(bucket_sum.y, temp.y);
                        bucket_sum.infinity = temp.infinity;
                    }
                }
            }
        }
        
        Traits::field_copy(shared_buckets[bucket_idx].x, bucket_sum.x);
        Traits::field_copy(shared_buckets[bucket_idx].y, bucket_sum.y);
        shared_buckets[bucket_idx].infinity = bucket_sum.infinity;
    }
    __syncthreads();
    
    // Phase 3: Write block's bucket contributions to global memory
    if (threadIdx.x < MSM_BUCKET_COUNT) {
        int bucket_idx = threadIdx.x;
        int block_bucket_idx = blockIdx.x * MSM_BUCKET_COUNT + bucket_idx;
        const PointType& bucket = shared_buckets[bucket_idx];
        Traits::field_copy(block_buckets[block_bucket_idx].x, bucket.x);
        Traits::field_copy(block_buckets[block_bucket_idx].y, bucket.y);
        block_buckets[block_bucket_idx].infinity = bucket.infinity;
    }
}

// Pippenger kernel: Accumulate points into buckets for multi-limb scalars
template<typename PointType>
__global__ void kernel_accumulate_buckets_multi(
    PointType* buckets,
    const PointType* points,
    const uint64_t* scalars,
    int scalar_limbs,
    int n,
    int window_idx,
    int window_size
) {
    using Traits = PointTraits<PointType>;
    
    // Same approach as u64 version: process each bucket sequentially
    int bucket_idx = blockIdx.x;
    if (bucket_idx == 0 || bucket_idx >= MSM_BUCKET_COUNT) return;
    
    PointType bucket_sum;
    Traits::point_at_infinity(bucket_sum);
    
    int points_per_thread = (n + blockDim.x - 1) / blockDim.x;
    int start_idx = threadIdx.x * points_per_thread;
    int end_idx = min(start_idx + points_per_thread, n);
    
    for (int i = start_idx; i < end_idx; i++) {
        int point_bucket = extract_window_multi(scalars + i * scalar_limbs, scalar_limbs, window_idx, window_size);
        if (point_bucket == bucket_idx) {
            if (Traits::is_infinity(bucket_sum)) {
                Traits::field_copy(bucket_sum.x, points[i].x);
                Traits::field_copy(bucket_sum.y, points[i].y);
                bucket_sum.infinity = points[i].infinity;
            } else {
                PointType temp;
                point_add(temp, bucket_sum, points[i]);
                Traits::field_copy(bucket_sum.x, temp.x);
                Traits::field_copy(bucket_sum.y, temp.y);
                bucket_sum.infinity = temp.infinity;
            }
        }
    }
    
    // Reduce within block using dynamic shared memory
    extern __shared__ char shared_mem[];
    PointType* shared_sums = (PointType*)shared_mem;
    shared_sums[threadIdx.x] = bucket_sum;
    __syncthreads();
    
    // Thread 0 reduces all thread results
    if (threadIdx.x == 0) {
        PointType total_sum;
        Traits::point_at_infinity(total_sum);
        for (int i = 0; i < blockDim.x; i++) {
            if (!Traits::is_infinity(shared_sums[i])) {
                if (Traits::is_infinity(total_sum)) {
                    Traits::field_copy(total_sum.x, shared_sums[i].x);
                    Traits::field_copy(total_sum.y, shared_sums[i].y);
                    total_sum.infinity = shared_sums[i].infinity;
                } else {
                    PointType temp;
                    point_add(temp, total_sum, shared_sums[i]);
                    Traits::field_copy(total_sum.x, temp.x);
                    Traits::field_copy(total_sum.y, temp.y);
                    total_sum.infinity = temp.infinity;
                }
            }
        }
        Traits::field_copy(buckets[bucket_idx].x, total_sum.x);
        Traits::field_copy(buckets[bucket_idx].y, total_sum.y);
        buckets[bucket_idx].infinity = total_sum.infinity;
    }
}

// Pippenger kernel: Combine buckets for a window and accumulate into result
// Standard Pippenger: window_sum = bucket[1] * 1 + bucket[2] * 2 + ... + bucket[15] * 15
// Using Horner's method: window_sum = bucket[1] + 2 * (bucket[2] + 2 * (bucket[3] + ... + 2 * bucket[15]))
// Then result = result * 2^window_size + window_sum
// Helper function: Compute k * P using binary method (for small k, 1 <= k <= 15)
template<typename PointType>
__device__ void point_scalar_mul_small(PointType& result, const PointType& P, int k) {
    using Traits = PointTraits<PointType>;
    
    if (k == 0 || Traits::is_infinity(P)) {
        Traits::point_at_infinity(result);
        return;
    }
    
    if (k == 1) {
        Traits::field_copy(result.x, P.x);
        Traits::field_copy(result.y, P.y);
        result.infinity = P.infinity;
        return;
    }
    
    // Binary scalar multiplication: k * P
    // Start with P, then for each bit from MSB-1 to LSB: double, then add P if bit is set
    PointType acc;
    Traits::field_copy(acc.x, P.x);
    Traits::field_copy(acc.y, P.y);
    acc.infinity = P.infinity;
    
    // Find the MSB of k
    int msb = 31 - __clz(k);  // __clz counts leading zeros, so msb is the highest set bit
    
    // Process bits from msb-1 down to 0
    for (int bit = msb - 1; bit >= 0; bit--) {
        point_double(acc, acc);
        if (k & (1 << bit)) {
            PointType temp;
            point_add(temp, acc, P);
            Traits::field_copy(acc.x, temp.x);
            Traits::field_copy(acc.y, temp.y);
            acc.infinity = temp.infinity;
        }
    }
    
    Traits::field_copy(result.x, acc.x);
    Traits::field_copy(result.y, acc.y);
    result.infinity = acc.infinity;
}

template<typename PointType>
__global__ void kernel_combine_buckets(
    PointType* result,
    PointType* buckets,
    int num_buckets,
    int window_idx
) {
    using Traits = PointTraits<PointType>;
    
    // Shared memory for storing weighted buckets and reduction tree
    extern __shared__ char shared_mem[];
    PointType* shared_weighted = (PointType*)shared_mem;
    
    // Each thread processes one bucket (bucket index = threadIdx.x + 1, since bucket[0] is not used)
    int bucket_idx = threadIdx.x + 1;
    
    // Compute i * bucket[i] for this thread's bucket using binary scalar multiplication
    if (bucket_idx < num_buckets) {
        if (!Traits::is_infinity(buckets[bucket_idx])) {
            point_scalar_mul_small(shared_weighted[threadIdx.x], buckets[bucket_idx], bucket_idx);
        } else {
            Traits::point_at_infinity(shared_weighted[threadIdx.x]);
        }
    } else {
        // Threads beyond num_buckets-1 set to infinity
        Traits::point_at_infinity(shared_weighted[threadIdx.x]);
    }
    
    __syncthreads();
    
    // Reduction tree: combine all weighted buckets
    // Use standard parallel reduction pattern
    int active_threads = num_buckets - 1;  // Number of buckets to process (buckets 1 to num_buckets-1)
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && threadIdx.x + stride < active_threads) {
            if (!Traits::is_infinity(shared_weighted[threadIdx.x + stride])) {
                if (Traits::is_infinity(shared_weighted[threadIdx.x])) {
                    Traits::field_copy(shared_weighted[threadIdx.x].x, shared_weighted[threadIdx.x + stride].x);
                    Traits::field_copy(shared_weighted[threadIdx.x].y, shared_weighted[threadIdx.x + stride].y);
                    shared_weighted[threadIdx.x].infinity = shared_weighted[threadIdx.x + stride].infinity;
                } else {
                    PointType temp;
                    point_add(temp, shared_weighted[threadIdx.x], shared_weighted[threadIdx.x + stride]);
                    Traits::field_copy(shared_weighted[threadIdx.x].x, temp.x);
                    Traits::field_copy(shared_weighted[threadIdx.x].y, temp.y);
                    shared_weighted[threadIdx.x].infinity = temp.infinity;
                }
            }
        }
        __syncthreads();
    }
    
    // Thread 0 has the final window_sum, add it to result
    if (threadIdx.x == 0) {
        PointType window_sum = shared_weighted[0];
        
        // Add window sum to result
        // For windows processed from MSB to LSB:
        // - First window (MSB, highest window_idx): result = window_sum (no multiplication)
        // - Subsequent windows: result = result * 2^window_size + window_sum
        if (!Traits::is_infinity(window_sum)) {
            if (Traits::is_infinity(*result)) {
                // First non-zero window: just copy window_sum
                Traits::field_copy(result->x, window_sum.x);
                Traits::field_copy(result->y, window_sum.y);
                result->infinity = window_sum.infinity;
            } else {
                // Multiply result by 2^window_size before adding window_sum
                for (int i = 0; i < MSM_WINDOW_SIZE; i++) {
                    point_double(*result, *result);
                }
                // Add window_sum to result
                PointType temp;
                point_add(temp, *result, window_sum);
                Traits::field_copy(result->x, temp.x);
                Traits::field_copy(result->y, temp.y);
                result->infinity = temp.infinity;
            }
        } else if (!Traits::is_infinity(*result)) {
            // Window sum is zero but result is not: still need to multiply result
            for (int i = 0; i < MSM_WINDOW_SIZE; i++) {
                point_double(*result, *result);
            }
        }
    }
}

// Legacy kernels for backward compatibility (kept for reference, but not used in new implementation)
// Template kernel: Compute scalar[i] * points[i] with 64-bit scalars
template<typename PointType>
__global__ void kernel_scalar_mul_u64_array(
    PointType* results,
    const PointType* points,
    const uint64_t* scalars,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        point_scalar_mul(results[idx], points[idx], &scalars[idx], 1);
    }
}

// Template kernel: Compute scalar[i] * points[i] with multi-limb scalars
template<typename PointType>
__global__ void kernel_scalar_mul_array(
    PointType* results,
    const PointType* points,
    const uint64_t* scalars,
    int scalar_limbs,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        point_scalar_mul(results[idx], points[idx], scalars + idx * scalar_limbs, scalar_limbs);
    }
}

// Template kernel: Reduce array of points by addition
template<typename PointType>
__global__ void kernel_reduce_sum(
    PointType* result,
    const PointType* points,
    int n
) {
    using Traits = PointTraits<PointType>;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        Traits::point_at_infinity(*result);
        for (int i = 0; i < n; i++) {
            PointType temp;
            point_add(temp, *result, points[i]);
            Traits::field_copy(result->x, temp.x);
            Traits::field_copy(result->y, temp.y);
            result->infinity = temp.infinity;
        }
    }
}

// Template kernels for array operations (replacing legacy g1_* and g2_* kernels)

// Template kernel: Compute scalar[i] * points[i] with 64-bit scalars
template<typename PointType>
__global__ void kernel_point_scalar_mul_u64_array(
    PointType* results,
    const PointType* points,
    const uint64_t* scalars,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        point_scalar_mul(results[idx], points[idx], &scalars[idx], 1);
    }
}

// Template kernel: Compute scalar[i] * points[i] with multi-limb scalars
template<typename PointType>
__global__ void kernel_point_scalar_mul_array(
    PointType* results,
    const PointType* points,
    const uint64_t* scalars,
    int scalar_limbs,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        point_scalar_mul(results[idx], points[idx], scalars + idx * scalar_limbs, scalar_limbs);
    }
}

// Template kernel: Reduce array of points by addition
template<typename PointType>
__global__ void kernel_point_reduce_sum(
    PointType* result,
    const PointType* points,
    int n
) {
    using Traits = PointTraits<PointType>;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        Traits::point_at_infinity(*result);
        for (int i = 0; i < n; i++) {
            PointType temp;
            point_add(temp, *result, points[i]);
            Traits::field_copy(result->x, temp.x);
            Traits::field_copy(result->y, temp.y);
            result->infinity = temp.infinity;
        }
    }
}

// ============================================================================
// Template Kernels for async/sync API (work on device pointers)
// ============================================================================

// Template kernel: Point addition
template<typename PointType>
__global__ void kernel_point_add(PointType* result, const PointType* p1, const PointType* p2) {
    point_add(*result, *p1, *p2);
}

// Template kernel: Point doubling
template<typename PointType>
__global__ void kernel_point_double(PointType* result, const PointType* p) {
    point_double(*result, *p);
}

// Template kernel: Point negation
template<typename PointType>
__global__ void kernel_point_neg(PointType* result, const PointType* p) {
    point_neg(*result, *p);
}

// Template kernel: Point at infinity
template<typename PointType>
__global__ void kernel_point_at_infinity(PointType* result) {
    using Traits = PointTraits<PointType>;
    Traits::point_at_infinity(*result);
}

// Template kernel: Convert point to Montgomery form
template<typename PointType>
__global__ void kernel_point_to_montgomery(PointType* result, const PointType* point) {
    using Traits = PointTraits<PointType>;
    if (point->infinity) {
        result->infinity = true;
        Traits::field_zero(result->x);
        Traits::field_zero(result->y);
    } else {
        Traits::field_to_montgomery(result->x, point->x);
        Traits::field_to_montgomery(result->y, point->y);
        result->infinity = false;
    }
}

// Template kernel: Convert point from Montgomery form
template<typename PointType>
__global__ void kernel_point_from_montgomery(PointType* result, const PointType* point) {
    using Traits = PointTraits<PointType>;
    if (point->infinity) {
        result->infinity = true;
        Traits::field_zero(result->x);
        Traits::field_zero(result->y);
    } else {
        Traits::field_from_montgomery(result->x, point->x);
        Traits::field_from_montgomery(result->y, point->y);
        result->infinity = false;
    }
}

// Template kernel: Batch convert points to Montgomery form
template<typename PointType>
__global__ void kernel_point_to_montgomery_batch(PointType* points, int n) {
    using Traits = PointTraits<PointType>;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (!points[idx].infinity) {
            Traits::field_to_montgomery(points[idx].x, points[idx].x);
            Traits::field_to_montgomery(points[idx].y, points[idx].y);
        }
    }
}

// Template kernel: Batch convert points from Montgomery form
template<typename PointType>
__global__ void kernel_point_from_montgomery_batch(PointType* points, int n) {
    using Traits = PointTraits<PointType>;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (!points[idx].infinity) {
            Traits::field_from_montgomery(points[idx].x, points[idx].x);
            Traits::field_from_montgomery(points[idx].y, points[idx].y);
        }
    }
}

// Template kernel: Scalar multiplication with 64-bit scalar
template<typename PointType>
__global__ void kernel_point_scalar_mul_u64(PointType* result, const PointType* point, uint64_t scalar) {
    point_scalar_mul(*result, *point, &scalar, 1);
}

// Template kernel: Scalar multiplication with multi-limb scalar
template<typename PointType>
__global__ void kernel_point_scalar_mul(PointType* result, const PointType* point, const uint64_t* scalar, int scalar_limbs) {
    point_scalar_mul(*result, *point, scalar, scalar_limbs);
}


// ============================================================================
// Template Async/Sync API implementations
// ============================================================================

// Template function: Point addition
template<typename PointType>
void point_add_async(cudaStream_t stream, uint32_t gpu_index, PointType* d_result, const PointType* d_p1, const PointType* d_p2) {
    PANIC_IF_FALSE(d_result != nullptr && d_p1 != nullptr && d_p2 != nullptr, "point_add_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_point_add<PointType><<<1, 1, 0, stream>>>(d_result, d_p1, d_p2);
    check_cuda_error(cudaGetLastError());
}

template<typename PointType>
void point_add(cudaStream_t stream, uint32_t gpu_index, PointType* d_result, const PointType* d_p1, const PointType* d_p2) {
    point_add_async<PointType>(stream, gpu_index, d_result, d_p1, d_p2);
    cuda_synchronize_stream(stream, gpu_index);
}

// Template function: Point doubling
template<typename PointType>
void point_double_async(cudaStream_t stream, uint32_t gpu_index, PointType* d_result, const PointType* d_p) {
    PANIC_IF_FALSE(d_result != nullptr && d_p != nullptr, "point_double_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_point_double<PointType><<<1, 1, 0, stream>>>(d_result, d_p);
    check_cuda_error(cudaGetLastError());
}

template<typename PointType>
void point_double(cudaStream_t stream, uint32_t gpu_index, PointType* d_result, const PointType* d_p) {
    point_double_async<PointType>(stream, gpu_index, d_result, d_p);
    cuda_synchronize_stream(stream, gpu_index);
}

// Template function: Point negation
template<typename PointType>
void point_neg_async(cudaStream_t stream, uint32_t gpu_index, PointType* d_result, const PointType* d_p) {
    PANIC_IF_FALSE(d_result != nullptr && d_p != nullptr, "point_neg_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_point_neg<PointType><<<1, 1, 0, stream>>>(d_result, d_p);
    check_cuda_error(cudaGetLastError());
}

template<typename PointType>
void point_neg(cudaStream_t stream, uint32_t gpu_index, PointType* d_result, const PointType* d_p) {
    point_neg_async<PointType>(stream, gpu_index, d_result, d_p);
    cuda_synchronize_stream(stream, gpu_index);
}

// Template function: Point at infinity
template<typename PointType>
void point_at_infinity_async(cudaStream_t stream, uint32_t gpu_index, PointType* d_result) {
    PANIC_IF_FALSE(d_result != nullptr, "point_at_infinity_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_point_at_infinity<PointType><<<1, 1, 0, stream>>>(d_result);
    check_cuda_error(cudaGetLastError());
}

template<typename PointType>
void point_at_infinity(cudaStream_t stream, uint32_t gpu_index, PointType* d_result) {
    point_at_infinity_async<PointType>(stream, gpu_index, d_result);
    cuda_synchronize_stream(stream, gpu_index);
}

// Template function: Convert point to Montgomery form
template<typename PointType>
void point_to_montgomery_async(cudaStream_t stream, uint32_t gpu_index, PointType* d_result, const PointType* d_point) {
    PANIC_IF_FALSE(d_result != nullptr && d_point != nullptr, "point_to_montgomery_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_point_to_montgomery<PointType><<<1, 1, 0, stream>>>(d_result, d_point);
    check_cuda_error(cudaGetLastError());
}

template<typename PointType>
void point_to_montgomery(cudaStream_t stream, uint32_t gpu_index, PointType* d_result, const PointType* d_point) {
    point_to_montgomery_async<PointType>(stream, gpu_index, d_result, d_point);
    cuda_synchronize_stream(stream, gpu_index);
}

// Template function: Convert point from Montgomery form
template<typename PointType>
void point_from_montgomery_async(cudaStream_t stream, uint32_t gpu_index, PointType* d_result, const PointType* d_point) {
    PANIC_IF_FALSE(d_result != nullptr && d_point != nullptr, "point_from_montgomery_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_point_from_montgomery<PointType><<<1, 1, 0, stream>>>(d_result, d_point);
    check_cuda_error(cudaGetLastError());
}

template<typename PointType>
void point_from_montgomery(cudaStream_t stream, uint32_t gpu_index, PointType* d_result, const PointType* d_point) {
    point_from_montgomery_async<PointType>(stream, gpu_index, d_result, d_point);
    cuda_synchronize_stream(stream, gpu_index);
}

// Template function: Scalar multiplication with 64-bit scalar
template<typename PointType>
void point_scalar_mul_u64_async(cudaStream_t stream, uint32_t gpu_index, PointType* d_result, const PointType* d_point, uint64_t scalar) {
    PANIC_IF_FALSE(d_result != nullptr && d_point != nullptr, "point_scalar_mul_u64_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_point_scalar_mul_u64<PointType><<<1, 1, 0, stream>>>(d_result, d_point, scalar);
    check_cuda_error(cudaGetLastError());
}

template<typename PointType>
void point_scalar_mul_u64(cudaStream_t stream, uint32_t gpu_index, PointType* d_result, const PointType* d_point, uint64_t scalar) {
    point_scalar_mul_u64_async<PointType>(stream, gpu_index, d_result, d_point, scalar);
    cuda_synchronize_stream(stream, gpu_index);
}

// Template function: Scalar multiplication with multi-limb scalar
template<typename PointType>
void point_scalar_mul_async(cudaStream_t stream, uint32_t gpu_index, PointType* d_result, const PointType* d_point, const uint64_t* d_scalar, int scalar_limbs) {
    PANIC_IF_FALSE(d_result != nullptr && d_point != nullptr && d_scalar != nullptr, "point_scalar_mul_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_point_scalar_mul<PointType><<<1, 1, 0, stream>>>(d_result, d_point, d_scalar, scalar_limbs);
    check_cuda_error(cudaGetLastError());
}

template<typename PointType>
void point_scalar_mul(cudaStream_t stream, uint32_t gpu_index, PointType* d_result, const PointType* d_point, const uint64_t* d_scalar, int scalar_limbs) {
    point_scalar_mul_async<PointType>(stream, gpu_index, d_result, d_point, d_scalar, scalar_limbs);
    cuda_synchronize_stream(stream, gpu_index);
}

// Template function: Batch convert points to Montgomery form
template<typename PointType>
void point_to_montgomery_batch_async(cudaStream_t stream, uint32_t gpu_index, PointType* d_points, int n) {
    PANIC_IF_FALSE(d_points != nullptr, "point_to_montgomery_batch_async: null pointer argument");
    PANIC_IF_FALSE(n >= 0, "point_to_montgomery_batch_async: invalid size n=%d", n);
    if (n == 0) return;
    
    cuda_set_device(gpu_index);
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    kernel_point_to_montgomery_batch<PointType><<<blocks, threadsPerBlock, 0, stream>>>(d_points, n);
    check_cuda_error(cudaGetLastError());
}

template<typename PointType>
void point_to_montgomery_batch(cudaStream_t stream, uint32_t gpu_index, PointType* d_points, int n) {
    point_to_montgomery_batch_async<PointType>(stream, gpu_index, d_points, n);
    cuda_synchronize_stream(stream, gpu_index);
}


// ============================================================================
// Refactored MSM API (device pointers only, no allocations/copies/frees)
// ============================================================================

// Helper function to get optimal threads per block for MSM based on point type
template<typename PointType>
constexpr int get_msm_threads_per_block() {
    // G1Point is smaller (Fp fields), use 256 threads
    // G2Point is larger (Fp2 fields), use 128 threads to avoid exceeding shared memory limits
    return sizeof(PointType) <= sizeof(G1Point) ? 256 : 128;
}

// Template MSM with 64-bit scalars - async version (Pippenger algorithm)
template<typename PointType>
void point_msm_u64_async(cudaStream_t stream, uint32_t gpu_index, PointType* d_result, const PointType* d_points, const uint64_t* d_scalars, PointType* d_scratch, int n) {
    if (n == 0) {
        point_at_infinity_async<PointType>(stream, gpu_index, d_result);
        return;
    }
    
    PANIC_IF_FALSE(n > 0, "point_msm_u64_async: invalid size n=%d", n);
    PANIC_IF_FALSE(d_result != nullptr && d_points != nullptr && d_scalars != nullptr && d_scratch != nullptr, "point_msm_u64_async: null pointer argument");
    
    cuda_set_device(gpu_index);
    
    // Calculate number of windows (64 bits / window_size)
    int num_windows = (64 + MSM_WINDOW_SIZE - 1) / MSM_WINDOW_SIZE;
    
    // Initialize result to point at infinity
    point_at_infinity_async<PointType>(stream, gpu_index, d_result);
    
    // Process each window from MSB to LSB
    int threadsPerBlock = get_msm_threads_per_block<PointType>();
    int num_blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Scratch space layout:
    // - d_scratch[0 .. num_blocks * MSM_BUCKET_COUNT - 1]: per-block bucket accumulations
    // - d_scratch[num_blocks * MSM_BUCKET_COUNT .. (num_blocks + 1) * MSM_BUCKET_COUNT - 1]: final buckets
    PointType* d_block_buckets = d_scratch;
    PointType* d_final_buckets = d_scratch + num_blocks * MSM_BUCKET_COUNT;
    
    for (int window_idx = num_windows - 1; window_idx >= 0; window_idx--) {
        // Clear final buckets
        int clear_blocks = (MSM_BUCKET_COUNT + threadsPerBlock - 1) / threadsPerBlock;
        kernel_clear_buckets<PointType><<<clear_blocks, threadsPerBlock, 0, stream>>>(d_final_buckets, MSM_BUCKET_COUNT);
        check_cuda_error(cudaGetLastError());
        
        // Clear block buckets
        int clear_block_blocks = (num_blocks * MSM_BUCKET_COUNT + threadsPerBlock - 1) / threadsPerBlock;
        kernel_clear_buckets<PointType><<<clear_block_blocks, threadsPerBlock, 0, stream>>>(d_block_buckets, num_blocks * MSM_BUCKET_COUNT);
        check_cuda_error(cudaGetLastError());
        
        // Phase 1: Accumulate points into per-block buckets (single-pass, all points processed in parallel)
        // Shared memory: MSM_BUCKET_COUNT buckets + threadsPerBlock points + threadsPerBlock bucket indices
        size_t shared_mem_size = MSM_BUCKET_COUNT * sizeof(PointType) + threadsPerBlock * sizeof(PointType) + threadsPerBlock * sizeof(int);
        kernel_accumulate_buckets_u64<PointType><<<num_blocks, threadsPerBlock, shared_mem_size, stream>>>(
            d_block_buckets, d_points, d_scalars, n, window_idx, MSM_WINDOW_SIZE
        );
        check_cuda_error(cudaGetLastError());
        
        // Phase 2: Reduce per-block bucket contributions to final buckets
        int reduce_threads = threadsPerBlock;
        int reduce_blocks = (MSM_BUCKET_COUNT + reduce_threads - 1) / reduce_threads;
        kernel_reduce_buckets<PointType><<<reduce_blocks, reduce_threads, 0, stream>>>(
            d_final_buckets, d_block_buckets, num_blocks, MSM_BUCKET_COUNT
        );
        check_cuda_error(cudaGetLastError());
        
        // Combine final buckets and accumulate into result
        // Use 16 threads (one per bucket) with shared memory for parallel binary scalar multiplication and reduction
        size_t combine_shared_mem = MSM_BUCKET_COUNT * sizeof(PointType);
        kernel_combine_buckets<PointType><<<1, MSM_BUCKET_COUNT, combine_shared_mem, stream>>>(
            d_result, d_final_buckets, MSM_BUCKET_COUNT, window_idx
        );
        check_cuda_error(cudaGetLastError());
    }
}

template<typename PointType>
void point_msm_u64(cudaStream_t stream, uint32_t gpu_index, PointType* d_result, const PointType* d_points, const uint64_t* d_scalars, PointType* d_scratch, int n) {
    point_msm_u64_async<PointType>(stream, gpu_index, d_result, d_points, d_scalars, d_scratch, n);
    cuda_synchronize_stream(stream, gpu_index);
}

// Template MSM with multi-limb scalars - async version (Pippenger algorithm)
template<typename PointType>
void point_msm_async(cudaStream_t stream, uint32_t gpu_index, PointType* d_result, const PointType* d_points, const uint64_t* d_scalars, int scalar_limbs, PointType* d_scratch, int n) {
    if (n == 0) {
        point_at_infinity_async<PointType>(stream, gpu_index, d_result);
        return;
    }
    
    PANIC_IF_FALSE(n > 0, "point_msm_async: invalid size n=%d", n);
    PANIC_IF_FALSE(d_result != nullptr && d_points != nullptr && d_scalars != nullptr && d_scratch != nullptr, "point_msm_async: null pointer argument");
    
    cuda_set_device(gpu_index);
    
    // Use d_scratch as bucket storage (need MSM_BUCKET_COUNT buckets)
    PointType* d_buckets = d_scratch;
    
    // Calculate number of windows
    int total_bits = scalar_limbs * 64;
    int num_windows = (total_bits + MSM_WINDOW_SIZE - 1) / MSM_WINDOW_SIZE;
    
    // Initialize result to point at infinity
    point_at_infinity_async<PointType>(stream, gpu_index, d_result);
    
    // Process each window from MSB to LSB
    int threadsPerBlock = get_msm_threads_per_block<PointType>();
    for (int window_idx = num_windows - 1; window_idx >= 0; window_idx--) {
        // Clear buckets
        int clear_blocks = (MSM_BUCKET_COUNT + threadsPerBlock - 1) / threadsPerBlock;
        kernel_clear_buckets<PointType><<<clear_blocks, threadsPerBlock, 0, stream>>>(d_buckets, MSM_BUCKET_COUNT);
        check_cuda_error(cudaGetLastError());
        
        // Accumulate points into buckets (one block per bucket)
        size_t shared_mem_size = threadsPerBlock * sizeof(PointType);
        kernel_accumulate_buckets_multi<PointType><<<MSM_BUCKET_COUNT, threadsPerBlock, shared_mem_size, stream>>>(
            d_buckets, d_points, d_scalars, scalar_limbs, n, window_idx, MSM_WINDOW_SIZE
        );
        check_cuda_error(cudaGetLastError());
        
        // Combine buckets and accumulate into result
        // Use 16 threads (one per bucket) with shared memory for parallel binary scalar multiplication and reduction
        size_t combine_shared_mem = MSM_BUCKET_COUNT * sizeof(PointType);
        kernel_combine_buckets<PointType><<<1, MSM_BUCKET_COUNT, combine_shared_mem, stream>>>(
            d_result, d_buckets, MSM_BUCKET_COUNT, window_idx
        );
        check_cuda_error(cudaGetLastError());
    }
}

template<typename PointType>
void point_msm(cudaStream_t stream, uint32_t gpu_index, PointType* d_result, const PointType* d_points, const uint64_t* d_scalars, int scalar_limbs, PointType* d_scratch, int n) {
    point_msm_async<PointType>(stream, gpu_index, d_result, d_points, d_scalars, scalar_limbs, d_scratch, n);
    cuda_synchronize_stream(stream, gpu_index);
}

// ============================================================================
// Explicit template instantiations (needed for external linkage)
// ============================================================================

// Async/Sync API instantiations
template void point_add_async<G1Point>(cudaStream_t, uint32_t, G1Point*, const G1Point*, const G1Point*);
template void point_add<G1Point>(cudaStream_t, uint32_t, G1Point*, const G1Point*, const G1Point*);
template void point_double_async<G1Point>(cudaStream_t, uint32_t, G1Point*, const G1Point*);
template void point_double<G1Point>(cudaStream_t, uint32_t, G1Point*, const G1Point*);
template void point_neg_async<G1Point>(cudaStream_t, uint32_t, G1Point*, const G1Point*);
template void point_neg<G1Point>(cudaStream_t, uint32_t, G1Point*, const G1Point*);
template void point_at_infinity_async<G1Point>(cudaStream_t, uint32_t, G1Point*);
template void point_at_infinity<G1Point>(cudaStream_t, uint32_t, G1Point*);
template void point_to_montgomery_async<G1Point>(cudaStream_t, uint32_t, G1Point*, const G1Point*);
template void point_to_montgomery<G1Point>(cudaStream_t, uint32_t, G1Point*, const G1Point*);
template void point_from_montgomery_async<G1Point>(cudaStream_t, uint32_t, G1Point*, const G1Point*);
template void point_from_montgomery<G1Point>(cudaStream_t, uint32_t, G1Point*, const G1Point*);
template void point_scalar_mul_u64_async<G1Point>(cudaStream_t, uint32_t, G1Point*, const G1Point*, uint64_t);
template void point_scalar_mul_u64<G1Point>(cudaStream_t, uint32_t, G1Point*, const G1Point*, uint64_t);
template void point_scalar_mul_async<G1Point>(cudaStream_t, uint32_t, G1Point*, const G1Point*, const uint64_t*, int);
template void point_scalar_mul<G1Point>(cudaStream_t, uint32_t, G1Point*, const G1Point*, const uint64_t*, int);
template void point_to_montgomery_batch_async<G1Point>(cudaStream_t, uint32_t, G1Point*, int);
template void point_to_montgomery_batch<G1Point>(cudaStream_t, uint32_t, G1Point*, int);
template void point_msm_u64_async<G1Point>(cudaStream_t, uint32_t, G1Point*, const G1Point*, const uint64_t*, G1Point*, int);
template void point_msm_u64<G1Point>(cudaStream_t, uint32_t, G1Point*, const G1Point*, const uint64_t*, G1Point*, int);
template void point_msm_async<G1Point>(cudaStream_t, uint32_t, G1Point*, const G1Point*, const uint64_t*, int, G1Point*, int);
template void point_msm<G1Point>(cudaStream_t, uint32_t, G1Point*, const G1Point*, const uint64_t*, int, G1Point*, int);

template void point_add_async<G2Point>(cudaStream_t, uint32_t, G2Point*, const G2Point*, const G2Point*);
template void point_add<G2Point>(cudaStream_t, uint32_t, G2Point*, const G2Point*, const G2Point*);
template void point_double_async<G2Point>(cudaStream_t, uint32_t, G2Point*, const G2Point*);
template void point_double<G2Point>(cudaStream_t, uint32_t, G2Point*, const G2Point*);
template void point_neg_async<G2Point>(cudaStream_t, uint32_t, G2Point*, const G2Point*);
template void point_neg<G2Point>(cudaStream_t, uint32_t, G2Point*, const G2Point*);
template void point_at_infinity_async<G2Point>(cudaStream_t, uint32_t, G2Point*);
template void point_at_infinity<G2Point>(cudaStream_t, uint32_t, G2Point*);
template void point_to_montgomery_async<G2Point>(cudaStream_t, uint32_t, G2Point*, const G2Point*);
template void point_to_montgomery<G2Point>(cudaStream_t, uint32_t, G2Point*, const G2Point*);
template void point_from_montgomery_async<G2Point>(cudaStream_t, uint32_t, G2Point*, const G2Point*);
template void point_from_montgomery<G2Point>(cudaStream_t, uint32_t, G2Point*, const G2Point*);
template void point_scalar_mul_u64_async<G2Point>(cudaStream_t, uint32_t, G2Point*, const G2Point*, uint64_t);
template void point_scalar_mul_u64<G2Point>(cudaStream_t, uint32_t, G2Point*, const G2Point*, uint64_t);
template void point_scalar_mul_async<G2Point>(cudaStream_t, uint32_t, G2Point*, const G2Point*, const uint64_t*, int);
template void point_scalar_mul<G2Point>(cudaStream_t, uint32_t, G2Point*, const G2Point*, const uint64_t*, int);
template void point_to_montgomery_batch_async<G2Point>(cudaStream_t, uint32_t, G2Point*, int);
template void point_to_montgomery_batch<G2Point>(cudaStream_t, uint32_t, G2Point*, int);
template void point_msm_u64_async<G2Point>(cudaStream_t, uint32_t, G2Point*, const G2Point*, const uint64_t*, G2Point*, int);
template void point_msm_u64<G2Point>(cudaStream_t, uint32_t, G2Point*, const G2Point*, const uint64_t*, G2Point*, int);
template void point_msm_async<G2Point>(cudaStream_t, uint32_t, G2Point*, const G2Point*, const uint64_t*, int, G2Point*, int);
template void point_msm<G2Point>(cudaStream_t, uint32_t, G2Point*, const G2Point*, const uint64_t*, int, G2Point*, int);

