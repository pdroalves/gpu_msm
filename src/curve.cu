#include "curve.h"
#include "fp.h"
#include "fp2.h"
#include "device.h"
#include <cstring>
#include <cstdio>

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
// G1 Point Operations
// ============================================================================

// Point negation: result = -p = (x, -y)
__host__ __device__ void g1_neg(G1Point& result, const G1Point& p) {
    if (g1_is_infinity(p)) {
        g1_point_at_infinity(result);
        return;
    }
    fp_copy(result.x, p.x);
    fp_neg(result.y, p.y);
    result.infinity = false;
}

// Point doubling: result = 2 * p
// Using the standard formula for doubling on y^2 = x^3 + b
// lambda = 3*x^2 / (2*y) = (3*x^2) * (2*y)^(-1)
// x_result = lambda^2 - 2*x
// y_result = lambda*(x - x_result) - y
__host__ __device__ void g1_double(G1Point& result, const G1Point& p) {
    if (g1_is_infinity(p) || fp_is_zero(p.y)) {
        g1_point_at_infinity(result);
        return;
    }
    
    // Compute lambda = 3*x^2 / (2*y)
    Fp x_squared, three_x_squared, two_y, lambda;
    fp_mul(x_squared, p.x, p.x);
    fp_add(three_x_squared, x_squared, x_squared);  // 2*x^2
    fp_add(three_x_squared, three_x_squared, x_squared);  // 3*x^2
    
    fp_add(two_y, p.y, p.y);  // 2*y
    fp_inv(lambda, two_y);  // 1/(2*y)
    fp_mul(lambda, lambda, three_x_squared);  // 3*x^2 / (2*y)
    
    // x_result = lambda^2 - 2*x
    Fp lambda_squared, two_x, x_result;
    fp_mul(lambda_squared, lambda, lambda);
    fp_add(two_x, p.x, p.x);  // 2*x
    fp_sub(x_result, lambda_squared, two_x);
    
    // y_result = lambda*(x - x_result) - y
    Fp x_minus_xr, y_result;
    fp_sub(x_minus_xr, p.x, x_result);
    fp_mul(y_result, lambda, x_minus_xr);
    fp_sub(y_result, y_result, p.y);
    
    fp_copy(result.x, x_result);
    fp_copy(result.y, y_result);
    result.infinity = false;
}

// Point addition: result = p1 + p2
// Using the standard elliptic curve addition formula
__host__ __device__ void g1_add(G1Point& result, const G1Point& p1, const G1Point& p2) {
    // Handle infinity cases
    if (g1_is_infinity(p1)) {
        fp_copy(result.x, p2.x);
        fp_copy(result.y, p2.y);
        result.infinity = p2.infinity;
        return;
    }
    if (g1_is_infinity(p2)) {
        fp_copy(result.x, p1.x);
        fp_copy(result.y, p1.y);
        result.infinity = p1.infinity;
        return;
    }
    
    // Check if p1 == -p2 (same x, opposite y)
    Fp neg_y2;
    fp_neg(neg_y2, p2.y);
    if (fp_cmp(p1.x, p2.x) == 0 && fp_cmp(p1.y, neg_y2) == 0) {
        g1_point_at_infinity(result);
        return;
    }
    
    // Check if p1 == p2 (use doubling)
    if (fp_cmp(p1.x, p2.x) == 0 && fp_cmp(p1.y, p2.y) == 0) {
        g1_double(result, p1);
        return;
    }
    
    // Standard addition: lambda = (y2 - y1) / (x2 - x1)
    Fp dx, dy, lambda, lambda_squared, x_result;
    fp_sub(dx, p2.x, p1.x);
    fp_sub(dy, p2.y, p1.y);
    fp_inv(lambda, dx);  // 1 / (x2 - x1)
    fp_mul(lambda, lambda, dy);  // (y2 - y1) / (x2 - x1)
    
    // x_result = lambda^2 - x1 - x2
    fp_mul(lambda_squared, lambda, lambda);
    fp_sub(x_result, lambda_squared, p1.x);
    fp_sub(x_result, x_result, p2.x);
    
    // y_result = lambda * (x1 - x_result) - y1
    Fp x1_minus_xr, y_result;
    fp_sub(x1_minus_xr, p1.x, x_result);
    fp_mul(y_result, lambda, x1_minus_xr);
    fp_sub(y_result, y_result, p1.y);
    
    fp_copy(result.x, x_result);
    fp_copy(result.y, y_result);
    result.infinity = false;
}

// Scalar multiplication: result = scalar * point
// Uses binary method (double-and-add)
__host__ __device__ void g1_scalar_mul(G1Point& result, const G1Point& point, const uint64_t* scalar, int scalar_limbs) {
    // Start with point at infinity
    g1_point_at_infinity(result);
    
    if (g1_is_infinity(point)) {
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
    
    G1Point current;
    fp_copy(current.x, point.x);
    fp_copy(current.y, point.y);
    current.infinity = point.infinity;
    
    // Process bits from MSB to LSB
    for (int limb = scalar_limbs - 1; limb >= 0; limb--) {
        for (int bit = 63; bit >= 0; bit--) {
            g1_double(result, result);
            
            if ((scalar[limb] >> bit) & 1) {
                g1_add(result, result, current);
            }
        }
    }
}

// Scalar multiplication with 64-bit scalar
__host__ __device__ void g1_scalar_mul_u64(G1Point& result, const G1Point& point, uint64_t scalar) {
    g1_scalar_mul(result, point, &scalar, 1);
}

// ============================================================================
// G2 Point Operations
// ============================================================================

// Point negation: result = -p = (x, -y)
__host__ __device__ void g2_neg(G2Point& result, const G2Point& p) {
    if (g2_is_infinity(p)) {
        g2_point_at_infinity(result);
        return;
    }
    fp2_copy(result.x, p.x);
    fp2_neg(result.y, p.y);
    result.infinity = false;
}

// Point doubling: result = 2 * p
// Using the standard formula for doubling on y^2 = x^3 + b'
__host__ __device__ void g2_double(G2Point& result, const G2Point& p) {
    if (g2_is_infinity(p) || fp2_is_zero(p.y)) {
        g2_point_at_infinity(result);
        return;
    }
    
    // Compute lambda = 3*x^2 / (2*y)
    Fp2 x_squared, three_x_squared, two_y, lambda;
    fp2_mul(x_squared, p.x, p.x);
    fp2_add(three_x_squared, x_squared, x_squared);  // 2*x^2
    fp2_add(three_x_squared, three_x_squared, x_squared);  // 3*x^2
    
    fp2_add(two_y, p.y, p.y);  // 2*y
    fp2_inv(lambda, two_y);  // 1/(2*y)
    fp2_mul(lambda, lambda, three_x_squared);  // 3*x^2 / (2*y)
    
    // x_result = lambda^2 - 2*x
    Fp2 lambda_squared, two_x, x_result;
    fp2_mul(lambda_squared, lambda, lambda);
    fp2_add(two_x, p.x, p.x);  // 2*x
    fp2_sub(x_result, lambda_squared, two_x);
    
    // y_result = lambda*(x - x_result) - y
    Fp2 x_minus_xr, y_result;
    fp2_sub(x_minus_xr, p.x, x_result);
    fp2_mul(y_result, lambda, x_minus_xr);
    fp2_sub(y_result, y_result, p.y);
    
    fp2_copy(result.x, x_result);
    fp2_copy(result.y, y_result);
    result.infinity = false;
}

// Point addition: result = p1 + p2
__host__ __device__ void g2_add(G2Point& result, const G2Point& p1, const G2Point& p2) {
    // Handle infinity cases
    if (g2_is_infinity(p1)) {
        fp2_copy(result.x, p2.x);
        fp2_copy(result.y, p2.y);
        result.infinity = p2.infinity;
        return;
    }
    if (g2_is_infinity(p2)) {
        fp2_copy(result.x, p1.x);
        fp2_copy(result.y, p1.y);
        result.infinity = p1.infinity;
        return;
    }
    
    // Check if p1 == -p2 (same x, opposite y)
    Fp2 neg_y2;
    fp2_neg(neg_y2, p2.y);
    if (fp2_cmp(p1.x, p2.x) == 0 && fp2_cmp(p1.y, neg_y2) == 0) {
        g2_point_at_infinity(result);
        return;
    }
    
    // Check if p1 == p2 (use doubling)
    if (fp2_cmp(p1.x, p2.x) == 0 && fp2_cmp(p1.y, p2.y) == 0) {
        g2_double(result, p1);
        return;
    }
    
    // Standard addition: lambda = (y2 - y1) / (x2 - x1)
    Fp2 dx, dy, lambda, lambda_squared, x_result;
    fp2_sub(dx, p2.x, p1.x);
    fp2_sub(dy, p2.y, p1.y);
    fp2_inv(lambda, dx);  // 1 / (x2 - x1)
    fp2_mul(lambda, lambda, dy);  // (y2 - y1) / (x2 - x1)
    
    // x_result = lambda^2 - x1 - x2
    fp2_mul(lambda_squared, lambda, lambda);
    fp2_sub(x_result, lambda_squared, p1.x);
    fp2_sub(x_result, x_result, p2.x);
    
    // y_result = lambda * (x1 - x_result) - y1
    Fp2 x1_minus_xr, y_result;
    fp2_sub(x1_minus_xr, p1.x, x_result);
    fp2_mul(y_result, lambda, x1_minus_xr);
    fp2_sub(y_result, y_result, p1.y);
    
    fp2_copy(result.x, x_result);
    fp2_copy(result.y, y_result);
    result.infinity = false;
}

// Scalar multiplication: result = scalar * point
__host__ __device__ void g2_scalar_mul(G2Point& result, const G2Point& point, const uint64_t* scalar, int scalar_limbs) {
    // Start with point at infinity
    g2_point_at_infinity(result);
    
    if (g2_is_infinity(point)) {
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
    
    G2Point current;
    fp2_copy(current.x, point.x);
    fp2_copy(current.y, point.y);
    current.infinity = point.infinity;
    
    // Process bits from MSB to LSB
    for (int limb = scalar_limbs - 1; limb >= 0; limb--) {
        for (int bit = 63; bit >= 0; bit--) {
            g2_double(result, result);
            
            if ((scalar[limb] >> bit) & 1) {
                g2_add(result, result, current);
            }
        }
    }
}

// Scalar multiplication with 64-bit scalar
__host__ __device__ void g2_scalar_mul_u64(G2Point& result, const G2Point& point, uint64_t scalar) {
    g2_scalar_mul(result, point, &scalar, 1);
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
// Multi-Scalar Multiplication (MSM)
// ============================================================================

// CUDA kernels for MSM operations

// Kernel: Compute scalar[i] * points[i] for G1 with 64-bit scalars
__global__ void kernel_g1_scalar_mul_u64_array(
    G1Point* results,
    const G1Point* points,
    const uint64_t* scalars,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        g1_scalar_mul_u64(results[idx], points[idx], scalars[idx]);
    }
}

// Kernel: Compute scalar[i] * points[i] for G1 with multi-limb scalars
__global__ void kernel_g1_scalar_mul_array(
    G1Point* results,
    const G1Point* points,
    const uint64_t* scalars,
    int scalar_limbs,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        g1_scalar_mul(results[idx], points[idx], scalars + idx * scalar_limbs, scalar_limbs);
    }
}

// Kernel: Compute scalar[i] * points[i] for G2 with 64-bit scalars
__global__ void kernel_g2_scalar_mul_u64_array(
    G2Point* results,
    const G2Point* points,
    const uint64_t* scalars,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        g2_scalar_mul_u64(results[idx], points[idx], scalars[idx]);
    }
}

// Kernel: Compute scalar[i] * points[i] for G2 with multi-limb scalars
__global__ void kernel_g2_scalar_mul_array(
    G2Point* results,
    const G2Point* points,
    const uint64_t* scalars,
    int scalar_limbs,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        g2_scalar_mul(results[idx], points[idx], scalars + idx * scalar_limbs, scalar_limbs);
    }
}

// Kernel: Reduce array of G1 points by addition (tree reduction)
// This is a simple sequential reduction - can be optimized with parallel reduction later
__global__ void kernel_g1_reduce_sum(
    G1Point* result,
    const G1Point* points,
    int n
) {
    // For now, use a simple approach: first thread does sequential reduction
    // This can be optimized with proper parallel reduction later
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        g1_point_at_infinity(*result);
        for (int i = 0; i < n; i++) {
            G1Point temp;
            g1_add(temp, *result, points[i]);
            fp_copy(result->x, temp.x);
            fp_copy(result->y, temp.y);
            result->infinity = temp.infinity;
        }
    }
}

// Kernel: Reduce array of G2 points by addition
__global__ void kernel_g2_reduce_sum(
    G2Point* result,
    const G2Point* points,
    int n
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        g2_point_at_infinity(*result);
        for (int i = 0; i < n; i++) {
            G2Point temp;
            g2_add(temp, *result, points[i]);
            fp2_copy(result->x, temp.x);
            fp2_copy(result->y, temp.y);
            result->infinity = temp.infinity;
        }
    }
}

// Host function for G1 MSM with 64-bit scalars
// points and scalars are host pointers
void g1_msm_u64(cudaStream_t stream, uint32_t gpu_index, G1Point& result, const G1Point* points, const uint64_t* scalars, int n) {
    if (n == 0) {
        g1_point_at_infinity(result);
        return;
    }
    
    PANIC_IF_FALSE(n > 0, "g1_msm_u64: invalid size n=%d", n);
    PANIC_IF_FALSE(points != nullptr && scalars != nullptr, "g1_msm_u64: null pointer argument");
    
    cuda_set_device(gpu_index);
    
    // Allocate device memory (asynchronous with stream)
    G1Point* d_points = (G1Point*)cuda_malloc_async(n * sizeof(G1Point), stream, gpu_index);
    uint64_t* d_scalars = (uint64_t*)cuda_malloc_async(n * sizeof(uint64_t), stream, gpu_index);
    G1Point* d_results = (G1Point*)cuda_malloc_async(n * sizeof(G1Point), stream, gpu_index);
    G1Point* d_sum = (G1Point*)cuda_malloc_async(sizeof(G1Point), stream, gpu_index);
    
    // Copy to device (asynchronous with stream)
    cuda_memcpy_async_to_gpu(d_points, points, n * sizeof(G1Point), stream, gpu_index);
    cuda_memcpy_async_to_gpu(d_scalars, scalars, n * sizeof(uint64_t), stream, gpu_index);
    
    // Launch kernel to compute scalar[i] * points[i] for each i
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    kernel_g1_scalar_mul_u64_array<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_results, d_points, d_scalars, n);
    check_cuda_error(cudaGetLastError());
    
    // Reduce sum on device
    kernel_g1_reduce_sum<<<1, 1, 0, stream>>>(d_sum, d_results, n);
    check_cuda_error(cudaGetLastError());
    
    // Copy result back (synchronous after stream sync)
    cuda_synchronize_stream(stream, gpu_index);
    cuda_memcpy_async_to_cpu(&result, d_sum, sizeof(G1Point), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    
    // Free device memory (asynchronous with stream)
    cuda_drop_async(d_points, stream, gpu_index);
    cuda_drop_async(d_scalars, stream, gpu_index);
    cuda_drop_async(d_results, stream, gpu_index);
    cuda_drop_async(d_sum, stream, gpu_index);
}

// G1 MSM with multi-limb scalars
// points and scalars are host pointers
void g1_msm(cudaStream_t stream, uint32_t gpu_index, G1Point& result, const G1Point* points, const uint64_t* scalars, int scalar_limbs, int n) {
    if (n == 0) {
        g1_point_at_infinity(result);
        return;
    }
    
    PANIC_IF_FALSE(n > 0, "g1_msm: invalid size n=%d", n);
    PANIC_IF_FALSE(points != nullptr && scalars != nullptr, "g1_msm: null pointer argument");
    
    cuda_set_device(gpu_index);
    
    // Allocate device memory (asynchronous with stream)
    G1Point* d_points = (G1Point*)cuda_malloc_async(n * sizeof(G1Point), stream, gpu_index);
    uint64_t* d_scalars = (uint64_t*)cuda_malloc_async(n * scalar_limbs * sizeof(uint64_t), stream, gpu_index);
    G1Point* d_results = (G1Point*)cuda_malloc_async(n * sizeof(G1Point), stream, gpu_index);
    G1Point* d_sum = (G1Point*)cuda_malloc_async(sizeof(G1Point), stream, gpu_index);
    
    // Copy to device (asynchronous with stream)
    cuda_memcpy_async_to_gpu(d_points, points, n * sizeof(G1Point), stream, gpu_index);
    cuda_memcpy_async_to_gpu(d_scalars, scalars, n * scalar_limbs * sizeof(uint64_t), stream, gpu_index);
    
    // Launch kernel to compute scalar[i] * points[i] for each i
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    kernel_g1_scalar_mul_array<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_results, d_points, d_scalars, scalar_limbs, n);
    check_cuda_error(cudaGetLastError());
    
    // Reduce sum on device
    kernel_g1_reduce_sum<<<1, 1, 0, stream>>>(d_sum, d_results, n);
    check_cuda_error(cudaGetLastError());
    
    // Copy result back (synchronous after stream sync)
    cuda_synchronize_stream(stream, gpu_index);
    cuda_memcpy_async_to_cpu(&result, d_sum, sizeof(G1Point), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    
    // Free device memory (asynchronous with stream)
    cuda_drop_async(d_points, stream, gpu_index);
    cuda_drop_async(d_scalars, stream, gpu_index);
    cuda_drop_async(d_results, stream, gpu_index);
    cuda_drop_async(d_sum, stream, gpu_index);
}

// G2 MSM with 64-bit scalars
// points and scalars are host pointers
void g2_msm_u64(cudaStream_t stream, uint32_t gpu_index, G2Point& result, const G2Point* points, const uint64_t* scalars, int n) {
    if (n == 0) {
        g2_point_at_infinity(result);
        return;
    }
    
    PANIC_IF_FALSE(n > 0, "g2_msm_u64: invalid size n=%d", n);
    PANIC_IF_FALSE(points != nullptr && scalars != nullptr, "g2_msm_u64: null pointer argument");
    
    cuda_set_device(gpu_index);
    
    // Allocate device memory (asynchronous with stream)
    G2Point* d_points = (G2Point*)cuda_malloc_async(n * sizeof(G2Point), stream, gpu_index);
    uint64_t* d_scalars = (uint64_t*)cuda_malloc_async(n * sizeof(uint64_t), stream, gpu_index);
    G2Point* d_results = (G2Point*)cuda_malloc_async(n * sizeof(G2Point), stream, gpu_index);
    G2Point* d_sum = (G2Point*)cuda_malloc_async(sizeof(G2Point), stream, gpu_index);
    
    // Copy to device (asynchronous with stream)
    cuda_memcpy_async_to_gpu(d_points, points, n * sizeof(G2Point), stream, gpu_index);
    cuda_memcpy_async_to_gpu(d_scalars, scalars, n * sizeof(uint64_t), stream, gpu_index);
    
    // Launch kernel to compute scalar[i] * points[i] for each i
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    kernel_g2_scalar_mul_u64_array<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_results, d_points, d_scalars, n);
    check_cuda_error(cudaGetLastError());
    
    // Reduce sum on device
    kernel_g2_reduce_sum<<<1, 1, 0, stream>>>(d_sum, d_results, n);
    check_cuda_error(cudaGetLastError());
    
    // Copy result back (synchronous after stream sync)
    cuda_synchronize_stream(stream, gpu_index);
    cuda_memcpy_async_to_cpu(&result, d_sum, sizeof(G2Point), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    
    // Free device memory (asynchronous with stream)
    cuda_drop_async(d_points, stream, gpu_index);
    cuda_drop_async(d_scalars, stream, gpu_index);
    cuda_drop_async(d_results, stream, gpu_index);
    cuda_drop_async(d_sum, stream, gpu_index);
}

// G2 MSM with multi-limb scalars
// points and scalars are host pointers
void g2_msm(cudaStream_t stream, uint32_t gpu_index, G2Point& result, const G2Point* points, const uint64_t* scalars, int scalar_limbs, int n) {
    if (n == 0) {
        g2_point_at_infinity(result);
        return;
    }
    
    PANIC_IF_FALSE(n > 0, "g2_msm: invalid size n=%d", n);
    PANIC_IF_FALSE(points != nullptr && scalars != nullptr, "g2_msm: null pointer argument");
    
    cuda_set_device(gpu_index);
    
    // Allocate device memory (asynchronous with stream)
    G2Point* d_points = (G2Point*)cuda_malloc_async(n * sizeof(G2Point), stream, gpu_index);
    uint64_t* d_scalars = (uint64_t*)cuda_malloc_async(n * scalar_limbs * sizeof(uint64_t), stream, gpu_index);
    G2Point* d_results = (G2Point*)cuda_malloc_async(n * sizeof(G2Point), stream, gpu_index);
    G2Point* d_sum = (G2Point*)cuda_malloc_async(sizeof(G2Point), stream, gpu_index);
    
    // Copy to device (asynchronous with stream)
    cuda_memcpy_async_to_gpu(d_points, points, n * sizeof(G2Point), stream, gpu_index);
    cuda_memcpy_async_to_gpu(d_scalars, scalars, n * scalar_limbs * sizeof(uint64_t), stream, gpu_index);
    
    // Launch kernel to compute scalar[i] * points[i] for each i
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    kernel_g2_scalar_mul_array<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_results, d_points, d_scalars, scalar_limbs, n);
    check_cuda_error(cudaGetLastError());
    
    // Reduce sum on device
    kernel_g2_reduce_sum<<<1, 1, 0, stream>>>(d_sum, d_results, n);
    check_cuda_error(cudaGetLastError());
    
    // Copy result back (synchronous after stream sync)
    cuda_synchronize_stream(stream, gpu_index);
    cuda_memcpy_async_to_cpu(&result, d_sum, sizeof(G2Point), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    
    // Free device memory (asynchronous with stream)
    cuda_drop_async(d_points, stream, gpu_index);
    cuda_drop_async(d_scalars, stream, gpu_index);
    cuda_drop_async(d_results, stream, gpu_index);
    cuda_drop_async(d_sum, stream, gpu_index);
}

// ============================================================================
// Kernels for async/sync API (work on device pointers)
// ============================================================================

// G1 kernels
__global__ void kernel_g1_add(G1Point* result, const G1Point* p1, const G1Point* p2) {
    g1_add(*result, *p1, *p2);
}

__global__ void kernel_g1_double(G1Point* result, const G1Point* p) {
    g1_double(*result, *p);
}

__global__ void kernel_g1_neg(G1Point* result, const G1Point* p) {
    g1_neg(*result, *p);
}

__global__ void kernel_g1_point_at_infinity(G1Point* result) {
    g1_point_at_infinity(*result);
}

__global__ void kernel_g1_scalar_mul_u64(G1Point* result, const G1Point* point, uint64_t scalar) {
    g1_scalar_mul_u64(*result, *point, scalar);
}

__global__ void kernel_g1_scalar_mul(G1Point* result, const G1Point* point, const uint64_t* scalar, int scalar_limbs) {
    g1_scalar_mul(*result, *point, scalar, scalar_limbs);
}

// G2 kernels
__global__ void kernel_g2_add(G2Point* result, const G2Point* p1, const G2Point* p2) {
    g2_add(*result, *p1, *p2);
}

__global__ void kernel_g2_double(G2Point* result, const G2Point* p) {
    g2_double(*result, *p);
}

__global__ void kernel_g2_neg(G2Point* result, const G2Point* p) {
    g2_neg(*result, *p);
}

__global__ void kernel_g2_point_at_infinity(G2Point* result) {
    g2_point_at_infinity(*result);
}

__global__ void kernel_g2_scalar_mul_u64(G2Point* result, const G2Point* point, uint64_t scalar) {
    g2_scalar_mul_u64(*result, *point, scalar);
}

__global__ void kernel_g2_scalar_mul(G2Point* result, const G2Point* point, const uint64_t* scalar, int scalar_limbs) {
    g2_scalar_mul(*result, *point, scalar, scalar_limbs);
}

// ============================================================================
// Async/Sync API implementations for G1
// ============================================================================

void g1_add_async(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_p1, const G1Point* d_p2) {
    PANIC_IF_FALSE(d_result != nullptr && d_p1 != nullptr && d_p2 != nullptr, "g1_add_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_g1_add<<<1, 1, 0, stream>>>(d_result, d_p1, d_p2);
    check_cuda_error(cudaGetLastError());
}

void g1_add(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_p1, const G1Point* d_p2) {
    g1_add_async(stream, gpu_index, d_result, d_p1, d_p2);
    cuda_synchronize_stream(stream, gpu_index);
}

void g1_double_async(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_p) {
    PANIC_IF_FALSE(d_result != nullptr && d_p != nullptr, "g1_double_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_g1_double<<<1, 1, 0, stream>>>(d_result, d_p);
    check_cuda_error(cudaGetLastError());
}

void g1_double(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_p) {
    g1_double_async(stream, gpu_index, d_result, d_p);
    cuda_synchronize_stream(stream, gpu_index);
}

void g1_neg_async(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_p) {
    PANIC_IF_FALSE(d_result != nullptr && d_p != nullptr, "g1_neg_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_g1_neg<<<1, 1, 0, stream>>>(d_result, d_p);
    check_cuda_error(cudaGetLastError());
}

void g1_neg(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_p) {
    g1_neg_async(stream, gpu_index, d_result, d_p);
    cuda_synchronize_stream(stream, gpu_index);
}

void g1_point_at_infinity_async(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result) {
    PANIC_IF_FALSE(d_result != nullptr, "g1_point_at_infinity_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_g1_point_at_infinity<<<1, 1, 0, stream>>>(d_result);
    check_cuda_error(cudaGetLastError());
}

void g1_point_at_infinity(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result) {
    g1_point_at_infinity_async(stream, gpu_index, d_result);
    cuda_synchronize_stream(stream, gpu_index);
}

void g1_scalar_mul_u64_async(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_point, uint64_t scalar) {
    PANIC_IF_FALSE(d_result != nullptr && d_point != nullptr, "g1_scalar_mul_u64_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_g1_scalar_mul_u64<<<1, 1, 0, stream>>>(d_result, d_point, scalar);
    check_cuda_error(cudaGetLastError());
}

void g1_scalar_mul_u64(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_point, uint64_t scalar) {
    g1_scalar_mul_u64_async(stream, gpu_index, d_result, d_point, scalar);
    cuda_synchronize_stream(stream, gpu_index);
}

void g1_scalar_mul_async(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_point, const uint64_t* d_scalar, int scalar_limbs) {
    PANIC_IF_FALSE(d_result != nullptr && d_point != nullptr && d_scalar != nullptr, "g1_scalar_mul_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_g1_scalar_mul<<<1, 1, 0, stream>>>(d_result, d_point, d_scalar, scalar_limbs);
    check_cuda_error(cudaGetLastError());
}

void g1_scalar_mul(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_point, const uint64_t* d_scalar, int scalar_limbs) {
    g1_scalar_mul_async(stream, gpu_index, d_result, d_point, d_scalar, scalar_limbs);
    cuda_synchronize_stream(stream, gpu_index);
}

// ============================================================================
// Async/Sync API implementations for G2
// ============================================================================

void g2_add_async(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_p1, const G2Point* d_p2) {
    PANIC_IF_FALSE(d_result != nullptr && d_p1 != nullptr && d_p2 != nullptr, "g2_add_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_g2_add<<<1, 1, 0, stream>>>(d_result, d_p1, d_p2);
    check_cuda_error(cudaGetLastError());
}

void g2_add(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_p1, const G2Point* d_p2) {
    g2_add_async(stream, gpu_index, d_result, d_p1, d_p2);
    cuda_synchronize_stream(stream, gpu_index);
}

void g2_double_async(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_p) {
    PANIC_IF_FALSE(d_result != nullptr && d_p != nullptr, "g2_double_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_g2_double<<<1, 1, 0, stream>>>(d_result, d_p);
    check_cuda_error(cudaGetLastError());
}

void g2_double(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_p) {
    g2_double_async(stream, gpu_index, d_result, d_p);
    cuda_synchronize_stream(stream, gpu_index);
}

void g2_neg_async(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_p) {
    PANIC_IF_FALSE(d_result != nullptr && d_p != nullptr, "g2_neg_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_g2_neg<<<1, 1, 0, stream>>>(d_result, d_p);
    check_cuda_error(cudaGetLastError());
}

void g2_neg(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_p) {
    g2_neg_async(stream, gpu_index, d_result, d_p);
    cuda_synchronize_stream(stream, gpu_index);
}

void g2_point_at_infinity_async(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result) {
    PANIC_IF_FALSE(d_result != nullptr, "g2_point_at_infinity_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_g2_point_at_infinity<<<1, 1, 0, stream>>>(d_result);
    check_cuda_error(cudaGetLastError());
}

void g2_point_at_infinity(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result) {
    g2_point_at_infinity_async(stream, gpu_index, d_result);
    cuda_synchronize_stream(stream, gpu_index);
}

void g2_scalar_mul_u64_async(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_point, uint64_t scalar) {
    PANIC_IF_FALSE(d_result != nullptr && d_point != nullptr, "g2_scalar_mul_u64_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_g2_scalar_mul_u64<<<1, 1, 0, stream>>>(d_result, d_point, scalar);
    check_cuda_error(cudaGetLastError());
}

void g2_scalar_mul_u64(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_point, uint64_t scalar) {
    g2_scalar_mul_u64_async(stream, gpu_index, d_result, d_point, scalar);
    cuda_synchronize_stream(stream, gpu_index);
}

void g2_scalar_mul_async(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_point, const uint64_t* d_scalar, int scalar_limbs) {
    PANIC_IF_FALSE(d_result != nullptr && d_point != nullptr && d_scalar != nullptr, "g2_scalar_mul_async: null pointer argument");
    cuda_set_device(gpu_index);
    kernel_g2_scalar_mul<<<1, 1, 0, stream>>>(d_result, d_point, d_scalar, scalar_limbs);
    check_cuda_error(cudaGetLastError());
}

void g2_scalar_mul(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_point, const uint64_t* d_scalar, int scalar_limbs) {
    g2_scalar_mul_async(stream, gpu_index, d_result, d_point, d_scalar, scalar_limbs);
    cuda_synchronize_stream(stream, gpu_index);
}

// ============================================================================
// Refactored MSM API (device pointers only, no allocations/copies/frees)
// ============================================================================

// G1 MSM with 64-bit scalars - async version
void g1_msm_u64_async(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_points, const uint64_t* d_scalars, G1Point* d_scratch, int n) {
    if (n == 0) {
        g1_point_at_infinity_async(stream, gpu_index, d_result);
        return;
    }
    
    PANIC_IF_FALSE(n > 0, "g1_msm_u64_async: invalid size n=%d", n);
    PANIC_IF_FALSE(d_result != nullptr && d_points != nullptr && d_scalars != nullptr && d_scratch != nullptr, "g1_msm_u64_async: null pointer argument");
    
    cuda_set_device(gpu_index);
    
    // Launch kernel to compute scalar[i] * points[i] for each i
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    kernel_g1_scalar_mul_u64_array<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_scratch, d_points, d_scalars, n);
    check_cuda_error(cudaGetLastError());
    
    // Reduce sum directly into result (uses async g1_add internally)
    kernel_g1_reduce_sum<<<1, 1, 0, stream>>>(d_result, d_scratch, n);
    check_cuda_error(cudaGetLastError());
}

void g1_msm_u64(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_points, const uint64_t* d_scalars, G1Point* d_scratch, int n) {
    g1_msm_u64_async(stream, gpu_index, d_result, d_points, d_scalars, d_scratch, n);
    cuda_synchronize_stream(stream, gpu_index);
}

// G1 MSM with multi-limb scalars - async version
void g1_msm_async(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_points, const uint64_t* d_scalars, int scalar_limbs, G1Point* d_scratch, int n) {
    if (n == 0) {
        g1_point_at_infinity_async(stream, gpu_index, d_result);
        return;
    }
    
    PANIC_IF_FALSE(n > 0, "g1_msm_async: invalid size n=%d", n);
    PANIC_IF_FALSE(d_result != nullptr && d_points != nullptr && d_scalars != nullptr && d_scratch != nullptr, "g1_msm_async: null pointer argument");
    
    cuda_set_device(gpu_index);
    
    // Launch kernel to compute scalar[i] * points[i] for each i
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    kernel_g1_scalar_mul_array<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_scratch, d_points, d_scalars, scalar_limbs, n);
    check_cuda_error(cudaGetLastError());
    
    // Reduce sum directly into result
    kernel_g1_reduce_sum<<<1, 1, 0, stream>>>(d_result, d_scratch, n);
    check_cuda_error(cudaGetLastError());
}

void g1_msm(cudaStream_t stream, uint32_t gpu_index, G1Point* d_result, const G1Point* d_points, const uint64_t* d_scalars, int scalar_limbs, G1Point* d_scratch, int n) {
    g1_msm_async(stream, gpu_index, d_result, d_points, d_scalars, scalar_limbs, d_scratch, n);
    cuda_synchronize_stream(stream, gpu_index);
}

// G2 MSM with 64-bit scalars - async version
void g2_msm_u64_async(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_points, const uint64_t* d_scalars, G2Point* d_scratch, int n) {
    if (n == 0) {
        g2_point_at_infinity_async(stream, gpu_index, d_result);
        return;
    }
    
    PANIC_IF_FALSE(n > 0, "g2_msm_u64_async: invalid size n=%d", n);
    PANIC_IF_FALSE(d_result != nullptr && d_points != nullptr && d_scalars != nullptr && d_scratch != nullptr, "g2_msm_u64_async: null pointer argument");
    
    cuda_set_device(gpu_index);
    
    // Launch kernel to compute scalar[i] * points[i] for each i
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    kernel_g2_scalar_mul_u64_array<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_scratch, d_points, d_scalars, n);
    check_cuda_error(cudaGetLastError());
    
    // Reduce sum directly into result
    kernel_g2_reduce_sum<<<1, 1, 0, stream>>>(d_result, d_scratch, n);
    check_cuda_error(cudaGetLastError());
}

void g2_msm_u64(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_points, const uint64_t* d_scalars, G2Point* d_scratch, int n) {
    g2_msm_u64_async(stream, gpu_index, d_result, d_points, d_scalars, d_scratch, n);
    cuda_synchronize_stream(stream, gpu_index);
}

// G2 MSM with multi-limb scalars - async version
void g2_msm_async(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_points, const uint64_t* d_scalars, int scalar_limbs, G2Point* d_scratch, int n) {
    if (n == 0) {
        g2_point_at_infinity_async(stream, gpu_index, d_result);
        return;
    }
    
    PANIC_IF_FALSE(n > 0, "g2_msm_async: invalid size n=%d", n);
    PANIC_IF_FALSE(d_result != nullptr && d_points != nullptr && d_scalars != nullptr && d_scratch != nullptr, "g2_msm_async: null pointer argument");
    
    cuda_set_device(gpu_index);
    
    // Launch kernel to compute scalar[i] * points[i] for each i
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    kernel_g2_scalar_mul_array<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_scratch, d_points, d_scalars, scalar_limbs, n);
    check_cuda_error(cudaGetLastError());
    
    // Reduce sum directly into result
    kernel_g2_reduce_sum<<<1, 1, 0, stream>>>(d_result, d_scratch, n);
    check_cuda_error(cudaGetLastError());
}

void g2_msm(cudaStream_t stream, uint32_t gpu_index, G2Point* d_result, const G2Point* d_points, const uint64_t* d_scalars, int scalar_limbs, G2Point* d_scratch, int n) {
    g2_msm_async(stream, gpu_index, d_result, d_points, d_scalars, scalar_limbs, d_scratch, n);
    cuda_synchronize_stream(stream, gpu_index);
}

