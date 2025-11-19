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

// Optimized point accumulation for MSM: optimized version that handles common cases efficiently
// Skips infinity checks (assumes already checked) but handles equality/negation cases
template<typename PointType>
__host__ __device__ void point_accumulate_fast(PointType& result, const PointType& p1, const PointType& p2) {
    using Traits = PointTraits<PointType>;
    using FieldType = typename Traits::FieldType;
    
    // Fast check: if x coordinates are equal, handle specially
    int x_cmp = Traits::field_cmp(p1.x, p2.x);
    if (x_cmp == 0) {
        // Same x coordinate - check if same point (doubling) or opposite (infinity)
        int y_cmp = Traits::field_cmp(p1.y, p2.y);
        if (y_cmp == 0) {
            // Same point - use doubling
            point_double(result, p1);
            return;
        } else {
            // Check if opposite (p1.y == -p2.y)
            FieldType neg_y2;
            Traits::field_neg(neg_y2, p2.y);
            if (Traits::field_cmp(p1.y, neg_y2) == 0) {
                Traits::point_at_infinity(result);
                return;
            }
        }
    }
    
    // Standard addition: lambda = (y2 - y1) / (x2 - x1)
    FieldType dx, dy, lambda, lambda_squared, x_result;
    Traits::field_sub(dx, p2.x, p1.x);
    Traits::field_sub(dy, p2.y, p1.y);
    Traits::field_inv(lambda, dx);  // 1 / (x2 - x1) - this is expensive but necessary
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
// Projective Point Operations
// ============================================================================

// Convert affine point to projective: (x, y) -> (X, Y, Z) where X=x, Y=y, Z=1
// Template version removed - using explicit specializations only

// Specialization for G1
__host__ __device__ void affine_to_projective(G1ProjectivePoint& proj, const G1Point& affine) {
    if (g1_is_infinity(affine)) {
        fp_zero(proj.X);
        fp_zero(proj.Y);
        fp_zero(proj.Z);
    } else {
        // Affine coordinates are already in Montgomery form (converted before MSM)
        fp_copy(proj.X, affine.x);
        fp_copy(proj.Y, affine.y);
        fp_one_montgomery(proj.Z);  // Z = 1 in Montgomery form
    }
}

// Specialization for G2
__host__ __device__ void affine_to_projective(G2ProjectivePoint& proj, const G2Point& affine) {
    if (g2_is_infinity(affine)) {
        fp2_zero(proj.X);
        fp2_zero(proj.Y);
        fp2_zero(proj.Z);
    } else {
        // Affine coordinates are already in Montgomery form (converted before MSM)
        fp2_copy(proj.X, affine.x);
        fp2_copy(proj.Y, affine.y);
        // Z = 1 in Montgomery form for Fp2 (1 + 0*i)
        Fp one;
        fp_one_montgomery(one);
        fp_copy(proj.Z.c0, one);
        fp_zero(proj.Z.c1);
    }
}

// Check if projective point is at infinity (Z == 0)
__host__ __device__ bool g1_projective_is_infinity(const G1ProjectivePoint& p) {
    return fp_is_zero(p.Z);
}

__host__ __device__ bool g2_projective_is_infinity(const G2ProjectivePoint& p) {
    return fp2_is_zero(p.Z);
}

// Projective point addition: result = p1 + p2 (no inversions!) - G1 specialization
__host__ __device__ void projective_point_add(G1ProjectivePoint& result, const G1ProjectivePoint& p1, const G1ProjectivePoint& p2) {
    // Handle infinity cases
    if (fp_is_zero(p1.Z)) {
        result = p2;
        return;
    }
    if (fp_is_zero(p2.Z)) {
        result = p1;
        return;
    }
    
    // G1 projective addition using complete formulas
    Fp Y1Z2, X1Z2, Z1Z2, u, uu, v, vv, vvv, R, A;
        
        // Y1Z2 = Y1 * Z2
        fp_mont_mul(Y1Z2, p1.Y, p2.Z);
        // X1Z2 = X1 * Z2
        fp_mont_mul(X1Z2, p1.X, p2.Z);
        // Z1Z2 = Z1 * Z2
        fp_mont_mul(Z1Z2, p1.Z, p2.Z);
        
        // u = Y2 * Z1 - Y1 * Z2 = Y2*Z1 - Y1Z2
        Fp Y2Z1;
        fp_mont_mul(Y2Z1, p2.Y, p1.Z);
        fp_sub(u, Y2Z1, Y1Z2);
        
        // uu = u^2
        fp_mont_mul(uu, u, u);
        
        // v = X2 * Z1 - X1 * Z2 = X2*Z1 - X1Z2
        Fp X2Z1;
        fp_mont_mul(X2Z1, p2.X, p1.Z);
        fp_sub(v, X2Z1, X1Z2);
        
        // Check if this is actually a doubling case (p1 == p2)
        // When u == 0 and v == 0, the points are equal
        if (fp_is_zero(u) && fp_is_zero(v)) {
            projective_point_double(result, p1);
            return;
        }
        
        // vv = v^2
        fp_mont_mul(vv, v, v);
        // vvv = v * vv
        fp_mont_mul(vvv, v, vv);
        
        // R = vv * X1Z2
        fp_mont_mul(R, vv, X1Z2);
        
        // A = uu * Z1Z2 - vvv - 2*R
        Fp temp1, temp2, two_R;
        fp_mont_mul(temp1, uu, Z1Z2);
        fp_sub(temp2, temp1, vvv);
        // Compute 2*R in Montgomery form
        Fp two_val;
        fp_one(two_val);
        two_val.limb[0] = 2;
        fp_to_montgomery(two_val, two_val);
        fp_mont_mul(two_R, two_val, R);
        fp_sub(A, temp2, two_R);
        
        // X3 = v * A
        fp_mont_mul(result.X, v, A);
        
        // Y3 = u * (R - A) - vvv * Y1Z2
        Fp R_minus_A;
        fp_sub(R_minus_A, R, A);
        Fp uR_minus_A;
        fp_mont_mul(uR_minus_A, u, R_minus_A);
        Fp vvvY1Z2;
        fp_mont_mul(vvvY1Z2, vvv, Y1Z2);
        fp_sub(result.Y, uR_minus_A, vvvY1Z2);
        
        // Z3 = vvv * Z1Z2
        fp_mont_mul(result.Z, vvv, Z1Z2);
}

// Projective point addition: result = p1 + p2 (no inversions!) - G2 specialization
__host__ __device__ void projective_point_add(G2ProjectivePoint& result, const G2ProjectivePoint& p1, const G2ProjectivePoint& p2) {
    // Handle infinity cases
    if (fp2_is_zero(p1.Z)) {
        result = p2;
        return;
    }
    if (fp2_is_zero(p2.Z)) {
        result = p1;
        return;
    }
    
    // G2 projective addition (same algorithm with Fp2)
    Fp2 Y1Z2, X1Z2, Z1Z2, u, uu, v, vv, vvv, R, A;
    
    fp2_mont_mul(Y1Z2, p1.Y, p2.Z);
    fp2_mont_mul(X1Z2, p1.X, p2.Z);
    fp2_mont_mul(Z1Z2, p1.Z, p2.Z);
    
    Fp2 Y2Z1;
    fp2_mont_mul(Y2Z1, p2.Y, p1.Z);
    fp2_sub(u, Y2Z1, Y1Z2);
    
    fp2_mont_mul(uu, u, u);
    
    Fp2 X2Z1;
    fp2_mont_mul(X2Z1, p2.X, p1.Z);
    fp2_sub(v, X2Z1, X1Z2);
    
    // Check if this is actually a doubling case (p1 == p2)
    // When u == 0 and v == 0, the points are equal
    if (fp2_is_zero(u) && fp2_is_zero(v)) {
        projective_point_double(result, p1);
        return;
    }
    
    fp2_mont_mul(vv, v, v);
    fp2_mont_mul(vvv, v, vv);
    
    fp2_mont_mul(R, vv, X1Z2);
    
    // A = uu * Z1Z2 - vvv - 2*R
    Fp2 temp1, temp2, two_R;
    fp2_mont_mul(temp1, uu, Z1Z2);
    fp2_sub(temp2, temp1, vvv);
    // Compute 2*R in Montgomery form
    Fp2 two_val;
    Fp two_fp;
    fp_one(two_fp);
    two_fp.limb[0] = 2;
    fp_to_montgomery(two_fp, two_fp);
    fp2_zero(two_val);
    fp_copy(two_val.c0, two_fp);
    fp2_mont_mul(two_R, two_val, R);
    fp2_sub(A, temp2, two_R);
    
    fp2_mont_mul(result.X, v, A);
    
    Fp2 R_minus_A;
    fp2_sub(R_minus_A, R, A);
    Fp2 uR_minus_A;
    fp2_mont_mul(uR_minus_A, u, R_minus_A);
    Fp2 vvvY1Z2;
    fp2_mont_mul(vvvY1Z2, vvv, Y1Z2);
    fp2_sub(result.Y, uR_minus_A, vvvY1Z2);
    
    fp2_mont_mul(result.Z, vvv, Z1Z2);
}

// Projective point doubling: result = 2 * p (no inversions!) - G1 specialization
__host__ __device__ void projective_point_double(G1ProjectivePoint& result, const G1ProjectivePoint& p) {
    // Handle infinity
    if (fp_is_zero(p.Z)) {
        result = p;
        return;
    }
    
    // G1 projective doubling using hyperelliptic.org formula
    // For curves y^2 = x^3 + a_4*x + b with a_4 = 0
    // Formula: http://hyperelliptic.org/EFD/g1p/auto-shortw-projective.html
    
    // Precompute constants in Montgomery form
    Fp two_mont, three_mont, four_mont, eight_mont;
    Fp one;
    fp_one(one);
    one.limb[0] = 2;
    fp_to_montgomery(two_mont, one);
    one.limb[0] = 3;
    fp_to_montgomery(three_mont, one);
    one.limb[0] = 4;
    fp_to_montgomery(four_mont, one);
    one.limb[0] = 8;
    fp_to_montgomery(eight_mont, one);
    
    // A = 3 * X^2 (for a_4 = 0, otherwise A = a_4*Z^2 + 3*X^2)
    Fp X_sq, A;
    fp_mont_mul(X_sq, p.X, p.X);
    fp_mont_mul(A, three_mont, X_sq);
    
    // B = Y * Z
    Fp B;
    fp_mont_mul(B, p.Y, p.Z);
    
    // C = X * Y * B
    Fp XY, C;
    fp_mont_mul(XY, p.X, p.Y);
    fp_mont_mul(C, XY, B);
    
    // D = A^2 - 8*C
    Fp A_sq, eight_C, D;
    fp_mont_mul(A_sq, A, A);
    fp_mont_mul(eight_C, eight_mont, C);
    fp_sub(D, A_sq, eight_C);
    
    // X₃ = 2 * B * D
    Fp BD;
    fp_mont_mul(BD, B, D);
    fp_mont_mul(result.X, two_mont, BD);
    
    // Y₃ = A * (4*C - D) - 8 * Y^2 * B^2
    Fp four_C, four_C_minus_D, A_times_diff;
    fp_mont_mul(four_C, four_mont, C);
    fp_sub(four_C_minus_D, four_C, D);
    fp_mont_mul(A_times_diff, A, four_C_minus_D);
    
    Fp Y_sq, B_sq, Y_sq_B_sq, eight_Y_sq_B_sq;
    fp_mont_mul(Y_sq, p.Y, p.Y);
    fp_mont_mul(B_sq, B, B);
    fp_mont_mul(Y_sq_B_sq, Y_sq, B_sq);
    fp_mont_mul(eight_Y_sq_B_sq, eight_mont, Y_sq_B_sq);
    fp_sub(result.Y, A_times_diff, eight_Y_sq_B_sq);
    
    // Z₃ = 8 * B^3
    Fp B_cu;
    fp_mont_mul(B_cu, B_sq, B);  // B^3
    fp_mont_mul(result.Z, eight_mont, B_cu);
}

// Projective point doubling: result = 2 * p (no inversions!) - G2 specialization
__host__ __device__ void projective_point_double(G2ProjectivePoint& result, const G2ProjectivePoint& p) {
    // Handle infinity
    if (fp2_is_zero(p.Z)) {
        result = p;
        return;
    }
    
    // G2 projective doubling using hyperelliptic.org formula (same as G1 but with Fp2)
    
    // Precompute constants in Montgomery form (for Fp2, just set c0)
    Fp2 two_mont, three_mont, four_mont, eight_mont;
    Fp one;
    fp_one(one);
    one.limb[0] = 2;
    fp_to_montgomery(one, one);
    fp2_zero(two_mont);
    fp_copy(two_mont.c0, one);
    
    fp_one(one);
    one.limb[0] = 3;
    fp_to_montgomery(one, one);
    fp2_zero(three_mont);
    fp_copy(three_mont.c0, one);
    
    fp_one(one);
    one.limb[0] = 4;
    fp_to_montgomery(one, one);
    fp2_zero(four_mont);
    fp_copy(four_mont.c0, one);
    
    fp_one(one);
    one.limb[0] = 8;
    fp_to_montgomery(one, one);
    fp2_zero(eight_mont);
    fp_copy(eight_mont.c0, one);
    
    // A = 3 * X^2
    Fp2 X_sq, A;
    fp2_mont_mul(X_sq, p.X, p.X);
    fp2_mont_mul(A, three_mont, X_sq);
    
    // B = Y * Z
    Fp2 B;
    fp2_mont_mul(B, p.Y, p.Z);
    
    // C = X * Y * B
    Fp2 XY, C;
    fp2_mont_mul(XY, p.X, p.Y);
    fp2_mont_mul(C, XY, B);
    
    // D = A^2 - 8*C
    Fp2 A_sq, eight_C, D;
    fp2_mont_mul(A_sq, A, A);
    fp2_mont_mul(eight_C, eight_mont, C);
    fp2_sub(D, A_sq, eight_C);
    
    // X₃ = 2 * B * D
    Fp2 BD;
    fp2_mont_mul(BD, B, D);
    fp2_mont_mul(result.X, two_mont, BD);
    
    // Y₃ = A * (4*C - D) - 8 * Y^2 * B^2
    Fp2 four_C, four_C_minus_D, A_times_diff;
    fp2_mont_mul(four_C, four_mont, C);
    fp2_sub(four_C_minus_D, four_C, D);
    fp2_mont_mul(A_times_diff, A, four_C_minus_D);
    
    Fp2 Y_sq, B_sq, Y_sq_B_sq, eight_Y_sq_B_sq;
    fp2_mont_mul(Y_sq, p.Y, p.Y);
    fp2_mont_mul(B_sq, B, B);
    fp2_mont_mul(Y_sq_B_sq, Y_sq, B_sq);
    fp2_mont_mul(eight_Y_sq_B_sq, eight_mont, Y_sq_B_sq);
    fp2_sub(result.Y, A_times_diff, eight_Y_sq_B_sq);
    
    // Z₃ = 8 * B^3
    Fp2 B_cu;
    fp2_mont_mul(B_cu, B_sq, B);  // B^3
    fp2_mont_mul(result.Z, eight_mont, B_cu);
}

// Set projective point to infinity (Z = 0)
__host__ __device__ void g1_projective_point_at_infinity(G1ProjectivePoint& p) {
    fp_zero(p.X);
    fp_zero(p.Y);
    fp_zero(p.Z);
}

__host__ __device__ void g2_projective_point_at_infinity(G2ProjectivePoint& p) {
    fp2_zero(p.X);
    fp2_zero(p.Y);
    fp2_zero(p.Z);
}

// Convert projective point to affine: (X, Y, Z) -> (x, y) where x = X/Z, y = Y/Z
__host__ __device__ void projective_to_affine_g1(G1Point& affine, const G1ProjectivePoint& proj) {
    if (fp_is_zero(proj.Z)) {
        g1_point_at_infinity(affine);
        return;
    }
    // x = X * Z^(-1)
    Fp Z_inv;
    fp_mont_inv(Z_inv, proj.Z);
    fp_mont_mul(affine.x, proj.X, Z_inv);
    fp_mont_mul(affine.y, proj.Y, Z_inv);
    affine.infinity = false;
}

__host__ __device__ void projective_to_affine_g2(G2Point& affine, const G2ProjectivePoint& proj) {
    if (fp2_is_zero(proj.Z)) {
        g2_point_at_infinity(affine);
        return;
    }
    // x = X * Z^(-1)
    Fp2 Z_inv;
    fp2_mont_inv(Z_inv, proj.Z);
    fp2_mont_mul(affine.x, proj.X, Z_inv);
    fp2_mont_mul(affine.y, proj.Y, Z_inv);
    affine.infinity = false;
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
    // Optimize memory access: use coalesced reads from global memory
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx < n) {
        int bucket_idx = extract_window_u64(scalars[point_idx], window_idx, window_size);
        // Coalesced read: threads read consecutive memory locations
        thread_points[threadIdx.x] = points[point_idx];
        thread_buckets[threadIdx.x] = bucket_idx;
    } else {
        Traits::point_at_infinity(thread_points[threadIdx.x]);
        thread_buckets[threadIdx.x] = 0;
    }
    __syncthreads();
    
    // Pre-fetch and cache bucket indices in registers for faster access
    // This reduces shared memory bank conflicts
    
    // Phase 2: Highly optimized two-stage reduction
    // Stage 1: Each warp reduces its points (one accumulator per bucket per warp)
    // Stage 2: Single thread per bucket reduces across warps
    // This eliminates redundant work while maintaining correctness
    
    const int WARP_SIZE = 32;
    const int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Per-warp buckets: only allocate if we have multiple warps
    size_t warp_buckets_offset = MSM_BUCKET_COUNT * sizeof(PointType) + blockDim.x * sizeof(PointType) + blockDim.x * sizeof(int);
    PointType* warp_buckets = num_warps > 1 ? (PointType*)(shared_mem + warp_buckets_offset) : nullptr;
    
    if (num_warps > 1 && warp_buckets != nullptr) {
        // Initialize per-warp buckets
        for (int i = threadIdx.x; i < num_warps * MSM_BUCKET_COUNT; i += blockDim.x) {
            Traits::point_at_infinity(warp_buckets[i]);
        }
    }
    __syncthreads();
    
    // Stage 1: Warp-level reduction - only one thread per bucket per warp accumulates
    int my_bucket = thread_buckets[threadIdx.x];
    const PointType& my_point = thread_points[threadIdx.x];
    bool my_valid = (point_idx < n) && (my_bucket > 0) && (my_bucket < MSM_BUCKET_COUNT) && !Traits::is_infinity(my_point);
    
    if (my_valid) {
        // Find first lane with this bucket in warp
        unsigned int mask = 0;
        for (int i = 0; i < WARP_SIZE; i++) {
            int t = warp_id * WARP_SIZE + i;
            if (t < blockDim.x && thread_buckets[t] == my_bucket) {
                mask |= (1U << i);
            }
        }
        unsigned int first = __ffs(mask) - 1;
        
        if (lane_id == first) {
            // I accumulate all points with this bucket in my warp
            // Optimize: collect all points first, then accumulate in one pass
            // This improves register usage and may help with instruction scheduling
            PointType sum = my_point;
            
            // Pre-compute which threads in warp have this bucket to reduce redundant checks
            // Use a more efficient scan pattern
            int count = 1;  // We already have my_point
            PointType points_to_add[WARP_SIZE];
            points_to_add[0] = my_point;  // Already included
            
            // Collect all points with this bucket (excluding self)
            for (int i = 0; i < WARP_SIZE; i++) {
                int t = warp_id * WARP_SIZE + i;
                if (t < blockDim.x && i != lane_id && thread_buckets[t] == my_bucket) {
                    int global_t = blockIdx.x * blockDim.x + t;
                    if (global_t < n) {
                        const PointType& pt = thread_points[t];
                        if (!Traits::is_infinity(pt)) {
                            points_to_add[count++] = pt;
                        }
                    }
                }
            }
            
            // Now accumulate all collected points
            // This allows better instruction scheduling and register usage
            for (int i = 1; i < count; i++) {
                PointType temp;
                point_accumulate_fast(temp, sum, points_to_add[i]);
                sum = temp;
            }
            
            // Write to warp bucket (or directly to shared if single warp)
            if (num_warps > 1 && warp_buckets != nullptr) {
                int idx = warp_id * MSM_BUCKET_COUNT + my_bucket;
                Traits::field_copy(warp_buckets[idx].x, sum.x);
                Traits::field_copy(warp_buckets[idx].y, sum.y);
                warp_buckets[idx].infinity = sum.infinity;
            } else {
                // Single warp: write directly to shared_buckets
                PointType* dst = &shared_buckets[my_bucket];
                PointType curr = *dst;
                if (Traits::is_infinity(curr)) {
                    Traits::field_copy(dst->x, sum.x);
                    Traits::field_copy(dst->y, sum.y);
                    dst->infinity = sum.infinity;
                } else {
                    // Use fast accumulation - both points are valid
                    PointType temp;
                    point_accumulate_fast(temp, curr, sum);
                    Traits::field_copy(dst->x, temp.x);
                    Traits::field_copy(dst->y, temp.y);
                    dst->infinity = temp.infinity;
                }
            }
        }
    }
    
    __syncthreads();
    
    // Stage 2: Reduce warp buckets to final buckets (only if multiple warps)
    if (num_warps > 1 && threadIdx.x < MSM_BUCKET_COUNT && warp_buckets != nullptr) {
        int bucket_idx = threadIdx.x;
        PointType final_sum;
        Traits::point_at_infinity(final_sum);
        
        for (int w = 0; w < num_warps; w++) {
            int idx = w * MSM_BUCKET_COUNT + bucket_idx;
            if (!Traits::is_infinity(warp_buckets[idx])) {
                if (Traits::is_infinity(final_sum)) {
                    final_sum = warp_buckets[idx];
                } else {
                    // Use fast accumulation - both points are valid
                    PointType temp;
                    point_accumulate_fast(temp, final_sum, warp_buckets[idx]);
                    final_sum = temp;
                }
            }
        }
        
        Traits::field_copy(shared_buckets[bucket_idx].x, final_sum.x);
        Traits::field_copy(shared_buckets[bucket_idx].y, final_sum.y);
        shared_buckets[bucket_idx].infinity = final_sum.infinity;
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

// Pippenger kernel: Accumulate points into buckets using projective coordinates (64-bit scalars) - G1 specialization
// This version uses projective coordinates internally to avoid expensive inversions
// Input: affine points, Output: projective points
__global__ void kernel_accumulate_buckets_u64_projective_g1(
    G1ProjectivePoint* block_buckets,
    const G1Point* points,
    const uint64_t* scalars,
    int n,
    int window_idx,
    int window_size
) {
    // Shared memory layout:
    // - shared_buckets[MSM_BUCKET_COUNT]: per-bucket accumulations (projective)
    // - thread_points[blockDim.x]: projective points processed by each thread
    // - thread_buckets[blockDim.x]: bucket index for each thread's point
    extern __shared__ char shared_mem[];
    G1ProjectivePoint* shared_buckets = (G1ProjectivePoint*)shared_mem;
    G1ProjectivePoint* thread_points = (G1ProjectivePoint*)(shared_mem + MSM_BUCKET_COUNT * sizeof(G1ProjectivePoint));
    int* thread_buckets = (int*)(shared_mem + MSM_BUCKET_COUNT * sizeof(G1ProjectivePoint) + blockDim.x * sizeof(G1ProjectivePoint));
    
    // Initialize shared memory buckets to infinity (Z = 0)
    for (int i = threadIdx.x; i < MSM_BUCKET_COUNT; i += blockDim.x) {
        g1_projective_point_at_infinity(shared_buckets[i]);
    }
    __syncthreads();
    
    // Phase 1: Each thread processes one point, converts to projective, stores in shared memory
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx < n) {
        int bucket_idx = extract_window_u64(scalars[point_idx], window_idx, window_size);
        // Convert affine to projective
        affine_to_projective(thread_points[threadIdx.x], points[point_idx]);
        thread_buckets[threadIdx.x] = bucket_idx;
    } else {
        // Point at infinity in projective form (Z = 0)
        g1_projective_point_at_infinity(thread_points[threadIdx.x]);
        thread_buckets[threadIdx.x] = 0;
    }
    __syncthreads();
    
    // Phase 2: Two-stage reduction using projective point addition (no inversions!)
    const int WARP_SIZE = 32;
    const int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Per-warp buckets: only allocate if we have multiple warps
    size_t warp_buckets_offset = MSM_BUCKET_COUNT * sizeof(G1ProjectivePoint) + blockDim.x * sizeof(G1ProjectivePoint) + blockDim.x * sizeof(int);
    G1ProjectivePoint* warp_buckets = num_warps > 1 ? (G1ProjectivePoint*)(shared_mem + warp_buckets_offset) : nullptr;
    
    if (num_warps > 1 && warp_buckets != nullptr) {
        // Initialize per-warp buckets
        for (int i = threadIdx.x; i < num_warps * MSM_BUCKET_COUNT; i += blockDim.x) {
            g1_projective_point_at_infinity(warp_buckets[i]);
        }
    }
    __syncthreads();
    
    // Stage 1: Warp-level reduction - only one thread per bucket per warp accumulates
    int my_bucket = thread_buckets[threadIdx.x];
    const G1ProjectivePoint& my_point = thread_points[threadIdx.x];
    bool my_valid = (point_idx < n) && (my_bucket > 0) && (my_bucket < MSM_BUCKET_COUNT) && !fp_is_zero(my_point.Z);
    
    if (my_valid) {
        // Find first lane with this bucket in warp
        unsigned int mask = 0;
        for (int i = 0; i < WARP_SIZE; i++) {
            int t = warp_id * WARP_SIZE + i;
            if (t < blockDim.x && thread_buckets[t] == my_bucket) {
                mask |= (1U << i);
            }
        }
        unsigned int first = __ffs(mask) - 1;
        
        if (lane_id == first) {
            // I accumulate all points with this bucket in my warp using projective addition
            G1ProjectivePoint sum = my_point;
            
            // Collect all points with this bucket (excluding self)
            int count = 1;
            G1ProjectivePoint points_to_add[WARP_SIZE];
            points_to_add[0] = my_point;
            
            for (int i = 0; i < WARP_SIZE; i++) {
                int t = warp_id * WARP_SIZE + i;
                if (t < blockDim.x && i != lane_id && thread_buckets[t] == my_bucket) {
                    int global_t = blockIdx.x * blockDim.x + t;
                    if (global_t < n) {
                        const G1ProjectivePoint& pt = thread_points[t];
                        if (!fp_is_zero(pt.Z)) {
                            points_to_add[count++] = pt;
                        }
                    }
                }
            }
            
            // Accumulate all collected points using projective addition (no inversions!)
            for (int i = 1; i < count; i++) {
                G1ProjectivePoint temp;
                projective_point_add(temp, sum, points_to_add[i]);
                sum = temp;
            }
            
            // Write to warp bucket (or directly to shared if single warp)
            if (num_warps > 1 && warp_buckets != nullptr) {
                int idx = warp_id * MSM_BUCKET_COUNT + my_bucket;
                fp_copy(warp_buckets[idx].X, sum.X);
                fp_copy(warp_buckets[idx].Y, sum.Y);
                fp_copy(warp_buckets[idx].Z, sum.Z);
            } else {
                // Single warp: write directly to shared_buckets
                G1ProjectivePoint* dst = &shared_buckets[my_bucket];
                if (fp_is_zero(dst->Z)) {
                    fp_copy(dst->X, sum.X);
                    fp_copy(dst->Y, sum.Y);
                    fp_copy(dst->Z, sum.Z);
                } else {
                    G1ProjectivePoint temp;
                    projective_point_add(temp, *dst, sum);
                    fp_copy(dst->X, temp.X);
                    fp_copy(dst->Y, temp.Y);
                    fp_copy(dst->Z, temp.Z);
                }
            }
        }
    }
    
    __syncthreads();
    
    // Stage 2: Reduce warp buckets to final buckets (only if multiple warps)
    if (num_warps > 1 && threadIdx.x < MSM_BUCKET_COUNT && warp_buckets != nullptr) {
        int bucket_idx = threadIdx.x;
        G1ProjectivePoint final_sum;
        g1_projective_point_at_infinity(final_sum);
        
        for (int w = 0; w < num_warps; w++) {
            int idx = w * MSM_BUCKET_COUNT + bucket_idx;
            if (!fp_is_zero(warp_buckets[idx].Z)) {
                if (fp_is_zero(final_sum.Z)) {
                    fp_copy(final_sum.X, warp_buckets[idx].X);
                    fp_copy(final_sum.Y, warp_buckets[idx].Y);
                    fp_copy(final_sum.Z, warp_buckets[idx].Z);
                } else {
                    G1ProjectivePoint temp;
                    projective_point_add(temp, final_sum, warp_buckets[idx]);
                    final_sum = temp;
                }
            }
        }
        
        fp_copy(shared_buckets[bucket_idx].X, final_sum.X);
        fp_copy(shared_buckets[bucket_idx].Y, final_sum.Y);
        fp_copy(shared_buckets[bucket_idx].Z, final_sum.Z);
    }
    
    __syncthreads();
    
    // Phase 3: Write block's bucket contributions to global memory (projective points)
    if (threadIdx.x < MSM_BUCKET_COUNT) {
        int bucket_idx = threadIdx.x;
        int block_bucket_idx = blockIdx.x * MSM_BUCKET_COUNT + bucket_idx;
        const G1ProjectivePoint& bucket = shared_buckets[bucket_idx];
        fp_copy(block_buckets[block_bucket_idx].X, bucket.X);
        fp_copy(block_buckets[block_bucket_idx].Y, bucket.Y);
        fp_copy(block_buckets[block_bucket_idx].Z, bucket.Z);
    }
}

// G2 specialization
__global__ void kernel_accumulate_buckets_u64_projective_g2(
    G2ProjectivePoint* block_buckets,
    const G2Point* points,
    const uint64_t* scalars,
    int n,
    int window_idx,
    int window_size
) {
    extern __shared__ char shared_mem[];
    G2ProjectivePoint* shared_buckets = (G2ProjectivePoint*)shared_mem;
    G2ProjectivePoint* thread_points = (G2ProjectivePoint*)(shared_mem + MSM_BUCKET_COUNT * sizeof(G2ProjectivePoint));
    int* thread_buckets = (int*)(shared_mem + MSM_BUCKET_COUNT * sizeof(G2ProjectivePoint) + blockDim.x * sizeof(G2ProjectivePoint));
    
    for (int i = threadIdx.x; i < MSM_BUCKET_COUNT; i += blockDim.x) {
        g2_projective_point_at_infinity(shared_buckets[i]);
    }
    __syncthreads();
    
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx < n) {
        int bucket_idx = extract_window_u64(scalars[point_idx], window_idx, window_size);
        affine_to_projective(thread_points[threadIdx.x], points[point_idx]);
        thread_buckets[threadIdx.x] = bucket_idx;
    } else {
        g2_projective_point_at_infinity(thread_points[threadIdx.x]);
        thread_buckets[threadIdx.x] = 0;
    }
    __syncthreads();
    
    const int WARP_SIZE = 32;
    const int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    size_t warp_buckets_offset = MSM_BUCKET_COUNT * sizeof(G2ProjectivePoint) + blockDim.x * sizeof(G2ProjectivePoint) + blockDim.x * sizeof(int);
    G2ProjectivePoint* warp_buckets = num_warps > 1 ? (G2ProjectivePoint*)(shared_mem + warp_buckets_offset) : nullptr;
    
    if (num_warps > 1 && warp_buckets != nullptr) {
        for (int i = threadIdx.x; i < num_warps * MSM_BUCKET_COUNT; i += blockDim.x) {
            g2_projective_point_at_infinity(warp_buckets[i]);
        }
    }
    __syncthreads();
    
    int my_bucket = thread_buckets[threadIdx.x];
    const G2ProjectivePoint& my_point = thread_points[threadIdx.x];
    bool my_valid = (point_idx < n) && (my_bucket > 0) && (my_bucket < MSM_BUCKET_COUNT) && !fp2_is_zero(my_point.Z);
    
    if (my_valid) {
        unsigned int mask = 0;
        for (int i = 0; i < WARP_SIZE; i++) {
            int t = warp_id * WARP_SIZE + i;
            if (t < blockDim.x && thread_buckets[t] == my_bucket) {
                mask |= (1U << i);
            }
        }
        unsigned int first = __ffs(mask) - 1;
        
        if (lane_id == first) {
            G2ProjectivePoint sum = my_point;
            int count = 1;
            G2ProjectivePoint points_to_add[WARP_SIZE];
            points_to_add[0] = my_point;
            
            for (int i = 0; i < WARP_SIZE; i++) {
                int t = warp_id * WARP_SIZE + i;
                if (t < blockDim.x && i != lane_id && thread_buckets[t] == my_bucket) {
                    int global_t = blockIdx.x * blockDim.x + t;
                    if (global_t < n) {
                        const G2ProjectivePoint& pt = thread_points[t];
                        if (!fp2_is_zero(pt.Z)) {
                            points_to_add[count++] = pt;
                        }
                    }
                }
            }
            
            for (int i = 1; i < count; i++) {
                G2ProjectivePoint temp;
                projective_point_add(temp, sum, points_to_add[i]);
                sum = temp;
            }
            
            if (num_warps > 1 && warp_buckets != nullptr) {
                int idx = warp_id * MSM_BUCKET_COUNT + my_bucket;
                fp2_copy(warp_buckets[idx].X, sum.X);
                fp2_copy(warp_buckets[idx].Y, sum.Y);
                fp2_copy(warp_buckets[idx].Z, sum.Z);
            } else {
                G2ProjectivePoint* dst = &shared_buckets[my_bucket];
                if (fp2_is_zero(dst->Z)) {
                    fp2_copy(dst->X, sum.X);
                    fp2_copy(dst->Y, sum.Y);
                    fp2_copy(dst->Z, sum.Z);
                } else {
                    G2ProjectivePoint temp;
                    projective_point_add(temp, *dst, sum);
                    fp2_copy(dst->X, temp.X);
                    fp2_copy(dst->Y, temp.Y);
                    fp2_copy(dst->Z, temp.Z);
                }
            }
        }
    }
    
    __syncthreads();
    
    if (num_warps > 1 && threadIdx.x < MSM_BUCKET_COUNT && warp_buckets != nullptr) {
        int bucket_idx = threadIdx.x;
        G2ProjectivePoint final_sum;
        g2_projective_point_at_infinity(final_sum);
        
        for (int w = 0; w < num_warps; w++) {
            int idx = w * MSM_BUCKET_COUNT + bucket_idx;
            if (!fp2_is_zero(warp_buckets[idx].Z)) {
                if (fp2_is_zero(final_sum.Z)) {
                    fp2_copy(final_sum.X, warp_buckets[idx].X);
                    fp2_copy(final_sum.Y, warp_buckets[idx].Y);
                    fp2_copy(final_sum.Z, warp_buckets[idx].Z);
                } else {
                    G2ProjectivePoint temp;
                    projective_point_add(temp, final_sum, warp_buckets[idx]);
                    final_sum = temp;
                }
            }
        }
        
        fp2_copy(shared_buckets[bucket_idx].X, final_sum.X);
        fp2_copy(shared_buckets[bucket_idx].Y, final_sum.Y);
        fp2_copy(shared_buckets[bucket_idx].Z, final_sum.Z);
    }
    
    __syncthreads();
    
    if (threadIdx.x < MSM_BUCKET_COUNT) {
        int bucket_idx = threadIdx.x;
        int block_bucket_idx = blockIdx.x * MSM_BUCKET_COUNT + bucket_idx;
        const G2ProjectivePoint& bucket = shared_buckets[bucket_idx];
        fp2_copy(block_buckets[block_bucket_idx].X, bucket.X);
        fp2_copy(block_buckets[block_bucket_idx].Y, bucket.Y);
        fp2_copy(block_buckets[block_bucket_idx].Z, bucket.Z);
    }
}

// Legacy template version removed - using specialized kernels instead
// Old template version that used std::is_same has been replaced by:
// - kernel_accumulate_buckets_u64_projective_g1
// - kernel_accumulate_buckets_u64_projective_g2

// Helper function to clear projective buckets
__global__ void kernel_clear_buckets_projective_g1(G1ProjectivePoint* buckets, int num_buckets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_buckets) {
        g1_projective_point_at_infinity(buckets[idx]);
    }
}

__global__ void kernel_clear_buckets_projective_g2(G2ProjectivePoint* buckets, int num_buckets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_buckets) {
        g2_projective_point_at_infinity(buckets[idx]);
    }
}

// Helper function to reduce projective buckets from multiple blocks
__global__ void kernel_reduce_buckets_projective_g1(
    G1ProjectivePoint* final_buckets,
    const G1ProjectivePoint* block_buckets,
    int num_blocks,
    int num_buckets
) {
    int bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_idx == 0 || bucket_idx >= num_buckets) return;
    
    G1ProjectivePoint bucket_sum;
    g1_projective_point_at_infinity(bucket_sum);
    
    for (int block = 0; block < num_blocks; block++) {
        int idx = block * num_buckets + bucket_idx;
        if (!fp_is_zero(block_buckets[idx].Z)) {
            if (fp_is_zero(bucket_sum.Z)) {
                fp_copy(bucket_sum.X, block_buckets[idx].X);
                fp_copy(bucket_sum.Y, block_buckets[idx].Y);
                fp_copy(bucket_sum.Z, block_buckets[idx].Z);
            } else {
                G1ProjectivePoint temp;
                projective_point_add(temp, bucket_sum, block_buckets[idx]);
                bucket_sum = temp;
            }
        }
    }
    
    fp_copy(final_buckets[bucket_idx].X, bucket_sum.X);
    fp_copy(final_buckets[bucket_idx].Y, bucket_sum.Y);
    fp_copy(final_buckets[bucket_idx].Z, bucket_sum.Z);
}

__global__ void kernel_reduce_buckets_projective_g2(
    G2ProjectivePoint* final_buckets,
    const G2ProjectivePoint* block_buckets,
    int num_blocks,
    int num_buckets
) {
    int bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_idx == 0 || bucket_idx >= num_buckets) return;
    
    G2ProjectivePoint bucket_sum;
    g2_projective_point_at_infinity(bucket_sum);
    
    for (int block = 0; block < num_blocks; block++) {
        int idx = block * num_buckets + bucket_idx;
        if (!fp2_is_zero(block_buckets[idx].Z)) {
            if (fp2_is_zero(bucket_sum.Z)) {
                fp2_copy(bucket_sum.X, block_buckets[idx].X);
                fp2_copy(bucket_sum.Y, block_buckets[idx].Y);
                fp2_copy(bucket_sum.Z, block_buckets[idx].Z);
            } else {
                G2ProjectivePoint temp;
                projective_point_add(temp, bucket_sum, block_buckets[idx]);
                bucket_sum = temp;
            }
        }
    }
    
    fp2_copy(final_buckets[bucket_idx].X, bucket_sum.X);
    fp2_copy(final_buckets[bucket_idx].Y, bucket_sum.Y);
    fp2_copy(final_buckets[bucket_idx].Z, bucket_sum.Z);
}

// Old template version removed - using specialized kernels instead

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
// Device version
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

// Host version: Compute k * P using binary method (for small k, 1 <= k <= 15)
template<typename PointType>
__host__ void point_scalar_mul_small_host(PointType& result, const PointType& P, int k) {
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
    
    // Find the MSB of k (using portable method for host)
    int msb = 0;
    int temp_k = k;
    while (temp_k > 1) {
        temp_k >>= 1;
        msb++;
    }
    
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

// CPU version: Combine buckets for a window and accumulate into result (affine version, for multi-limb scalars)
// This is more efficient than the GPU kernel for small bucket counts (16 buckets)
template<typename PointType>
__host__ void combine_buckets_cpu(
    PointType& result,
    const PointType* buckets,
    int num_buckets,
    int window_idx
) {
    (void)window_idx;  // Unused parameter, kept for API consistency
    using Traits = PointTraits<PointType>;
    
    // Compute weighted buckets: i * bucket[i] for i = 1 to num_buckets-1
    PointType weighted_buckets[MSM_BUCKET_COUNT];
    for (int i = 1; i < num_buckets; i++) {
        if (!Traits::is_infinity(buckets[i])) {
            point_scalar_mul_small_host(weighted_buckets[i-1], buckets[i], i);
        } else {
            Traits::point_at_infinity(weighted_buckets[i-1]);
        }
    }
    
    // Reduce all weighted buckets into window_sum
    PointType window_sum;
    Traits::point_at_infinity(window_sum);
    
    for (int i = 0; i < num_buckets - 1; i++) {
        if (!Traits::is_infinity(weighted_buckets[i])) {
            if (Traits::is_infinity(window_sum)) {
                Traits::field_copy(window_sum.x, weighted_buckets[i].x);
                Traits::field_copy(window_sum.y, weighted_buckets[i].y);
                window_sum.infinity = weighted_buckets[i].infinity;
            } else {
                PointType temp;
                point_add(temp, window_sum, weighted_buckets[i]);
                Traits::field_copy(window_sum.x, temp.x);
                Traits::field_copy(window_sum.y, temp.y);
                window_sum.infinity = temp.infinity;
            }
        }
    }
    
    // Add window sum to result
    // For windows processed from MSB to LSB:
    // - First window (MSB, highest window_idx): result = window_sum (no multiplication)
    // - Subsequent windows: result = result * 2^window_size + window_sum
    if (!Traits::is_infinity(window_sum)) {
        if (Traits::is_infinity(result)) {
            // First non-zero window: just copy window_sum
            Traits::field_copy(result.x, window_sum.x);
            Traits::field_copy(result.y, window_sum.y);
            result.infinity = window_sum.infinity;
        } else {
            // Multiply result by 2^window_size before adding window_sum
            for (int i = 0; i < MSM_WINDOW_SIZE; i++) {
                point_double(result, result);
            }
            // Add window_sum to result
            PointType temp;
            point_add(temp, result, window_sum);
            Traits::field_copy(result.x, temp.x);
            Traits::field_copy(result.y, temp.y);
            result.infinity = temp.infinity;
        }
    } else if (!Traits::is_infinity(result)) {
        // Window sum is zero but result is not: still need to multiply result
        for (int i = 0; i < MSM_WINDOW_SIZE; i++) {
            point_double(result, result);
        }
    }
}

// Projective scalar multiplication: result = k * P (small scalar, host version)
// Uses binary method with projective coordinates (no inversions!)
__host__ void projective_point_scalar_mul_small_host_g1(G1ProjectivePoint& result, const G1ProjectivePoint& P, int k) {
    if (k == 0 || fp_is_zero(P.Z)) {
        g1_projective_point_at_infinity(result);
        return;
    }
    if (k == 1) {
        fp_copy(result.X, P.X);
        fp_copy(result.Y, P.Y);
        fp_copy(result.Z, P.Z);
        return;
    }
    G1ProjectivePoint acc;
    fp_copy(acc.X, P.X);
    fp_copy(acc.Y, P.Y);
    fp_copy(acc.Z, P.Z);
    int msb = 0;
    int temp_k = k;
    while (temp_k > 1) {
        temp_k >>= 1;
        msb++;
    }
    for (int bit = msb - 1; bit >= 0; bit--) {
        projective_point_double(acc, acc);
        if (k & (1 << bit)) {
            G1ProjectivePoint temp;
            projective_point_add(temp, acc, P);
            acc = temp;
        }
    }
    fp_copy(result.X, acc.X);
    fp_copy(result.Y, acc.Y);
    fp_copy(result.Z, acc.Z);
}

__host__ void projective_point_scalar_mul_small_host_g2(G2ProjectivePoint& result, const G2ProjectivePoint& P, int k) {
    if (k == 0 || fp2_is_zero(P.Z)) {
        g2_projective_point_at_infinity(result);
        return;
    }
    if (k == 1) {
        fp2_copy(result.X, P.X);
        fp2_copy(result.Y, P.Y);
        fp2_copy(result.Z, P.Z);
        return;
    }
    G2ProjectivePoint acc;
    fp2_copy(acc.X, P.X);
    fp2_copy(acc.Y, P.Y);
    fp2_copy(acc.Z, P.Z);
    int msb = 0;
    int temp_k = k;
    while (temp_k > 1) {
        temp_k >>= 1;
        msb++;
    }
    for (int bit = msb - 1; bit >= 0; bit--) {
        projective_point_double(acc, acc);
        if (k & (1 << bit)) {
            G2ProjectivePoint temp;
            projective_point_add(temp, acc, P);
            acc = temp;
        }
    }
    fp2_copy(result.X, acc.X);
    fp2_copy(result.Y, acc.Y);
    fp2_copy(result.Z, acc.Z);
}

// CPU version: Combine buckets for a window and accumulate into result (projective version)
// This is more efficient than the GPU kernel for small bucket counts (16 buckets)
__host__ void combine_buckets_cpu_projective_g1(
    G1ProjectivePoint& result,
    const G1ProjectivePoint* buckets,
    int num_buckets,
    int window_idx
) {
    (void)window_idx;  // Unused parameter, kept for API consistency
    
    // Compute weighted buckets: i * bucket[i] for i = 1 to num_buckets-1
    G1ProjectivePoint weighted_buckets[MSM_BUCKET_COUNT];
    for (int i = 1; i < num_buckets; i++) {
        if (!fp_is_zero(buckets[i].Z)) {
            projective_point_scalar_mul_small_host_g1(weighted_buckets[i-1], buckets[i], i);
        } else {
            g1_projective_point_at_infinity(weighted_buckets[i-1]);
        }
    }
    
    // Reduce all weighted buckets into window_sum
    G1ProjectivePoint window_sum;
    g1_projective_point_at_infinity(window_sum);
    
    for (int i = 0; i < num_buckets - 1; i++) {
        if (!fp_is_zero(weighted_buckets[i].Z)) {
            if (fp_is_zero(window_sum.Z)) {
                fp_copy(window_sum.X, weighted_buckets[i].X);
                fp_copy(window_sum.Y, weighted_buckets[i].Y);
                fp_copy(window_sum.Z, weighted_buckets[i].Z);
            } else {
                G1ProjectivePoint temp;
                projective_point_add(temp, window_sum, weighted_buckets[i]);
                window_sum = temp;
            }
        }
    }
    
    // Add window sum to result
    // For windows processed from MSB to LSB:
    // - First window (MSB, highest window_idx): result = window_sum (no multiplication)
    // - Subsequent windows: result = result * 2^window_size + window_sum
    if (!fp_is_zero(window_sum.Z)) {
        if (fp_is_zero(result.Z)) {
            // First non-zero window: just copy window_sum
            fp_copy(result.X, window_sum.X);
            fp_copy(result.Y, window_sum.Y);
            fp_copy(result.Z, window_sum.Z);
        } else {
            // Multiply result by 2^window_size before adding window_sum
            for (int i = 0; i < MSM_WINDOW_SIZE; i++) {
                projective_point_double(result, result);
            }
            // Add window_sum to result
            G1ProjectivePoint temp;
            projective_point_add(temp, result, window_sum);
            result = temp;
        }
    } else if (!fp_is_zero(result.Z)) {
        // Window sum is zero but result is not: still need to multiply result
        for (int i = 0; i < MSM_WINDOW_SIZE; i++) {
            projective_point_double(result, result);
        }
    }
}

__host__ void combine_buckets_cpu_projective_g2(
    G2ProjectivePoint& result,
    const G2ProjectivePoint* buckets,
    int num_buckets,
    int window_idx
) {
    (void)window_idx;  // Unused parameter, kept for API consistency
    
    // Compute weighted buckets: i * bucket[i] for i = 1 to num_buckets-1
    G2ProjectivePoint weighted_buckets[MSM_BUCKET_COUNT];
    for (int i = 1; i < num_buckets; i++) {
        if (!fp2_is_zero(buckets[i].Z)) {
            projective_point_scalar_mul_small_host_g2(weighted_buckets[i-1], buckets[i], i);
        } else {
            g2_projective_point_at_infinity(weighted_buckets[i-1]);
        }
    }
    
    // Reduce all weighted buckets into window_sum
    G2ProjectivePoint window_sum;
    g2_projective_point_at_infinity(window_sum);
    
    for (int i = 0; i < num_buckets - 1; i++) {
        if (!fp2_is_zero(weighted_buckets[i].Z)) {
            if (fp2_is_zero(window_sum.Z)) {
                fp2_copy(window_sum.X, weighted_buckets[i].X);
                fp2_copy(window_sum.Y, weighted_buckets[i].Y);
                fp2_copy(window_sum.Z, weighted_buckets[i].Z);
            } else {
                G2ProjectivePoint temp;
                projective_point_add(temp, window_sum, weighted_buckets[i]);
                window_sum = temp;
            }
        }
    }
    
    // Add window sum to result
    if (!fp2_is_zero(window_sum.Z)) {
        if (fp2_is_zero(result.Z)) {
            fp2_copy(result.X, window_sum.X);
            fp2_copy(result.Y, window_sum.Y);
            fp2_copy(result.Z, window_sum.Z);
        } else {
            for (int i = 0; i < MSM_WINDOW_SIZE; i++) {
                projective_point_double(result, result);
            }
            G2ProjectivePoint temp;
            projective_point_add(temp, result, window_sum);
            result = temp;
        }
    } else if (!fp2_is_zero(result.Z)) {
        for (int i = 0; i < MSM_WINDOW_SIZE; i++) {
            projective_point_double(result, result);
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

// MSM with 64-bit scalars - G1 version (projective output)
void point_msm_u64_async_g1(cudaStream_t stream, uint32_t gpu_index, G1ProjectivePoint* d_result, const G1Point* d_points, const uint64_t* d_scalars, G1ProjectivePoint* d_scratch, int n) {
    if (n == 0) {
        // Set result to infinity (Z = 0)
        cuda_set_device(gpu_index);
        kernel_clear_buckets_projective_g1<<<1, 1, 0, stream>>>(d_result, 1);
        check_cuda_error(cudaGetLastError());
        return;
    }
    
    PANIC_IF_FALSE(n > 0, "point_msm_u64_async_g1: invalid size n=%d", n);
    PANIC_IF_FALSE(d_result != nullptr && d_points != nullptr && d_scalars != nullptr && d_scratch != nullptr, "point_msm_u64_async_g1: null pointer argument");
    
    cuda_set_device(gpu_index);
    
    // Calculate number of windows (64 bits / window_size)
    int num_windows = (64 + MSM_WINDOW_SIZE - 1) / MSM_WINDOW_SIZE;
    
    // Initialize result to point at infinity (Z = 0)
    kernel_clear_buckets_projective_g1<<<1, 1, 0, stream>>>(d_result, 1);
    check_cuda_error(cudaGetLastError());
    
    // Process each window from MSB to LSB
    // Use 128 threads to fit in shared memory (projective points are larger)
    int threadsPerBlock = 128;
    int num_blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Scratch space layout:
    // - d_scratch[0 .. num_blocks * MSM_BUCKET_COUNT - 1]: per-block bucket accumulations (projective)
    // - d_scratch[num_blocks * MSM_BUCKET_COUNT .. (num_blocks + 1) * MSM_BUCKET_COUNT - 1]: final buckets (projective)
    G1ProjectivePoint* d_block_buckets = d_scratch;
    G1ProjectivePoint* d_final_buckets = d_scratch + num_blocks * MSM_BUCKET_COUNT;
    
    for (int window_idx = num_windows - 1; window_idx >= 0; window_idx--) {
        // Clear final buckets
        int clear_blocks = (MSM_BUCKET_COUNT + threadsPerBlock - 1) / threadsPerBlock;
        kernel_clear_buckets_projective_g1<<<clear_blocks, threadsPerBlock, 0, stream>>>(d_final_buckets, MSM_BUCKET_COUNT);
        check_cuda_error(cudaGetLastError());
        
        // Clear block buckets
        int clear_block_blocks = (num_blocks * MSM_BUCKET_COUNT + threadsPerBlock - 1) / threadsPerBlock;
        kernel_clear_buckets_projective_g1<<<clear_block_blocks, threadsPerBlock, 0, stream>>>(d_block_buckets, num_blocks * MSM_BUCKET_COUNT);
        check_cuda_error(cudaGetLastError());
        
        // Phase 1: Accumulate points into per-block buckets using projective coordinates (no inversions!)
        const int WARP_SIZE = 32;
        int num_warps = (threadsPerBlock + WARP_SIZE - 1) / WARP_SIZE;
        size_t shared_mem_size = MSM_BUCKET_COUNT * sizeof(G1ProjectivePoint) +  // Final buckets
                                 threadsPerBlock * sizeof(G1ProjectivePoint) +    // Thread points
                                 threadsPerBlock * sizeof(int);                  // Thread bucket indices
        if (num_warps > 1) {
            shared_mem_size += num_warps * MSM_BUCKET_COUNT * sizeof(G1ProjectivePoint);  // Per-warp buckets
        }
        kernel_accumulate_buckets_u64_projective_g1<<<num_blocks, threadsPerBlock, shared_mem_size, stream>>>(
            d_block_buckets, d_points, d_scalars, n, window_idx, MSM_WINDOW_SIZE
        );
        check_cuda_error(cudaGetLastError());
        
        // Phase 2: Reduce per-block bucket contributions to final buckets
        int reduce_threads = threadsPerBlock;
        int reduce_blocks = (MSM_BUCKET_COUNT + reduce_threads - 1) / reduce_threads;
        kernel_reduce_buckets_projective_g1<<<reduce_blocks, reduce_threads, 0, stream>>>(
            d_final_buckets, d_block_buckets, num_blocks, MSM_BUCKET_COUNT
        );
        check_cuda_error(cudaGetLastError());
        
        // Combine final buckets and accumulate into result (CPU version, projective)
        G1ProjectivePoint h_buckets[MSM_BUCKET_COUNT];
        G1ProjectivePoint h_result;
        
        // Synchronize stream before copying
        cuda_synchronize_stream(stream, gpu_index);
        
        // Copy buckets from device to host
        check_cuda_error(cudaMemcpy(h_buckets, d_final_buckets, MSM_BUCKET_COUNT * sizeof(G1ProjectivePoint), cudaMemcpyDeviceToHost));
        
        // Copy current result from device to host
        check_cuda_error(cudaMemcpy(&h_result, d_result, sizeof(G1ProjectivePoint), cudaMemcpyDeviceToHost));
        
        // Combine buckets on CPU (projective version)
        combine_buckets_cpu_projective_g1(h_result, h_buckets, MSM_BUCKET_COUNT, window_idx);
        
        // Copy result back to device
        check_cuda_error(cudaMemcpy(d_result, &h_result, sizeof(G1ProjectivePoint), cudaMemcpyHostToDevice));
    }
}

// MSM with 64-bit scalars - G2 version (projective output)
void point_msm_u64_async_g2(cudaStream_t stream, uint32_t gpu_index, G2ProjectivePoint* d_result, const G2Point* d_points, const uint64_t* d_scalars, G2ProjectivePoint* d_scratch, int n) {
    if (n == 0) {
        cuda_set_device(gpu_index);
        kernel_clear_buckets_projective_g2<<<1, 1, 0, stream>>>(d_result, 1);
        check_cuda_error(cudaGetLastError());
        return;
    }
    
    PANIC_IF_FALSE(n > 0, "point_msm_u64_async_g2: invalid size n=%d", n);
    PANIC_IF_FALSE(d_result != nullptr && d_points != nullptr && d_scalars != nullptr && d_scratch != nullptr, "point_msm_u64_async_g2: null pointer argument");
    
    cuda_set_device(gpu_index);
    
    int num_windows = (64 + MSM_WINDOW_SIZE - 1) / MSM_WINDOW_SIZE;
    
    kernel_clear_buckets_projective_g2<<<1, 1, 0, stream>>>(d_result, 1);
    check_cuda_error(cudaGetLastError());
    
    // Use 64 threads for G2 (projective points are even larger - Fp2)
    int threadsPerBlock = 64;
    int num_blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    G2ProjectivePoint* d_block_buckets = d_scratch;
    G2ProjectivePoint* d_final_buckets = d_scratch + num_blocks * MSM_BUCKET_COUNT;
    
    for (int window_idx = num_windows - 1; window_idx >= 0; window_idx--) {
        int clear_blocks = (MSM_BUCKET_COUNT + threadsPerBlock - 1) / threadsPerBlock;
        kernel_clear_buckets_projective_g2<<<clear_blocks, threadsPerBlock, 0, stream>>>(d_final_buckets, MSM_BUCKET_COUNT);
        check_cuda_error(cudaGetLastError());
        
        int clear_block_blocks = (num_blocks * MSM_BUCKET_COUNT + threadsPerBlock - 1) / threadsPerBlock;
        kernel_clear_buckets_projective_g2<<<clear_block_blocks, threadsPerBlock, 0, stream>>>(d_block_buckets, num_blocks * MSM_BUCKET_COUNT);
        check_cuda_error(cudaGetLastError());
        
        const int WARP_SIZE = 32;
        int num_warps = (threadsPerBlock + WARP_SIZE - 1) / WARP_SIZE;
        size_t shared_mem_size = MSM_BUCKET_COUNT * sizeof(G2ProjectivePoint) +
                                 threadsPerBlock * sizeof(G2ProjectivePoint) +
                                 threadsPerBlock * sizeof(int);
        if (num_warps > 1) {
            shared_mem_size += num_warps * MSM_BUCKET_COUNT * sizeof(G2ProjectivePoint);
        }
        kernel_accumulate_buckets_u64_projective_g2<<<num_blocks, threadsPerBlock, shared_mem_size, stream>>>(
            d_block_buckets, d_points, d_scalars, n, window_idx, MSM_WINDOW_SIZE
        );
        check_cuda_error(cudaGetLastError());
        
        int reduce_threads = threadsPerBlock;
        int reduce_blocks = (MSM_BUCKET_COUNT + reduce_threads - 1) / reduce_threads;
        kernel_reduce_buckets_projective_g2<<<reduce_blocks, reduce_threads, 0, stream>>>(
            d_final_buckets, d_block_buckets, num_blocks, MSM_BUCKET_COUNT
        );
        check_cuda_error(cudaGetLastError());
        
        G2ProjectivePoint h_buckets[MSM_BUCKET_COUNT];
        G2ProjectivePoint h_result;
        
        cuda_synchronize_stream(stream, gpu_index);
        
        check_cuda_error(cudaMemcpy(h_buckets, d_final_buckets, MSM_BUCKET_COUNT * sizeof(G2ProjectivePoint), cudaMemcpyDeviceToHost));
        check_cuda_error(cudaMemcpy(&h_result, d_result, sizeof(G2ProjectivePoint), cudaMemcpyDeviceToHost));
        
        combine_buckets_cpu_projective_g2(h_result, h_buckets, MSM_BUCKET_COUNT, window_idx);
        
        check_cuda_error(cudaMemcpy(d_result, &h_result, sizeof(G2ProjectivePoint), cudaMemcpyHostToDevice));
    }
}

void point_msm_u64_g1(cudaStream_t stream, uint32_t gpu_index, G1ProjectivePoint* d_result, const G1Point* d_points, const uint64_t* d_scalars, G1ProjectivePoint* d_scratch, int n) {
    point_msm_u64_async_g1(stream, gpu_index, d_result, d_points, d_scalars, d_scratch, n);
    cuda_synchronize_stream(stream, gpu_index);
}

void point_msm_u64_g2(cudaStream_t stream, uint32_t gpu_index, G2ProjectivePoint* d_result, const G2Point* d_points, const uint64_t* d_scalars, G2ProjectivePoint* d_scratch, int n) {
    point_msm_u64_async_g2(stream, gpu_index, d_result, d_points, d_scalars, d_scratch, n);
    cuda_synchronize_stream(stream, gpu_index);
}

// Old template version removed - use point_msm_u64_g1 or point_msm_u64_g2 instead

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
        
        // Combine buckets and accumulate into result (CPU version for better performance)
        // Copy buckets and result to host, compute on CPU, then copy result back
        PointType h_buckets[MSM_BUCKET_COUNT];
        PointType h_result;
        
        // Synchronize stream before copying
        cuda_synchronize_stream(stream, gpu_index);
        
        // Copy buckets from device to host
        check_cuda_error(cudaMemcpy(h_buckets, d_buckets, MSM_BUCKET_COUNT * sizeof(PointType), cudaMemcpyDeviceToHost));
        
        // Copy current result from device to host
        check_cuda_error(cudaMemcpy(&h_result, d_result, sizeof(PointType), cudaMemcpyDeviceToHost));
        
        // Combine buckets on CPU
        combine_buckets_cpu<PointType>(h_result, h_buckets, MSM_BUCKET_COUNT, window_idx);
        
        // Copy result back to device
        check_cuda_error(cudaMemcpy(d_result, &h_result, sizeof(PointType), cudaMemcpyHostToDevice));
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
// point_msm_u64_async/g1/g2 are now non-template functions - no instantiation needed
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
// point_msm_u64_async/g1/g2 are now non-template functions - no instantiation needed
template void point_msm_async<G2Point>(cudaStream_t, uint32_t, G2Point*, const G2Point*, const uint64_t*, int, G2Point*, int);
template void point_msm<G2Point>(cudaStream_t, uint32_t, G2Point*, const G2Point*, const uint64_t*, int, G2Point*, int);

