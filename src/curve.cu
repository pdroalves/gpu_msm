#include "curve.h"
#include "fp.h"
#include "fp2.h"
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
    cudaError_t err = cudaSetDevice(gpu_index);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error setting device %u: %s\n", gpu_index, cudaGetErrorString(err));
        return;
    }
    
    Fp host_b_g1 = get_host_curve_b_g1();
    Fp2 host_b_g2 = get_host_curve_b_g2();
    
    // Note: cudaMemcpyToSymbolAsync doesn't exist for __constant__ memory
    // We must use synchronous cudaMemcpyToSymbol, but we ensure the device is set correctly
    err = cudaMemcpyToSymbol(DEVICE_CURVE_B_G1, &host_b_g1, sizeof(Fp));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error initializing device curve b_g1 on GPU %u: %s\n", gpu_index, cudaGetErrorString(err));
        return;
    }
    
    err = cudaMemcpyToSymbol(DEVICE_CURVE_B_G2, &host_b_g2, sizeof(Fp2));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error initializing device curve b_g2 on GPU %u: %s\n", gpu_index, cudaGetErrorString(err));
        return;
    }
    
    // Synchronize stream to ensure completion
    if (stream != nullptr) {
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error synchronizing stream on GPU %u: %s\n", gpu_index, cudaGetErrorString(err));
        }
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

