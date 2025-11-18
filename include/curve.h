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

