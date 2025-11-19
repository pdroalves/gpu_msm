#pragma once

#include "../include/curve.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// C-compatible wrapper functions for Rust FFI

// G1 affine to projective conversion
void affine_to_projective_g1_wrapper(G1ProjectivePoint* proj, const G1Point* affine);

// G2 affine to projective conversion
void affine_to_projective_g2_wrapper(G2ProjectivePoint* proj, const G2Point* affine);

// G1 projective to affine conversion
void projective_to_affine_g1_wrapper(G1Point* affine, const G1ProjectivePoint* proj);

// G2 projective to affine conversion
void projective_to_affine_g2_wrapper(G2Point* affine, const G2ProjectivePoint* proj);

// G1 point at infinity
void g1_point_at_infinity_wrapper(G1Point* point);

// G2 point at infinity
void g2_point_at_infinity_wrapper(G2Point* point);

// G1 projective point at infinity
void g1_projective_point_at_infinity_wrapper(G1ProjectivePoint* point);

// G2 projective point at infinity
void g2_projective_point_at_infinity_wrapper(G2ProjectivePoint* point);

// Check if G1 point is at infinity
bool g1_is_infinity_wrapper(const G1Point* point);

// Check if G2 point is at infinity
bool g2_is_infinity_wrapper(const G2Point* point);

// High-level MSM wrappers (handle CUDA setup internally)
// Returns 0 on success, non-zero on error
int g1_msm_wrapper(
    G1ProjectivePoint* result,
    const G1Point* points,
    const uint64_t* scalars,
    int n,
    uint32_t gpu_index
);

int g2_msm_wrapper(
    G2ProjectivePoint* result,
    const G2Point* points,
    const uint64_t* scalars,
    int n,
    uint32_t gpu_index
);

#ifdef __cplusplus
}
#endif

