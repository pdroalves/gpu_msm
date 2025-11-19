// C wrapper functions for Rust FFI
// These functions provide a C-compatible interface to the C++ functions

#include "../../include/curve.h"
#include <stdint.h>
#include <stdbool.h>

extern "C" {

// G1 affine to projective conversion
void affine_to_projective_g1_wrapper(G1ProjectivePoint* proj, const G1Point* affine) {
    affine_to_projective(*proj, *affine);
}

// G2 affine to projective conversion
void affine_to_projective_g2_wrapper(G2ProjectivePoint* proj, const G2Point* affine) {
    affine_to_projective(*proj, *affine);
}

// G1 projective to affine conversion
void projective_to_affine_g1_wrapper(G1Point* affine, const G1ProjectivePoint* proj) {
    projective_to_affine_g1(*affine, *proj);
}

// G2 projective to affine conversion
void projective_to_affine_g2_wrapper(G2Point* affine, const G2ProjectivePoint* proj) {
    projective_to_affine_g2(*affine, *proj);
}

// G1 point at infinity
void g1_point_at_infinity_wrapper(G1Point* point) {
    g1_point_at_infinity(*point);
}

// G2 point at infinity
void g2_point_at_infinity_wrapper(G2Point* point) {
    g2_point_at_infinity(*point);
}

// G1 projective point at infinity
void g1_projective_point_at_infinity_wrapper(G1ProjectivePoint* point) {
    g1_projective_point_at_infinity(*point);
}

// G2 projective point at infinity
void g2_projective_point_at_infinity_wrapper(G2ProjectivePoint* point) {
    g2_projective_point_at_infinity(*point);
}

// Check if G1 point is at infinity
bool g1_is_infinity_wrapper(const G1Point* point) {
    return g1_is_infinity(*point);
}

// Check if G2 point is at infinity
bool g2_is_infinity_wrapper(const G2Point* point) {
    return g2_is_infinity(*point);
}

} // extern "C"
