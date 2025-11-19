//! FFI bindings to the C++ library
//! 
//! This module contains the raw FFI bindings to the C++ CUDA library.
//! These are low-level bindings that should generally not be used directly.

use std::os::raw::c_uint;

// Fp structure: 7 limbs of 64 bits each
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Fp {
    pub limb: [u64; 7],
}

// Fp2 structure: two Fp elements
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Fp2 {
    pub c0: Fp,
    pub c1: Fp,
}

// G1 affine point
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct G1Point {
    pub x: Fp,
    pub y: Fp,
    pub infinity: bool,
}

// G2 affine point
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct G2Point {
    pub x: Fp2,
    pub y: Fp2,
    pub infinity: bool,
}

// G1 projective point
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct G1ProjectivePoint {
    pub X: Fp,
    pub Y: Fp,
    pub Z: Fp,
}

// G2 projective point
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct G2ProjectivePoint {
    pub X: Fp2,
    pub Y: Fp2,
    pub Z: Fp2,
}

// Opaque CUDA stream type
#[repr(C)]
pub struct cudaStream_t {
    _private: [u8; 0],
}

extern "C" {
    // Device initialization
    pub fn init_device_generators(stream: *const cudaStream_t, gpu_index: c_uint);
    
    // Point conversions (using C wrapper functions)
    pub fn affine_to_projective_g1_wrapper(proj: *mut G1ProjectivePoint, affine: *const G1Point);
    pub fn affine_to_projective_g2_wrapper(proj: *mut G2ProjectivePoint, affine: *const G2Point);
    pub fn projective_to_affine_g1_wrapper(affine: *mut G1Point, proj: *const G1ProjectivePoint);
    pub fn projective_to_affine_g2_wrapper(affine: *mut G2Point, proj: *const G2ProjectivePoint);
    
    // Point at infinity (using wrapper functions with C linkage)
    pub fn g1_point_at_infinity_wrapper(point: *mut G1Point);
    pub fn g2_point_at_infinity_wrapper(point: *mut G2Point);
    pub fn g1_projective_point_at_infinity_wrapper(point: *mut G1ProjectivePoint);
    pub fn g2_projective_point_at_infinity_wrapper(point: *mut G2ProjectivePoint);
    
    // Infinity checks (using wrapper functions with C linkage)
    pub fn g1_is_infinity_wrapper(point: *const G1Point) -> bool;
    pub fn g2_is_infinity_wrapper(point: *const G2Point) -> bool;
    
    // MSM functions
    pub fn g1_msm_wrapper(
        result: *mut G1ProjectivePoint,
        points: *const G1Point,
        scalars: *const u64,
        n: std::os::raw::c_int,
        gpu_index: c_uint,
    ) -> std::os::raw::c_int;
    
    pub fn g2_msm_wrapper(
        result: *mut G2ProjectivePoint,
        points: *const G2Point,
        scalars: *const u64,
        n: std::os::raw::c_int,
        gpu_index: c_uint,
    ) -> std::os::raw::c_int;
    
    // Montgomery form conversion functions
    pub fn g1_from_montgomery_wrapper(result: *mut G1Point, point: *const G1Point);
    pub fn g2_from_montgomery_wrapper(result: *mut G2Point, point: *const G2Point);
    pub fn fp_to_montgomery_wrapper(result: *mut Fp, value: *const Fp);
}

