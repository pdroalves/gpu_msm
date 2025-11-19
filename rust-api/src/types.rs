//! Rust wrapper types for G1 and G2 points
//! 
//! This module provides safe Rust wrappers around the FFI types,
//! with proper memory management and error handling.

use crate::ffi::{Fp, Fp2, G1Point, G2Point, G1ProjectivePoint, G2ProjectivePoint};
use std::fmt;

/// G1 affine point on the BLS12-446 curve
#[derive(Clone, Copy)]
pub struct G1Affine {
    inner: G1Point,
}

impl G1Affine {
    /// Create a new G1 affine point from coordinates
    pub fn new(x: [u64; 7], y: [u64; 7], infinity: bool) -> Self {
        Self {
            inner: G1Point {
                x: Fp { limb: x },
                y: Fp { limb: y },
                infinity,
            },
        }
    }
    
    /// Create the point at infinity
    pub fn infinity() -> Self {
        let mut point = G1Point {
            x: Fp { limb: [0; 7] },
            y: Fp { limb: [0; 7] },
            infinity: true,
        };
        unsafe {
            crate::ffi::g1_point_at_infinity_wrapper(&mut point);
        }
        Self { inner: point }
    }
    
    /// Check if this point is at infinity
    pub fn is_infinity(&self) -> bool {
        unsafe {
            crate::ffi::g1_is_infinity_wrapper(&self.inner)
        }
    }
    
    /// Get the x coordinate
    pub fn x(&self) -> [u64; 7] {
        self.inner.x.limb
    }
    
    /// Get the y coordinate
    pub fn y(&self) -> [u64; 7] {
        self.inner.y.limb
    }
    
    /// Get the inner FFI type (for internal use)
    pub(crate) fn inner(&self) -> &G1Point {
        &self.inner
    }
    
    /// Get a mutable reference to the inner FFI type (for internal use)
    pub(crate) fn inner_mut(&mut self) -> &mut G1Point {
        &mut self.inner
    }
    
    /// Convert to projective coordinates
    pub fn to_projective(&self) -> G1Projective {
        let mut proj = G1ProjectivePoint {
            X: Fp { limb: [0; 7] },
            Y: Fp { limb: [0; 7] },
            Z: Fp { limb: [0; 7] },
        };
        unsafe {
            crate::ffi::affine_to_projective_g1_wrapper(&mut proj, &self.inner);
        }
        G1Projective { inner: proj }
    }
    
}

impl fmt::Display for G1Affine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_infinity() {
            write!(f, "Infinity")
        } else {
            // Convert from Montgomery to normal form
            let normal = crate::conversions::g1_affine_from_montgomery(self);
            
            let x_str = fp_to_decimal_string(&normal.inner().x.limb);
            let y_str = fp_to_decimal_string(&normal.inner().y.limb);
            
            write!(f, "({}, {})", x_str, y_str)
        }
    }
}

/// Convert an Fp value (represented as limbs) to a decimal string
/// Assumes the limbs are in normal form (not Montgomery)
fn fp_to_decimal_string(limbs: &[u64; 7]) -> String {
    // Check if all limbs are zero
    if limbs.iter().all(|&x| x == 0) {
        return "0".to_string();
    }
    
    // Create a working copy
    let mut working = *limbs;
    let mut result = String::new();
    
    // Repeatedly divide by 10 and collect remainders
    loop {
        // Check if all limbs are zero
        if working.iter().all(|&x| x == 0) {
            break;
        }
        
        // Divide the big integer by 10 and get remainder
        // Process from MSB to LSB
        let mut remainder = 0u64;
        for i in (0..7).rev() {
            // Combine remainder with current limb: value = remainder * 2^64 + limbs[i]
            // Use 128-bit arithmetic
            let value = ((remainder as u128) << 64) | (working[i] as u128);
            working[i] = (value / 10) as u64;
            remainder = (value % 10) as u64;
        }
        
        // The remainder is our digit
        result = format!("{}{}", remainder, result);
    }
    
    if result.is_empty() {
        "0".to_string()
    } else {
        result
    }
}

impl fmt::Debug for G1Affine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_infinity() {
            write!(f, "G1Affine(infinity)")
        } else {
            write!(f, "G1Affine(x: {:?}, y: {:?})", self.x(), self.y())
        }
    }
}

/// G2 affine point on the BLS12-446 curve
#[derive(Clone, Copy)]
pub struct G2Affine {
    inner: G2Point,
}

impl G2Affine {
    /// Create a new G2 affine point from coordinates
    pub fn new(x: ([u64; 7], [u64; 7]), y: ([u64; 7], [u64; 7]), infinity: bool) -> Self {
        Self {
            inner: G2Point {
                x: Fp2 {
                    c0: Fp { limb: x.0 },
                    c1: Fp { limb: x.1 },
                },
                y: Fp2 {
                    c0: Fp { limb: y.0 },
                    c1: Fp { limb: y.1 },
                },
                infinity,
            },
        }
    }
    
    /// Create the point at infinity
    pub fn infinity() -> Self {
        let mut point = G2Point {
            x: Fp2 {
                c0: Fp { limb: [0; 7] },
                c1: Fp { limb: [0; 7] },
            },
            y: Fp2 {
                c0: Fp { limb: [0; 7] },
                c1: Fp { limb: [0; 7] },
            },
            infinity: true,
        };
        unsafe {
            crate::ffi::g2_point_at_infinity_wrapper(&mut point);
        }
        Self { inner: point }
    }
    
    /// Check if this point is at infinity
    pub fn is_infinity(&self) -> bool {
        unsafe {
            crate::ffi::g2_is_infinity_wrapper(&self.inner)
        }
    }
    
    /// Get the x coordinate as (c0, c1)
    pub fn x(&self) -> ([u64; 7], [u64; 7]) {
        (self.inner.x.c0.limb, self.inner.x.c1.limb)
    }
    
    /// Get the y coordinate as (c0, c1)
    pub fn y(&self) -> ([u64; 7], [u64; 7]) {
        (self.inner.y.c0.limb, self.inner.y.c1.limb)
    }
    
    /// Get the inner FFI type (for internal use)
    pub(crate) fn inner(&self) -> &G2Point {
        &self.inner
    }
    
    /// Get a mutable reference to the inner FFI type (for internal use)
    pub(crate) fn inner_mut(&mut self) -> &mut G2Point {
        &mut self.inner
    }
    
    /// Convert to projective coordinates
    pub fn to_projective(&self) -> G2Projective {
        let mut proj = G2ProjectivePoint {
            X: Fp2 {
                c0: Fp { limb: [0; 7] },
                c1: Fp { limb: [0; 7] },
            },
            Y: Fp2 {
                c0: Fp { limb: [0; 7] },
                c1: Fp { limb: [0; 7] },
            },
            Z: Fp2 {
                c0: Fp { limb: [0; 7] },
                c1: Fp { limb: [0; 7] },
            },
        };
        unsafe {
            crate::ffi::affine_to_projective_g2_wrapper(&mut proj, &self.inner);
        }
        G2Projective { inner: proj }
    }
    
}

impl fmt::Display for G2Affine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_infinity() {
            write!(f, "Infinity")
        } else {
            // Convert from Montgomery to normal form
            let normal = crate::conversions::g2_affine_from_montgomery(self);
            
            let (x_c0, x_c1) = normal.x();
            let (y_c0, y_c1) = normal.y();
            
            let x_c0_str = fp_to_decimal_string(&x_c0);
            let x_c1_str = fp_to_decimal_string(&x_c1);
            let y_c0_str = fp_to_decimal_string(&y_c0);
            let y_c1_str = fp_to_decimal_string(&y_c1);
            
            write!(f, "(x: ({}, {}), y: ({}, {}))", x_c0_str, x_c1_str, y_c0_str, y_c1_str)
        }
    }
}

impl fmt::Debug for G2Affine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_infinity() {
            write!(f, "G2Affine(infinity)")
        } else {
            let (x0, x1) = self.x();
            let (y0, y1) = self.y();
            write!(f, "G2Affine(x: ({:?}, {:?}), y: ({:?}, {:?}))", x0, x1, y0, y1)
        }
    }
}

/// G1 projective point on the BLS12-446 curve
#[derive(Clone, Copy)]
pub struct G1Projective {
    inner: G1ProjectivePoint,
}

impl G1Projective {
    /// Create a new G1 projective point from coordinates
    pub fn new(X: [u64; 7], Y: [u64; 7], Z: [u64; 7]) -> Self {
        Self {
            inner: G1ProjectivePoint {
                X: Fp { limb: X },
                Y: Fp { limb: Y },
                Z: Fp { limb: Z },
            },
        }
    }
    
    /// Create the point at infinity (Z = 0)
    pub fn infinity() -> Self {
        let mut point = G1ProjectivePoint {
            X: Fp { limb: [0; 7] },
            Y: Fp { limb: [0; 7] },
            Z: Fp { limb: [0; 7] },
        };
        unsafe {
            crate::ffi::g1_projective_point_at_infinity_wrapper(&mut point);
        }
        Self { inner: point }
    }
    
    /// Get the X coordinate
    pub fn X(&self) -> [u64; 7] {
        self.inner.X.limb
    }
    
    /// Get the Y coordinate
    pub fn Y(&self) -> [u64; 7] {
        self.inner.Y.limb
    }
    
    /// Get the Z coordinate
    pub fn Z(&self) -> [u64; 7] {
        self.inner.Z.limb
    }
    
    /// Get the inner FFI type (for internal use)
    pub(crate) fn inner(&self) -> &G1ProjectivePoint {
        &self.inner
    }
    
    /// Get a mutable reference to the inner FFI type (for internal use)
    pub(crate) fn inner_mut(&mut self) -> &mut G1ProjectivePoint {
        &mut self.inner
    }
    
    /// Convert to affine coordinates
    pub fn to_affine(&self) -> G1Affine {
        let mut affine = G1Point {
            x: Fp { limb: [0; 7] },
            y: Fp { limb: [0; 7] },
            infinity: false,
        };
        unsafe {
            crate::ffi::projective_to_affine_g1_wrapper(&mut affine, &self.inner);
        }
        G1Affine { inner: affine }
    }
    
    /// Compute multi-scalar multiplication: result = sum(scalars[i] * points[i])
    /// Returns an error if MSM computation fails
    pub fn msm(points: &[G1Affine], scalars: &[u64], gpu_index: u32) -> Result<Self, String> {
        if points.len() != scalars.len() {
            return Err(format!("Points and scalars must have the same length: {} != {}", points.len(), scalars.len()));
        }
        if points.is_empty() {
            return Ok(Self::infinity());
        }
        let n = points.len() as i32;
        let points_ffi: Vec<G1Point> = points.iter().map(|p| p.inner).collect();
        let mut result = G1ProjectivePoint {
            X: Fp { limb: [0; 7] },
            Y: Fp { limb: [0; 7] },
            Z: Fp { limb: [0; 7] },
        };
        println!("points: {:?}", points);
        let ret = unsafe {
            crate::ffi::g1_msm_wrapper(
                &mut result,
                points_ffi.as_ptr(),
                scalars.as_ptr(),
                n,
                gpu_index,
            )
        };
        
        if ret != 0 {
            return Err(format!("MSM computation failed with error code: {}", ret));
        }
        
        Ok(Self { inner: result })
    }
}

impl fmt::Display for G1Projective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Convert to affine for display
        let affine = self.to_affine();
        write!(f, "{}", affine)
    }
}

impl fmt::Debug for G1Projective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "G1Projective(X: {:?}, Y: {:?}, Z: {:?})", self.X(), self.Y(), self.Z())
    }
}

/// G2 projective point on the BLS12-446 curve
#[derive(Clone, Copy)]
pub struct G2Projective {
    inner: G2ProjectivePoint,
}

impl G2Projective {
    /// Create a new G2 projective point from coordinates
    pub fn new(
        X: ([u64; 7], [u64; 7]),
        Y: ([u64; 7], [u64; 7]),
        Z: ([u64; 7], [u64; 7]),
    ) -> Self {
        Self {
            inner: G2ProjectivePoint {
                X: Fp2 {
                    c0: Fp { limb: X.0 },
                    c1: Fp { limb: X.1 },
                },
                Y: Fp2 {
                    c0: Fp { limb: Y.0 },
                    c1: Fp { limb: Y.1 },
                },
                Z: Fp2 {
                    c0: Fp { limb: Z.0 },
                    c1: Fp { limb: Z.1 },
                },
            },
        }
    }
    
    /// Create the point at infinity (Z = 0)
    pub fn infinity() -> Self {
        let mut point = G2ProjectivePoint {
            X: Fp2 {
                c0: Fp { limb: [0; 7] },
                c1: Fp { limb: [0; 7] },
            },
            Y: Fp2 {
                c0: Fp { limb: [0; 7] },
                c1: Fp { limb: [0; 7] },
            },
            Z: Fp2 {
                c0: Fp { limb: [0; 7] },
                c1: Fp { limb: [0; 7] },
            },
        };
        unsafe {
            crate::ffi::g2_projective_point_at_infinity_wrapper(&mut point);
        }
        Self { inner: point }
    }
    
    /// Get the X coordinate as (c0, c1)
    pub fn X(&self) -> ([u64; 7], [u64; 7]) {
        (self.inner.X.c0.limb, self.inner.X.c1.limb)
    }
    
    /// Get the Y coordinate as (c0, c1)
    pub fn Y(&self) -> ([u64; 7], [u64; 7]) {
        (self.inner.Y.c0.limb, self.inner.Y.c1.limb)
    }
    
    /// Get the Z coordinate as (c0, c1)
    pub fn Z(&self) -> ([u64; 7], [u64; 7]) {
        (self.inner.Z.c0.limb, self.inner.Z.c1.limb)
    }
    
    /// Get the inner FFI type (for internal use)
    pub(crate) fn inner(&self) -> &G2ProjectivePoint {
        &self.inner
    }
    
    /// Get a mutable reference to the inner FFI type (for internal use)
    pub(crate) fn inner_mut(&mut self) -> &mut G2ProjectivePoint {
        &mut self.inner
    }
    
    /// Convert to affine coordinates
    pub fn to_affine(&self) -> G2Affine {
        let mut affine = G2Point {
            x: Fp2 {
                c0: Fp { limb: [0; 7] },
                c1: Fp { limb: [0; 7] },
            },
            y: Fp2 {
                c0: Fp { limb: [0; 7] },
                c1: Fp { limb: [0; 7] },
            },
            infinity: false,
        };
        unsafe {
            crate::ffi::projective_to_affine_g2_wrapper(&mut affine, &self.inner);
        }
        G2Affine { inner: affine }
    }
    
    /// Compute multi-scalar multiplication: result = sum(scalars[i] * points[i])
    /// Returns an error if MSM computation fails
    pub fn msm(points: &[G2Affine], scalars: &[u64], gpu_index: u32) -> Result<Self, String> {
        if points.len() != scalars.len() {
            return Err(format!("Points and scalars must have the same length: {} != {}", points.len(), scalars.len()));
        }
        if points.is_empty() {
            return Ok(Self::infinity());
        }
        
        let n = points.len() as i32;
        let points_ffi: Vec<G2Point> = points.iter().map(|p| p.inner).collect();
        let mut result = G2ProjectivePoint {
            X: Fp2 {
                c0: Fp { limb: [0; 7] },
                c1: Fp { limb: [0; 7] },
            },
            Y: Fp2 {
                c0: Fp { limb: [0; 7] },
                c1: Fp { limb: [0; 7] },
            },
            Z: Fp2 {
                c0: Fp { limb: [0; 7] },
                c1: Fp { limb: [0; 7] },
            },
        };
        
        let ret = unsafe {
            crate::ffi::g2_msm_wrapper(
                &mut result,
                points_ffi.as_ptr(),
                scalars.as_ptr(),
                n,
                gpu_index,
            )
        };
        
        if ret != 0 {
            return Err(format!("MSM computation failed with error code: {}", ret));
        }
        
        Ok(Self { inner: result })
    }
}

impl fmt::Display for G2Projective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Convert to affine for display
        let affine = self.to_affine();
        write!(f, "{}", affine)
    }
}

impl fmt::Debug for G2Projective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (X0, X1) = self.X();
        let (Y0, Y1) = self.Y();
        let (Z0, Z1) = self.Z();
        write!(
            f,
            "G2Projective(X: ({:?}, {:?}), Y: ({:?}, {:?}), Z: ({:?}, {:?}))",
            X0, X1, Y0, Y1, Z0, Z1
        )
    }
}

