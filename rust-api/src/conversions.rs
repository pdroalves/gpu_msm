//! Conversion traits and functions for tfhe-zk-pok compatibility
//! 
//! This module provides conversion functions between our types and tfhe-zk-pok types.
//! The conversions are designed to be compatible with tfhe-zk-pok's curve_446 module.

use crate::types::{G1Affine, G1Projective, G2Affine, G2Projective};

// Re-export tfhe-zk-pok types for convenience
pub use tfhe_zk_pok::curve_446::g1::{G1Affine as TfheZkPokG1Affine, G1Projective as TfheZkPokG1Projective};
pub use tfhe_zk_pok::curve_446::g2::{G2Affine as TfheZkPokG2Affine, G2Projective as TfheZkPokG2Projective};
pub use tfhe_zk_pok::curve_446::{Fq, Fq2};

// Import arkworks traits that are re-exported through tfhe-zk-pok
use ark_ec::AffineRepr;

/// Trait for converting to tfhe-zk-pok G1Affine type
pub trait ToTfheZkPokG1Affine {
    fn to_tfhe_zk_pok(&self) -> TfheZkPokG1Affine;
}

/// Trait for converting from tfhe-zk-pok G1Affine type
pub trait FromTfheZkPokG1Affine {
    fn from_tfhe_zk_pok(other: &TfheZkPokG1Affine) -> Self;
}

/// Trait for converting to tfhe-zk-pok G2Affine type
pub trait ToTfheZkPokG2Affine {
    fn to_tfhe_zk_pok(&self) -> TfheZkPokG2Affine;
}

/// Trait for converting from tfhe-zk-pok G2Affine type
pub trait FromTfheZkPokG2Affine {
    fn from_tfhe_zk_pok(other: &TfheZkPokG2Affine) -> Self;
}

/// Trait for converting to tfhe-zk-pok G1Projective type
pub trait ToTfheZkPokG1Projective {
    fn to_tfhe_zk_pok(&self) -> TfheZkPokG1Projective;
}

/// Trait for converting from tfhe-zk-pok G1Projective type
pub trait FromTfheZkPokG1Projective {
    fn from_tfhe_zk_pok(other: &TfheZkPokG1Projective) -> Self;
}

/// Trait for converting to tfhe-zk-pok G2Projective type
pub trait ToTfheZkPokG2Projective {
    fn to_tfhe_zk_pok(&self) -> TfheZkPokG2Projective;
}

/// Trait for converting from tfhe-zk-pok G2Projective type
pub trait FromTfheZkPokG2Projective {
    fn from_tfhe_zk_pok(other: &TfheZkPokG2Projective) -> Self;
}

// Helper function to convert Fp (7 limbs) to tfhe-zk-pok's Fq
fn fp_to_fq(limbs: [u64; 7]) -> Fq {
    // Convert from little-endian limbs to Fq using arkworks' from_sign_and_limbs
    Fq::from_sign_and_limbs(true, &limbs)
}

// Helper function to convert tfhe-zk-pok's Fq to Fp (7 limbs)
fn fq_to_fp(fq: &Fq) -> [u64; 7] {
    // Extract limbs from Fq's internal representation
    // In arkworks, Fq has a `into_bigint()` method that gives us the limbs
    // IMPORTANT: arkworks stores field elements in Montgomery form internally!
    // The into_bigint() method returns the Montgomery representation
    use ark_ff::PrimeField;
    let bigint = fq.into_bigint();
    let mut limbs = [0u64; 7];
    for (i, limb) in bigint.as_ref().iter().take(7).enumerate() {
        limbs[i] = *limb;
    }
    // Note: These limbs are in Montgomery form, matching our internal representation
    limbs
}

// Helper function to convert G1Affine from Montgomery form to normal form
pub fn g1_affine_from_montgomery(g1_mont: &crate::types::G1Affine) -> crate::types::G1Affine {
    let mut result = crate::types::G1Affine::infinity();
    if !g1_mont.is_infinity() {
        unsafe {
            // Call the C wrapper to convert from Montgomery form
            crate::ffi::g1_from_montgomery_wrapper(result.inner_mut(), g1_mont.inner());
        }
    }
    result
}

// Helper function to convert G2Affine from Montgomery form to normal form
pub fn g2_affine_from_montgomery(g2_mont: &crate::types::G2Affine) -> crate::types::G2Affine {
    let mut result = crate::types::G2Affine::infinity();
    if !g2_mont.is_infinity() {
        unsafe {
            // Call the C wrapper to convert from Montgomery form
            crate::ffi::g2_from_montgomery_wrapper(result.inner_mut(), g2_mont.inner());
        }
    }
    result
}

// Helper function to convert Fp2 (two 7-limb arrays) to tfhe-zk-pok's Fq2
fn fp2_to_fq2(c0: [u64; 7], c1: [u64; 7]) -> Fq2 {
    let c0_fq = fp_to_fq(c0);
    let c1_fq = fp_to_fq(c1);
    Fq2::new(c0_fq, c1_fq)
}

// Helper function to convert tfhe-zk-pok's Fq2 to Fp2 (two 7-limb arrays)
fn fq2_to_fp2(fq2: &Fq2) -> ([u64; 7], [u64; 7]) {
    // Fq2 has c0 and c1 fields in arkworks
    let c0 = fq_to_fp(&fq2.c0);
    let c1 = fq_to_fp(&fq2.c1);
    (c0, c1)
}

// Implement conversions for G1Affine
impl ToTfheZkPokG1Affine for G1Affine {
    fn to_tfhe_zk_pok(&self) -> TfheZkPokG1Affine {
        if self.is_infinity() {
            return TfheZkPokG1Affine::default();
        }
        
        let x_fq = fp_to_fq(self.x());
        let y_fq = fp_to_fq(self.y());
        
        // Create G1Affine from coordinates using arkworks API
        // G1Affine::new(x, y, infinity_flag) - but we need to check the actual constructor
        // For arkworks, we typically use try_new or from
        TfheZkPokG1Affine::new(x_fq, y_fq)
    }
}

impl FromTfheZkPokG1Affine for G1Affine {
    fn from_tfhe_zk_pok(other: &TfheZkPokG1Affine) -> Self {
        if other.is_zero() {
            return G1Affine::infinity();
        }
        
        // Extract coordinates - in arkworks, these are fields, not methods
        let x_fq = &other.x;
        let y_fq = &other.y;
        
        let x = fq_to_fp(x_fq);
        let y = fq_to_fp(y_fq);
        
        G1Affine::new(x, y, false)
    }
}

// Implement conversions for G2Affine
impl ToTfheZkPokG2Affine for G2Affine {
    fn to_tfhe_zk_pok(&self) -> TfheZkPokG2Affine {
        if self.is_infinity() {
            return TfheZkPokG2Affine::default();
        }
        
        let (x0, x1) = self.x();
        let (y0, y1) = self.y();
        
        let x_fq2 = fp2_to_fq2(x0, x1);
        let y_fq2 = fp2_to_fq2(y0, y1);
        
        TfheZkPokG2Affine::new(x_fq2, y_fq2)
    }
}

impl FromTfheZkPokG2Affine for G2Affine {
    fn from_tfhe_zk_pok(other: &TfheZkPokG2Affine) -> Self {
        if other.is_zero() {
            return G2Affine::infinity();
        }
        
        // Extract coordinates - in arkworks, these are fields
        let x_fq2 = &other.x;
        let y_fq2 = &other.y;
        
        let (x0, x1) = fq2_to_fp2(x_fq2);
        let (y0, y1) = fq2_to_fp2(y_fq2);
        
        G2Affine::new((x0, x1), (y0, y1), false)
    }
}

// Implement conversions for G1Projective
impl ToTfheZkPokG1Projective for G1Projective {
    fn to_tfhe_zk_pok(&self) -> TfheZkPokG1Projective {
        let x_fq = fp_to_fq(self.X());
        let y_fq = fp_to_fq(self.Y());
        let z_fq = fp_to_fq(self.Z());
        
        TfheZkPokG1Projective::new(x_fq, y_fq, z_fq)
    }
}

impl FromTfheZkPokG1Projective for G1Projective {
    fn from_tfhe_zk_pok(other: &TfheZkPokG1Projective) -> Self {
        // Extract coordinates - in arkworks projective, these are fields
        let x_fq = &other.x;
        let y_fq = &other.y;
        let z_fq = &other.z;
        
        let x = fq_to_fp(x_fq);
        let y = fq_to_fp(y_fq);
        let z = fq_to_fp(z_fq);
        
        G1Projective::new(x, y, z)
    }
}

// Implement conversions for G2Projective
impl ToTfheZkPokG2Projective for G2Projective {
    fn to_tfhe_zk_pok(&self) -> TfheZkPokG2Projective {
        let (x0, x1) = self.X();
        let (y0, y1) = self.Y();
        let (z0, z1) = self.Z();
        
        let x_fq2 = fp2_to_fq2(x0, x1);
        let y_fq2 = fp2_to_fq2(y0, y1);
        let z_fq2 = fp2_to_fq2(z0, z1);
        
        TfheZkPokG2Projective::new(x_fq2, y_fq2, z_fq2)
    }
}

impl FromTfheZkPokG2Projective for G2Projective {
    fn from_tfhe_zk_pok(other: &TfheZkPokG2Projective) -> Self {
        // Extract coordinates - in arkworks projective, these are fields
        let x_fq2 = &other.x;
        let y_fq2 = &other.y;
        let z_fq2 = &other.z;
        
        let (x0, x1) = fq2_to_fp2(x_fq2);
        let (y0, y1) = fq2_to_fp2(y_fq2);
        let (z0, z1) = fq2_to_fp2(z_fq2);
        
        G2Projective::new((x0, x1), (y0, y1), (z0, z1))
    }
}

// Convenience conversion functions

/// Convert from our G1Affine to tfhe-zk-pok's G1Affine
pub fn g1_affine_to_tfhe_zk_pok(point: &G1Affine) -> TfheZkPokG1Affine {
    point.to_tfhe_zk_pok()
}

/// Convert from tfhe-zk-pok's G1Affine to our G1Affine
pub fn g1_affine_from_tfhe_zk_pok(point: &TfheZkPokG1Affine) -> G1Affine {
    G1Affine::from_tfhe_zk_pok(point)
}

/// Convert from our G2Affine to tfhe-zk-pok's G2Affine
pub fn g2_affine_to_tfhe_zk_pok(point: &G2Affine) -> TfheZkPokG2Affine {
    point.to_tfhe_zk_pok()
}

/// Convert from tfhe-zk-pok's G2Affine to our G2Affine
pub fn g2_affine_from_tfhe_zk_pok(point: &TfheZkPokG2Affine) -> G2Affine {
    G2Affine::from_tfhe_zk_pok(point)
}

/// Convert from our G1Projective to tfhe-zk-pok's G1Projective
pub fn g1_projective_to_tfhe_zk_pok(point: &G1Projective) -> TfheZkPokG1Projective {
    point.to_tfhe_zk_pok()
}

/// Convert from tfhe-zk-pok's G1Projective to our G1Projective
pub fn g1_projective_from_tfhe_zk_pok(point: &TfheZkPokG1Projective) -> G1Projective {
    G1Projective::from_tfhe_zk_pok(point)
}

/// Convert from our G2Projective to tfhe-zk-pok's G2Projective
pub fn g2_projective_to_tfhe_zk_pok(point: &G2Projective) -> TfheZkPokG2Projective {
    point.to_tfhe_zk_pok()
}

/// Convert from tfhe-zk-pok's G2Projective to our G2Projective
pub fn g2_projective_from_tfhe_zk_pok(point: &TfheZkPokG2Projective) -> G2Projective {
    G2Projective::from_tfhe_zk_pok(point)
}
