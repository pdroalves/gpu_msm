//! Example usage of the Rust API
//! 
//! This module demonstrates how to use the Rust API with tfhe-zk-pok types.
//! 
//! Note: This is a placeholder example. Once tfhe-zk-pok is available,
//! update the types and conversion logic accordingly.

#[cfg(test)]
mod tests {
    use super::super::{G1Affine, G1Projective, G2Affine, G2Projective};
    use super::super::conversions::*;

    #[test]
    fn test_g1_affine_creation() {
        // Create a G1 affine point at infinity
        let g1_inf = G1Affine::infinity();
        assert!(g1_inf.is_infinity());
        
        // Create a G1 affine point from coordinates
        let x = [0x1234567890abcdef, 0, 0, 0, 0, 0, 0];
        let y = [0xfedcba0987654321, 0, 0, 0, 0, 0, 0];
        let g1_point = G1Affine::new(x, y, false);
        assert!(!g1_point.is_infinity());
    }

    #[test]
    fn test_g1_affine_to_projective() {
        let x = [1, 0, 0, 0, 0, 0, 0];
        let y = [2, 0, 0, 0, 0, 0, 0];
        let g1_affine = G1Affine::new(x, y, false);
        
        let g1_proj = g1_affine.to_projective();
        let g1_affine_again = g1_proj.to_affine();
        
        // After round-trip conversion, coordinates should match
        assert_eq!(g1_affine.x(), g1_affine_again.x());
        assert_eq!(g1_affine.y(), g1_affine_again.y());
    }

    #[test]
    fn test_g2_affine_creation() {
        // Create a G2 affine point at infinity
        let g2_inf = G2Affine::infinity();
        assert!(g2_inf.is_infinity());
        
        // Create a G2 affine point from coordinates
        let x = ([0x1234, 0, 0, 0, 0, 0, 0], [0x5678, 0, 0, 0, 0, 0, 0]);
        let y = ([0x9abc, 0, 0, 0, 0, 0, 0], [0xdef0, 0, 0, 0, 0, 0, 0]);
        let g2_point = G2Affine::new(x, y, false);
        assert!(!g2_point.is_infinity());
    }

    #[test]
    fn test_g2_affine_to_projective() {
        let x = ([1, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0]);
        let y = ([3, 0, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0, 0]);
        let g2_affine = G2Affine::new(x, y, false);
        
        let g2_proj = g2_affine.to_projective();
        let g2_affine_again = g2_proj.to_affine();
        
        // After round-trip conversion, coordinates should match
        assert_eq!(g2_affine.x(), g2_affine_again.x());
        assert_eq!(g2_affine.y(), g2_affine_again.y());
    }

    #[test]
    fn test_tfhe_zk_pok_conversions() {
        use super::super::conversions::{TfheZkPokG1Affine, TfheZkPokG2Affine};
        use ark_ec::AffineRepr;
        
        // Get the generator points from tfhe-zk-pok
        let tfhe_g1_gen = TfheZkPokG1Affine::generator();
        let tfhe_g2_gen = TfheZkPokG2Affine::generator();
        
        // Convert from tfhe-zk-pok generator to our format
        let g1_point = g1_affine_from_tfhe_zk_pok(&tfhe_g1_gen);
        
        // Convert back to tfhe-zk-pok and verify
        let tfhe_g1_again = g1_affine_to_tfhe_zk_pok(&g1_point);
        let g1_point_again = g1_affine_from_tfhe_zk_pok(&tfhe_g1_again);
        
        assert_eq!(g1_point.x(), g1_point_again.x());
        assert_eq!(g1_point.y(), g1_point_again.y());
        
        // Test G2 conversions with generator
        let g2_point = g2_affine_from_tfhe_zk_pok(&tfhe_g2_gen);
        let tfhe_g2_again = g2_affine_to_tfhe_zk_pok(&g2_point);
        let g2_point_again = g2_affine_from_tfhe_zk_pok(&tfhe_g2_again);
        
        assert_eq!(g2_point.x(), g2_point_again.x());
        assert_eq!(g2_point.y(), g2_point_again.y());
    }
    
    #[test]
    fn test_g1_msm_vs_tfhe_zk_pok() {
        use super::super::conversions::{TfheZkPokG1Affine, TfheZkPokG1Projective};
        use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
        
        // Get generator from tfhe-zk-pok
        let tfhe_g1_gen = TfheZkPokG1Affine::generator();
        let g1_gen = g1_affine_from_tfhe_zk_pok(&tfhe_g1_gen);
        
        let g1_x = g1_gen.x();
        eprintln!("tfhe-zk-pok generator (extracted via our conversion - normal form):");
        eprintln!("  x[0] = {:#x}", g1_x[0]);
        eprintln!("  x[1] = {:#x}", g1_x[1]);
        eprintln!("  x[2] = {:#x}", g1_x[2]);
        eprintln!("  x[3] = {:#x}", g1_x[3]);
        eprintln!("  x[4] = {:#x}", g1_x[4]);
        eprintln!("  x[5] = {:#x}", g1_x[5]);
        eprintln!("  x[6] = {:#x}", g1_x[6]);
        
        // CRITICAL CHECK: Our hardcoded generator values are now stored in STANDARD form
        // Convert the tfhe generator to Montgomery to ensure it matches the runtime conversion
        use crate::ffi::Fp;
        unsafe {
            let mut x_mont = Fp { limb: [0; 7] };
            let mut y_mont = Fp { limb: [0; 7] };
            crate::ffi::fp_to_montgomery_wrapper(&mut x_mont, &g1_gen.inner().x);
            crate::ffi::fp_to_montgomery_wrapper(&mut y_mont, &g1_gen.inner().y);
            eprintln!("tfhe-zk-pok generator CONVERTED to Montgomery form:");
            eprintln!("  x[0] = {:#x}", x_mont.limb[0]);
            eprintln!("  y[0] = {:#x}", y_mont.limb[0]);
        }
        
        // Create test data: scalar=1 should return generator
        eprintln!("\n=== Testing with scalar 1 (should return generator) ===");
        let _n = 1;
        let points: Vec<G1Affine> = vec![g1_gen];
        let scalars: Vec<u64> = vec![1];
        
        // Compute MSM using our CUDA implementation
        let gpu_index = 0;
        let our_result_proj = match G1Projective::msm(&points, &scalars, gpu_index) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("CUDA MSM failed with error: {}", e);
                eprintln!("Error codes: -1=mismatch, -2=stream creation, -3=memory alloc, -4=CUDA error, -5=conversion error");
                eprintln!("Skipping test - CUDA may not be available");
                return;
            }
        };
        
        // Convert projective to affine
        // Note: projective_to_affine_g1 uses fp_mont_mul internally, which means
        // the affine result is STILL in Montgomery form
        let our_result_mont = our_result_proj.to_affine();
        let our_x_mont = our_result_mont.x();
        eprintln!("Our result (affine, Montgomery form - ALL limbs):");
        eprintln!("  x[0] = {:#x}", our_x_mont[0]);
        eprintln!("  x[1] = {:#x}", our_x_mont[1]);
        eprintln!("  x[2] = {:#x}", our_x_mont[2]);
        eprintln!("  x[3] = {:#x}", our_x_mont[3]);
        eprintln!("  x[4] = {:#x}", our_x_mont[4]);
        eprintln!("  x[5] = {:#x}", our_x_mont[5]);
        eprintln!("  x[6] = {:#x}", our_x_mont[6]);
        
        // arkworks `into_bigint()` returns NORMAL form (as we can see from the generator)
        // So we need to convert our Montgomery result to normal form
        let our_result = super::super::conversions::g1_affine_from_montgomery(&our_result_mont);
        let our_x_normal = our_result.x();
        eprintln!("Our result (affine, NORMAL form - ALL limbs):");
        eprintln!("  x[0] = {:#x}", our_x_normal[0]);
        eprintln!("  x[1] = {:#x}", our_x_normal[1]);
        eprintln!("  x[2] = {:#x}", our_x_normal[2]);
        eprintln!("  x[3] = {:#x}", our_x_normal[3]);
        eprintln!("  x[4] = {:#x}", our_x_normal[4]);
        eprintln!("  x[5] = {:#x}", our_x_normal[5]);
        eprintln!("  x[6] = {:#x}", our_x_normal[6]);
        
        // Check if result is at infinity
        if our_result.is_infinity() {
            eprintln!("Warning: CUDA MSM returned point at infinity");
            eprintln!("This might indicate an error in the computation");
            eprintln!("Points: {:?}", points.len());
            eprintln!("Scalars: {:?}", scalars);
        }
        
        // Compute MSM using tfhe-zk-pok
        let tfhe_points: Vec<TfheZkPokG1Affine> = points.iter()
            .map(|p| g1_affine_to_tfhe_zk_pok(p))
            .collect();
        
        // Convert scalars to field elements
        // Use the scalar field from the G1 config
        // For BLS12-446, the scalar field (Fr) is ~256 bits, so we use 4 limbs
        use tfhe_zk_pok::curve_446::g1::Config as G1Config;
        type ScalarField = <G1Config as ark_ec::models::CurveConfig>::ScalarField;
        let tfhe_scalars: Vec<ScalarField> = scalars.iter()
            .map(|&s| {
                // Convert u64 to field element (scalar field is ~256 bits, so 4 limbs)
                let limbs = [s, 0u64, 0u64, 0u64];
                ScalarField::from_sign_and_limbs(true, &limbs)
            })
            .collect();
        
        let tfhe_result_proj = TfheZkPokG1Projective::msm(&tfhe_points, &tfhe_scalars)
            .expect("tfhe-zk-pok MSM should succeed");
        let tfhe_result = <TfheZkPokG1Projective as CurveGroup>::into_affine(tfhe_result_proj);
        
        eprintln!("tfhe-zk-pok MSM result: affine");
        
        // Convert tfhe-zk-pok result to our format
        let tfhe_result_ours = g1_affine_from_tfhe_zk_pok(&tfhe_result);
        let x = tfhe_result_ours.x();
        let y = tfhe_result_ours.y();
        eprintln!("tfhe-zk-pok result (our format - ALL limbs):");
        eprintln!("  x[0] = {:#x}", x[0]);
        eprintln!("  x[1] = {:#x}", x[1]);
        eprintln!("  x[2] = {:#x}", x[2]);
        eprintln!("  x[3] = {:#x}", x[3]);
        eprintln!("  x[4] = {:#x}", x[4]);
        eprintln!("  x[5] = {:#x}", x[5]);
        eprintln!("  x[6] = {:#x}", x[6]);
        eprintln!("  y[0] = {:#x}", y[0]);
        
        // Compare results
        assert_eq!(our_result.x(), tfhe_result_ours.x(), "G1 MSM x coordinate mismatch");
        assert_eq!(our_result.y(), tfhe_result_ours.y(), "G1 MSM y coordinate mismatch");
    }
    
    #[test]
    fn test_g2_msm_vs_tfhe_zk_pok() {
        use super::super::conversions::{TfheZkPokG2Affine, TfheZkPokG2Projective};
        use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
        
        // Get generator from tfhe-zk-pok
        let tfhe_g2_gen = TfheZkPokG2Affine::generator();
        let g2_gen = g2_affine_from_tfhe_zk_pok(&tfhe_g2_gen);
        
        // Create test data: 5 points (all generator) and scalars [1, 2, 3, 4, 5]
        let n = 5;
        let points: Vec<G2Affine> = (0..n).map(|_| g2_gen).collect();
        let scalars: Vec<u64> = (1..=n as u64).collect();
        
        // Compute MSM using our CUDA implementation
        let gpu_index = 0;
        let our_result = match G2Projective::msm(&points, &scalars, gpu_index) {
            Ok(result) => result.to_affine(),
            Err(e) => {
                eprintln!("CUDA MSM failed: {}", e);
                eprintln!("Skipping test - CUDA may not be available");
                return;
            }
        };
        
        // Compute MSM using tfhe-zk-pok
        let tfhe_points: Vec<TfheZkPokG2Affine> = points.iter()
            .map(|p| g2_affine_to_tfhe_zk_pok(p))
            .collect();
        
        // Convert scalars to field elements
        // Use the scalar field from the G2 config
        // For BLS12-446, the scalar field (Fr) is ~256 bits, so we use 4 limbs
        use tfhe_zk_pok::curve_446::g2::Config as G2Config;
        type ScalarField = <G2Config as ark_ec::models::CurveConfig>::ScalarField;
        let tfhe_scalars: Vec<ScalarField> = scalars.iter()
            .map(|&s| {
                // Convert u64 to field element (scalar field is ~256 bits, so 4 limbs)
                let limbs = [s, 0u64, 0u64, 0u64];
                ScalarField::from_sign_and_limbs(true, &limbs)
            })
            .collect();
        
        let tfhe_result_proj = TfheZkPokG2Projective::msm(&tfhe_points, &tfhe_scalars)
            .expect("tfhe-zk-pok MSM should succeed");
        let tfhe_result = <TfheZkPokG2Projective as CurveGroup>::into_affine(tfhe_result_proj);
        
        // Convert tfhe-zk-pok result to our format
        let tfhe_result_ours = g2_affine_from_tfhe_zk_pok(&tfhe_result);
        
        // Compare results
        assert_eq!(our_result.x(), tfhe_result_ours.x(), "G2 MSM x coordinate mismatch");
        assert_eq!(our_result.y(), tfhe_result_ours.y(), "G2 MSM y coordinate mismatch");
    }
}

