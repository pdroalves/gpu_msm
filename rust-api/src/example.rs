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
}

