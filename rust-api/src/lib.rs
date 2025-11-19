//! Rust API for tower field arithmetic
//! 
//! This crate provides Rust bindings for the CUDA-based BLS12-446 curve operations,
//! with compatibility for tfhe-zk-pok types.
//! 
//! ## Overview
//! 
//! The API exposes G1 and G2 points in both affine and projective coordinates,
//! with conversion functions to/from tfhe-zk-pok types.
//! 
//! ## Example
//! 
//! ```rust,no_run
//! use tower_field_arithmetic::{G1Affine, G1Projective};
//! 
//! // Create a G1 affine point
//! let g1_affine = G1Affine::new(
//!     [0x1234, 0, 0, 0, 0, 0, 0],  // x coordinate
//!     [0x5678, 0, 0, 0, 0, 0, 0],  // y coordinate
//!     false  // not at infinity
//! );
//! 
//! // Convert to projective coordinates
//! let g1_proj = g1_affine.to_projective();
//! 
//! // Convert back to affine
//! let g1_affine_again = g1_proj.to_affine();
//! ```

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub mod ffi;
pub mod types;
pub mod conversions;

#[cfg(test)]
mod example;

#[cfg(test)]
pub mod debug_msm;

pub use types::*;
pub use conversions::*;

