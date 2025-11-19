# Rust API for Tower Field Arithmetic

This crate provides Rust bindings for the CUDA-based BLS12-446 curve operations, with compatibility for tfhe-zk-pok types.

## Overview

The Rust API exposes:
- **G1Affine** and **G1Projective**: G1 group points in affine and projective coordinates
- **G2Affine** and **G2Projective**: G2 group points in affine and projective coordinates
- Conversion functions to/from tfhe-zk-pok types

## Building

### Prerequisites

1. Build the C++/CUDA library first:
   ```bash
   cd ..
   mkdir -p build
   cd build
   cmake ..
   make
   ```

2. Ensure you have Rust installed (stable or nightly)

### Building the Rust API

```bash
cd rust-api
cargo build
```

For release builds:
```bash
cargo build --release
```

## Usage

### Basic Usage

```rust
use tower_field_arithmetic::{G1Affine, G1Projective, G2Affine, G2Projective};

// Create a G1 affine point at infinity
let g1_inf = G1Affine::infinity();

// Create a G1 affine point from coordinates
let g1_point = G1Affine::new(
    [0x1234, 0, 0, 0, 0, 0, 0],  // x coordinate (7 limbs)
    [0x5678, 0, 0, 0, 0, 0, 0],  // y coordinate (7 limbs)
    false  // not at infinity
);

// Convert to projective coordinates
let g1_proj = g1_point.to_projective();

// Convert back to affine
let g1_affine_again = g1_proj.to_affine();

// Similar for G2 points
let g2_point = G2Affine::new(
    ([0x1234, 0, 0, 0, 0, 0, 0], [0x5678, 0, 0, 0, 0, 0, 0]),  // x = (c0, c1)
    ([0x9abc, 0, 0, 0, 0, 0, 0], [0xdef0, 0, 0, 0, 0, 0, 0]),  // y = (c0, c1)
    false
);
```

### Integration with tfhe-zk-pok

The API provides conversion functions to/from tfhe-zk-pok types. Once you have tfhe-zk-pok available, you can use:

```rust
use tower_field_arithmetic::{
    G1Affine, G2Affine,
    g1_affine_to_tfhe_zk_pok, g1_affine_from_tfhe_zk_pok,
    g2_affine_to_tfhe_zk_pok, g2_affine_from_tfhe_zk_pok,
};

// Convert our G1Affine to tfhe-zk-pok's G1Affine
let our_point = G1Affine::new(/* ... */);
let tfhe_point = g1_affine_to_tfhe_zk_pok(&our_point);

// Convert back
let our_point_again = g1_affine_from_tfhe_zk_pok(&tfhe_point);
```

### Updating for Actual tfhe-zk-pok Types

When tfhe-zk-pok is available, update `src/conversions.rs` to replace the placeholder types:

```rust
// Replace this:
pub struct TfheZkPokG1Affine { ... }

// With this:
pub use tfhe_zk_pok::curve_446::G1Affine as TfheZkPokG1Affine;
```

Then update the conversion implementations to work with the actual types.

## Type Structure

### Fp (Field Element)
- 7 limbs of 64 bits each (446-bit prime field)
- Little-endian: `limb[0]` is the least significant word

### G1Affine
- `x: [u64; 7]` - x coordinate in Fp
- `y: [u64; 7]` - y coordinate in Fp
- `infinity: bool` - true if point at infinity

### G2Affine
- `x: ([u64; 7], [u64; 7])` - x coordinate in Fp2 (c0, c1)
- `y: ([u64; 7], [u64; 7])` - y coordinate in Fp2 (c0, c1)
- `infinity: bool` - true if point at infinity

### G1Projective
- `X: [u64; 7]` - X coordinate in Fp
- `Y: [u64; 7]` - Y coordinate in Fp
- `Z: [u64; 7]` - Z coordinate in Fp
- Represents affine point (X/Z, Y/Z)

### G2Projective
- `X: ([u64; 7], [u64; 7])` - X coordinate in Fp2
- `Y: ([u64; 7], [u64; 7])` - Y coordinate in Fp2
- `Z: ([u64; 7], [u64; 7])` - Z coordinate in Fp2
- Represents affine point (X/Z, Y/Z)

## Notes

- All coordinates are stored in little-endian format
- The infinity flag is separate from coordinate values
- Projective coordinates use Z=0 to represent infinity
- Conversions between affine and projective are provided for both G1 and G2

## License

Same as the parent project.

