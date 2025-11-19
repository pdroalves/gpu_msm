# Fp Arithmetic for BLS12-446 on CUDA

This project implements multi-precision finite field arithmetic for the BLS12-446 elliptic curve on CUDA.

## Overview

BLS12-446 uses a 446-bit prime field, which requires multi-precision arithmetic since it exceeds the 64-bit native integer size. This implementation uses 7 limbs of 64 bits each (448 bits total, with 2 bits of headroom).

## Structure

```
.
├── include/           # Header files
│   ├── fp.h          # Fp struct and function declarations
│   └── fp_kernels.h  # CUDA kernel declarations
├── src/              # Implementation files
│   ├── fp.cu         # Multi-precision arithmetic operations
│   └── fp_kernels.cu # CUDA kernel implementations
└── tests/            # Test programs
    └── test_fp.cu    # Basic test suite
```

## Fp Structure

```cpp
struct Fp {
    uint64_t limb[7];  // Little-endian: limb[0] is LSB
};
```

## Operations

All operations are CUDA-compatible (can be called from host or device code):

- **Comparison**: `fp_cmp()`, `fp_is_zero()`, `fp_is_one()`
- **Basic arithmetic**: `fp_add()`, `fp_sub()`, `fp_mul()`, `fp_neg()`
- **Raw operations** (without reduction): `fp_add_raw()`, `fp_sub_raw()`, `fp_mul_raw()`
- **Modular reduction**: `fp_reduce()`
- **Utility**: `fp_zero()`, `fp_one()`, `fp_copy()`, `fp_cmov()`

## Multi-Precision Arithmetic

### Addition/Subtraction
- Uses standard carry/borrow propagation
- Handles overflow/underflow correctly
- Modular reduction ensures result < p

### Multiplication
- Uses schoolbook method (O(n²) for n limbs)
- 64×64 → 128-bit multiplication using CUDA intrinsics (`__umul64hi()`) on device or `__uint128_t` on host
- Result stored in double-width (14 limbs)
- Then reduced modulo p using Montgomery reduction

### Modular Reduction
- **Montgomery Reduction**: Implemented with R = 2^448 (matching tfhe-rs)
- Uses precomputed constants: R^2 mod p, R_INV mod p, and p' = -p^(-1) mod 2^64
- All multiplications use Montgomery form internally for efficiency
- Conversion functions: `fp_to_montgomery()` and `fp_from_montgomery()`
- Direct Montgomery multiplication: `fp_mont_mul()` (for values already in Montgomery form)

## Modulus

The BLS12-446 Fq modulus is already set from the tfhe-rs reference:
- **Modulus**: `172824703542857155980071276579495962243492693522789898437834836356385656662277472896902502740297183690175962001546428467344062165330603`
- **Source**: [tfhe-rs/tfhe-zk-pok/src/curve_446/mod.rs](https://github.com/zama-ai/tfhe-rs/blob/main/tfhe-zk-pok/src/curve_446/mod.rs)

The modulus and curve constants are automatically initialized for both host and device code (hardcoded at compile time). Call `init_device_generators()` once before using generator points in CUDA kernels.

## Building and Testing

### Build

```bash
mkdir build
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75  # Adjust for your GPU
cmake --build .
```

### Run Tests

```bash
# Run all tests
./test_fp

# List all available tests
./test_fp --gtest_list_tests

# Run specific test(s) with filter
./test_fp --gtest_filter="*Montgomery*"

# Or use CTest for detailed output
ctest --output-on-failure

# Run with verbose output
ctest --verbose

# Run specific test via CTest
ctest -R Montgomery
```

To adjust the CUDA architecture, set `CMAKE_CUDA_ARCHITECTURES`:
- `60`: Pascal (GTX 10xx)
- `70`: Volta (V100)
- `75`: Turing (RTX 20xx, GTX 16xx)
- `80`: Ampere (A100, RTX 30xx)
- `86`: Ampere (RTX 30xx consumer)
- `89`: Ada Lovelace (RTX 40xx)
- `90`: Hopper (H100)

Or edit `CMakeLists.txt` to change the default.

## Tests

The project uses Google Test framework with comprehensive test coverage:

### Test Suite (`test_fp`)

The test suite includes 22 comprehensive tests covering:

**Basic Operations:**
- **Addition**: Basic addition, carry propagation, and commutativity
- **Subtraction**: Basic subtraction, borrow propagation, and inverse operations
- **Multiplication**: Multi-precision multiplication with Montgomery reduction
- **Negation**: Modular negation (a + (-a) = 0)

**Montgomery Form:**
- **Montgomery Round-Trip**: Conversion to/from Montgomery form
- **Montgomery Multiplication**: Direct multiplication in Montgomery form

**Utility Functions:**
- **Comparison**: Element comparison operations (`fp_cmp`, `fp_is_zero`, `fp_is_one`)
- **Zero and One**: Special element initialization and verification
- **Copy**: Element copying
- **Conditional Move**: Constant-time conditional move

**Edge Cases:**
- **Multiplication by Zero**: Verifies a × 0 = 0
- **Multiplication by One**: Verifies a × 1 = a

**Large Value Tests (10 hardcoded tests with values near modulus):**
1. **LargeAddition1**: Addition without overflow (commutativity check)
2. **LargeAddition2WithReduction**: (p-1) + 1 = 0 (modular wrapping)
3. **LargeSubtraction1**: Subtraction verification via (a-b)+b = a
4. **LargeSubtraction2WithBorrow**: Subtraction with borrow (50 - 100 = p - 50)
5. **LargeMultiplication1**: 2^200 × 2^100 (consistency and commutativity)
6. **LargeMultiplication2ModulusMinus1**: (p-1) × (p-1) = 1
7. **LargeMultiplication3Half**: Verify a × 2 = a + a
8. **LargeMultiplication4Square**: Large number squared (consistency check)
9. **LargeAddition3Chain**: Addition chain: (p-1)+1=0, 0+1=1
10. **LargeMultiplication5Complex**: Complex multiplication with large random-like values

All tests pass and are automatically discovered by CTest.

## Performance Considerations

1. **Montgomery Reduction**: Currently implemented with R = 2^448
   - Efficient for repeated multiplications
   - Uses CUDA intrinsics for 64×64→128 bit multiplication
   - Consider exploiting BLS12 prime structure for further optimization

2. **Multiplication**: Current implementation uses schoolbook method
   - Consider Karatsuba for better asymptotic complexity (O(n^1.58) vs O(n²))
   - Could exploit special structure of BLS12-446 prime

3. **Memory**: 
   - Use shared memory for frequently accessed values in kernels
   - Consider warp-level primitives for parallel operations
   - Constant memory already used for modulus and Montgomery constants

## Next Steps

1. Add modular inversion (using extended Euclidean or Fermat's little theorem)
2. Add square root (for point decompression)
3. Implement elliptic curve point operations (addition, doubling, scalar multiplication)
4. Add comprehensive test suite with edge cases
5. Benchmark and optimize for specific GPU architectures
6. Consider Karatsuba multiplication for better performance

## References

- BLS12 curves: [Pairing-Friendly Curves](https://eprint.iacr.org/2006/372.pdf)
- Multi-precision arithmetic: [Handbook of Applied Cryptography](https://cacr.uwaterloo.ca/hac/)
- CUDA optimization: [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

