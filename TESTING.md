# Testing Strategy

## Overview

This document describes the testing approach for the BLS12-446 Fp arithmetic implementation.

## Test Framework

The project uses **Google Test** for all unit testing, integrated via CMake with automatic test discovery.

## Test Suite

The test suite consists of **22 comprehensive tests** organized into categories:

### 1. Basic Arithmetic Operations (4 tests)
- **Addition**: Tests basic addition and commutativity
- **Subtraction**: Tests basic subtraction and inverse operations  
- **Multiplication**: Tests Montgomery-based multiplication
- **Negation**: Tests modular negation (a + (-a) = 0)

### 2. Montgomery Form Operations (2 tests)
- **MontgomeryRoundTrip**: Tests conversion to/from Montgomery form
- **MontgomeryMultiplication**: Tests direct multiplication in Montgomery form

### 3. Utility Functions (5 tests)
- **Comparison**: Tests `fp_cmp`, `fp_is_zero`, `fp_is_one`
- **ZeroAndOne**: Tests initialization of special elements
- **Copy**: Tests element copying
- **ConditionalMove**: Tests constant-time conditional move
- **MultiplicationByZero/One**: Tests edge cases

### 4. Large Value Tests (10 tests)

These tests use hardcoded large values near the modulus to verify correct behavior with:
- **Modular reduction** (when sum/product exceeds p)
- **Carry propagation** across all limbs
- **Borrow handling** in subtraction
- **Edge cases** like (p-1) operations

Specific test cases:

1. **LargeAddition1**: Large value addition with commutativity check
2. **LargeAddition2WithReduction**: Tests (p-1) + 1 = 0 (wrapping)
3. **LargeSubtraction1**: Verifies (a-b)+b = a for large values
4. **LargeSubtraction2WithBorrow**: Tests 50 - 100 = p - 50
5. **LargeMultiplication1**: Tests 2^200 × 2^100 with commutativity
6. **LargeMultiplication2ModulusMinus1**: Tests (p-1) × (p-1) = 1
7. **LargeMultiplication3Half**: Verifies a × 2 = a + a
8. **LargeMultiplication4Square**: Tests squaring of large values
9. **LargeAddition3Chain**: Tests (p-1)+1=0, then 0+1=1
10. **LargeMultiplication5Complex**: Complex multiplication with pseudo-random values

## Why Hardcoded Tests?

Initially, we attempted to use the RELIC cryptographic library for comparison testing. However, we discovered that RELIC's default 446-bit prime differs from the BLS12-446 prime used in this implementation. This made direct value comparisons impossible.

Instead, we use **hardcoded test vectors** that:
- Verify specific known values (like (p-1) × (p-1) = 1)
- Test algebraic properties (commutativity, distributivity)
- Ensure modular reduction works correctly near the modulus boundary
- Cover edge cases that might not appear in random testing

## Running Tests

```bash
cd build

# Run all tests with detailed output
./test_fp

# Run with verbose Google Test output
./test_fp --gtest_color=yes

# Run specific test(s)
./test_fp --gtest_filter="*Montgomery*"

# List all tests
./test_fp --gtest_list_tests

# Use CTest (CMake's test runner)
ctest --output-on-failure
```

## Test Results

All 22 tests pass consistently:

```
[==========] Running 22 tests from 1 test suite.
[----------] 22 tests from FpArithmeticTest
...
[  PASSED  ] 22 tests.
```

## Coverage

The test suite provides comprehensive coverage of:
- ✅ All public API functions
- ✅ Basic arithmetic (add, sub, mul, neg)
- ✅ Montgomery form conversions and operations
- ✅ Edge cases (zero, one, p-1)
- ✅ Carry/borrow propagation across limb boundaries
- ✅ Modular reduction when results exceed p
- ✅ Algebraic properties (commutativity, identity elements)

## Future Improvements

Potential enhancements to the test suite:
1. **Randomized testing**: Generate random test cases (ensuring they're < p)
2. **Property-based testing**: More extensive algebraic property checks
3. **Performance benchmarks**: Measure operations/second
4. **CUDA kernel tests**: Verify device code execution
5. **Fuzz testing**: Test with malformed inputs for robustness

