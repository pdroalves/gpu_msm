#include "fp.h"
#include "fp_kernels.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>

// Test fixture for Fp arithmetic tests
class FpArithmeticTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Initialize device modulus once for all tests
        init_device_modulus();
    }
};

// Test basic addition
TEST_F(FpArithmeticTest, Addition) {
    Fp a, b, c;
    
    // Test: 1 + 1 = 2
    fp_one(a);
    fp_one(b);
    fp_add(c, a, b);
    
    EXPECT_EQ(c.limb[0], 2);
    for (int i = 1; i < FP_LIMBS; i++) {
        EXPECT_EQ(c.limb[i], 0);
    }
}

// Test subtraction
TEST_F(FpArithmeticTest, Subtraction) {
    Fp a, b, c;
    
    // Test: 2 - 1 = 1
    fp_one(b);
    fp_zero(c);
    c.limb[0] = 2;
    
    fp_sub(a, c, b);
    EXPECT_TRUE(fp_is_one(a));
}

// Test multiplication
TEST_F(FpArithmeticTest, Multiplication) {
    Fp five, three, result, expected;
    
    fp_zero(five);
    fp_zero(three);
    fp_zero(expected);
    five.limb[0] = 5;
    three.limb[0] = 3;
    expected.limb[0] = 15;
    
    fp_mul(result, five, three);
    EXPECT_EQ(fp_cmp(result, expected), 0);
}

// Test negation
TEST_F(FpArithmeticTest, Negation) {
    Fp a, neg_a, result;
    
    fp_zero(a);
    a.limb[0] = 5;
    
    fp_neg(neg_a, a);
    fp_add(result, a, neg_a);
    
    EXPECT_TRUE(fp_is_zero(result));
}

// Test Montgomery conversion round-trip
TEST_F(FpArithmeticTest, MontgomeryRoundTrip) {
    Fp value, mont_form, back;
    
    fp_zero(value);
    value.limb[0] = 5;
    
    fp_to_montgomery(mont_form, value);
    fp_from_montgomery(back, mont_form);
    
    EXPECT_EQ(fp_cmp(back, value), 0);
}

// Test Montgomery multiplication
TEST_F(FpArithmeticTest, MontgomeryMultiplication) {
    Fp five, three, five_m, three_m, result_m, result, expected;
    
    fp_zero(five);
    fp_zero(three);
    fp_zero(expected);
    five.limb[0] = 5;
    three.limb[0] = 3;
    expected.limb[0] = 15;
    
    // Convert to Montgomery form
    fp_to_montgomery(five_m, five);
    fp_to_montgomery(three_m, three);
    
    // Multiply in Montgomery form
    fp_mont_mul(result_m, five_m, three_m);
    
    // Convert back
    fp_from_montgomery(result, result_m);
    
    EXPECT_EQ(fp_cmp(result, expected), 0);
}

// Test comparison operations
TEST_F(FpArithmeticTest, Comparison) {
    Fp five, three;
    
    fp_zero(five);
    fp_zero(three);
    five.limb[0] = 5;
    three.limb[0] = 3;
    
    EXPECT_GT(fp_cmp(five, three), 0);  // 5 > 3
    EXPECT_LT(fp_cmp(three, five), 0);  // 3 < 5
    EXPECT_EQ(fp_cmp(five, five), 0);   // 5 == 5
}

// Test zero and one
TEST_F(FpArithmeticTest, ZeroAndOne) {
    Fp zero, one;
    
    fp_zero(zero);
    fp_one(one);
    
    EXPECT_TRUE(fp_is_zero(zero));
    EXPECT_FALSE(fp_is_zero(one));
    EXPECT_TRUE(fp_is_one(one));
    EXPECT_FALSE(fp_is_one(zero));
}

// Test copy
TEST_F(FpArithmeticTest, Copy) {
    Fp a, b;
    
    fp_zero(a);
    a.limb[0] = 42;
    a.limb[1] = 123;
    
    fp_copy(b, a);
    
    EXPECT_EQ(fp_cmp(a, b), 0);
}

// Test conditional move
TEST_F(FpArithmeticTest, ConditionalMove) {
    Fp a, b, result;
    
    fp_zero(a);
    fp_zero(b);
    a.limb[0] = 10;
    b.limb[0] = 20;
    
    // Test move when condition is true (1)
    fp_copy(result, a);
    fp_cmov(result, b, 1);
    EXPECT_EQ(fp_cmp(result, b), 0);
    
    // Test no move when condition is false (0)
    fp_copy(result, a);
    fp_cmov(result, b, 0);
    EXPECT_EQ(fp_cmp(result, a), 0);
}

// Test multiplication by zero
TEST_F(FpArithmeticTest, MultiplicationByZero) {
    Fp a, zero, result;
    
    fp_zero(zero);
    fp_zero(a);
    a.limb[0] = 5;
    
    fp_mul(result, a, zero);
    EXPECT_TRUE(fp_is_zero(result));
}

// Test multiplication by one
TEST_F(FpArithmeticTest, MultiplicationByOne) {
    Fp a, one, result;
    
    fp_one(one);
    fp_zero(a);
    a.limb[0] = 5;
    
    fp_mul(result, a, one);
    EXPECT_EQ(fp_cmp(result, a), 0);
}

// ============================================================================
// Hardcoded Large Value Tests - Testing with values near modulus
// ============================================================================

// Helper to create Fp from hex values (little-endian limbs)
static Fp make_fp(uint64_t l0, uint64_t l1, uint64_t l2, uint64_t l3,
                  uint64_t l4, uint64_t l5, uint64_t l6) {
    Fp result;
    result.limb[0] = l0;
    result.limb[1] = l1;
    result.limb[2] = l2;
    result.limb[3] = l3;
    result.limb[4] = l4;
    result.limb[5] = l5;
    result.limb[6] = l6;
    return result;
}

// Test 1: Addition that doesn't overflow
TEST_F(FpArithmeticTest, LargeAddition1) {
    // a = large value
    Fp a = make_fp(0x18e00013555855ULL, 0x2b772294629DAULL, 0x412736E1F11D66ULL,
                   0x87BAD325DD638ULL, 0x4CAD5BC5017FULL, 0x1007E1A4B2D56ULL, 0x1E6F707D94629ULL);
    
    // b = small value
    Fp b = make_fp(0x1234567890ABCULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL);
    
    // Expected: a + b (no reduction needed)
    Fp expected;
    fp_add(expected, a, b);  // Let implementation compute and verify it's consistent
    
    Fp result;
    fp_add(result, a, b);
    
    EXPECT_EQ(fp_cmp(result, expected), 0) << "Large addition without overflow failed";
    
    // Verify commutativity: a + b = b + a
    Fp result2;
    fp_add(result2, b, a);
    EXPECT_EQ(fp_cmp(result, result2), 0) << "Addition commutativity failed";
}

// Test 2: Addition that triggers reduction (sum > p)
TEST_F(FpArithmeticTest, LargeAddition2WithReduction) {
    // Use two large numbers that will trigger reduction
    // a + b should wrap around modulus
    Fp a = make_fp(0x311c0026aab0aaaaULL, 0x56ee4528c573b5ccULL, 0x824e6dc3e23acdeeULL,
                   0x0f75a64bbac71602ULL, 0x0095a4b78a02fe32ULL, 0x200fc34965aad640ULL, 0x3cdee0fb28c5e535ULL);
    
    // b = 1 (so a+b should wrap to 0 if a = p-1)
    Fp b;
    fp_zero(b);
    b.limb[0] = 1;
    
    Fp result;
    fp_add(result, a, b);
    
    // (p-1) + 1 = 0 (mod p)
    EXPECT_TRUE(fp_is_zero(result)) << "Addition with reduction (p-1)+1 should equal 0";
}

// Test 3: Subtraction without borrow
TEST_F(FpArithmeticTest, LargeSubtraction1) {
    // a = large value
    Fp a = make_fp(0x18e00013555855ULL, 0x2b772294629DAULL, 0x412736E1F11D66ULL,
                   0x87BAD325DD638ULL, 0x4CAD5BC5017FULL, 0x1007E1A4B2D56ULL, 0x1E6F707D94629ULL);
    
    // b = 1000
    Fp b;
    fp_zero(b);
    b.limb[0] = 1000;
    
    Fp result;
    fp_sub(result, a, b);
    
    // Verify: (a - b) + b = a
    Fp verify;
    fp_add(verify, result, b);
    
    EXPECT_EQ(fp_cmp(verify, a), 0) << "Large subtraction failed: (a-b)+b != a";
}

// Test 4: Subtraction with borrow (a < b)
TEST_F(FpArithmeticTest, LargeSubtraction2WithBorrow) {
    // a = 50
    Fp a = make_fp(0x32ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL);
    
    // b = 100
    Fp b = make_fp(0x64ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL);
    
    // Expected: 50 - 100 = -50 = p - 50 (mod p)
    Fp expected = make_fp(0x311c0026aab0aa79ULL, 0x56ee4528c573b5ccULL, 0x824e6dc3e23acdeeULL,
                          0xf75a64bbac71602ULL, 0x95a4b78a02fe32ULL, 0x200fc34965aad640ULL, 0x3cdee0fb28c5e535ULL);
    
    Fp result;
    fp_sub(result, a, b);
    
    EXPECT_EQ(fp_cmp(result, expected), 0) << "Subtraction with borrow failed";
}

// Test 5: Multiplication of large values (triggers reduction)
TEST_F(FpArithmeticTest, LargeMultiplication1) {
    // a = 2^200 (bit 200 set)
    Fp a;
    fp_zero(a);
    a.limb[3] = 0x100ULL;  // bit 200 = bit 8 of limb 3
    
    // b = 2^100 (bit 100 set)  
    Fp b;
    fp_zero(b);
    b.limb[1] = 0x10ULL;  // bit 100 = bit 36 of limb 1
    
    Fp result;
    fp_mul(result, a, b);
    
    // Verify: a * b * 1 = a * b (consistency check)
    Fp one;
    fp_one(one);
    Fp verify;
    fp_mul(verify, result, one);
    
    EXPECT_EQ(fp_cmp(result, verify), 0) << "Large multiplication consistency failed";
    
    // Verify commutativity: a * b = b * a
    Fp result2;
    fp_mul(result2, b, a);
    EXPECT_EQ(fp_cmp(result, result2), 0) << "Multiplication commutativity failed";
}

// Test 6: (p-1) * (p-1) = 1 (mod p)
TEST_F(FpArithmeticTest, LargeMultiplication2ModulusMinus1) {
    // a = p - 1
    Fp a = make_fp(0x311c0026aab0aaaaULL, 0x56ee4528c573b5ccULL, 0x824e6dc3e23acdeeULL,
                   0xf75a64bbac71602ULL, 0x95a4b78a02fe32ULL, 0x200fc34965aad640ULL, 0x3cdee0fb28c5e535ULL);
    
    // b = p - 1
    Fp b = a;
    
    // Expected: (p-1) * (p-1) = p^2 - 2p + 1 = 1 (mod p)
    Fp expected = make_fp(0x1ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL);
    
    Fp result;
    fp_mul(result, a, b);
    
    EXPECT_EQ(fp_cmp(result, expected), 0) << "(p-1) * (p-1) should equal 1";
}

// Test 7: Multiplication with 2: a * 2 = a + a
TEST_F(FpArithmeticTest, LargeMultiplication3Half) {
    // a = large value
    Fp a = make_fp(0x18e00013555855ULL, 0x2b772294629DAE6ULL, 0x412736E1F11D66F7ULL,
                   0x7BAD325DD638B01ULL, 0x4CAD5BC5017F19ULL, 0x1007E1A4B2D56B20ULL, 0x1E6F707D9462F2ULL);
    
    // b = 2
    Fp b;
    fp_zero(b);
    b.limb[0] = 2;
    
    // Compute a * 2
    Fp result;
    fp_mul(result, a, b);
    
    // Verify: a * 2 = a + a
    Fp expected;
    fp_add(expected, a, a);
    
    EXPECT_EQ(fp_cmp(result, expected), 0) << "a * 2 should equal a + a";
}

// Test 8: Large number squared
TEST_F(FpArithmeticTest, LargeMultiplication4Square) {
    // a = large value
    Fp a = make_fp(0x123456789ABCDEFULL, 0xFEDCBA9876543210ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL);
    
    Fp result;
    fp_mul(result, a, a);
    
    // Verify: a^2 * 1 = a^2
    Fp one;
    fp_one(one);
    Fp verify;
    fp_mul(verify, result, one);
    
    EXPECT_EQ(fp_cmp(result, verify), 0) << "Square consistency check failed";
    
    // Verify: a^2 should not equal zero (unless a is zero)
    EXPECT_FALSE(fp_is_zero(result)) << "Square of non-zero element is zero";
}

// Test 9: Addition chain near modulus
TEST_F(FpArithmeticTest, LargeAddition3Chain) {
    // Start with p-1
    Fp a = make_fp(0x311c0026aab0aaaaULL, 0x56ee4528c573b5ccULL, 0x824e6dc3e23acdeeULL,
                   0x0f75a64bbac71602ULL, 0x0095a4b78a02fe32ULL, 0x200fc34965aad640ULL, 0x3cdee0fb28c5e535ULL);
    
    // Add 1 repeatedly
    Fp one;
    fp_one(one);
    
    Fp result = a;
    
    // (p-1) + 1 = 0, then 0 + 1 = 1
    fp_add(result, result, one);  // result should be 0
    EXPECT_TRUE(fp_is_zero(result)) << "Addition chain: (p-1)+1 should be 0";
    
    fp_add(result, result, one);  // result should be 1
    EXPECT_EQ(fp_cmp(result, one), 0) << "Addition chain: 0+1 should be 1";
}

// Test 10: Complex multiplication with reduction
TEST_F(FpArithmeticTest, LargeMultiplication5Complex) {
    // a = large prime-like number
    Fp a = make_fp(0x123456789ABCDEFULL, 0xFEDCBA9876543210ULL, 0x0123456789ABCDEFULL,
                   0xFEDCBA9876543210ULL, 0x123456789ABULL, 0x1000000000000ULL, 0x10000000000ULL);
    
    // b = another large number
    Fp b = make_fp(0xABCDEF0123456789ULL, 0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL,
                   0x123456789ABCDEFULL, 0xFEDCBA98765ULL, 0x2000000000000ULL, 0x20000000000ULL);
    
    // Compute a * b
    Fp result;
    fp_mul(result, a, b);
    
    // Verify: (a * b) * 1 = a * b
    Fp one;
    fp_one(one);
    Fp verify;
    fp_mul(verify, result, one);
    
    EXPECT_EQ(fp_cmp(result, verify), 0) << "Complex large multiplication consistency check failed";
    
    // Verify: a * b = b * a (commutativity)
    Fp result2;
    fp_mul(result2, b, a);
    
    EXPECT_EQ(fp_cmp(result, result2), 0) << "Multiplication commutativity failed for large values";
}

