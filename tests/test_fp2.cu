#include "fp2.h"
#include "fp2_kernels.h"
#include "fp.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>

// ============================================================================
// Test Utilities
// ============================================================================

namespace test_utils_fp2 {

// Helper to create Fp2 from Fp values
Fp2 make_fp2(const Fp& c0, const Fp& c1) {
    Fp2 result;
    fp_copy(result.c0, c0);
    fp_copy(result.c1, c1);
    return result;
}

// Helper to create Fp2 from small integers
Fp2 make_fp2_simple(uint64_t c0_val, uint64_t c1_val) {
    Fp2 result;
    fp_zero(result.c0);
    fp_zero(result.c1);
    result.c0.limb[0] = c0_val;
    result.c1.limb[0] = c1_val;
    return result;
}

// Generate a random Fp2 value
Fp2 random_fp2(std::mt19937_64& rng) {
    Fp2 result;
    Fp p;
    // Get modulus
    p.limb[0] = 0x311c0026aab0aaabULL;
    p.limb[1] = 0x56ee4528c573b5ccULL;
    p.limb[2] = 0x824e6dc3e23acdeeULL;
    p.limb[3] = 0x0f75a64bbac71602ULL;
    p.limb[4] = 0x0095a4b78a02fe32ULL;
    p.limb[5] = 0x200fc34965aad640ULL;
    p.limb[6] = 0x3cdee0fb28c5e535ULL;
    
    // Generate random Fp values for c0 and c1
    for (int i = 0; i < FP_LIMBS; i++) {
        result.c0.limb[i] = rng();
        result.c1.limb[i] = rng();
    }
    
    // Reduce if needed
    if (fp_cmp(result.c0, p) >= 0) {
        while (fp_cmp(result.c0, p) >= 0) {
            Fp reduced;
            fp_sub_raw(reduced, result.c0, p);
            fp_copy(result.c0, reduced);
        }
    }
    if (fp_cmp(result.c1, p) >= 0) {
        while (fp_cmp(result.c1, p) >= 0) {
            Fp reduced;
            fp_sub_raw(reduced, result.c1, p);
            fp_copy(result.c1, reduced);
        }
    }
    
    return result;
}

} // namespace test_utils_fp2

// ============================================================================
// Test Fixtures
// ============================================================================

// Base fixture for all Fp2 arithmetic tests
class Fp2ArithmeticTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Initialize device modulus once for all tests
        init_device_modulus();
    }
    
    // Common test values
    Fp2 zero, one, i_unit;  // 0, 1, i
    
    void SetUp() override {
        fp2_zero(zero);
        fp2_one(one);
        fp2_zero(i_unit);
        fp_one(i_unit.c1);  // i = 0 + 1*i
    }
};

// Fixture for property-based tests with random number generator
class Fp2PropertyTest : public Fp2ArithmeticTest {
protected:
    std::mt19937_64 rng;
    
    void SetUp() override {
        Fp2ArithmeticTest::SetUp();
        rng.seed(42);  // Fixed seed for reproducibility
    }
    
    Fp2 random_value() {
        return test_utils_fp2::random_fp2(rng);
    }
};

// Fixture for CUDA kernel tests
class Fp2CudaKernelTest : public Fp2ArithmeticTest {
protected:
    std::mt19937_64 rng;
    
    void SetUp() override {
        Fp2ArithmeticTest::SetUp();
        rng.seed(42);  // Fixed seed for reproducibility
    }
    
    // Helper to check CUDA errors
    void checkCudaError(cudaError_t err, const char* msg) {
        if (err != cudaSuccess) {
            FAIL() << msg << ": " << cudaGetErrorString(err);
        }
    }
};

// ============================================================================
// Basic Operation Tests
// ============================================================================

// Test basic addition
TEST_F(Fp2ArithmeticTest, Addition) {
    Fp2 a, b, c;
    
    // Test: (1 + 0*i) + (1 + 0*i) = (2 + 0*i)
    fp2_one(a);
    fp2_one(b);
    fp2_add(c, a, b);
    
    // Expected: (2 + 0*i)
    EXPECT_EQ(c.c0.limb[0], 2);
    EXPECT_TRUE(fp_is_zero(c.c1));
}

// Test subtraction
TEST_F(Fp2ArithmeticTest, Subtraction) {
    Fp2 a, b, c;
    
    // Test: (2 + 0*i) - (1 + 0*i) = (1 + 0*i)
    fp2_one(b);
    fp2_zero(c);
    c.c0.limb[0] = 2;
    
    fp2_sub(a, c, b);
    EXPECT_TRUE(fp2_is_one(a));
}

// Test multiplication
TEST_F(Fp2ArithmeticTest, Multiplication) {
    Fp2 a, b, result, expected;
    
    // Test: (1 + 1*i) * (1 + 1*i) = (0 + 2*i)
    // (1 + i) * (1 + i) = 1 + 2i + i^2 = 1 + 2i - 1 = 2i
    a = test_utils_fp2::make_fp2_simple(1, 1);
    b = test_utils_fp2::make_fp2_simple(1, 1);
    
    fp2_mul(result, a, b);
    
    // Expected: (0 + 2*i)
    fp2_zero(expected);
    expected.c1.limb[0] = 2;
    
    EXPECT_TRUE(fp_is_zero(result.c0));
    EXPECT_EQ(result.c1.limb[0], 2);
}

// Test i * i = -1
TEST_F(Fp2ArithmeticTest, I_Squared) {
    Fp2 i_val, result, expected;
    
    // i = 0 + 1*i
    fp2_zero(i_val);
    fp_one(i_val.c1);
    
    // i * i = -1
    fp2_mul(result, i_val, i_val);
    
    // Expected: -1 = (p-1) + 0*i
    fp2_one(expected);
    fp_neg(expected.c0, expected.c0);
    
    EXPECT_EQ(fp_cmp(result.c0, expected.c0), 0);
    EXPECT_TRUE(fp_is_zero(result.c1));
}

// Test negation
TEST_F(Fp2ArithmeticTest, Negation) {
    Fp2 a, neg_a, result;
    
    a = test_utils_fp2::make_fp2_simple(5, 3);
    
    fp2_neg(neg_a, a);
    fp2_add(result, a, neg_a);
    
    EXPECT_TRUE(fp2_is_zero(result));
}

// Test conjugation
TEST_F(Fp2ArithmeticTest, Conjugation) {
    Fp2 a, conj, result;
    
    a = test_utils_fp2::make_fp2_simple(5, 3);
    
    fp2_conjugate(conj, a);
    
    // conj should be (5 - 3*i)
    EXPECT_EQ(a.c0.limb[0], conj.c0.limb[0]);
    Fp neg_c1;
    fp_neg(neg_c1, a.c1);
    EXPECT_EQ(fp_cmp(conj.c1, neg_c1), 0);
    
    // a * conj should be real (norm)
    fp2_mul(result, a, conj);
    EXPECT_TRUE(fp_is_zero(result.c1));
}

// Test squaring
TEST_F(Fp2ArithmeticTest, Squaring) {
    Fp2 a, square;
    
    // Test: (1 + 1*i)^2 = 2*i
    a = test_utils_fp2::make_fp2_simple(1, 1);
    
    fp2_square(square, a);
    
    // Expected: (0 + 2*i)
    EXPECT_TRUE(fp_is_zero(square.c0));
    EXPECT_EQ(square.c1.limb[0], 2);
}

// Test zero and one
TEST_F(Fp2ArithmeticTest, ZeroAndOne) {
    Fp2 zero_val, one_val;
    
    fp2_zero(zero_val);
    fp2_one(one_val);
    
    EXPECT_TRUE(fp2_is_zero(zero_val));
    EXPECT_FALSE(fp2_is_zero(one_val));
    EXPECT_TRUE(fp2_is_one(one_val));
    EXPECT_FALSE(fp2_is_one(zero_val));
}

// Test copy
TEST_F(Fp2ArithmeticTest, Copy) {
    Fp2 a, b;
    
    a = test_utils_fp2::make_fp2_simple(42, 123);
    
    fp2_copy(b, a);
    
    EXPECT_EQ(fp_cmp(a.c0, b.c0), 0);
    EXPECT_EQ(fp_cmp(a.c1, b.c1), 0);
}

// Test conditional move
TEST_F(Fp2ArithmeticTest, ConditionalMove) {
    Fp2 a, b, result;
    
    a = test_utils_fp2::make_fp2_simple(10, 20);
    b = test_utils_fp2::make_fp2_simple(30, 40);
    
    // Test move when condition is true (1)
    fp2_copy(result, a);
    fp2_cmov(result, b, 1);
    EXPECT_EQ(fp_cmp(result.c0, b.c0), 0);
    EXPECT_EQ(fp_cmp(result.c1, b.c1), 0);
    
    // Test no move when condition is false (0)
    fp2_copy(result, a);
    fp2_cmov(result, b, 0);
    EXPECT_EQ(fp_cmp(result.c0, a.c0), 0);
    EXPECT_EQ(fp_cmp(result.c1, a.c1), 0);
}

// Test multiplication by zero
TEST_F(Fp2ArithmeticTest, MultiplicationByZero) {
    Fp2 a, zero_val, result;
    
    fp2_zero(zero_val);
    a = test_utils_fp2::make_fp2_simple(5, 3);
    
    fp2_mul(result, a, zero_val);
    EXPECT_TRUE(fp2_is_zero(result));
}

// Test multiplication by one
TEST_F(Fp2ArithmeticTest, MultiplicationByOne) {
    Fp2 a, one_val, result;
    
    fp2_one(one_val);
    a = test_utils_fp2::make_fp2_simple(5, 3);
    
    fp2_mul(result, a, one_val);
    EXPECT_EQ(fp_cmp(result.c0, a.c0), 0);
    EXPECT_EQ(fp_cmp(result.c1, a.c1), 0);
}

// Test inversion
TEST_F(Fp2ArithmeticTest, Inversion) {
    Fp2 a, a_inv, result;
    
    a = test_utils_fp2::make_fp2_simple(5, 3);
    
    fp2_inv(a_inv, a);
    fp2_mul(result, a, a_inv);
    
    // a * a^(-1) should equal 1
    EXPECT_TRUE(fp2_is_one(result));
}

// Test division
TEST_F(Fp2ArithmeticTest, Division) {
    Fp2 a, b, quotient, result;
    
    a = test_utils_fp2::make_fp2_simple(10, 6);
    b = test_utils_fp2::make_fp2_simple(5, 3);
    
    fp2_div(quotient, a, b);
    fp2_mul(result, quotient, b);
    
    // quotient * b should equal a
    EXPECT_EQ(fp_cmp(result.c0, a.c0), 0);
    EXPECT_EQ(fp_cmp(result.c1, a.c1), 0);
}

// Test multiply by i
TEST_F(Fp2ArithmeticTest, MultiplyByI) {
    Fp2 a, result;
    
    // Test: (a + b*i) * i = -b + a*i
    a = test_utils_fp2::make_fp2_simple(5, 3);
    
    fp2_mul_by_i(result, a);
    
    // Expected: (-3 + 5*i)
    Fp neg_three;
    fp_zero(neg_three);
    neg_three.limb[0] = 3;
    fp_neg(neg_three, neg_three);
    
    EXPECT_EQ(fp_cmp(result.c0, neg_three), 0);
    EXPECT_EQ(result.c1.limb[0], 5);
}

// Test Frobenius map
TEST_F(Fp2ArithmeticTest, Frobenius) {
    Fp2 a, frob, conj;
    
    a = test_utils_fp2::make_fp2_simple(5, 3);
    
    fp2_frobenius(frob, a);
    fp2_conjugate(conj, a);
    
    // Frobenius should equal conjugation for Fp2
    EXPECT_EQ(fp_cmp(frob.c0, conj.c0), 0);
    EXPECT_EQ(fp_cmp(frob.c1, conj.c1), 0);
}

// ============================================================================
// Property-Based Tests
// ============================================================================

// Test addition associativity: (a + b) + c = a + (b + c)
TEST_F(Fp2PropertyTest, AdditionAssociativity) {
    for (int i = 0; i < 100; i++) {
        Fp2 a = random_value();
        Fp2 b = random_value();
        Fp2 c = random_value();
        
        Fp2 result1, result2, temp;
        
        // (a + b) + c
        fp2_add(temp, a, b);
        fp2_add(result1, temp, c);
        
        // a + (b + c)
        fp2_add(temp, b, c);
        fp2_add(result2, a, temp);
        
        EXPECT_EQ(fp_cmp(result1.c0, result2.c0), 0) 
            << "Addition associativity failed: (a+b)+c != a+(b+c) (c0)";
        EXPECT_EQ(fp_cmp(result1.c1, result2.c1), 0) 
            << "Addition associativity failed: (a+b)+c != a+(b+c) (c1)";
    }
}

// Test multiplication associativity: (a * b) * c = a * (b * c)
TEST_F(Fp2PropertyTest, MultiplicationAssociativity) {
    for (int i = 0; i < 50; i++) {
        Fp2 a = random_value();
        Fp2 b = random_value();
        Fp2 c = random_value();
        
        Fp2 result1, result2, temp;
        
        // (a * b) * c
        fp2_mul(temp, a, b);
        fp2_mul(result1, temp, c);
        
        // a * (b * c)
        fp2_mul(temp, b, c);
        fp2_mul(result2, a, temp);
        
        EXPECT_EQ(fp_cmp(result1.c0, result2.c0), 0) 
            << "Multiplication associativity failed: (a*b)*c != a*(b*c) (c0)";
        EXPECT_EQ(fp_cmp(result1.c1, result2.c1), 0) 
            << "Multiplication associativity failed: (a*b)*c != a*(b*c) (c1)";
    }
}

// Test distributivity: a * (b + c) = a*b + a*c
TEST_F(Fp2PropertyTest, MultiplicationDistributivity) {
    for (int i = 0; i < 50; i++) {
        Fp2 a = random_value();
        Fp2 b = random_value();
        Fp2 c = random_value();
        
        Fp2 result1, result2, temp1, temp2;
        
        // a * (b + c)
        fp2_add(temp1, b, c);
        fp2_mul(result1, a, temp1);
        
        // a*b + a*c
        fp2_mul(temp1, a, b);
        fp2_mul(temp2, a, c);
        fp2_add(result2, temp1, temp2);
        
        EXPECT_EQ(fp_cmp(result1.c0, result2.c0), 0) 
            << "Distributivity failed: a*(b+c) != a*b + a*c (c0)";
        EXPECT_EQ(fp_cmp(result1.c1, result2.c1), 0) 
            << "Distributivity failed: a*(b+c) != a*b + a*c (c1)";
    }
}

// Test addition commutativity
TEST_F(Fp2PropertyTest, AdditionCommutativity) {
    for (int i = 0; i < 100; i++) {
        Fp2 a = random_value();
        Fp2 b = random_value();
        
        Fp2 result1, result2;
        fp2_add(result1, a, b);
        fp2_add(result2, b, a);
        
        EXPECT_EQ(fp_cmp(result1.c0, result2.c0), 0) 
            << "Addition commutativity failed: a+b != b+a (c0)";
        EXPECT_EQ(fp_cmp(result1.c1, result2.c1), 0) 
            << "Addition commutativity failed: a+b != b+a (c1)";
    }
}

// Test multiplication commutativity
TEST_F(Fp2PropertyTest, MultiplicationCommutativity) {
    for (int i = 0; i < 50; i++) {
        Fp2 a = random_value();
        Fp2 b = random_value();
        
        Fp2 result1, result2;
        fp2_mul(result1, a, b);
        fp2_mul(result2, b, a);
        
        EXPECT_EQ(fp_cmp(result1.c0, result2.c0), 0) 
            << "Multiplication commutativity failed: a*b != b*a (c0)";
        EXPECT_EQ(fp_cmp(result1.c1, result2.c1), 0) 
            << "Multiplication commutativity failed: a*b != b*a (c1)";
    }
}

// Test additive identity: a + 0 = a
TEST_F(Fp2PropertyTest, AdditiveIdentity) {
    for (int i = 0; i < 100; i++) {
        Fp2 a = random_value();
        Fp2 result;
        
        fp2_add(result, a, zero);
        EXPECT_EQ(fp_cmp(result.c0, a.c0), 0) << "Additive identity failed: a + 0 != a (c0)";
        EXPECT_EQ(fp_cmp(result.c1, a.c1), 0) << "Additive identity failed: a + 0 != a (c1)";
    }
}

// Test multiplicative identity: a * 1 = a
TEST_F(Fp2PropertyTest, MultiplicativeIdentity) {
    for (int i = 0; i < 100; i++) {
        Fp2 a = random_value();
        Fp2 result;
        
        fp2_mul(result, a, one);
        EXPECT_EQ(fp_cmp(result.c0, a.c0), 0) << "Multiplicative identity failed: a * 1 != a (c0)";
        EXPECT_EQ(fp_cmp(result.c1, a.c1), 0) << "Multiplicative identity failed: a * 1 != a (c1)";
    }
}

// Test additive inverse: a + (-a) = 0
TEST_F(Fp2PropertyTest, AdditiveInverse) {
    for (int i = 0; i < 100; i++) {
        Fp2 a = random_value();
        Fp2 neg_a, result;
        
        fp2_neg(neg_a, a);
        fp2_add(result, a, neg_a);
        
        EXPECT_TRUE(fp2_is_zero(result)) << "Additive inverse failed: a + (-a) != 0";
    }
}

// Test multiplicative inverse: a * a^(-1) = 1
TEST_F(Fp2PropertyTest, MultiplicativeInverse) {
    for (int i = 0; i < 50; i++) {
        Fp2 a = random_value();
        // Skip zero
        if (fp2_is_zero(a)) continue;
        
        Fp2 a_inv, result;
        
        fp2_inv(a_inv, a);
        fp2_mul(result, a, a_inv);
        
        EXPECT_TRUE(fp2_is_one(result)) << "Multiplicative inverse failed: a * a^(-1) != 1";
    }
}

// Test square vs multiply by self: a^2 = a * a
TEST_F(Fp2PropertyTest, SquareVsMultiply) {
    for (int i = 0; i < 50; i++) {
        Fp2 a = random_value();
        
        Fp2 square, multiply;
        fp2_square(square, a);
        fp2_mul(multiply, a, a);
        
        EXPECT_EQ(fp_cmp(square.c0, multiply.c0), 0) 
            << "Square vs multiply failed: a^2 != a*a (c0)";
        EXPECT_EQ(fp_cmp(square.c1, multiply.c1), 0) 
            << "Square vs multiply failed: a^2 != a*a (c1)";
    }
}

// ============================================================================
// CUDA Kernel Tests
// ============================================================================

// Test CUDA kernel: array addition
TEST_F(Fp2CudaKernelTest, CudaKernelArrayAdd) {
    const int n = 1000;
    Fp2* h_a = new Fp2[n];
    Fp2* h_b = new Fp2[n];
    Fp2* h_c = new Fp2[n];
    Fp2* h_expected = new Fp2[n];
    
    // Initialize with random values
    for (int i = 0; i < n; i++) {
        h_a[i] = test_utils_fp2::random_fp2(rng);
        h_b[i] = test_utils_fp2::random_fp2(rng);
        // Compute expected result on host
        fp2_add(h_expected[i], h_a[i], h_b[i]);
    }
    
    // Launch GPU kernel
    fp2_add_array_host(h_c, h_a, h_b, n);
    
    // Check CUDA errors
    cudaError_t err = cudaDeviceSynchronize();
    checkCudaError(err, "CUDA kernel execution failed");
    
    // Verify results match host computation
    for (int i = 0; i < n; i++) {
        EXPECT_EQ(fp_cmp(h_c[i].c0, h_expected[i].c0), 0) 
            << "GPU result mismatch at index " << i << " (c0)";
        EXPECT_EQ(fp_cmp(h_c[i].c1, h_expected[i].c1), 0) 
            << "GPU result mismatch at index " << i << " (c1)";
    }
    
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_expected;
}

// Test CUDA kernel: array multiplication
TEST_F(Fp2CudaKernelTest, CudaKernelArrayMul) {
    const int n = 1000;
    Fp2* h_a = new Fp2[n];
    Fp2* h_b = new Fp2[n];
    Fp2* h_c = new Fp2[n];
    Fp2* h_expected = new Fp2[n];
    
    // Initialize with random values
    for (int i = 0; i < n; i++) {
        h_a[i] = test_utils_fp2::random_fp2(rng);
        h_b[i] = test_utils_fp2::random_fp2(rng);
        // Compute expected result on host
        fp2_mul(h_expected[i], h_a[i], h_b[i]);
    }
    
    // Launch GPU kernel
    fp2_mul_array_host(h_c, h_a, h_b, n);
    
    // Check CUDA errors
    cudaError_t err = cudaDeviceSynchronize();
    checkCudaError(err, "CUDA kernel execution failed");
    
    // Verify results match host computation
    for (int i = 0; i < n; i++) {
        EXPECT_EQ(fp_cmp(h_c[i].c0, h_expected[i].c0), 0) 
            << "GPU result mismatch at index " << i << " (c0)";
        EXPECT_EQ(fp_cmp(h_c[i].c1, h_expected[i].c1), 0) 
            << "GPU result mismatch at index " << i << " (c1)";
    }
    
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_expected;
}

