#include "curve.h"
#include "fp.h"
#include "fp2.h"
#include "device.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <string>

// Test fixture for MSM tests
class MSMTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA
        if (!cuda_is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
        
        gpu_index = 0;
        stream = cuda_create_stream(gpu_index);
        
        // Initialize device modulus and curve
        init_device_modulus(stream, gpu_index);
        init_device_curve(stream, gpu_index);
        init_device_generators(stream, gpu_index);
        
        cuda_synchronize_stream(stream, gpu_index);
    }
    
    void TearDown() override {
        if (stream != nullptr) {
            cuda_destroy_stream(stream, gpu_index);
        }
    }
    
    uint32_t gpu_index;
    cudaStream_t stream;
};

// Helper function to compute N * (N+1) / 2
uint64_t triangular_number(uint64_t N) {
    if (N % 2 == 0) {
        return (N / 2) * (N + 1);
    } else {
        return N * ((N + 1) / 2);
    }
}

// Helper function to convert Fp to decimal string
// Converts from Montgomery form to normal form, then to decimal string
std::string fp_to_decimal_string(const Fp& val_montgomery) {
    // Convert from Montgomery form to normal form
    Fp val_normal;
    fp_from_montgomery(val_normal, val_montgomery);
    
    // Convert big integer to decimal string using repeated division by 10
    std::string result;
    if (fp_is_zero(val_normal)) {
        return "0";
    }
    
    // Work with limbs directly for integer division
    // We'll use a simple approach: extract digits from the lowest limb,
    // then shift right (divide by 10) using proper big integer division
    
    // For now, use a simpler approach: convert the big integer to a string
    // by processing it as a base-10 number
    
    // Since Fp is a 448-bit number, we can represent it as a string
    // We'll do this by repeatedly dividing by 10 and collecting remainders
    
    // Create a working copy as an array of limbs for easier manipulation
    uint64_t limbs[FP_LIMBS];
    for (int i = 0; i < FP_LIMBS; i++) {
        limbs[i] = val_normal.limb[i];
    }
    
    // Repeatedly divide by 10 and collect remainders
    while (true) {
        // Check if all limbs are zero
        bool all_zero = true;
        for (int i = 0; i < FP_LIMBS; i++) {
            if (limbs[i] != 0) {
                all_zero = false;
                break;
            }
        }
        if (all_zero) {
            break;
        }
        
        // Divide the big integer by 10 and get remainder
        // We do this by processing from MSB to LSB
        uint64_t remainder = 0;
        for (int i = FP_LIMBS - 1; i >= 0; i--) {
            // Combine remainder with current limb: value = remainder * 2^64 + limbs[i]
            // We can't represent this exactly in 64 bits, so we use a 128-bit approach
            __uint128_t value = ((__uint128_t)remainder << 64) | limbs[i];
            limbs[i] = value / 10;
            remainder = value % 10;
        }
        
        // The remainder is our digit
        result = std::to_string(remainder) + result;
    }
    
    return result.empty() ? "0" : result;
}

// Helper function to print Fp value as decimal scalar
void print_fp(const char* label, const Fp& val) {
    std::string decimal = fp_to_decimal_string(val);
    std::cout << label << ": " << decimal << std::endl;
}

// Helper function to print G1Point
void print_g1_point(const char* label, const G1Point& point) {
    std::cout << label << ":" << std::endl;
    if (point.infinity) {
        std::cout << "  Infinity: true" << std::endl;
    } else {
        std::cout << "  Infinity: false" << std::endl;
        print_fp("  X", point.x);
        print_fp("  Y", point.y);
    }
}

// Test G1 MSM with generator point
// For N points, scalars are [1, 2, 3, ..., N]
// Expected result: G * (1 + 2 + ... + N) = G * N * (N+1) / 2
TEST_F(MSMTest, G1MSMWithGenerator) {
    const uint64_t N = 10;  // Test with 10 points
    
    // Get generator point
    const G1Point& G = g1_generator();
    
    // Check that generator is not at infinity (should be set from tfhe-rs)
    if (g1_is_infinity(G)) {
        GTEST_SKIP() << "G1 generator not set - please provide generator points from tfhe-rs";
    }
    
    // Create array of N points, all equal to generator G
    G1Point* h_points = (G1Point*)malloc(N * sizeof(G1Point));
    for (uint64_t i = 0; i < N; i++) {
        fp_copy(h_points[i].x, G.x);
        fp_copy(h_points[i].y, G.y);
        h_points[i].infinity = G.infinity;
    }
    
    // Create scalar array: [1, 2, 3, ..., N]
    uint64_t* h_scalars = (uint64_t*)malloc(N * sizeof(uint64_t));
    for (uint64_t i = 0; i < N; i++) {
        h_scalars[i] = i + 1;
    }
    
    // Compute MSM (function handles device memory internally)
    G1Point msm_result;
    g1_msm_u64(stream, gpu_index, msm_result, h_points, h_scalars, N);
    
    // Compute expected result: G * (N * (N+1) / 2)
    uint64_t expected_scalar = triangular_number(N);
    G1Point expected_result;
    g1_scalar_mul_u64(expected_result, G, expected_scalar);
    
    // Compare results
    EXPECT_EQ(msm_result.infinity, expected_result.infinity);
    if (!msm_result.infinity && !expected_result.infinity) {
        EXPECT_EQ(fp_cmp(msm_result.x, expected_result.x), 0) 
            << "MSM x-coordinate mismatch";
        EXPECT_EQ(fp_cmp(msm_result.y, expected_result.y), 0) 
            << "MSM y-coordinate mismatch";
    }
    
    // Cleanup
    free(h_points);
    free(h_scalars);
}

#ifdef DEBUG
// Test G1 MSM with generator point (N=1) with printing
// This is a basic test that prints the inputs and outputs for debugging
// Only compiled and run when DEBUG flag is enabled
TEST_F(MSMTest, G1MSMWithGeneratorBasicTest) {
    const uint64_t N = 1;  // Test with 1 point
    
    // Get generator point
    const G1Point& G = g1_generator();
    
    // Check that generator is not at infinity (should be set from tfhe-rs)
    if (g1_is_infinity(G)) {
        GTEST_SKIP() << "G1 generator not set - please provide generator points from tfhe-rs";
    }
    
    std::cout << "\n=== G1MSMWithGeneratorBasicTest (N=" << N << ") ===" << std::endl;
    
    // Create array of N points, all equal to generator G
    G1Point* h_points = (G1Point*)malloc(N * sizeof(G1Point));
    for (uint64_t i = 0; i < N; i++) {
        fp_copy(h_points[i].x, G.x);
        fp_copy(h_points[i].y, G.y);
        h_points[i].infinity = G.infinity;
    }
    
    // Create scalar array: [1, 2, 3, ..., N]
    uint64_t* h_scalars = (uint64_t*)malloc(N * sizeof(uint64_t));
    for (uint64_t i = 0; i < N; i++) {
        h_scalars[i] = i + 1;
    }
    
    // Print base array (points)
    std::cout << "\nBase array (points):" << std::endl;
    for (uint64_t i = 0; i < N; i++) {
        std::cout << "  Point[" << i << "]:" << std::endl;
        print_g1_point("    ", h_points[i]);
    }
    
    // Print scalar array
    std::cout << "\nScalar array:" << std::endl;
    for (uint64_t i = 0; i < N; i++) {
        std::cout << "  Scalar[" << i << "] = " << h_scalars[i] << std::endl;
    }
    
    // Compute MSM (function handles device memory internally)
    G1Point msm_result;
    g1_msm_u64(stream, gpu_index, msm_result, h_points, h_scalars, N);
    
    // Print result
    std::cout << "\nMSM Result:" << std::endl;
    print_g1_point("  ", msm_result);
    
    // Compute expected result: G * (N * (N+1) / 2)
    uint64_t expected_scalar = triangular_number(N);
    G1Point expected_result;
    g1_scalar_mul_u64(expected_result, G, expected_scalar);
    
    std::cout << "\nExpected Result (G * " << expected_scalar << "):" << std::endl;
    print_g1_point("  ", expected_result);
    
    // Compare results
    EXPECT_EQ(msm_result.infinity, expected_result.infinity);
    if (!msm_result.infinity && !expected_result.infinity) {
        EXPECT_EQ(fp_cmp(msm_result.x, expected_result.x), 0) 
            << "MSM x-coordinate mismatch";
        EXPECT_EQ(fp_cmp(msm_result.y, expected_result.y), 0) 
            << "MSM y-coordinate mismatch";
    }
    
    std::cout << "\n=== Test completed ===" << std::endl << std::endl;
    
    // Cleanup
    free(h_points);
    free(h_scalars);
}
#endif // DEBUG

// Test G2 MSM with generator point
TEST_F(MSMTest, G2MSMWithGenerator) {
    const uint64_t N = 10;  // Test with 10 points
    
    // Get generator point
    const G2Point& G = g2_generator();
    
    // Check that generator is not at infinity (should be set from tfhe-rs)
    if (g2_is_infinity(G)) {
        GTEST_SKIP() << "G2 generator not set - please provide generator points from tfhe-rs";
    }
    
    // Create array of N points, all equal to generator G
    G2Point* h_points = (G2Point*)malloc(N * sizeof(G2Point));
    for (uint64_t i = 0; i < N; i++) {
        fp2_copy(h_points[i].x, G.x);
        fp2_copy(h_points[i].y, G.y);
        h_points[i].infinity = G.infinity;
    }
    
    // Create scalar array: [1, 2, 3, ..., N]
    uint64_t* h_scalars = (uint64_t*)malloc(N * sizeof(uint64_t));
    for (uint64_t i = 0; i < N; i++) {
        h_scalars[i] = i + 1;
    }
    
    // Compute MSM (function handles device memory internally)
    G2Point msm_result;
    g2_msm_u64(stream, gpu_index, msm_result, h_points, h_scalars, N);
    
    // Compute expected result: G * (N * (N+1) / 2)
    uint64_t expected_scalar = triangular_number(N);
    G2Point expected_result;
    g2_scalar_mul_u64(expected_result, G, expected_scalar);
    
    // Compare results
    EXPECT_EQ(msm_result.infinity, expected_result.infinity);
    if (!msm_result.infinity && !expected_result.infinity) {
        EXPECT_EQ(fp2_cmp(msm_result.x, expected_result.x), 0) 
            << "MSM x-coordinate mismatch";
        EXPECT_EQ(fp2_cmp(msm_result.y, expected_result.y), 0) 
            << "MSM y-coordinate mismatch";
    }
    
    // Cleanup
    free(h_points);
    free(h_scalars);
}

// Test with larger N to verify correctness
TEST_F(MSMTest, G1MSMLargeN) {
    const uint64_t N = 100;
    
    const G1Point& G = g1_generator();
    if (g1_is_infinity(G)) {
        GTEST_SKIP() << "G1 generator not set";
    }
    
    G1Point* h_points = (G1Point*)malloc(N * sizeof(G1Point));
    for (uint64_t i = 0; i < N; i++) {
        fp_copy(h_points[i].x, G.x);
        fp_copy(h_points[i].y, G.y);
        h_points[i].infinity = G.infinity;
    }
    
    uint64_t* h_scalars = (uint64_t*)malloc(N * sizeof(uint64_t));
    for (uint64_t i = 0; i < N; i++) {
        h_scalars[i] = i + 1;
    }
    
    G1Point msm_result;
    g1_msm_u64(stream, gpu_index, msm_result, h_points, h_scalars, N);
    
    uint64_t expected_scalar = triangular_number(N);
    G1Point expected_result;
    g1_scalar_mul_u64(expected_result, G, expected_scalar);
    
    EXPECT_EQ(msm_result.infinity, expected_result.infinity);
    if (!msm_result.infinity && !expected_result.infinity) {
        EXPECT_EQ(fp_cmp(msm_result.x, expected_result.x), 0);
        EXPECT_EQ(fp_cmp(msm_result.y, expected_result.y), 0);
    }
    
    free(h_points);
    free(h_scalars);
}

