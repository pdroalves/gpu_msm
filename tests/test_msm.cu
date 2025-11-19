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
    
    // Calculate required scratch space: (num_blocks + 1) * MSM_BUCKET_COUNT (projective points)
    int threadsPerBlock = 256;
    int num_blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t scratch_size = (num_blocks + 1) * MSM_BUCKET_COUNT * sizeof(G1ProjectivePoint);
    
    // Allocate device memory for points and scalars
    G1Point* d_points = (G1Point*)cuda_malloc_async(N * sizeof(G1Point), stream, gpu_index);
    uint64_t* d_scalars = (uint64_t*)cuda_malloc_async(N * sizeof(uint64_t), stream, gpu_index);
    G1ProjectivePoint* d_result = (G1ProjectivePoint*)cuda_malloc_async(sizeof(G1ProjectivePoint), stream, gpu_index);
    G1ProjectivePoint* d_scratch = (G1ProjectivePoint*)cuda_malloc_async(scratch_size, stream, gpu_index);
    G1Point* d_expected = (G1Point*)cuda_malloc_async(sizeof(G1Point), stream, gpu_index);
    G1Point* d_G = (G1Point*)cuda_malloc_async(sizeof(G1Point), stream, gpu_index);
    
    // Initialize allocated memory to zero to avoid uninitialized access warnings
    cuda_memset_async(d_result, 0, sizeof(G1ProjectivePoint), stream, gpu_index);
    cuda_memset_async(d_expected, 0, sizeof(G1Point), stream, gpu_index);
    cuda_memset_async(d_G, 0, sizeof(G1Point), stream, gpu_index);
    
    // Prepare host data
    // Convert generator back to normal form for test points (generator is in Montgomery)
    G1Point G_normal;
    fp_from_montgomery(G_normal.x, G.x);
    fp_from_montgomery(G_normal.y, G.y);
    G_normal.infinity = G.infinity;
    
    G1Point* h_points = (G1Point*)malloc(N * sizeof(G1Point));
    for (uint64_t i = 0; i < N; i++) {
        fp_copy(h_points[i].x, G_normal.x);
        fp_copy(h_points[i].y, G_normal.y);
        h_points[i].infinity = G_normal.infinity;
    }
    
    uint64_t* h_scalars = (uint64_t*)malloc(N * sizeof(uint64_t));
    for (uint64_t i = 0; i < N; i++) {
        h_scalars[i] = i + 1;
    }
    
    // Copy to device
    cuda_memcpy_async_to_gpu(d_points, h_points, N * sizeof(G1Point), stream, gpu_index);
    cuda_memcpy_async_to_gpu(d_scalars, h_scalars, N * sizeof(uint64_t), stream, gpu_index);
    
    // Convert points to Montgomery form (required for performance)
    int threadsPerBlock_conv = 256;
    int blocks_conv = (N + threadsPerBlock_conv - 1) / threadsPerBlock_conv;
    point_to_montgomery_batch<G1Point>(stream, gpu_index, d_points, N);
    check_cuda_error(cudaGetLastError());
    
    // Copy generator to device (generator is already in Montgomery form from host)
    cuda_memcpy_async_to_gpu(d_G, &G, sizeof(G1Point), stream, gpu_index);
    
    // Compute MSM on device (returns projective point)
    point_msm_u64_g1(stream, gpu_index, d_result, d_points, d_scalars, d_scratch, N);
    
    // Compute expected result on device: G * (N * (N+1) / 2)
    uint64_t expected_scalar = triangular_number(N);
    point_scalar_mul_u64<G1Point>(stream, gpu_index, d_expected, d_G, expected_scalar);
    
    // Convert projective result to affine, then from Montgomery form before comparing
    G1ProjectivePoint* d_result_proj = d_result;
    G1Point* d_result_affine = (G1Point*)cuda_malloc_async(sizeof(G1Point), stream, gpu_index);
    G1Point* d_expected_normal = (G1Point*)cuda_malloc_async(sizeof(G1Point), stream, gpu_index);
    
    // Convert projective to affine on device
    cuda_synchronize_stream(stream, gpu_index);
    G1ProjectivePoint h_result_proj;
    cuda_memcpy_async_to_cpu(&h_result_proj, d_result_proj, sizeof(G1ProjectivePoint), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    G1Point h_result_affine;
    projective_to_affine_g1(h_result_affine, h_result_proj);
    cuda_memcpy_async_to_gpu(d_result_affine, &h_result_affine, sizeof(G1Point), stream, gpu_index);
    
    point_from_montgomery<G1Point>(stream, gpu_index, d_result_affine, d_result_affine);
    point_from_montgomery<G1Point>(stream, gpu_index, d_expected_normal, d_expected);
    
    // Synchronize and copy results back
    cuda_synchronize_stream(stream, gpu_index);
    G1Point msm_result;
    G1Point expected_result;
    cuda_memcpy_async_to_cpu(&msm_result, d_result_affine, sizeof(G1Point), stream, gpu_index);
    cuda_memcpy_async_to_cpu(&expected_result, d_expected_normal, sizeof(G1Point), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    
    // Cleanup
    cuda_drop_async(d_result_affine, stream, gpu_index);
    cuda_drop_async(d_expected_normal, stream, gpu_index);
    
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
    cuda_drop_async(d_points, stream, gpu_index);
    cuda_drop_async(d_scalars, stream, gpu_index);
    cuda_drop_async(d_result, stream, gpu_index);
    cuda_drop_async(d_scratch, stream, gpu_index);
    cuda_drop_async(d_expected, stream, gpu_index);
    cuda_drop_async(d_G, stream, gpu_index);
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
    
    // Calculate required scratch space: (num_blocks + 1) * MSM_BUCKET_COUNT
    int threadsPerBlock = 256;
    int num_blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t scratch_size = (num_blocks + 1) * MSM_BUCKET_COUNT * sizeof(G1Point);
    
    // Allocate device memory
    G1Point* d_points = (G1Point*)cuda_malloc_async(N * sizeof(G1Point), stream, gpu_index);
    uint64_t* d_scalars = (uint64_t*)cuda_malloc_async(N * sizeof(uint64_t), stream, gpu_index);
    G1Point* d_result = (G1Point*)cuda_malloc_async(sizeof(G1Point), stream, gpu_index);
    G1Point* d_scratch = (G1Point*)cuda_malloc_async(scratch_size, stream, gpu_index);
    G1Point* d_expected = (G1Point*)cuda_malloc_async(sizeof(G1Point), stream, gpu_index);
    G1Point* d_G = (G1Point*)cuda_malloc_async(sizeof(G1Point), stream, gpu_index);
    
    // Prepare host data
    // Convert generator back to normal form for test points (generator is in Montgomery)
    G1Point G_normal;
    fp_from_montgomery(G_normal.x, G.x);
    fp_from_montgomery(G_normal.y, G.y);
    G_normal.infinity = G.infinity;
    
    G1Point* h_points = (G1Point*)malloc(N * sizeof(G1Point));
    for (uint64_t i = 0; i < N; i++) {
        fp_copy(h_points[i].x, G_normal.x);
        fp_copy(h_points[i].y, G_normal.y);
        h_points[i].infinity = G_normal.infinity;
    }
    
    uint64_t* h_scalars = (uint64_t*)malloc(N * sizeof(uint64_t));
    for (uint64_t i = 0; i < N; i++) {
        h_scalars[i] = i + 1;
    }
    
    // Print base array (points) - before copying to device
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
    
    // Copy to device
    cuda_memcpy_async_to_gpu(d_points, h_points, N * sizeof(G1Point), stream, gpu_index);
    cuda_memcpy_async_to_gpu(d_scalars, h_scalars, N * sizeof(uint64_t), stream, gpu_index);
    
    // Convert points to Montgomery form (required for performance)
    int threadsPerBlock_conv = 256;
    int blocks_conv = (N + threadsPerBlock_conv - 1) / threadsPerBlock_conv;
    point_to_montgomery_batch<G1Point>(stream, gpu_index, d_points, N);
    check_cuda_error(cudaGetLastError());
    
    // Copy generator to device (generator is already in Montgomery from host)
    cuda_memcpy_async_to_gpu(d_G, &G, sizeof(G1Point), stream, gpu_index);
    
    // Compute MSM on device
    g1_msm_u64(stream, gpu_index, d_result, d_points, d_scalars, d_scratch, N);
    
    // Compute expected result on device: G * (N * (N+1) / 2)
    uint64_t expected_scalar = triangular_number(N);
    g1_scalar_mul_u64(stream, gpu_index, d_expected, d_G, expected_scalar);
    
    // Synchronize and copy results back
    cuda_synchronize_stream(stream, gpu_index);
    G1Point msm_result;
    G1Point expected_result;
    cuda_memcpy_async_to_cpu(&msm_result, d_result, sizeof(G1Point), stream, gpu_index);
    cuda_memcpy_async_to_cpu(&expected_result, d_expected, sizeof(G1Point), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    
    // Print result
    std::cout << "\nMSM Result:" << std::endl;
    print_g1_point("  ", msm_result);
    
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
    cuda_drop_async(d_points, stream, gpu_index);
    cuda_drop_async(d_scalars, stream, gpu_index);
    cuda_drop_async(d_result, stream, gpu_index);
    cuda_drop_async(d_scratch, stream, gpu_index);
    cuda_drop_async(d_expected, stream, gpu_index);
    cuda_drop_async(d_G, stream, gpu_index);
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
    
    // Calculate required scratch space: (num_blocks + 1) * MSM_BUCKET_COUNT (projective points)
    int threadsPerBlock = 128;
    int num_blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t scratch_size = (num_blocks + 1) * MSM_BUCKET_COUNT * sizeof(G2ProjectivePoint);
    
    // Allocate device memory
    G2Point* d_points = (G2Point*)cuda_malloc_async(N * sizeof(G2Point), stream, gpu_index);
    uint64_t* d_scalars = (uint64_t*)cuda_malloc_async(N * sizeof(uint64_t), stream, gpu_index);
    G2ProjectivePoint* d_result = (G2ProjectivePoint*)cuda_malloc_async(sizeof(G2ProjectivePoint), stream, gpu_index);
    G2ProjectivePoint* d_scratch = (G2ProjectivePoint*)cuda_malloc_async(scratch_size, stream, gpu_index);
    G2Point* d_expected = (G2Point*)cuda_malloc_async(sizeof(G2Point), stream, gpu_index);
    G2Point* d_G = (G2Point*)cuda_malloc_async(sizeof(G2Point), stream, gpu_index);
    
    // Initialize allocated memory to zero to avoid uninitialized access warnings
    cuda_memset_async(d_result, 0, sizeof(G2ProjectivePoint), stream, gpu_index);
    cuda_memset_async(d_expected, 0, sizeof(G2Point), stream, gpu_index);
    cuda_memset_async(d_G, 0, sizeof(G2Point), stream, gpu_index);
    
    // Prepare host data
    // Convert generator back to normal form for test points (generator is in Montgomery)
    G2Point G_normal;
    fp_from_montgomery(G_normal.x.c0, G.x.c0);
    fp_from_montgomery(G_normal.x.c1, G.x.c1);
    fp_from_montgomery(G_normal.y.c0, G.y.c0);
    fp_from_montgomery(G_normal.y.c1, G.y.c1);
    G_normal.infinity = G.infinity;
    
    G2Point* h_points = (G2Point*)malloc(N * sizeof(G2Point));
    for (uint64_t i = 0; i < N; i++) {
        fp2_copy(h_points[i].x, G_normal.x);
        fp2_copy(h_points[i].y, G_normal.y);
        h_points[i].infinity = G_normal.infinity;
    }
    
    uint64_t* h_scalars = (uint64_t*)malloc(N * sizeof(uint64_t));
    for (uint64_t i = 0; i < N; i++) {
        h_scalars[i] = i + 1;
    }
    
    // Copy to device
    cuda_memcpy_async_to_gpu(d_points, h_points, N * sizeof(G2Point), stream, gpu_index);
    cuda_memcpy_async_to_gpu(d_scalars, h_scalars, N * sizeof(uint64_t), stream, gpu_index);
    
    // Convert points to Montgomery form (required for performance)
    int threadsPerBlock_conv = 128;
    int blocks_conv = (N + threadsPerBlock_conv - 1) / threadsPerBlock_conv;
    point_to_montgomery_batch<G2Point>(stream, gpu_index, d_points, N);
    check_cuda_error(cudaGetLastError());
    
    // Copy generator to device (generator is already in Montgomery form from host)
    cuda_memcpy_async_to_gpu(d_G, &G, sizeof(G2Point), stream, gpu_index);
    
    // Compute MSM on device (returns projective point)
    point_msm_u64_g2(stream, gpu_index, d_result, d_points, d_scalars, d_scratch, N);
    
    // Compute expected result on device: G * (N * (N+1) / 2)
    uint64_t expected_scalar = triangular_number(N);
    point_scalar_mul_u64<G2Point>(stream, gpu_index, d_expected, d_G, expected_scalar);
    
    // Convert projective result to affine, then from Montgomery form before comparing
    G2ProjectivePoint* d_result_proj = d_result;
    G2Point* d_result_affine = (G2Point*)cuda_malloc_async(sizeof(G2Point), stream, gpu_index);
    G2Point* d_expected_normal = (G2Point*)cuda_malloc_async(sizeof(G2Point), stream, gpu_index);
    
    // Convert projective to affine on device
    cuda_synchronize_stream(stream, gpu_index);
    G2ProjectivePoint h_result_proj;
    cuda_memcpy_async_to_cpu(&h_result_proj, d_result_proj, sizeof(G2ProjectivePoint), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    G2Point h_result_affine;
    projective_to_affine_g2(h_result_affine, h_result_proj);
    cuda_memcpy_async_to_gpu(d_result_affine, &h_result_affine, sizeof(G2Point), stream, gpu_index);
    
    point_from_montgomery<G2Point>(stream, gpu_index, d_result_affine, d_result_affine);
    point_from_montgomery<G2Point>(stream, gpu_index, d_expected_normal, d_expected);
    
    // Synchronize and copy results back
    cuda_synchronize_stream(stream, gpu_index);
    G2Point msm_result;
    G2Point expected_result;
    cuda_memcpy_async_to_cpu(&msm_result, d_result_affine, sizeof(G2Point), stream, gpu_index);
    cuda_memcpy_async_to_cpu(&expected_result, d_expected_normal, sizeof(G2Point), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    
    // Cleanup
    cuda_drop_async(d_result_affine, stream, gpu_index);
    cuda_drop_async(d_expected_normal, stream, gpu_index);
    
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
    cuda_drop_async(d_points, stream, gpu_index);
    cuda_drop_async(d_scalars, stream, gpu_index);
    cuda_drop_async(d_result, stream, gpu_index);
    cuda_drop_async(d_scratch, stream, gpu_index);
    cuda_drop_async(d_expected, stream, gpu_index);
    cuda_drop_async(d_G, stream, gpu_index);
}

// Test with larger N to verify correctness
TEST_F(MSMTest, G1MSMLargeN) {
    const uint64_t N = 100;
    
    const G1Point& G = g1_generator();
    if (g1_is_infinity(G)) {
        GTEST_SKIP() << "G1 generator not set";
    }
    
    // Calculate required scratch space: (num_blocks + 1) * MSM_BUCKET_COUNT (projective points)
    int threadsPerBlock = 256;
    int num_blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t scratch_size = (num_blocks + 1) * MSM_BUCKET_COUNT * sizeof(G1ProjectivePoint);
    
    // Allocate device memory
    G1Point* d_points = (G1Point*)cuda_malloc_async(N * sizeof(G1Point), stream, gpu_index);
    uint64_t* d_scalars = (uint64_t*)cuda_malloc_async(N * sizeof(uint64_t), stream, gpu_index);
    G1ProjectivePoint* d_result = (G1ProjectivePoint*)cuda_malloc_async(sizeof(G1ProjectivePoint), stream, gpu_index);
    G1ProjectivePoint* d_scratch = (G1ProjectivePoint*)cuda_malloc_async(scratch_size, stream, gpu_index);
    G1Point* d_expected = (G1Point*)cuda_malloc_async(sizeof(G1Point), stream, gpu_index);
    G1Point* d_G = (G1Point*)cuda_malloc_async(sizeof(G1Point), stream, gpu_index);
    
    // Prepare host data
    // Convert generator back to normal form for test points (generator is in Montgomery)
    G1Point G_normal;
    fp_from_montgomery(G_normal.x, G.x);
    fp_from_montgomery(G_normal.y, G.y);
    G_normal.infinity = G.infinity;
    
    G1Point* h_points = (G1Point*)malloc(N * sizeof(G1Point));
    for (uint64_t i = 0; i < N; i++) {
        fp_copy(h_points[i].x, G_normal.x);
        fp_copy(h_points[i].y, G_normal.y);
        h_points[i].infinity = G_normal.infinity;
    }
    
    uint64_t* h_scalars = (uint64_t*)malloc(N * sizeof(uint64_t));
    for (uint64_t i = 0; i < N; i++) {
        h_scalars[i] = i + 1;
    }
    
    // Copy to device
    cuda_memcpy_async_to_gpu(d_points, h_points, N * sizeof(G1Point), stream, gpu_index);
    cuda_memcpy_async_to_gpu(d_scalars, h_scalars, N * sizeof(uint64_t), stream, gpu_index);
    
    // Convert points to Montgomery form (required for performance)
    int threadsPerBlock_conv = 256;
    int blocks_conv = (N + threadsPerBlock_conv - 1) / threadsPerBlock_conv;
    point_to_montgomery_batch<G1Point>(stream, gpu_index, d_points, N);
    check_cuda_error(cudaGetLastError());
    
    // Copy generator to device (generator is already in Montgomery from host)
    cuda_memcpy_async_to_gpu(d_G, &G, sizeof(G1Point), stream, gpu_index);
    
    // Compute MSM on device (returns projective point)
    point_msm_u64_g1(stream, gpu_index, d_result, d_points, d_scalars, d_scratch, N);
    
    // Compute expected result on device
    uint64_t expected_scalar = triangular_number(N);
    point_scalar_mul_u64<G1Point>(stream, gpu_index, d_expected, d_G, expected_scalar);
    
    // Convert projective result to affine, then from Montgomery form before comparing
    G1Point* d_result_affine = (G1Point*)cuda_malloc_async(sizeof(G1Point), stream, gpu_index);
    G1Point* d_expected_normal = (G1Point*)cuda_malloc_async(sizeof(G1Point), stream, gpu_index);
    
    // Convert projective to affine on host
    cuda_synchronize_stream(stream, gpu_index);
    G1ProjectivePoint h_result_proj;
    cuda_memcpy_async_to_cpu(&h_result_proj, d_result, sizeof(G1ProjectivePoint), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    G1Point h_result_affine;
    projective_to_affine_g1(h_result_affine, h_result_proj);
    cuda_memcpy_async_to_gpu(d_result_affine, &h_result_affine, sizeof(G1Point), stream, gpu_index);
    
    point_from_montgomery<G1Point>(stream, gpu_index, d_result_affine, d_result_affine);
    point_from_montgomery<G1Point>(stream, gpu_index, d_expected_normal, d_expected);
    
    // Synchronize and copy results back
    cuda_synchronize_stream(stream, gpu_index);
    G1Point msm_result;
    G1Point expected_result;
    cuda_memcpy_async_to_cpu(&msm_result, d_result_affine, sizeof(G1Point), stream, gpu_index);
    cuda_memcpy_async_to_cpu(&expected_result, d_expected_normal, sizeof(G1Point), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    
    // Cleanup
    cuda_drop_async(d_result_affine, stream, gpu_index);
    cuda_drop_async(d_expected_normal, stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    
    EXPECT_EQ(msm_result.infinity, expected_result.infinity);
    if (!msm_result.infinity && !expected_result.infinity) {
        EXPECT_EQ(fp_cmp(msm_result.x, expected_result.x), 0);
        EXPECT_EQ(fp_cmp(msm_result.y, expected_result.y), 0);
    }
    
    free(h_points);
    free(h_scalars);
    cuda_drop_async(d_points, stream, gpu_index);
    cuda_drop_async(d_scalars, stream, gpu_index);
    cuda_drop_async(d_result, stream, gpu_index);
    cuda_drop_async(d_scratch, stream, gpu_index);
    cuda_drop_async(d_expected, stream, gpu_index);
    cuda_drop_async(d_G, stream, gpu_index);
}

