// C wrapper functions for Rust FFI
// These functions provide a C-compatible interface to the C++ functions

#include "../../include/curve.h"
#include "../../include/device.h"
#include <stdint.h>
#include <stdbool.h>

extern "C" {

// G1 affine to projective conversion
void affine_to_projective_g1_wrapper(G1ProjectivePoint* proj, const G1Point* affine) {
    affine_to_projective(*proj, *affine);
}

// G2 affine to projective conversion
void affine_to_projective_g2_wrapper(G2ProjectivePoint* proj, const G2Point* affine) {
    affine_to_projective(*proj, *affine);
}

// G1 projective to affine conversion
void projective_to_affine_g1_wrapper(G1Point* affine, const G1ProjectivePoint* proj) {
    projective_to_affine_g1(*affine, *proj);
}

// G2 projective to affine conversion
void projective_to_affine_g2_wrapper(G2Point* affine, const G2ProjectivePoint* proj) {
    projective_to_affine_g2(*affine, *proj);
}

// G1 point at infinity
void g1_point_at_infinity_wrapper(G1Point* point) {
    g1_point_at_infinity(*point);
}

// G2 point at infinity
void g2_point_at_infinity_wrapper(G2Point* point) {
    g2_point_at_infinity(*point);
}

// G1 projective point at infinity
void g1_projective_point_at_infinity_wrapper(G1ProjectivePoint* point) {
    g1_projective_point_at_infinity(*point);
}

// G2 projective point at infinity
void g2_projective_point_at_infinity_wrapper(G2ProjectivePoint* point) {
    g2_projective_point_at_infinity(*point);
}

// Check if G1 point is at infinity
bool g1_is_infinity_wrapper(const G1Point* point) {
    return g1_is_infinity(*point);
}

// Check if G2 point is at infinity
bool g2_is_infinity_wrapper(const G2Point* point) {
    return g2_is_infinity(*point);
}

// High-level MSM wrapper for G1 (handles CUDA setup internally)
// This allocates device memory, copies data, runs MSM, and copies result back
// Returns 0 on success, non-zero on error
int g1_msm_wrapper(
    G1ProjectivePoint* result,
    const G1Point* points,
    const uint64_t* scalars,
    int n,
    uint32_t gpu_index
) {
    if (n <= 0) {
        return -1;
    }
    
    // Create stream
    cudaStream_t stream = cuda_create_stream(gpu_index);
    if (stream == nullptr) {
        return -2;
    }
    
    // Initialize device
    init_device_modulus(stream, gpu_index);
    init_device_curve(stream, gpu_index);
    init_device_generators(stream, gpu_index);
    
    // Calculate scratch size: (num_blocks + 1) * MSM_BUCKET_COUNT
    // MSM internally uses 128 threads per block (see point_msm_u64_async_g1)
    // The scratch size MUST match what MSM expects
    int threadsPerBlock = 128;  // Must match MSM internal threadsPerBlock
    int num_blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    size_t scratch_size = (num_blocks + 1) * MSM_BUCKET_COUNT * sizeof(G1ProjectivePoint);
    printf("DEBUG: Scratch calculation - threadsPerBlock=%d, num_blocks=%d, scratch_size=%zu\n",
           threadsPerBlock, num_blocks, scratch_size);
    
    // Allocate device memory
    G1Point* d_points = (G1Point*)cuda_malloc_async(n * sizeof(G1Point), stream, gpu_index);
    uint64_t* d_scalars = (uint64_t*)cuda_malloc_async(n * sizeof(uint64_t), stream, gpu_index);
    G1ProjectivePoint* d_result = (G1ProjectivePoint*)cuda_malloc_async(sizeof(G1ProjectivePoint), stream, gpu_index);
    G1ProjectivePoint* d_scratch = (G1ProjectivePoint*)cuda_malloc_async(scratch_size, stream, gpu_index);
    
    if (!d_points || !d_scalars || !d_result || !d_scratch) {
        cuda_destroy_stream(stream, gpu_index);
        return -3;
    }
    
    // DEBUG: Print first point to verify data
    if (n > 0) {
        printf("DEBUG: First point before MSM:\n");
        printf("  x[0] = 0x%016llx\n", points[0].x.limb[0]);
        printf("  x[1] = 0x%016llx\n", points[0].x.limb[1]);
        printf("  y[0] = 0x%016llx\n", points[0].y.limb[0]);
        printf("  infinity = %d\n", points[0].infinity);
        printf("  scalar[0] = %llu\n", scalars[0]);
    }
    
    // WORKAROUND: Device-side Montgomery conversion isn't working for unknown reasons
    // Convert points to Montgomery form on HOST and copy already-converted points
    G1Point* h_points_mont = (G1Point*)malloc(n * sizeof(G1Point));
    for (int i = 0; i < n; i++) {
        h_points_mont[i].infinity = points[i].infinity;
        if (!points[i].infinity) {
            fp_to_montgomery(h_points_mont[i].x, points[i].x);
            fp_to_montgomery(h_points_mont[i].y, points[i].y);
        } else {
            fp_zero(h_points_mont[i].x);
            fp_zero(h_points_mont[i].y);
        }
    }
    
    printf("DEBUG: After HOST Montgomery conversion:\n");
    printf("  x[0] = 0x%016llx\n", h_points_mont[0].x.limb[0]);
    printf("  y[0] = 0x%016llx\n", h_points_mont[0].y.limb[0]);
    
    // Copy Montgomery-form data to device
    cuda_memcpy_async_to_gpu(d_points, h_points_mont, n * sizeof(G1Point), stream, gpu_index);
    cuda_memcpy_async_to_gpu(d_scalars, scalars, n * sizeof(uint64_t), stream, gpu_index);
    free(h_points_mont);
    
    // Synchronize before MSM
    cuda_synchronize_stream(stream, gpu_index);
    
    // DEBUG: Read back scalars from device to verify they were copied correctly
    uint64_t* h_scalars_check = (uint64_t*)malloc(n * sizeof(uint64_t));
    cuda_memcpy_async_to_cpu(h_scalars_check, d_scalars, n * sizeof(uint64_t), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    printf("DEBUG: Scalars on device (read back):\n");
    for (int i = 0; i < n; i++) {
        printf("  scalars[%d] = %llu\n", i, h_scalars_check[i]);
    }
    free(h_scalars_check);
    
    // Debug: verify pointers before MSM
    printf("DEBUG: Before MSM - d_result=%p, d_points=%p, d_scalars=%p, d_scratch=%p, n=%d\n",
           d_result, d_points, d_scalars, d_scratch, n);
    
    // Initialize result to a known pattern (not zero) to verify MSM writes to it
    G1ProjectivePoint test_pattern;
    for (int i = 0; i < 7; i++) {
        test_pattern.X.limb[i] = 0xDEADBEEFCAFEBABEULL;
        test_pattern.Y.limb[i] = 0xFEEDFACEDEADC0DEULL;
        test_pattern.Z.limb[i] = 0x1234567890ABCDEFUL;
    }
    cuda_memcpy_async_to_gpu(d_result, &test_pattern, sizeof(G1ProjectivePoint), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    printf("DEBUG: Initialized d_result with test pattern\n");
    
    // Note: MSM function initializes result to infinity internally
    // Run MSM
    point_msm_u64_g1(stream, gpu_index, d_result, d_points, d_scalars, d_scratch, n);
    
    // point_msm_u64_g1 is the SYNC version (internally calls cuda_synchronize_stream)
    // So the result should be ready now
    printf("DEBUG: After MSM call (sync version, should be complete)\n");
    
    // Check for CUDA errors after MSM
    cudaError_t err = cudaGetLastError();
    printf("DEBUG: CUDA error after MSM: %s\n", cudaGetErrorString(err));
    if (err != cudaSuccess) {
        cuda_drop_async(d_points, stream, gpu_index);
        cuda_drop_async(d_scalars, stream, gpu_index);
        cuda_drop_async(d_result, stream, gpu_index);
        cuda_drop_async(d_scratch, stream, gpu_index);
        cuda_destroy_stream(stream, gpu_index);
        return -4; // CUDA error
    }
    
    // Synchronize before copying result back
    cuda_synchronize_stream(stream, gpu_index);
    
    // DEBUG: Read result from device first
    G1ProjectivePoint h_result_check;
    cuda_memcpy_async_to_cpu(&h_result_check, d_result, sizeof(G1ProjectivePoint), stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    printf("DEBUG: Result from device (h_result_check):\n");
    printf("  X[0] = 0x%016llx\n", h_result_check.X.limb[0]);
    printf("  Y[0] = 0x%016llx\n", h_result_check.Y.limb[0]);
    printf("  Z[0] = 0x%016llx\n", h_result_check.Z.limb[0]);
    
    // Copy result back to the provided pointer
    cuda_memcpy_async_to_cpu(result, d_result, sizeof(G1ProjectivePoint), stream, gpu_index);
    
    // Synchronize again to ensure copy completes
    cuda_synchronize_stream(stream, gpu_index);
    
    // DEBUG: Print result and convert to normal form to compare with tfhe-zk-pok
    printf("DEBUG: MSM result (projective, Montgomery form):\n");
    printf("  X[0] = 0x%016llx\n", result->X.limb[0]);
    printf("  Y[0] = 0x%016llx\n", result->Y.limb[0]);
    printf("  Z[0] = 0x%016llx\n", result->Z.limb[0]);
    
    // Convert to affine
    G1Point affine_result_mont;
    projective_to_affine_g1(affine_result_mont, *result);
    printf("DEBUG: MSM result (affine, Montgomery form):\n");
    printf("  x[0] = 0x%016llx\n", affine_result_mont.x.limb[0]);
    printf("  y[0] = 0x%016llx\n", affine_result_mont.y.limb[0]);
    
    // Convert to normal form
    G1Point affine_result_normal;
    fp_from_montgomery(affine_result_normal.x, affine_result_mont.x);
    fp_from_montgomery(affine_result_normal.y, affine_result_mont.y);
    affine_result_normal.infinity = affine_result_mont.infinity;
    printf("DEBUG: MSM result (affine, normal form - ALL limbs):\n");
    printf("  x[0] = 0x%016llx\n", affine_result_normal.x.limb[0]);
    printf("  x[1] = 0x%016llx\n", affine_result_normal.x.limb[1]);
    printf("  x[2] = 0x%016llx\n", affine_result_normal.x.limb[2]);
    printf("  x[3] = 0x%016llx\n", affine_result_normal.x.limb[3]);
    printf("  x[4] = 0x%016llx\n", affine_result_normal.x.limb[4]);
    printf("  x[5] = 0x%016llx\n", affine_result_normal.x.limb[5]);
    printf("  x[6] = 0x%016llx\n", affine_result_normal.x.limb[6]);
    printf("  y[0] = 0x%016llx\n", affine_result_normal.y.limb[0]);
    
    // Cleanup
    cuda_drop_async(d_points, stream, gpu_index);
    cuda_drop_async(d_scalars, stream, gpu_index);
    cuda_drop_async(d_result, stream, gpu_index);
    cuda_drop_async(d_scratch, stream, gpu_index);
    cuda_destroy_stream(stream, gpu_index);
    
    return 0;
}

// High-level MSM wrapper for G2 (handles CUDA setup internally)
int g2_msm_wrapper(
    G2ProjectivePoint* result,
    const G2Point* points,
    const uint64_t* scalars,
    int n,
    uint32_t gpu_index
) {
    if (n <= 0) {
        return -1;
    }
    
    // Create stream
    cudaStream_t stream = cuda_create_stream(gpu_index);
    if (stream == nullptr) {
        return -2;
    }
    
    // Initialize device
    init_device_modulus(stream, gpu_index);
    init_device_curve(stream, gpu_index);
    init_device_generators(stream, gpu_index);
    
    // Calculate scratch size
    // Use 256 threads per block to match the test
    int threadsPerBlock = 256;  // Match test_msm.cu
    int num_blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    size_t scratch_size = (num_blocks + 1) * MSM_BUCKET_COUNT * sizeof(G2ProjectivePoint);
    
    // Allocate device memory
    G2Point* d_points = (G2Point*)cuda_malloc_async(n * sizeof(G2Point), stream, gpu_index);
    uint64_t* d_scalars = (uint64_t*)cuda_malloc_async(n * sizeof(uint64_t), stream, gpu_index);
    G2ProjectivePoint* d_result = (G2ProjectivePoint*)cuda_malloc_async(sizeof(G2ProjectivePoint), stream, gpu_index);
    G2ProjectivePoint* d_scratch = (G2ProjectivePoint*)cuda_malloc_async(scratch_size, stream, gpu_index);
    
    if (!d_points || !d_scalars || !d_result || !d_scratch) {
        cuda_destroy_stream(stream, gpu_index);
        return -3;
    }
    
    // Copy data to device
    cuda_memcpy_async_to_gpu(d_points, points, n * sizeof(G2Point), stream, gpu_index);
    cuda_memcpy_async_to_gpu(d_scalars, scalars, n * sizeof(uint64_t), stream, gpu_index);
    
    // Convert points to Montgomery form (required for MSM performance)
    // Points from tfhe-zk-pok are in normal form, but MSM expects Montgomery form
    point_to_montgomery_batch<G2Point>(stream, gpu_index, d_points, n);
    cudaError_t conv_err = cudaGetLastError();
    if (conv_err != cudaSuccess) {
        cuda_drop_async(d_points, stream, gpu_index);
        cuda_drop_async(d_scalars, stream, gpu_index);
        cuda_drop_async(d_result, stream, gpu_index);
        cuda_drop_async(d_scratch, stream, gpu_index);
        cuda_destroy_stream(stream, gpu_index);
        return -5; // Conversion error
    }
    
    // Synchronize after conversion (like the test does)
    cuda_synchronize_stream(stream, gpu_index);
    
    // Note: MSM function initializes result to infinity internally
    // Run MSM
    point_msm_u64_g2(stream, gpu_index, d_result, d_points, d_scalars, d_scratch, n);
    
    // Check for CUDA errors after MSM
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cuda_drop_async(d_points, stream, gpu_index);
        cuda_drop_async(d_scalars, stream, gpu_index);
        cuda_drop_async(d_result, stream, gpu_index);
        cuda_drop_async(d_scratch, stream, gpu_index);
        cuda_destroy_stream(stream, gpu_index);
        return -4; // CUDA error
    }
    
    // Synchronize before copying result back
    cuda_synchronize_stream(stream, gpu_index);
    
    // Copy result back
    cuda_memcpy_async_to_cpu(result, d_result, sizeof(G2ProjectivePoint), stream, gpu_index);
    
    // Synchronize again to ensure copy completes
    cuda_synchronize_stream(stream, gpu_index);
    
    // Cleanup
    cuda_drop_async(d_points, stream, gpu_index);
    cuda_drop_async(d_scalars, stream, gpu_index);
    cuda_drop_async(d_result, stream, gpu_index);
    cuda_drop_async(d_scratch, stream, gpu_index);
    cuda_destroy_stream(stream, gpu_index);
    
    return 0;
}

// Convert G1 point from Montgomery form to normal form
void g1_from_montgomery_wrapper(G1Point* result, const G1Point* point) {
    if (point->infinity) {
        g1_point_at_infinity(*result);
        return;
    }
    fp_from_montgomery(result->x, point->x);
    fp_from_montgomery(result->y, point->y);
    result->infinity = false;
}

// Convert G2 point from Montgomery form to normal form
void g2_from_montgomery_wrapper(G2Point* result, const G2Point* point) {
    if (point->infinity) {
        g2_point_at_infinity(*result);
        return;
    }
    // Convert each component of Fp2 from Montgomery form
    fp_from_montgomery(result->x.c0, point->x.c0);
    fp_from_montgomery(result->x.c1, point->x.c1);
    fp_from_montgomery(result->y.c0, point->y.c0);
    fp_from_montgomery(result->y.c1, point->y.c1);
    result->infinity = false;
}

// Convert Fp to Montgomery form
void fp_to_montgomery_wrapper(Fp* result, const Fp* value) {
    fp_to_montgomery(*result, *value);
}

} // extern "C"
