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
    
    // Set device before creating stream (like C++ test does in SetUp)
    cuda_set_device(gpu_index);
    
    // Create stream
    auto stream = cuda_create_stream(gpu_index);
    if (stream == nullptr) {
        return -2;
    }
    
    // Initialize device generators (converts from standard to Montgomery form)
    init_device_generators(stream, gpu_index);
    
    // Synchronize to ensure initialization completes
    cuda_synchronize_stream(stream, gpu_index);
    
    // Calculate scratch size: (num_blocks + 1) * MSM_BUCKET_COUNT
    // IMPORTANT: Must match the num_blocks calculation used in point_msm_u64_async_g1
    // MSM function uses threadsPerBlock = 128, so we must use the same for num_blocks calculation
    const auto threadsPerBlock_msm = 128;  // Match MSM function (point_msm_u64_async_g1 uses 128)
    const auto num_blocks = (n + threadsPerBlock_msm - 1) / threadsPerBlock_msm;
    const auto scratch_size = (num_blocks + 1) * MSM_BUCKET_COUNT * sizeof(G1ProjectivePoint);
    
    // Debug: Verify buffer size calculation
    printf("DEBUG wrapper: n=%d, num_blocks=%d, scratch_size=%zu bytes\n", n, num_blocks, scratch_size);
    
    // Ensure device is set before allocation
    cuda_set_device(gpu_index);
    
    // Allocate device memory
    auto* d_points = static_cast<G1Point*>(cuda_malloc_async(n * sizeof(G1Point), stream, gpu_index));
    auto* d_scalars = static_cast<uint64_t*>(cuda_malloc_async(n * sizeof(uint64_t), stream, gpu_index));
    auto* d_result = static_cast<G1ProjectivePoint*>(cuda_malloc_async(sizeof(G1ProjectivePoint), stream, gpu_index));
    auto* d_scratch = static_cast<G1ProjectivePoint*>(cuda_malloc_async(scratch_size, stream, gpu_index));
    
    if (!d_points || !d_scalars || !d_result || !d_scratch) {
        cuda_destroy_stream(stream, gpu_index);
        return -3;
    }
    
    // Zero-initialize result (like C++ test does, even though MSM initializes it internally)
    cuda_memset_async(d_result, 0, sizeof(G1ProjectivePoint), stream, gpu_index);
    
    // Copy points and scalars to device (points are in standard form from arkworks)
    cuda_memcpy_async_to_gpu(d_points, points, n * sizeof(G1Point), stream, gpu_index);
    cuda_memcpy_async_to_gpu(d_scalars, scalars, n * sizeof(uint64_t), stream, gpu_index);
    
    // CRITICAL: Synchronize to ensure copy completes before conversion
    cuda_synchronize_stream(stream, gpu_index);
    
    // CRITICAL: Convert points to Montgomery form (required for MSM)
    // MSM expects points in Montgomery form, just like the C++ test does
    // Use the synchronous version which handles synchronization internally
    point_to_montgomery_batch<G1Point>(stream, gpu_index, d_points, n);
    auto conv_err = cudaGetLastError();
    if (conv_err != cudaSuccess) {
        printf("ERROR: point_to_montgomery_batch failed: %s\n", cudaGetErrorString(conv_err));
        cuda_drop_async(d_points, stream, gpu_index);
        cuda_drop_async(d_scalars, stream, gpu_index);
        cuda_drop_async(d_result, stream, gpu_index);
        cuda_drop_async(d_scratch, stream, gpu_index);
        cuda_destroy_stream(stream, gpu_index);
        return -4;
    }
    
    
    // Run MSM (matches C++ test pattern exactly)
    // point_msm_u64_g1 internally synchronizes the stream, so no need to sync here
    point_msm_u64_g1(stream, gpu_index, d_result, d_points, d_scalars, d_scratch, n);
    
    // Check for CUDA errors after MSM
    auto err = cudaGetLastError();
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
    
    // Copy result back to the provided pointer
    cuda_memcpy_async_to_cpu(result, d_result, sizeof(G1ProjectivePoint), stream, gpu_index);
    
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
    auto stream = cuda_create_stream(gpu_index);
    if (stream == nullptr) {
        return -2;
    }
    
    // Initialize device
    cuda_set_device(gpu_index);
    // Initialize device generators (converts from standard to Montgomery form)
    init_device_generators(stream, gpu_index);
    cuda_synchronize_stream(stream, gpu_index);
    
    // Calculate scratch size
    // IMPORTANT: Must match the num_blocks calculation used in point_msm_u64_async_g2
    // MSM function uses threadsPerBlock = 64 for G2, so we must use the same for num_blocks calculation
    const auto threadsPerBlock_msm = 64;  // Match MSM function (point_msm_u64_async_g2 uses 64)
    const auto num_blocks = (n + threadsPerBlock_msm - 1) / threadsPerBlock_msm;
    const auto scratch_size = (num_blocks + 1) * MSM_BUCKET_COUNT * sizeof(G2ProjectivePoint);
    
    // Debug: Verify buffer size calculation
    printf("DEBUG wrapper G2: n=%d, num_blocks=%d, scratch_size=%zu bytes\n", n, num_blocks, scratch_size);
    
    // Allocate device memory
    auto* d_points = static_cast<G2Point*>(cuda_malloc_async(n * sizeof(G2Point), stream, gpu_index));
    auto* d_scalars = static_cast<uint64_t*>(cuda_malloc_async(n * sizeof(uint64_t), stream, gpu_index));
    auto* d_result = static_cast<G2ProjectivePoint*>(cuda_malloc_async(sizeof(G2ProjectivePoint), stream, gpu_index));
    auto* d_scratch = static_cast<G2ProjectivePoint*>(cuda_malloc_async(scratch_size, stream, gpu_index));
    
    if (!d_points || !d_scalars || !d_result || !d_scratch) {
        cuda_destroy_stream(stream, gpu_index);
        return -3;
    }
    
    // Copy data to device
    cuda_memcpy_async_to_gpu(d_points, points, n * sizeof(G2Point), stream, gpu_index);
    cuda_memcpy_async_to_gpu(d_scalars, scalars, n * sizeof(uint64_t), stream, gpu_index);
    
    // Points are already in Montgomery form (from arkworks' into_bigint())
    // No need to convert - they're already in the correct form for MSM
    
    // Note: MSM function initializes result to infinity internally
    // Run MSM
    point_msm_u64_g2(stream, gpu_index, d_result, d_points, d_scalars, d_scratch, n);
    
    // Check for CUDA errors after MSM
    auto err = cudaGetLastError();
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
