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
    int threadsPerBlock = 128;
    int num_blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    size_t scratch_size = (num_blocks + 1) * MSM_BUCKET_COUNT * sizeof(G1ProjectivePoint);
    
    // Allocate device memory
    G1Point* d_points = (G1Point*)cuda_malloc_async(n * sizeof(G1Point), stream, gpu_index);
    uint64_t* d_scalars = (uint64_t*)cuda_malloc_async(n * sizeof(uint64_t), stream, gpu_index);
    G1ProjectivePoint* d_result = (G1ProjectivePoint*)cuda_malloc_async(sizeof(G1ProjectivePoint), stream, gpu_index);
    G1ProjectivePoint* d_scratch = (G1ProjectivePoint*)cuda_malloc_async(scratch_size, stream, gpu_index);
    
    if (!d_points || !d_scalars || !d_result || !d_scratch) {
        cuda_destroy_stream(stream, gpu_index);
        return -3;
    }
    
    // Copy data to device
    cuda_memcpy_async_to_gpu(d_points, points, n * sizeof(G1Point), stream, gpu_index);
    cuda_memcpy_async_to_gpu(d_scalars, scalars, n * sizeof(uint64_t), stream, gpu_index);
    
    // Initialize result to infinity
    g1_projective_point_at_infinity_wrapper(d_result);
    
    // Run MSM
    point_msm_u64_g1(stream, gpu_index, d_result, d_points, d_scalars, d_scratch, n);
    
    // Copy result back
    cuda_memcpy_async_to_cpu(result, d_result, sizeof(G1ProjectivePoint), stream, gpu_index);
    
    // Synchronize
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
    cudaStream_t stream = cuda_create_stream(gpu_index);
    if (stream == nullptr) {
        return -2;
    }
    
    // Initialize device
    init_device_modulus(stream, gpu_index);
    init_device_curve(stream, gpu_index);
    init_device_generators(stream, gpu_index);
    
    // Calculate scratch size
    int threadsPerBlock = 128;
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
    
    // Initialize result to infinity
    g2_projective_point_at_infinity_wrapper(d_result);
    
    // Run MSM
    point_msm_u64_g2(stream, gpu_index, d_result, d_points, d_scalars, d_scratch, n);
    
    // Copy result back
    cuda_memcpy_async_to_cpu(result, d_result, sizeof(G2ProjectivePoint), stream, gpu_index);
    
    // Synchronize
    cuda_synchronize_stream(stream, gpu_index);
    
    // Cleanup
    cuda_drop_async(d_points, stream, gpu_index);
    cuda_drop_async(d_scalars, stream, gpu_index);
    cuda_drop_async(d_result, stream, gpu_index);
    cuda_drop_async(d_scratch, stream, gpu_index);
    cuda_destroy_stream(stream, gpu_index);
    
    return 0;
}

} // extern "C"
