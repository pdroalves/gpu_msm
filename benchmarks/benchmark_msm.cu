#include "curve.h"
#include "fp.h"
#include "fp2.h"
#include "device.h"
#include <benchmark/benchmark.h>
#include <random>
#include <cuda_runtime.h>
#include <cstdint>

// Helper to get modulus (duplicated from test utilities)
static Fp get_modulus() {
    Fp p;
    p.limb[0] = 0x311c0026aab0aaabULL;
    p.limb[1] = 0x56ee4528c573b5ccULL;
    p.limb[2] = 0x824e6dc3e23acdeeULL;
    p.limb[3] = 0x0f75a64bbac71602ULL;
    p.limb[4] = 0x0095a4b78a02fe32ULL;
    p.limb[5] = 0x200fc34965aad640ULL;
    p.limb[6] = 0x3cdee0fb28c5e535ULL;
    return p;
}

// Global stream and gpu_index for benchmarks
static cudaStream_t g_benchmark_stream = nullptr;
static uint32_t g_gpu_index = 0;

// Initialize device modulus, curve, and generators
static void init_benchmark() {
    static bool initialized = false;
    if (!initialized) {
        g_gpu_index = 0;
        
        // Create a CUDA stream using library function
        g_benchmark_stream = cuda_create_stream(g_gpu_index);
        
        // Initialize device generators (converts from standard to Montgomery form)
        init_device_generators(g_benchmark_stream, g_gpu_index);
        
        // Synchronize to ensure initialization completes
        cuda_synchronize_stream(g_benchmark_stream, g_gpu_index);
        
        initialized = true;
    }
}

// Helper to generate random Fp value
static Fp random_fp_value(std::mt19937_64& rng) {
    Fp result;
    Fp p = get_modulus();
    
    // Generate random limbs
    for (int i = 0; i < FP_LIMBS; i++) {
        result.limb[i] = rng();
    }
    
    // Reduce if needed
    while (fp_cmp(result, p) >= 0) {
        Fp reduced;
        fp_sub_raw(reduced, result, p);
        fp_copy(result, reduced);
    }
    
    return result;
}

// Helper to generate random G1 point (not necessarily on curve, but valid coordinates)
static G1Point random_g1_point(std::mt19937_64& rng) {
    G1Point point;
    point.infinity = false;
    point.x = random_fp_value(rng);
    point.y = random_fp_value(rng);
    return point;
}

// Helper to generate random G2 point
static G2Point random_g2_point(std::mt19937_64& rng) {
    G2Point point;
    point.infinity = false;
    point.x.c0 = random_fp_value(rng);
    point.x.c1 = random_fp_value(rng);
    point.y.c0 = random_fp_value(rng);
    point.y.c1 = random_fp_value(rng);
    return point;
}

// Helper to generate random 64-bit scalar
static uint64_t random_scalar_u64(std::mt19937_64& rng) {
    return rng();
}

// Benchmark G1 scalar multiplication (single point)
static void BM_G1_ScalarMul(benchmark::State& state) {
    init_benchmark();
    
    std::mt19937_64 rng(42);
    G1Point h_point = random_g1_point(rng);
    uint64_t scalar = random_scalar_u64(rng);
    
    // Allocate device memory
    G1Point* d_point = (G1Point*)cuda_malloc_async(sizeof(G1Point), g_benchmark_stream, g_gpu_index);
    G1Point* d_result = (G1Point*)cuda_malloc_async(sizeof(G1Point), g_benchmark_stream, g_gpu_index);
    
    // Copy point to device
    cuda_memcpy_async_to_gpu(d_point, &h_point, sizeof(G1Point), g_benchmark_stream, g_gpu_index);
    cuda_synchronize_stream(g_benchmark_stream, g_gpu_index);
    
    for (auto _ : state) {
        point_scalar_mul_u64<G1Point>(g_benchmark_stream, g_gpu_index, d_result, d_point, scalar);
        benchmark::ClobberMemory();
    }
    
    cuda_synchronize_stream(g_benchmark_stream, g_gpu_index);
    state.SetItemsProcessed(state.iterations());
    
    cuda_drop_async(d_point, g_benchmark_stream, g_gpu_index);
    cuda_drop_async(d_result, g_benchmark_stream, g_gpu_index);
}

// Benchmark G2 scalar multiplication (single point)
static void BM_G2_ScalarMul(benchmark::State& state) {
    init_benchmark();
    
    std::mt19937_64 rng(42);
    G2Point h_point = random_g2_point(rng);
    uint64_t scalar = random_scalar_u64(rng);
    
    // Allocate device memory
    G2Point* d_point = (G2Point*)cuda_malloc_async(sizeof(G2Point), g_benchmark_stream, g_gpu_index);
    G2Point* d_result = (G2Point*)cuda_malloc_async(sizeof(G2Point), g_benchmark_stream, g_gpu_index);
    
    // Copy point to device
    cuda_memcpy_async_to_gpu(d_point, &h_point, sizeof(G2Point), g_benchmark_stream, g_gpu_index);
    cuda_synchronize_stream(g_benchmark_stream, g_gpu_index);
    
    for (auto _ : state) {
        point_scalar_mul_u64<G2Point>(g_benchmark_stream, g_gpu_index, d_result, d_point, scalar);
        benchmark::ClobberMemory();
    }
    
    cuda_synchronize_stream(g_benchmark_stream, g_gpu_index);
    state.SetItemsProcessed(state.iterations());
    
    cuda_drop_async(d_point, g_benchmark_stream, g_gpu_index);
    cuda_drop_async(d_result, g_benchmark_stream, g_gpu_index);
}

// Benchmark G1 MSM with 64-bit scalars
static void BM_G1_MSM_U64(benchmark::State& state) {
    init_benchmark();
    
    const int n = static_cast<int>(state.range(0));
    std::mt19937_64 rng(42);
    
    // Calculate required scratch space: (num_blocks + 1) * MSM_BUCKET_COUNT (projective points)
    const int threadsPerBlock = 128;  // Reduced for projective points
    const int num_blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    const size_t scratch_size = (num_blocks + 1) * MSM_BUCKET_COUNT * sizeof(G1ProjectivePoint);
    
    // Allocate device memory
    G1Point* d_points = (G1Point*)cuda_malloc_async(n * sizeof(G1Point), g_benchmark_stream, g_gpu_index);
    uint64_t* d_scalars = (uint64_t*)cuda_malloc_async(n * sizeof(uint64_t), g_benchmark_stream, g_gpu_index);
    G1ProjectivePoint* d_result = (G1ProjectivePoint*)cuda_malloc_async(sizeof(G1ProjectivePoint), g_benchmark_stream, g_gpu_index);
    G1ProjectivePoint* d_scratch = (G1ProjectivePoint*)cuda_malloc_async(scratch_size, g_benchmark_stream, g_gpu_index);
    
    // Prepare host data
    G1Point* h_points = new G1Point[n];
    uint64_t* h_scalars = new uint64_t[n];
    
    // Initialize with random values
    for (int i = 0; i < n; i++) {
        h_points[i] = random_g1_point(rng);
        h_scalars[i] = random_scalar_u64(rng);
    }
    
    // Copy to device (once, before benchmark loop)
    cuda_memcpy_async_to_gpu(d_points, h_points, n * sizeof(G1Point), g_benchmark_stream, g_gpu_index);
    cuda_memcpy_async_to_gpu(d_scalars, h_scalars, n * sizeof(uint64_t), g_benchmark_stream, g_gpu_index);
    
    // Convert points to Montgomery form (required for performance - all operations use Montgomery)
    int threadsPerBlock_conv = 256;
    int blocks_conv = (n + threadsPerBlock_conv - 1) / threadsPerBlock_conv;
    point_to_montgomery_batch<G1Point>(g_benchmark_stream, g_gpu_index, d_points, n);
    check_cuda_error(cudaGetLastError());
    
    // Initialize result and scratch memory to zero (once, before benchmark loop)
    cuda_memset_async(d_result, 0, sizeof(G1Point), g_benchmark_stream, g_gpu_index);
    cuda_memset_async(d_scratch, 0, scratch_size, g_benchmark_stream, g_gpu_index);
    
    // Synchronize once before benchmark loop to ensure all setup is complete
    cuda_synchronize_stream(g_benchmark_stream, g_gpu_index);
    
    // Benchmark loop: only measure the MSM computation, no memory operations
    for (auto _ : state) {
        point_msm_u64_g1(g_benchmark_stream, g_gpu_index, d_result, d_points, d_scalars, d_scratch, n);
        benchmark::ClobberMemory();
    }
    
    // Synchronize once after benchmark loop to ensure all iterations complete
    cuda_synchronize_stream(g_benchmark_stream, g_gpu_index);
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * (sizeof(G1Point) + sizeof(uint64_t)));
    
    delete[] h_points;
    delete[] h_scalars;
    cuda_drop_async(d_points, g_benchmark_stream, g_gpu_index);
    cuda_drop_async(d_scalars, g_benchmark_stream, g_gpu_index);
    cuda_drop_async(d_result, g_benchmark_stream, g_gpu_index);
    cuda_drop_async(d_scratch, g_benchmark_stream, g_gpu_index);
}

// Benchmark G2 MSM with 64-bit scalars
static void BM_G2_MSM_U64(benchmark::State& state) {
    init_benchmark();
    
    const int n = static_cast<int>(state.range(0));
    std::mt19937_64 rng(42);
    
    // Calculate required scratch space: (num_blocks + 1) * MSM_BUCKET_COUNT (projective points)
    int threadsPerBlock = 64;  // Reduced for G2 projective points
    int num_blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    size_t scratch_size = (num_blocks + 1) * MSM_BUCKET_COUNT * sizeof(G2ProjectivePoint);
    
    // Allocate device memory
    G2Point* d_points = (G2Point*)cuda_malloc_async(n * sizeof(G2Point), g_benchmark_stream, g_gpu_index);
    uint64_t* d_scalars = (uint64_t*)cuda_malloc_async(n * sizeof(uint64_t), g_benchmark_stream, g_gpu_index);
    G2ProjectivePoint* d_result = (G2ProjectivePoint*)cuda_malloc_async(sizeof(G2ProjectivePoint), g_benchmark_stream, g_gpu_index);
    G2ProjectivePoint* d_scratch = (G2ProjectivePoint*)cuda_malloc_async(scratch_size, g_benchmark_stream, g_gpu_index);
    
    // Prepare host data
    G2Point* h_points = new G2Point[n];
    uint64_t* h_scalars = new uint64_t[n];
    
    // Initialize with random values
    for (int i = 0; i < n; i++) {
        h_points[i] = random_g2_point(rng);
        h_scalars[i] = random_scalar_u64(rng);
    }
    
    // Copy to device (once, before benchmark loop)
    cuda_memcpy_async_to_gpu(d_points, h_points, n * sizeof(G2Point), g_benchmark_stream, g_gpu_index);
    cuda_memcpy_async_to_gpu(d_scalars, h_scalars, n * sizeof(uint64_t), g_benchmark_stream, g_gpu_index);
    
    // Convert points to Montgomery form (required for performance - all operations use Montgomery)
    int threadsPerBlock_conv = 256;
    int blocks_conv = (n + threadsPerBlock_conv - 1) / threadsPerBlock_conv;
    point_to_montgomery_batch<G2Point>(g_benchmark_stream, g_gpu_index, d_points, n);
    check_cuda_error(cudaGetLastError());
    
    // Initialize result and scratch memory to zero (once, before benchmark loop)
    cuda_memset_async(d_result, 0, sizeof(G2ProjectivePoint), g_benchmark_stream, g_gpu_index);
    cuda_memset_async(d_scratch, 0, scratch_size, g_benchmark_stream, g_gpu_index);
    
    // Synchronize once before benchmark loop to ensure all setup is complete
    cuda_synchronize_stream(g_benchmark_stream, g_gpu_index);
    
    // Benchmark loop: only measure the MSM computation, no memory operations
    for (auto _ : state) {
        point_msm_u64_g2(g_benchmark_stream, g_gpu_index, d_result, d_points, d_scalars, d_scratch, n);
        benchmark::ClobberMemory();
    }
    
    // Synchronize once after benchmark loop to ensure all iterations complete
    cuda_synchronize_stream(g_benchmark_stream, g_gpu_index);
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * (sizeof(G2Point) + sizeof(uint64_t)));
    
    delete[] h_points;
    delete[] h_scalars;
    cuda_drop_async(d_points, g_benchmark_stream, g_gpu_index);
    cuda_drop_async(d_scalars, g_benchmark_stream, g_gpu_index);
    cuda_drop_async(d_result, g_benchmark_stream, g_gpu_index);
    cuda_drop_async(d_scratch, g_benchmark_stream, g_gpu_index);
}

// Benchmark G1 MSM with generator point (common use case)
static void BM_G1_MSM_Generator(benchmark::State& state) {
    init_benchmark();
    
    const int n = static_cast<int>(state.range(0));
    std::mt19937_64 rng(42);
    
    // Get generator point
    const G1Point& G = g1_generator();
    
    // Calculate required scratch space: (num_blocks + 1) * MSM_BUCKET_COUNT (projective points)
    int threadsPerBlock = 128;  // Reduced for projective points
    int num_blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    size_t scratch_size = (num_blocks + 1) * MSM_BUCKET_COUNT * sizeof(G1ProjectivePoint);
    
    // Allocate device memory
    G1Point* d_points = (G1Point*)cuda_malloc_async(n * sizeof(G1Point), g_benchmark_stream, g_gpu_index);
    uint64_t* d_scalars = (uint64_t*)cuda_malloc_async(n * sizeof(uint64_t), g_benchmark_stream, g_gpu_index);
    G1ProjectivePoint* d_result = (G1ProjectivePoint*)cuda_malloc_async(sizeof(G1ProjectivePoint), g_benchmark_stream, g_gpu_index);
    G1ProjectivePoint* d_scratch = (G1ProjectivePoint*)cuda_malloc_async(scratch_size, g_benchmark_stream, g_gpu_index);
    
    // Prepare host data - all points are the generator
    G1Point* h_points = new G1Point[n];
    uint64_t* h_scalars = new uint64_t[n];
    
    // Initialize: all points are generator, random scalars
    for (int i = 0; i < n; i++) {
        fp_copy(h_points[i].x, G.x);
        fp_copy(h_points[i].y, G.y);
        h_points[i].infinity = G.infinity;
        h_scalars[i] = random_scalar_u64(rng);
    }
    
    // Copy to device (once, before benchmark loop)
    cuda_memcpy_async_to_gpu(d_points, h_points, n * sizeof(G1Point), g_benchmark_stream, g_gpu_index);
    cuda_memcpy_async_to_gpu(d_scalars, h_scalars, n * sizeof(uint64_t), g_benchmark_stream, g_gpu_index);
    
    // Convert points to Montgomery form (required for performance - all operations use Montgomery)
    int threadsPerBlock_conv = 256;
    int blocks_conv = (n + threadsPerBlock_conv - 1) / threadsPerBlock_conv;
    point_to_montgomery_batch<G1Point>(g_benchmark_stream, g_gpu_index, d_points, n);
    check_cuda_error(cudaGetLastError());
    
    // Initialize result and scratch memory to zero (once, before benchmark loop)
    cuda_memset_async(d_result, 0, sizeof(G1ProjectivePoint), g_benchmark_stream, g_gpu_index);
    cuda_memset_async(d_scratch, 0, scratch_size, g_benchmark_stream, g_gpu_index);
    
    // Synchronize once before benchmark loop to ensure all setup is complete
    cuda_synchronize_stream(g_benchmark_stream, g_gpu_index);
    
    // Benchmark loop: only measure the MSM computation, no memory operations
    for (auto _ : state) {
        point_msm_u64_g1(g_benchmark_stream, g_gpu_index, d_result, d_points, d_scalars, d_scratch, n);
        benchmark::ClobberMemory();
    }
    
    // Synchronize once after benchmark loop to ensure all iterations complete
    cuda_synchronize_stream(g_benchmark_stream, g_gpu_index);
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * (sizeof(G1Point) + sizeof(uint64_t)));
    
    delete[] h_points;
    delete[] h_scalars;
    cuda_drop_async(d_points, g_benchmark_stream, g_gpu_index);
    cuda_drop_async(d_scalars, g_benchmark_stream, g_gpu_index);
    cuda_drop_async(d_result, g_benchmark_stream, g_gpu_index);
    cuda_drop_async(d_scratch, g_benchmark_stream, g_gpu_index);
}

// Benchmark G2 MSM with generator point (common use case)
static void BM_G2_MSM_Generator(benchmark::State& state) {
    init_benchmark();
    
    const int n = static_cast<int>(state.range(0));
    std::mt19937_64 rng(42);
    
    // Get generator point
    const G2Point& G = g2_generator();
    
    // Calculate required scratch space: (num_blocks + 1) * MSM_BUCKET_COUNT (projective points)
    int threadsPerBlock = 64;  // Reduced for G2 projective points
    int num_blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    size_t scratch_size = (num_blocks + 1) * MSM_BUCKET_COUNT * sizeof(G2ProjectivePoint);
    
    // Allocate device memory
    G2Point* d_points = (G2Point*)cuda_malloc_async(n * sizeof(G2Point), g_benchmark_stream, g_gpu_index);
    uint64_t* d_scalars = (uint64_t*)cuda_malloc_async(n * sizeof(uint64_t), g_benchmark_stream, g_gpu_index);
    G2ProjectivePoint* d_result = (G2ProjectivePoint*)cuda_malloc_async(sizeof(G2ProjectivePoint), g_benchmark_stream, g_gpu_index);
    G2ProjectivePoint* d_scratch = (G2ProjectivePoint*)cuda_malloc_async(scratch_size, g_benchmark_stream, g_gpu_index);
    
    // Prepare host data - all points are the generator
    G2Point* h_points = new G2Point[n];
    uint64_t* h_scalars = new uint64_t[n];
    
    // Initialize: all points are generator, random scalars
    for (int i = 0; i < n; i++) {
        fp2_copy(h_points[i].x, G.x);
        fp2_copy(h_points[i].y, G.y);
        h_points[i].infinity = G.infinity;
        h_scalars[i] = random_scalar_u64(rng);
    }
    
    // Copy to device (once, before benchmark loop)
    cuda_memcpy_async_to_gpu(d_points, h_points, n * sizeof(G2Point), g_benchmark_stream, g_gpu_index);
    cuda_memcpy_async_to_gpu(d_scalars, h_scalars, n * sizeof(uint64_t), g_benchmark_stream, g_gpu_index);
    
    // Convert points to Montgomery form (required for performance - all operations use Montgomery)
    int threadsPerBlock_conv = 256;
    int blocks_conv = (n + threadsPerBlock_conv - 1) / threadsPerBlock_conv;
    point_to_montgomery_batch<G2Point>(g_benchmark_stream, g_gpu_index, d_points, n);
    check_cuda_error(cudaGetLastError());
    
    // Initialize result and scratch memory to zero (once, before benchmark loop)
    cuda_memset_async(d_result, 0, sizeof(G2Point), g_benchmark_stream, g_gpu_index);
    cuda_memset_async(d_scratch, 0, scratch_size, g_benchmark_stream, g_gpu_index);
    
    // Synchronize once before benchmark loop to ensure all setup is complete
    cuda_synchronize_stream(g_benchmark_stream, g_gpu_index);
    
    // Benchmark loop: only measure the MSM computation, no memory operations
    for (auto _ : state) {
        point_msm_u64_g2(g_benchmark_stream, g_gpu_index, d_result, d_points, d_scalars, d_scratch, n);
        benchmark::ClobberMemory();
    }
    
    // Synchronize once after benchmark loop to ensure all iterations complete
    cuda_synchronize_stream(g_benchmark_stream, g_gpu_index);
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * (sizeof(G2Point) + sizeof(uint64_t)));
    
    delete[] h_points;
    delete[] h_scalars;
    cuda_drop_async(d_points, g_benchmark_stream, g_gpu_index);
    cuda_drop_async(d_scalars, g_benchmark_stream, g_gpu_index);
    cuda_drop_async(d_result, g_benchmark_stream, g_gpu_index);
    cuda_drop_async(d_scratch, g_benchmark_stream, g_gpu_index);
}

// Register scalar multiplication benchmarks
BENCHMARK(BM_G1_ScalarMul);
BENCHMARK(BM_G2_ScalarMul);

// Register MSM benchmarks with different sizes
// Range from 10 to 10,000 points
BENCHMARK(BM_G1_MSM_U64)->Range(10, 10000)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_G2_MSM_U64)->Range(10, 10000)->Unit(benchmark::kMillisecond);

// MSM with generator (common use case)
BENCHMARK(BM_G1_MSM_Generator)->Range(10, 10000)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_G2_MSM_Generator)->Range(10, 10000)->Unit(benchmark::kMillisecond);

// Run benchmarks
BENCHMARK_MAIN();

