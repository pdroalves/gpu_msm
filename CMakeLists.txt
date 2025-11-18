cmake_minimum_required(VERSION 3.18)
project(FpArithmetic VERSION 1.0.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture - adjust based on your GPU
# Common values:
# - sm_60: Pascal (GTX 10xx)
# - sm_70: Volta (V100)
# - sm_75: Turing (RTX 20xx, GTX 16xx)
# - sm_80: Ampere (A100, RTX 30xx)
# - sm_86: Ampere (RTX 30xx consumer)
# - sm_89: Ada Lovelace (RTX 40xx)
# - sm_90: Hopper (H100)
set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures to build for")

# Enable CUDA separable compilation for better optimization
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -O0 -G")

# Additional CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall -Xcompiler -Wextra")

# DEBUG flag for debug tests
option(DEBUG "Enable debug tests and verbose output" OFF)

# Enable testing
enable_testing()

# Fetch Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Fetch Google Benchmark
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# Source files
set(FP_SOURCES
    src/device.cu
    src/fp.cu
    src/fp_kernels.cu
    src/fp2.cu
    src/fp2_kernels.cu
    src/curve.cu
)

set(FP_HEADERS
    include/device.h
    include/fp.h
    include/fp_kernels.h
    include/fp2.h
    include/fp2_kernels.h
    include/curve.h
)

# Create library
add_library(fp_arithmetic STATIC ${FP_SOURCES} ${FP_HEADERS})

# Set target properties
set_target_properties(fp_arithmetic PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(fp_arithmetic
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Test executable
add_executable(test_fp tests/test_fp.cu)
target_link_libraries(test_fp 
    fp_arithmetic
    GTest::gtest_main
)

# Set CUDA properties for test executable
set_target_properties(test_fp PROPERTIES
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(test_fp)

# Test executable for Fp2
add_executable(test_fp2 tests/test_fp2.cu)
target_link_libraries(test_fp2 
    fp_arithmetic
    GTest::gtest_main
)

# Set CUDA properties for test executable
set_target_properties(test_fp2 PROPERTIES
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Register Fp2 tests with CTest
gtest_discover_tests(test_fp2)

# Test executable for MSM
add_executable(test_msm tests/test_msm.cu)
target_link_libraries(test_msm 
    fp_arithmetic
    GTest::gtest_main
)

# Set CUDA properties for MSM test executable
set_target_properties(test_msm PROPERTIES
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Add DEBUG preprocessor definition if DEBUG is enabled
if(DEBUG)
    target_compile_definitions(test_msm PRIVATE DEBUG=1)
    message(STATUS "DEBUG mode enabled: debug tests will be compiled")
else()
    message(STATUS "DEBUG mode disabled: debug tests will be skipped")
endif()

# Register MSM tests with CTest
gtest_discover_tests(test_msm)

# Benchmark executable for Fp
add_executable(benchmark_fp benchmarks/benchmark_fp.cu)
target_link_libraries(benchmark_fp 
    fp_arithmetic
    benchmark::benchmark
)

# Set CUDA properties for benchmark executable
set_target_properties(benchmark_fp PROPERTIES
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Benchmark executable for Fp2
add_executable(benchmark_fp2 benchmarks/benchmark_fp2.cu)
target_link_libraries(benchmark_fp2 
    fp_arithmetic
    benchmark::benchmark
)

# Set CUDA properties for benchmark executable
set_target_properties(benchmark_fp2 PROPERTIES
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Benchmark executable for MSM
add_executable(benchmark_msm benchmarks/benchmark_msm.cu)
target_link_libraries(benchmark_msm 
    fp_arithmetic
    benchmark::benchmark
)

# Set CUDA properties for MSM benchmark executable
set_target_properties(benchmark_msm PROPERTIES
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Installation (optional)
install(TARGETS fp_arithmetic
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES ${FP_HEADERS}
    DESTINATION include
)

# Print configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "DEBUG flag: ${DEBUG}")
