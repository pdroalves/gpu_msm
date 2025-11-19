use std::env;
use std::path::PathBuf;

fn main() {
    // Tell cargo to link against the C++ library
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let project_root = manifest_dir.parent().unwrap();
    let build_dir = project_root.join("build");
    
    // Add CUDA library paths first (before linking our library)
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    } else if let Ok(cuda_home) = env::var("CUDA_HOME") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
    } else {
        // Try common CUDA installation paths
        let common_paths = [
            "/usr/local/cuda/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/opt/cuda/lib64",
        ];
        for path in &common_paths {
            if PathBuf::from(path).exists() {
                println!("cargo:rustc-link-search=native={}", path);
            }
        }
    }
    
    // Library is in build/ directory, not build/lib/
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    // Use whole-archive to ensure CUDA device code registration symbols are included
    println!("cargo:rustc-link-arg=-Wl,--whole-archive");
    println!("cargo:rustc-link-lib=static=fp_arithmetic");
    println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");
    
    // Link CUDA runtime (after our library) to provide CUDA runtime functions
    // This must come after our library so CUDA runtime can resolve device registration symbols
    println!("cargo:rustc-link-lib=dylib=cudart");
    
    // Link against C++ standard library (required for CUDA code)
    println!("cargo:rustc-link-lib=stdc++");
    
    // CUDA device code registration symbols (__cudaRegisterLinkedBinary_*) are weak symbols
    // that should be provided by the CUDA runtime library (libcudart.so)
    // They are resolved at runtime when CUDA device code is initialized
    
    // Rebuild if the C++ library changes
    println!("cargo:rerun-if-changed={}", build_dir.join("libfp_arithmetic.a").display());
    
    // Rebuild if C wrapper changes
    println!("cargo:rerun-if-changed={}", manifest_dir.join("src").join("c_wrapper.cu").display());
    
    // Compile CUDA stub file to provide weak implementations of device registration symbols
    let cuda_stubs = manifest_dir.join("src").join("cuda_stubs.c");
    if cuda_stubs.exists() {
        println!("cargo:rerun-if-changed={}", cuda_stubs.display());
        cc::Build::new()
            .file(&cuda_stubs)
            .compile("cuda_stubs");
    }
}

