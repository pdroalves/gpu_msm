use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Tell cargo to link against the C++ library
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let project_root = manifest_dir.parent().unwrap();
    let build_dir = project_root.join("build");
    
    // Check if c_wrapper.cu is newer than libfp_arithmetic.a
    // If so, we need to rebuild the CMake project
    let c_wrapper = manifest_dir.join("src").join("c_wrapper.cu");
    let lib_file = build_dir.join("libfp_arithmetic.a");
    
    let needs_rebuild = if c_wrapper.exists() && lib_file.exists() {
        let c_wrapper_meta = std::fs::metadata(&c_wrapper).ok();
        let lib_meta = std::fs::metadata(&lib_file).ok();
        
        match (c_wrapper_meta, lib_meta) {
            (Some(cw_meta), Some(lib_meta)) => {
                if let (Ok(cw_time), Ok(lib_time)) = (cw_meta.modified(), lib_meta.modified()) {
                    cw_time > lib_time
                } else {
                    false
                }
            }
            _ => false,
        }
    } else {
        false
    };
    
    // If c_wrapper.cu is newer, trigger CMake rebuild
    if needs_rebuild {
        println!("cargo:warning=c_wrapper.cu is newer than libfp_arithmetic.a, rebuilding CMake project...");
        // First ensure CMake is configured
        if !build_dir.join("CMakeCache.txt").exists() {
            println!("cargo:warning=CMake not configured, running cmake...");
            let configure_status = Command::new("cmake")
                .arg("-B")
                .arg(&build_dir)
                .arg("-S")
                .arg(&project_root)
                .status();
            if let Ok(s) = configure_status {
                if !s.success() {
                    println!("cargo:warning=CMake configuration failed!");
                }
            }
        }
        
        let status = Command::new("cmake")
            .arg("--build")
            .arg(&build_dir)
            .arg("--target")
            .arg("fp_arithmetic")
            .status();
        
        if let Ok(s) = status {
            if !s.success() {
                println!("cargo:warning=CMake rebuild failed, but continuing...");
            } else {
                println!("cargo:warning=CMake rebuild completed successfully");
            }
        } else {
            println!("cargo:warning=Failed to run cmake --build");
        }
    }
    
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

