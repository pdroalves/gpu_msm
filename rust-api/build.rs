use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Tell cargo to link against the C++ library
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let project_root = manifest_dir.parent().unwrap();
    let build_dir = project_root.join("build");
    
    // Check if any CMake source files are newer than libfp_arithmetic.a
    // If so, we need to rebuild the CMake project
    let lib_file = build_dir.join("libfp_arithmetic.a");
    
    // List of all source files that CMake compiles (from CMakeLists.txt)
    let cmake_sources = [
        project_root.join("src").join("device.cu"),
        project_root.join("src").join("fp.cu"),
        project_root.join("src").join("fp_kernels.cu"),
        project_root.join("src").join("fp2.cu"),
        project_root.join("src").join("fp2_kernels.cu"),
        project_root.join("src").join("curve.cu"),
        manifest_dir.join("src").join("c_wrapper.cu"),
    ];
    
    let needs_rebuild = if lib_file.exists() {
        let lib_meta = std::fs::metadata(&lib_file).ok();
        let lib_time = lib_meta.and_then(|m| m.modified().ok());
        
        // Check if any source file is newer than the library
        cmake_sources.iter().any(|source| {
            if !source.exists() {
                return false;
            }
            let source_meta = std::fs::metadata(source).ok();
            let source_time = source_meta.and_then(|m| m.modified().ok());
            
            match (source_time, lib_time) {
                (Some(st), Some(lt)) => st > lt,
                _ => false,
            }
        })
    } else {
        // If library doesn't exist, we need to build it
        true
    };
    
    // If any source file is newer, trigger CMake rebuild
    if needs_rebuild {
        println!("cargo:warning=CMake source files are newer than libfp_arithmetic.a, rebuilding CMake project...");
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
    
    // Rebuild if any CMake source files change
    for source in &cmake_sources {
        if source.exists() {
            println!("cargo:rerun-if-changed={}", source.display());
        }
    }
    
    // Compile CUDA stub file to provide weak implementations of device registration symbols
    let cuda_stubs = manifest_dir.join("src").join("cuda_stubs.c");
    if cuda_stubs.exists() {
        println!("cargo:rerun-if-changed={}", cuda_stubs.display());
        cc::Build::new()
            .file(&cuda_stubs)
            .compile("cuda_stubs");
    }
}

