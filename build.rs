//! Build script for furiosa-optimizer
//!
//! Generates Rust code from ONNX protobuf definitions using prost-build.

use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_path = PathBuf::from("proto/onnx.proto");

    // Verify proto file exists
    if !proto_path.exists() {
        return Err(format!(
            "ONNX proto file not found at: {}\n\
             Please ensure proto/onnx.proto exists.\n\
             Download from: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto",
            proto_path.display()
        )
        .into());
    }

    // Get output directory from cargo
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);

    // Configure prost-build
    let mut config = prost_build::Config::new();

    // Performance: use BTreeMap for deterministic ordering
    config.btree_map(["."]);

    // Set output directory
    config.out_dir(&out_dir);

    // Compile ONNX proto
    config.compile_protos(&[&proto_path], &["proto/"])?;

    // Tell cargo to rerun if proto files change
    println!("cargo:rerun-if-changed=proto/onnx.proto");
    println!("cargo:rerun-if-changed=build.rs");

    Ok(())
}
