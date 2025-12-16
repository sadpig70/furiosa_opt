//! Example: ONNX model optimization
//!
//! This example demonstrates how to use furiosa-optimizer to optimize an ONNX model.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example optimize_model -- input.onnx output.onnx
//! ```

use std::env;

use furiosa_optimizer::io::{
    get_model_info, load_model, optimize_file, validate_model, OptimizeOptions,
};
use furiosa_optimizer::prelude::*;

fn main() -> OnnxResult<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <input.onnx> [output.onnx] [options]", args[0]);
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --no-fuse       Disable fusion passes (Conv+BN, etc.)");
        eprintln!("  --no-eliminate  Disable elimination passes (Identity, etc.)");
        eprintln!("  --info          Show model info only, don't optimize");
        eprintln!("  --validate      Validate model only, don't optimize");
        std::process::exit(1);
    }

    let input_path = &args[1];

    // Parse options
    let info_only = args.contains(&"--info".to_string());
    let validate_only = args.contains(&"--validate".to_string());
    let no_fuse = args.contains(&"--no-fuse".to_string());
    let no_eliminate = args.contains(&"--no-eliminate".to_string());

    // Info only mode
    if info_only {
        let info = get_model_info(input_path)?;
        println!("Model Information:");
        println!("  IR Version: {}", info.ir_version);
        println!(
            "  Producer: {} {}",
            info.producer_name, info.producer_version
        );
        println!("  Graph: {}", info.graph_name);
        println!("  Nodes: {}", info.node_count);
        println!("  Initializers: {}", info.initializer_count);
        println!("  Inputs: {:?}", info.inputs);
        println!("  Outputs: {:?}", info.outputs);
        println!("  Opsets: {:?}", info.opsets);
        return Ok(());
    }

    // Validate only mode
    if validate_only {
        let model = load_model(input_path)?;
        let result = validate_model(&model);

        if result.is_valid {
            println!("Model is valid");
        } else {
            println!("Model has errors:");
            for err in &result.errors {
                println!("  - {}", err);
            }
        }

        if !result.warnings.is_empty() {
            println!("Warnings:");
            for warn in &result.warnings {
                println!("  - {}", warn);
            }
        }

        return Ok(());
    }

    // Need output path for optimization
    if args.len() < 3 || args[2].starts_with("--") {
        eprintln!("Error: output path required for optimization");
        eprintln!("Usage: {} <input.onnx> <output.onnx>", args[0]);
        std::process::exit(1);
    }

    let output_path = &args[2];

    // Optimize
    println!("Optimizing {} -> {}", input_path, output_path);

    let options = OptimizeOptions {
        fuse: !no_fuse,
        eliminate: !no_eliminate,
        ..Default::default()
    };

    let stats = optimize_file(input_path, output_path, options)?;

    println!();
    println!("Optimization Results:");
    println!("  Original nodes:  {}", stats.original_nodes);
    println!("  Optimized nodes: {}", stats.optimized_nodes);
    println!(
        "  Nodes reduced:   {} ({:.1}%)",
        stats.nodes_reduced,
        stats.node_reduction_percent()
    );
    println!("  Original size:   {} bytes", stats.original_size);
    println!(
        "  Optimized size:  {} bytes ({:.1}% reduction)",
        stats.optimized_size,
        stats.size_reduction_percent()
    );
    println!();
    println!("Saved to {}", output_path);

    Ok(())
}
