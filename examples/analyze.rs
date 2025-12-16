//! Model analysis tool
//!
//! Analyze ONNX model structure and node distribution.
//!
//! Run with: cargo run --release --example analyze -- model.onnx

use std::collections::HashMap;
use std::env;
use std::path::Path;

use furiosa_optimizer::io::{load_model, validate_model};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <model.onnx>", args[0]);
        std::process::exit(1);
    }

    let path = Path::new(&args[1]);
    println!("Analyzing: {}", path.display());
    println!("{}", "=".repeat(60));

    let model = load_model(path)?;

    // Basic info
    println!("\n## Model Info");
    println!("  IR Version: {}", model.ir_version);
    println!(
        "  Producer: {} {}",
        model.producer_name, model.producer_version
    );

    // Opsets
    println!("\n## Opsets");
    for opset in &model.opset_import {
        let domain = if opset.domain.is_empty() {
            "ai.onnx"
        } else {
            &opset.domain
        };
        println!("  {}: {}", domain, opset.version);
    }

    // Graph analysis
    if let Some(graph) = &model.graph {
        println!("\n## Graph: {}", graph.name);
        println!("  Nodes: {}", graph.node.len());
        println!("  Initializers: {}", graph.initializer.len());
        println!("  Inputs: {}", graph.input.len());
        println!("  Outputs: {}", graph.output.len());

        // Node type distribution
        let mut op_counts: HashMap<&str, usize> = HashMap::new();
        for node in &graph.node {
            *op_counts.entry(&node.op_type).or_insert(0) += 1;
        }

        // Sort by count
        let mut sorted: Vec<_> = op_counts.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));

        println!("\n## Node Distribution");
        println!("  {:<25} {:>6}", "Op Type", "Count");
        println!("  {}", "-".repeat(35));
        for (op, count) in &sorted {
            let bar = "█".repeat((**count).min(20));
            println!("  {:<25} {:>6} {}", op, count, bar);
        }

        // Optimization opportunities
        println!("\n## Optimization Opportunities");

        let bn_count = op_counts.get("BatchNormalization").unwrap_or(&0);
        let conv_count = op_counts.get("Conv").unwrap_or(&0);
        let identity_count = op_counts.get("Identity").unwrap_or(&0);
        let dropout_count = op_counts.get("Dropout").unwrap_or(&0);

        if *bn_count > 0 && *conv_count > 0 {
            println!(
                "  ✅ Conv+BN Fusion: {} potential fusions",
                bn_count.min(conv_count)
            );
        }
        if *identity_count > 0 {
            println!("  ✅ Identity Elimination: {} nodes", identity_count);
        }
        if *dropout_count > 0 {
            println!("  ✅ Dropout Elimination: {} nodes", dropout_count);
        }

        // Squeeze/Unsqueeze without axes
        let squeeze_count = op_counts.get("Squeeze").unwrap_or(&0);
        let unsqueeze_count = op_counts.get("Unsqueeze").unwrap_or(&0);
        if *squeeze_count > 0 || *unsqueeze_count > 0 {
            println!(
                "  ✅ Axes Inference: {} Squeeze + {} Unsqueeze",
                squeeze_count, unsqueeze_count
            );
        }

        // Input/Output analysis
        println!("\n## Inputs");
        for input in &graph.input {
            // Skip initializers that appear in inputs
            if graph.initializer.iter().any(|i| i.name == input.name) {
                continue;
            }
            print!("  {} ", input.name);
            if let Some(ref type_proto) = input.r#type {
                if let Some(ref value) = type_proto.value {
                    match value {
                        furiosa_optimizer::proto::onnx::type_proto::Value::TensorType(t) => {
                            if let Some(ref shape) = t.shape {
                                let dims: Vec<String> = shape.dim.iter().map(|d| {
                                    match &d.value {
                                        Some(furiosa_optimizer::proto::onnx::tensor_shape_proto::dimension::Value::DimValue(v)) => v.to_string(),
                                        Some(furiosa_optimizer::proto::onnx::tensor_shape_proto::dimension::Value::DimParam(p)) => p.clone(),
                                        None => "?".to_string(),
                                    }
                                }).collect();
                                print!("[{}]", dims.join(", "));
                            }
                        }
                        _ => {}
                    }
                }
            }
            println!();
        }

        println!("\n## Outputs");
        for output in &graph.output {
            println!("  {}", output.name);
        }
    }

    // Validation
    println!("\n## Validation");
    let result = validate_model(&model);
    if result.is_valid {
        println!("  ✅ Model is valid");
    } else {
        println!("  ❌ Validation errors:");
        for err in &result.errors {
            println!("    - {}", err);
        }
    }
    if !result.warnings.is_empty() {
        println!("  ⚠️ Warnings:");
        for warn in &result.warnings {
            println!("    - {}", warn);
        }
    }

    Ok(())
}
