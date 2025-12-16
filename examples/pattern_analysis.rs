//! Pattern analysis tool for NLP models
//!
//! Analyze optimization opportunities in ONNX models.

use std::collections::HashMap;
use std::env;
use std::path::Path;

use furiosa_optimizer::graph::GraphContext;
use furiosa_optimizer::io::load_model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.onnx>", args[0]);
        std::process::exit(1);
    }

    let path = Path::new(&args[1]);
    println!("Pattern Analysis: {}", path.display());
    println!("{}", "=".repeat(60));

    let model = load_model(path)?;
    let graph = model.graph.as_ref().expect("No graph");
    let ctx = GraphContext::new(graph);

    // Collect node info
    let nodes: Vec<_> = ctx.nodes().collect();

    // Build producer map
    let mut producer_map: HashMap<&str, &str> = HashMap::new();
    for node in &nodes {
        for output in &node.output {
            producer_map.insert(output.as_str(), node.name.as_str());
        }
    }

    // Find MatMul -> Add patterns
    println!("\n## MatMul -> Add Patterns");
    let mut matmul_add_count = 0;
    let mut matmul_add_fusible = 0;

    for node in &nodes {
        if node.op_type == "Add" {
            for input in &node.input {
                if let Some(&producer_name) = producer_map.get(input.as_str()) {
                    if let Some(producer) = ctx.get_node(producer_name) {
                        if producer.op_type == "MatMul" {
                            matmul_add_count += 1;

                            // Check if other input is initializer
                            let other_input = if &node.input[0] == input {
                                &node.input[1]
                            } else {
                                &node.input[0]
                            };

                            let is_bias_initializer = ctx.get_initializer(other_input).is_some();
                            let is_single_use = ctx.is_single_use(&producer.output[0]);

                            if is_bias_initializer && is_single_use {
                                matmul_add_fusible += 1;
                            }

                            println!(
                                "  {} -> {} | bias_init: {}, single_use: {}",
                                producer_name, node.name, is_bias_initializer, is_single_use
                            );
                        }
                    }
                }
            }
        }
    }
    println!(
        "\n  Total MatMul->Add: {}, Fusible: {}",
        matmul_add_count, matmul_add_fusible
    );

    // Find Gemm -> Add patterns
    println!("\n## Gemm -> Add Patterns");
    let mut gemm_add_count = 0;

    for node in &nodes {
        if node.op_type == "Add" {
            for input in &node.input {
                if let Some(&producer_name) = producer_map.get(input.as_str()) {
                    if let Some(producer) = ctx.get_node(producer_name) {
                        if producer.op_type == "Gemm" {
                            gemm_add_count += 1;
                            let has_bias =
                                producer.input.len() >= 3 && !producer.input[2].is_empty();
                            println!(
                                "  {} -> {} | already_has_bias: {}",
                                producer_name, node.name, has_bias
                            );
                        }
                    }
                }
            }
        }
    }
    println!("\n  Total Gemm->Add: {}", gemm_add_count);

    // Find Transpose -> Transpose patterns
    println!("\n## Transpose -> Transpose Patterns");
    let mut trans_trans_count = 0;

    for node in &nodes {
        if node.op_type == "Transpose" {
            if let Some(input) = node.input.first() {
                if let Some(&producer_name) = producer_map.get(input.as_str()) {
                    if let Some(producer) = ctx.get_node(producer_name) {
                        if producer.op_type == "Transpose" {
                            trans_trans_count += 1;
                            println!("  {} -> {}", producer_name, node.name);
                        }
                    }
                }
            }
        }
    }
    println!("\n  Total Transpose->Transpose: {}", trans_trans_count);

    // Find Reshape -> Reshape patterns
    println!("\n## Reshape -> Reshape Patterns");
    let mut reshape_reshape_count = 0;

    for node in &nodes {
        if node.op_type == "Reshape" {
            if let Some(input) = node.input.first() {
                if let Some(&producer_name) = producer_map.get(input.as_str()) {
                    if let Some(producer) = ctx.get_node(producer_name) {
                        if producer.op_type == "Reshape" {
                            reshape_reshape_count += 1;
                        }
                    }
                }
            }
        }
    }
    println!("  Total Reshape->Reshape: {}", reshape_reshape_count);

    // Summary
    println!("\n## Summary of Optimization Opportunities");
    println!("  MatMul+Add fusible:        {}", matmul_add_fusible);
    println!("  Gemm+Add (no existing bias): {}", gemm_add_count);
    println!("  Transpose chains:          {}", trans_trans_count);
    println!("  Reshape chains:            {}", reshape_reshape_count);

    Ok(())
}
