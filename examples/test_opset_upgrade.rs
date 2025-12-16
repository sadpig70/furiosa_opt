//! Test opset upgrade on real models

use std::env;
use std::path::Path;
use std::time::Instant;

use furiosa_optimizer::builder::build_optimized_model;
use furiosa_optimizer::graph::GraphContext;
use furiosa_optimizer::io::{load_model, save_model, validation::validate_model};
use furiosa_optimizer::opset::{get_opset_version, upgrade_model};
use furiosa_optimizer::transformers::{FuseLayerNorm, OnnxTransformer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.onnx> [output.onnx]", args[0]);
        std::process::exit(1);
    }

    let input_path = Path::new(&args[1]);
    let output_path = args.get(2).map(|s| Path::new(s.as_str()));

    println!("Opset Upgrade Test: {}", input_path.display());
    println!("{}", "=".repeat(60));

    // Load model
    let model = load_model(input_path)?;
    let original_opset = get_opset_version(&model);
    let original_nodes = model.graph.as_ref().map(|g| g.node.len()).unwrap_or(0);

    println!("\n## Original Model");
    println!("  Opset version: {}", original_opset);
    println!("  Nodes: {}", original_nodes);

    // Upgrade to opset 17
    println!("\n## Upgrading to opset 17...");
    let start = Instant::now();
    let upgraded = upgrade_model(&model, 17)?;
    let upgrade_time = start.elapsed();

    let new_opset = get_opset_version(&upgraded);
    let upgraded_nodes = upgraded.graph.as_ref().map(|g| g.node.len()).unwrap_or(0);
    let new_initializers = upgraded
        .graph
        .as_ref()
        .map(|g| g.initializer.len())
        .unwrap_or(0);

    println!("  New opset version: {}", new_opset);
    println!("  Nodes: {}", upgraded_nodes);
    println!("  Initializers: {}", new_initializers);
    println!("  Upgrade time: {:?}", upgrade_time);

    // Validate upgraded model
    let validation = validate_model(&upgraded);
    if validation.is_valid {
        println!("  ✅ Valid ONNX model");
    } else {
        println!("  ❌ Invalid: {}", validation.errors.join("; "));
    }

    // Count Squeeze/Unsqueeze nodes with new input structure
    let graph = upgraded.graph.as_ref().unwrap();
    let squeeze_count = graph
        .node
        .iter()
        .filter(|n| n.op_type == "Squeeze" && n.input.len() == 2)
        .count();
    let unsqueeze_count = graph
        .node
        .iter()
        .filter(|n| n.op_type == "Unsqueeze" && n.input.len() == 2)
        .count();

    println!("\n## Upgraded Nodes");
    println!("  Squeeze (with axes input): {}", squeeze_count);
    println!("  Unsqueeze (with axes input): {}", unsqueeze_count);

    // Now apply LayerNorm fusion
    println!("\n## Applying LayerNorm Fusion...");
    let mut ctx = GraphContext::new(graph);
    let start = Instant::now();
    let result = FuseLayerNorm::new().transform(&mut ctx)?;
    let fusion_time = start.elapsed();

    let remaining_nodes: usize = ctx.nodes().count();
    let ln_count: usize = ctx
        .nodes()
        .filter(|n| n.op_type == "LayerNormalization")
        .count();

    println!("  LayerNorm patterns fused: {}", result.transforms_applied);
    println!("  Nodes eliminated: {}", result.nodes_eliminated);
    println!("  Fusion time: {:?}", fusion_time);
    println!("  LayerNormalization nodes created: {}", ln_count);
    println!("  Remaining nodes: {}", remaining_nodes);

    // Calculate total reduction
    let total_reduction = (original_nodes - remaining_nodes) as f64 / original_nodes as f64 * 100.0;
    println!("\n## Summary");
    println!("  Original nodes: {}", original_nodes);
    println!("  Final nodes: {}", remaining_nodes);
    println!("  Total reduction: {:.1}%", total_reduction);

    // Save if output path provided
    if let Some(out_path) = output_path {
        let final_model = build_optimized_model(&ctx, &upgraded);
        save_model(&final_model, out_path)?;
        println!("\n  Saved to: {}", out_path.display());
    }

    Ok(())
}
