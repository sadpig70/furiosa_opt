//! Test LayerNorm fusion on real models

use std::env;
use std::path::Path;
use std::time::Instant;

use furiosa_optimizer::graph::GraphContext;
use furiosa_optimizer::io::load_model;
use furiosa_optimizer::transformers::{FuseLayerNorm, OnnxTransformer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.onnx>", args[0]);
        std::process::exit(1);
    }

    let path = Path::new(&args[1]);
    println!("LayerNorm Fusion Test: {}", path.display());
    println!("{}", "=".repeat(60));

    let model = load_model(path)?;
    let graph = model.graph.as_ref().expect("No graph");

    let original_nodes = graph.node.len();
    println!("Original nodes: {}", original_nodes);

    let mut ctx = GraphContext::new(graph);

    let start = Instant::now();
    let result = FuseLayerNorm::new().transform(&mut ctx)?;
    let elapsed = start.elapsed();

    println!("\n## Fusion Results");
    println!("  Transforms applied: {}", result.transforms_applied);
    println!("  Nodes eliminated: {}", result.nodes_eliminated);
    println!("  Time: {:?}", elapsed);

    // Count remaining nodes
    let remaining_nodes: usize = ctx.nodes().count();
    let reduction = (original_nodes - remaining_nodes) as f64 / original_nodes as f64 * 100.0;

    println!("\n## Summary");
    println!("  Original: {} nodes", original_nodes);
    println!("  After fusion: {} nodes", remaining_nodes);
    println!("  Reduction: {:.1}%", reduction);

    // List LayerNormalization nodes
    let ln_nodes: Vec<_> = ctx
        .nodes()
        .filter(|n| n.op_type == "LayerNormalization")
        .collect();

    if !ln_nodes.is_empty() {
        println!("\n## Created LayerNormalization nodes: {}", ln_nodes.len());
        for (i, node) in ln_nodes.iter().take(5).enumerate() {
            println!("  {}: {} -> {:?}", i + 1, node.name, node.output.first());
        }
    }

    Ok(())
}
