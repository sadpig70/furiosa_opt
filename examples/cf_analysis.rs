//! Constant folding analysis tool

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
    println!("Constant Folding Analysis: {}", path.display());
    println!("{}", "=".repeat(60));

    let model = load_model(path)?;
    let graph = model.graph.as_ref().expect("No graph");
    let ctx = GraphContext::new(graph);

    // Count operations
    let mut op_counts: HashMap<&str, usize> = HashMap::new();
    for node in ctx.nodes() {
        *op_counts.entry(&node.op_type).or_insert(0) += 1;
    }

    // Analyze Constant nodes
    let constant_count = op_counts.get("Constant").copied().unwrap_or(0);
    println!("\n## Constant Nodes: {}", constant_count);

    // Find all constant output names
    let mut constant_outputs: Vec<String> = Vec::new();
    for node in ctx.nodes() {
        if node.op_type == "Constant" {
            for output in &node.output {
                constant_outputs.push(output.clone());
            }
        }
    }

    // Find nodes that consume constants
    let mut consumers_of_constants: HashMap<&str, usize> = HashMap::new();
    for node in ctx.nodes() {
        let const_inputs = node
            .input
            .iter()
            .filter(|inp| constant_outputs.contains(inp) || ctx.get_initializer(inp).is_some())
            .count();
        if const_inputs > 0 {
            *consumers_of_constants.entry(&node.op_type).or_insert(0) += 1;
        }
    }

    println!("\n## Nodes consuming constants:");
    let mut sorted: Vec<_> = consumers_of_constants.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (op, count) in sorted.iter().take(15) {
        println!("  {}: {}", op, count);
    }

    // Find fully constant nodes (all inputs are constants)
    let mut fully_constant: HashMap<&str, usize> = HashMap::new();
    for node in ctx.nodes() {
        if node.input.is_empty() {
            continue;
        }
        let all_const = node
            .input
            .iter()
            .filter(|inp| !inp.is_empty())
            .all(|inp| constant_outputs.contains(inp) || ctx.get_initializer(inp).is_some());
        if all_const {
            *fully_constant.entry(&node.op_type).or_insert(0) += 1;
        }
    }

    println!("\n## Fully constant nodes (foldable):");
    let mut sorted: Vec<_> = fully_constant.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    let total_foldable: usize = sorted.iter().map(|(_, c)| **c).sum();
    for (op, count) in sorted.iter().take(15) {
        println!("  {}: {}", op, count);
    }
    println!("  TOTAL: {}", total_foldable);

    // Analyze Shape nodes specifically
    println!("\n## Shape node analysis:");
    let mut shape_count = 0;
    let mut shape_foldable = 0;
    for node in ctx.nodes() {
        if node.op_type == "Shape" {
            shape_count += 1;
            if let Some(input) = node.input.first() {
                // Check if input has known static shape
                let has_init = ctx.get_initializer(input).is_some();
                let has_value_info = ctx.get_value_info(input).is_some();
                if has_init {
                    shape_foldable += 1;
                }
                if shape_count <= 5 {
                    println!(
                        "  Shape of '{}': init={}, vi={}",
                        input, has_init, has_value_info
                    );
                }
            }
        }
    }
    println!(
        "  Total Shape nodes: {}, Foldable: {}",
        shape_count, shape_foldable
    );

    // Analyze Gather nodes
    println!("\n## Gather node analysis:");
    let mut gather_count = 0;
    let mut gather_foldable = 0;
    for node in ctx.nodes() {
        if node.op_type == "Gather" {
            gather_count += 1;
            let data_const = node
                .input
                .first()
                .map(|inp| constant_outputs.contains(inp) || ctx.get_initializer(inp).is_some())
                .unwrap_or(false);
            let idx_const = node
                .input
                .get(1)
                .map(|inp| constant_outputs.contains(inp) || ctx.get_initializer(inp).is_some())
                .unwrap_or(false);
            if data_const && idx_const {
                gather_foldable += 1;
            }
            if gather_count <= 5 {
                println!(
                    "  Gather: data_const={}, idx_const={}",
                    data_const, idx_const
                );
            }
        }
    }
    println!(
        "  Total Gather nodes: {}, Foldable: {}",
        gather_count, gather_foldable
    );

    Ok(())
}
