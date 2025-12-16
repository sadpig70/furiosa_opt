//! LayerNorm pattern analysis tool

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
    println!("LayerNorm Pattern Analysis: {}", path.display());
    println!("{}", "=".repeat(60));

    let model = load_model(path)?;
    let graph = model.graph.as_ref().expect("No graph");
    let ctx = GraphContext::new(graph);

    // Find ReduceMean nodes (entry point for LayerNorm)
    println!("\n## ReduceMean Nodes Analysis");
    let mut reduce_mean_count = 0;
    let mut ln_candidates = Vec::new();

    for node in ctx.nodes() {
        if node.op_type == "ReduceMean" {
            reduce_mean_count += 1;

            // Check consumers of this ReduceMean
            let output = node.output.first().map(|s| s.as_str()).unwrap_or("");
            let consumers: Vec<_> = ctx
                .nodes()
                .filter(|n| n.input.contains(&output.to_string()))
                .map(|n| n.op_type.as_str())
                .collect();

            if reduce_mean_count <= 10 {
                println!("  {} -> [{}] consumers: {:?}", node.name, output, consumers);
            }

            // Check if it's consumed by Sub (potential LayerNorm start)
            if consumers.contains(&"Sub") {
                ln_candidates.push(node.name.clone());
            }
        }
    }
    println!("  Total ReduceMean: {}", reduce_mean_count);
    println!(
        "  LayerNorm candidates (ReduceMean→Sub): {}",
        ln_candidates.len()
    );

    // Trace LayerNorm pattern: ReduceMean → Sub → Pow → ReduceMean → Add → Sqrt → Div
    println!("\n## LayerNorm Pattern Tracing");
    let mut complete_patterns = 0;

    for (i, rm_name) in ln_candidates.iter().take(5).enumerate() {
        println!("\n  Pattern {}:", i + 1);

        if let Some(rm_node) = ctx.get_node(rm_name) {
            let rm_out = rm_node.output.first().map(|s| s.as_str()).unwrap_or("");
            println!("    1. ReduceMean: {} → {}", rm_name, rm_out);

            // Find Sub that uses this ReduceMean output
            let sub_node = ctx
                .nodes()
                .find(|n| n.op_type == "Sub" && n.input.contains(&rm_out.to_string()));

            if let Some(sub) = sub_node {
                let sub_out = sub.output.first().map(|s| s.as_str()).unwrap_or("");
                println!("    2. Sub: {} → {}", sub.name, sub_out);

                // Find Pow that uses Sub output
                let pow_node = ctx.nodes().find(|n| {
                    n.op_type == "Pow" && n.input.first().map(|s| s.as_str()) == Some(sub_out)
                });

                if let Some(pow) = pow_node {
                    let pow_out = pow.output.first().map(|s| s.as_str()).unwrap_or("");
                    println!("    3. Pow: {} → {}", pow.name, pow_out);

                    // Find ReduceMean that uses Pow output
                    let rm2_node = ctx.nodes().find(|n| {
                        n.op_type == "ReduceMean"
                            && n.input.first().map(|s| s.as_str()) == Some(pow_out)
                    });

                    if let Some(rm2) = rm2_node {
                        let rm2_out = rm2.output.first().map(|s| s.as_str()).unwrap_or("");
                        println!("    4. ReduceMean: {} → {}", rm2.name, rm2_out);

                        // Find Add (epsilon addition)
                        let add_node = ctx
                            .nodes()
                            .find(|n| n.op_type == "Add" && n.input.contains(&rm2_out.to_string()));

                        if let Some(add) = add_node {
                            let add_out = add.output.first().map(|s| s.as_str()).unwrap_or("");
                            println!("    5. Add (epsilon): {} → {}", add.name, add_out);

                            // Find Sqrt
                            let sqrt_node = ctx.nodes().find(|n| {
                                n.op_type == "Sqrt"
                                    && n.input.first().map(|s| s.as_str()) == Some(add_out)
                            });

                            if let Some(sqrt) = sqrt_node {
                                let sqrt_out =
                                    sqrt.output.first().map(|s| s.as_str()).unwrap_or("");
                                println!("    6. Sqrt: {} → {}", sqrt.name, sqrt_out);

                                // Find Div
                                let div_node = ctx.nodes().find(|n| {
                                    n.op_type == "Div"
                                        && n.input.get(1).map(|s| s.as_str()) == Some(sqrt_out)
                                });

                                if let Some(div) = div_node {
                                    println!("    7. Div: {} → {:?}", div.name, div.output.first());

                                    // Check for scale (Mul) and bias (Add)
                                    let div_out =
                                        div.output.first().map(|s| s.as_str()).unwrap_or("");
                                    let mul_node = ctx.nodes().find(|n| {
                                        n.op_type == "Mul"
                                            && n.input.first().map(|s| s.as_str()) == Some(div_out)
                                    });

                                    if let Some(mul) = mul_node {
                                        println!(
                                            "    8. Mul (scale): {} → {:?}",
                                            mul.name,
                                            mul.output.first()
                                        );

                                        let mul_out =
                                            mul.output.first().map(|s| s.as_str()).unwrap_or("");
                                        let bias_add = ctx.nodes().find(|n| {
                                            n.op_type == "Add"
                                                && n.input.first().map(|s| s.as_str())
                                                    == Some(mul_out)
                                        });

                                        if let Some(ba) = bias_add {
                                            println!(
                                                "    9. Add (bias): {} → {:?}",
                                                ba.name,
                                                ba.output.first()
                                            );
                                            complete_patterns += 1;
                                            println!("    ✅ COMPLETE LayerNorm pattern!");
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    println!("\n## Summary");
    println!("  Complete LayerNorm patterns found: {}", complete_patterns);
    println!(
        "  Potential patterns (ReduceMean→Sub): {}",
        ln_candidates.len()
    );

    // Estimate nodes that could be fused
    // Each LayerNorm pattern = ~9 nodes → 1 node = 8 nodes saved
    let estimated_savings = ln_candidates.len() / 2 * 8; // /2 because each LN has 2 ReduceMean
    println!("  Estimated node savings: ~{}", estimated_savings);

    Ok(())
}
