//! Benchmark runner for real ONNX models
//!
//! Run with: cargo run --release --example benchmark

use std::fs;
use std::path::Path;
use std::time::Instant;

use furiosa_optimizer::io::{
    get_model_info, load_model, optimize_model, save_model, validate_model, OptimizeOptions,
};

/// Benchmark result for a single model
#[derive(Debug)]
struct BenchmarkResult {
    model_name: String,
    original_size: usize,
    optimized_size: usize,
    original_nodes: usize,
    optimized_nodes: usize,
    load_time_ms: f64,
    optimize_time_ms: f64,
    save_time_ms: f64,
    is_valid: bool,
    errors: Vec<String>,
}

impl BenchmarkResult {
    fn node_reduction_percent(&self) -> f64 {
        if self.original_nodes == 0 {
            0.0
        } else {
            let reduced = self.original_nodes.saturating_sub(self.optimized_nodes);
            (reduced as f64 / self.original_nodes as f64) * 100.0
        }
    }

    fn size_reduction_percent(&self) -> f64 {
        if self.original_size == 0 {
            0.0
        } else {
            let reduced = self.original_size.saturating_sub(self.optimized_size);
            (reduced as f64 / self.original_size as f64) * 100.0
        }
    }
}

fn benchmark_model(path: &Path) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    let model_name = path.file_stem().unwrap().to_string_lossy().to_string();
    println!();
    println!("{}", "=".repeat(60));
    println!("Benchmarking: {}", model_name);
    println!("{}", "=".repeat(60));

    // Get original file size
    let original_size = fs::metadata(path)?.len() as usize;

    // Load model
    let load_start = Instant::now();
    let model = load_model(path)?;
    let load_time_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    // Get original info
    let info = get_model_info(path)?;
    let original_nodes = info.node_count;

    println!(
        "  Original: {} nodes, {:.2} MB",
        original_nodes,
        original_size as f64 / 1_000_000.0
    );

    // Validate before
    let validation = validate_model(&model);
    if !validation.is_valid {
        println!("  ⚠ Validation errors: {:?}", validation.errors);
    }

    // Optimize
    let optimize_start = Instant::now();
    let options = OptimizeOptions::default();
    let (optimized, _stats) = optimize_model(&model, &options)?;
    let optimize_time_ms = optimize_start.elapsed().as_secs_f64() * 1000.0;

    // Get optimized info
    let optimized_nodes = optimized.graph.as_ref().map(|g| g.node.len()).unwrap_or(0);

    // Save to temp file to measure size
    let temp_path = format!("/tmp/optimized_{}.onnx", model_name);
    let save_start = Instant::now();
    save_model(&optimized, &temp_path)?;
    let save_time_ms = save_start.elapsed().as_secs_f64() * 1000.0;

    let optimized_size = fs::metadata(&temp_path)?.len() as usize;

    // Validate after
    let post_validation = validate_model(&optimized);

    // Cleanup
    fs::remove_file(&temp_path).ok();

    let result = BenchmarkResult {
        model_name,
        original_size,
        optimized_size,
        original_nodes,
        optimized_nodes,
        load_time_ms,
        optimize_time_ms,
        save_time_ms,
        is_valid: post_validation.is_valid,
        errors: post_validation.errors,
    };

    println!(
        "  Optimized: {} nodes, {:.2} MB",
        result.optimized_nodes,
        result.optimized_size as f64 / 1_000_000.0
    );
    println!("  Node reduction: {:.1}%", result.node_reduction_percent());
    println!("  Size reduction: {:.1}%", result.size_reduction_percent());
    println!(
        "  Timings: load={:.1}ms, optimize={:.1}ms, save={:.1}ms",
        result.load_time_ms, result.optimize_time_ms, result.save_time_ms
    );

    if !result.is_valid {
        println!(
            "  ❌ Post-optimization validation failed: {:?}",
            result.errors
        );
    } else {
        println!("  ✅ Valid");
    }

    Ok(result)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║       Furiosa Optimizer Benchmark Suite                  ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let models_dir = Path::new("benches/models");

    if !models_dir.exists() {
        eprintln!("Models directory not found: {:?}", models_dir);
        eprintln!("Please download models first.");
        return Ok(());
    }

    let mut results = Vec::new();

    // Find all ONNX models
    for entry in fs::read_dir(models_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().map(|e| e == "onnx").unwrap_or(false) {
            match benchmark_model(&path) {
                Ok(result) => results.push(result),
                Err(e) => {
                    eprintln!("Failed to benchmark {:?}: {}", path, e);
                }
            }
        }
    }

    // Summary
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║                    SUMMARY                               ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "{:<15} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Model", "Nodes", "→", "Reduced", "Time(ms)", "Valid"
    );
    println!("{}", "-".repeat(70));

    for r in &results {
        let valid_str = if r.is_valid { "✅" } else { "❌" };
        println!(
            "{:<15} {:>10} {:>10} {:>9.1}% {:>10.1} {:>10}",
            r.model_name,
            r.original_nodes,
            r.optimized_nodes,
            r.node_reduction_percent(),
            r.optimize_time_ms,
            valid_str
        );
    }

    println!("{}", "-".repeat(70));

    // Totals
    let total_original_nodes: usize = results.iter().map(|r| r.original_nodes).sum();
    let total_optimized_nodes: usize = results.iter().map(|r| r.optimized_nodes).sum();
    let total_time: f64 = results.iter().map(|r| r.optimize_time_ms).sum();
    let all_valid = results.iter().all(|r| r.is_valid);

    let total_reduction = if total_original_nodes > 0 {
        ((total_original_nodes - total_optimized_nodes) as f64 / total_original_nodes as f64)
            * 100.0
    } else {
        0.0
    };

    println!(
        "{:<15} {:>10} {:>10} {:>9.1}% {:>10.1} {:>10}",
        "TOTAL",
        total_original_nodes,
        total_optimized_nodes,
        total_reduction,
        total_time,
        if all_valid { "✅" } else { "❌" }
    );

    // Save results to JSON
    let results_json = serde_json::json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "models": results.iter().map(|r| {
            serde_json::json!({
                "name": r.model_name,
                "original_nodes": r.original_nodes,
                "optimized_nodes": r.optimized_nodes,
                "node_reduction_percent": r.node_reduction_percent(),
                "original_size_bytes": r.original_size,
                "optimized_size_bytes": r.optimized_size,
                "size_reduction_percent": r.size_reduction_percent(),
                "load_time_ms": r.load_time_ms,
                "optimize_time_ms": r.optimize_time_ms,
                "save_time_ms": r.save_time_ms,
                "is_valid": r.is_valid,
            })
        }).collect::<Vec<_>>(),
        "summary": {
            "total_original_nodes": total_original_nodes,
            "total_optimized_nodes": total_optimized_nodes,
            "total_reduction_percent": total_reduction,
            "total_time_ms": total_time,
            "all_valid": all_valid,
        }
    });

    fs::write(
        "benches/results/benchmark_results.json",
        serde_json::to_string_pretty(&results_json)?,
    )?;

    println!("\nResults saved to benches/results/benchmark_results.json");

    Ok(())
}
