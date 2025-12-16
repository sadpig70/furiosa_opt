//! ONNX I/O module
//!
//! This module provides functions for loading, saving, and validating ONNX models.
//!
//! # Example
//!
//! ```ignore
//! use furiosa_optimizer::io::{load_model, save_model, optimize_file};
//!
//! // Load and save
//! let model = load_model("input.onnx")?;
//! save_model(&model, "output.onnx")?;
//!
//! // One-shot optimization
//! let stats = optimize_file("input.onnx", "optimized.onnx", Default::default())?;
//! println!("Reduced {} nodes", stats.nodes_reduced);
//! ```

pub mod reader;
pub mod validation;
pub mod writer;

// Re-exports
pub use reader::{get_model_info, load_graph, load_model, load_model_from_bytes, ModelInfo};
pub use validation::{
    check_model, get_opset_version, is_opset_supported, validate_graph, validate_model,
    validate_model_with_options, ValidationOptions, ValidationResult, MAX_OPSET_VERSION,
    MIN_OPSET_VERSION,
};
pub use writer::{model_size, model_to_bytes, save_model, save_model_with_stats, SaveStats};

use std::path::Path;

use crate::builder::build_optimized_model;
use crate::error::OnnxResult;
use crate::graph::GraphContext;
use crate::proto::ModelProto;
use crate::transformers::{OptimizationPipeline, TransformResult};

/// Optimization statistics
#[derive(Debug, Clone, Default)]
pub struct OptimizeStats {
    /// Original node count
    pub original_nodes: usize,
    /// Optimized node count
    pub optimized_nodes: usize,
    /// Nodes reduced
    pub nodes_reduced: usize,
    /// Original file size (if available)
    pub original_size: usize,
    /// Optimized file size
    pub optimized_size: usize,
    /// Transform statistics
    pub transform: TransformResult,
}

impl OptimizeStats {
    /// Calculate reduction percentage
    pub fn node_reduction_percent(&self) -> f64 {
        if self.original_nodes == 0 {
            0.0
        } else {
            (self.nodes_reduced as f64 / self.original_nodes as f64) * 100.0
        }
    }

    /// Calculate size reduction percentage
    pub fn size_reduction_percent(&self) -> f64 {
        if self.original_size == 0 {
            0.0
        } else {
            let reduced = self.original_size.saturating_sub(self.optimized_size);
            (reduced as f64 / self.original_size as f64) * 100.0
        }
    }
}

/// Optimization options
#[derive(Debug, Clone)]
pub struct OptimizeOptions {
    /// Enable inference passes
    pub infer: bool,
    /// Enable fusion passes
    pub fuse: bool,
    /// Enable elimination passes
    pub eliminate: bool,
    /// Validate before optimization
    pub validate_input: bool,
    /// Validate after optimization
    pub validate_output: bool,
    /// Number of optimization iterations
    pub iterations: usize,
}

impl Default for OptimizeOptions {
    fn default() -> Self {
        Self {
            infer: true,
            fuse: true,
            eliminate: true,
            validate_input: true,
            validate_output: true,
            iterations: 3,
        }
    }
}

/// Optimize an ONNX model in memory
pub fn optimize_model(
    model: &ModelProto,
    options: &OptimizeOptions,
) -> OnnxResult<(ModelProto, OptimizeStats)> {
    // Validate input if requested
    if options.validate_input {
        check_model(model)?;
    }

    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| crate::error::TransformError::InvalidModel("No graph".to_string()))?;

    let original_nodes = graph.node.len();
    let original_size = model_size(model);

    // Create context and run optimization
    let mut ctx = GraphContext::new(graph);

    let pipeline = OptimizationPipeline {
        infer: options.infer,
        fuse: options.fuse,
        eliminate: options.eliminate,
        optimize_shape: true,
        constant_fold: true,
        iterations: options.iterations,
    };

    let transform_result = pipeline.run(&mut ctx)?;

    // Build optimized model
    let optimized = build_optimized_model(&ctx, model);

    let optimized_nodes = optimized.graph.as_ref().map(|g| g.node.len()).unwrap_or(0);
    let optimized_size = model_size(&optimized);

    // Validate output if requested
    if options.validate_output {
        check_model(&optimized)?;
    }

    let stats = OptimizeStats {
        original_nodes,
        optimized_nodes,
        nodes_reduced: original_nodes.saturating_sub(optimized_nodes),
        original_size,
        optimized_size,
        transform: transform_result,
    };

    Ok((optimized, stats))
}

/// Optimize an ONNX file and save to another file
///
/// This is the main entry point for file-based optimization.
///
/// # Example
///
/// ```ignore
/// use furiosa_optimizer::io::{optimize_file, OptimizeOptions};
///
/// let stats = optimize_file("model.onnx", "optimized.onnx", OptimizeOptions::default())?;
/// println!("Reduced {} nodes ({:.1}%)", stats.nodes_reduced, stats.node_reduction_percent());
/// ```
pub fn optimize_file<P1: AsRef<Path>, P2: AsRef<Path>>(
    input: P1,
    output: P2,
    options: OptimizeOptions,
) -> OnnxResult<OptimizeStats> {
    let model = load_model(input)?;
    let (optimized, stats) = optimize_model(&model, &options)?;
    save_model(&optimized, output)?;
    Ok(stats)
}

/// Quick optimization with default settings
pub fn optimize_file_default<P1: AsRef<Path>, P2: AsRef<Path>>(
    input: P1,
    output: P2,
) -> OnnxResult<OptimizeStats> {
    optimize_file(input, output, OptimizeOptions::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::{GraphProto, TensorProto, ValueInfoProto};

    fn make_test_model() -> ModelProto {
        // Create weight initializer
        let weight = TensorProto {
            name: "W".to_string(),
            dims: vec![1, 1, 3, 3],
            data_type: 1, // FLOAT
            float_data: vec![0.0; 9],
            ..Default::default()
        };

        ModelProto {
            ir_version: 8,
            producer_name: "test".to_string(),
            graph: Some(GraphProto {
                name: "test_graph".to_string(),
                node: vec![
                    make_node("Conv", &["X", "W"], &["conv_out"], "conv_0"),
                    make_node("Identity", &["conv_out"], &["id_out"], "identity_0"),
                    make_node("Relu", &["id_out"], &["Y"], "relu_0"),
                ],
                input: vec![ValueInfoProto {
                    name: "X".to_string(),
                    ..Default::default()
                }],
                output: vec![ValueInfoProto {
                    name: "Y".to_string(),
                    ..Default::default()
                }],
                initializer: vec![weight],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn test_optimize_model() {
        let model = make_test_model();
        let (optimized, stats) = optimize_model(&model, &OptimizeOptions::default()).unwrap();

        // Identity should be eliminated
        assert!(stats.nodes_reduced >= 1);
        assert!(stats.optimized_nodes < stats.original_nodes);

        // Model should still be valid
        assert!(check_model(&optimized).is_ok());
    }

    #[test]
    fn test_optimize_file() {
        let model = make_test_model();
        let input_path = format!("/tmp/test_input_{}.onnx", std::process::id());
        let output_path = format!("/tmp/test_output_{}.onnx", std::process::id());

        // Save input model
        save_model(&model, &input_path).unwrap();

        // Optimize
        let stats = optimize_file(&input_path, &output_path, OptimizeOptions::default()).unwrap();

        assert!(stats.nodes_reduced >= 1);

        // Load and verify output
        let loaded = load_model(&output_path).unwrap();
        assert!(check_model(&loaded).is_ok());

        // Cleanup
        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn test_optimize_stats() {
        let stats = OptimizeStats {
            original_nodes: 100,
            optimized_nodes: 80,
            nodes_reduced: 20,
            original_size: 1000,
            optimized_size: 800,
            ..Default::default()
        };

        assert!((stats.node_reduction_percent() - 20.0).abs() < 0.01);
        assert!((stats.size_reduction_percent() - 20.0).abs() < 0.01);
    }
}
