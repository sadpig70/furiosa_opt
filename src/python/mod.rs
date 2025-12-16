//! Python bindings for furiosa-optimizer using PyO3
//!
//! This module provides Python-callable functions for ONNX model optimization.
//!
//! # Usage from Python
//!
//! ```python
//! import furiosa_optimizer
//!
//! # Simple optimization
//! result = furiosa_optimizer.optimize_model("model.onnx", "optimized.onnx")
//! print(f"Reduced {result.nodes_removed} nodes ({result.reduction_percent:.1f}%)")
//!
//! # With custom config
//! config = furiosa_optimizer.OptimizationConfig(
//!     fuse_conv_bn=True,
//!     fuse_gemm_add=True,
//!     eliminate_identity=True,
//!     infer_axes=True,
//! )
//! result = furiosa_optimizer.optimize_model("model.onnx", "out.onnx", config)
//!
//! # Analysis only
//! info = furiosa_optimizer.analyze_model("model.onnx")
//! print(f"Model has {info.node_count} nodes")
//! ```

use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::path::Path;

use crate::builder::build_graph_from_context;
use crate::graph::GraphContext;
use crate::io::{load_model, save_model, validate_model as validate_onnx};
use crate::transformers::OptimizationPipeline;

// ============================================================================
// Python-exposed configuration class
// ============================================================================

/// Configuration for the optimization pipeline.
///
/// All options are enabled by default for maximum optimization.
#[pyclass(name = "OptimizationConfig")]
#[derive(Clone, Debug)]
pub struct PyOptimizationConfig {
    /// Fuse Conv + BatchNormalization
    #[pyo3(get, set)]
    pub fuse_conv_bn: bool,

    /// Fuse Gemm + Add (bias)
    #[pyo3(get, set)]
    pub fuse_gemm_add: bool,

    /// Fuse MatMul + Add → Gemm
    #[pyo3(get, set)]
    pub fuse_matmul_add: bool,

    /// Eliminate Identity nodes
    #[pyo3(get, set)]
    pub eliminate_identity: bool,

    /// Eliminate Dropout nodes (inference mode)
    #[pyo3(get, set)]
    pub eliminate_dropout: bool,

    /// Infer missing axes attributes
    #[pyo3(get, set)]
    pub infer_axes: bool,

    /// Optimize Reshape chains
    #[pyo3(get, set)]
    pub optimize_reshape: bool,

    /// Optimize Transpose chains
    #[pyo3(get, set)]
    pub optimize_transpose: bool,

    /// Number of optimization iterations
    #[pyo3(get, set)]
    pub iterations: usize,
}

#[pymethods]
impl PyOptimizationConfig {
    /// Create a new configuration with optional overrides.
    ///
    /// All optimizations are enabled by default.
    #[new]
    #[pyo3(signature = (
        fuse_conv_bn = true,
        fuse_gemm_add = true,
        fuse_matmul_add = true,
        eliminate_identity = true,
        eliminate_dropout = true,
        infer_axes = true,
        optimize_reshape = true,
        optimize_transpose = true,
        iterations = 3
    ))]
    fn new(
        fuse_conv_bn: bool,
        fuse_gemm_add: bool,
        fuse_matmul_add: bool,
        eliminate_identity: bool,
        eliminate_dropout: bool,
        infer_axes: bool,
        optimize_reshape: bool,
        optimize_transpose: bool,
        iterations: usize,
    ) -> Self {
        Self {
            fuse_conv_bn,
            fuse_gemm_add,
            fuse_matmul_add,
            eliminate_identity,
            eliminate_dropout,
            infer_axes,
            optimize_reshape,
            optimize_transpose,
            iterations,
        }
    }

    /// Create a minimal configuration (only essential optimizations).
    #[staticmethod]
    fn minimal() -> Self {
        Self {
            fuse_conv_bn: true,
            fuse_gemm_add: false,
            fuse_matmul_add: false,
            eliminate_identity: true,
            eliminate_dropout: true,
            infer_axes: false,
            optimize_reshape: false,
            optimize_transpose: false,
            iterations: 1,
        }
    }

    /// Create an aggressive configuration (maximum optimization).
    #[staticmethod]
    fn aggressive() -> Self {
        Self {
            fuse_conv_bn: true,
            fuse_gemm_add: true,
            fuse_matmul_add: true,
            eliminate_identity: true,
            eliminate_dropout: true,
            infer_axes: true,
            optimize_reshape: true,
            optimize_transpose: true,
            iterations: 5,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "OptimizationConfig(fuse_conv_bn={}, fuse_gemm_add={}, iterations={})",
            self.fuse_conv_bn, self.fuse_gemm_add, self.iterations
        )
    }
}

impl Default for PyOptimizationConfig {
    fn default() -> Self {
        Self::new(true, true, true, true, true, true, true, true, 3)
    }
}

// ============================================================================
// Python-exposed result classes
// ============================================================================

/// Result of an optimization operation.
#[pyclass(name = "OptimizationResult")]
#[derive(Clone, Debug)]
pub struct PyOptimizationResult {
    /// Number of nodes in original model
    #[pyo3(get)]
    pub original_nodes: usize,

    /// Number of nodes after optimization
    #[pyo3(get)]
    pub optimized_nodes: usize,

    /// Number of nodes removed
    #[pyo3(get)]
    pub nodes_removed: usize,

    /// Reduction percentage
    #[pyo3(get)]
    pub reduction_percent: f64,

    /// Optimization time in milliseconds
    #[pyo3(get)]
    pub optimize_time_ms: f64,

    /// Whether the optimized model is valid
    #[pyo3(get)]
    pub is_valid: bool,

    /// Output file path
    #[pyo3(get)]
    pub output_path: String,
}

#[pymethods]
impl PyOptimizationResult {
    fn __repr__(&self) -> String {
        format!(
            "OptimizationResult(nodes: {} → {}, reduction: {:.1}%, valid: {})",
            self.original_nodes, self.optimized_nodes, self.reduction_percent, self.is_valid
        )
    }

    /// Alias for optimized_nodes
    #[getter]
    fn final_nodes(&self) -> usize {
        self.optimized_nodes
    }

    /// Convert to dictionary for easy serialization.
    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);
        dict.set_item("original_nodes", self.original_nodes)?;
        dict.set_item("optimized_nodes", self.optimized_nodes)?;
        dict.set_item("final_nodes", self.optimized_nodes)?;
        dict.set_item("nodes_removed", self.nodes_removed)?;
        dict.set_item("reduction_percent", self.reduction_percent)?;
        dict.set_item("optimize_time_ms", self.optimize_time_ms)?;
        dict.set_item("is_valid", self.is_valid)?;
        dict.set_item("output_path", &self.output_path)?;
        Ok(dict.into())
    }
}

/// Model analysis information.
#[pyclass(name = "ModelInfo")]
#[derive(Clone, Debug)]
pub struct PyModelInfo {
    /// Model IR version
    #[pyo3(get)]
    pub ir_version: i64,

    /// ONNX opset version
    #[pyo3(get)]
    pub opset_version: i64,

    /// Producer name
    #[pyo3(get)]
    pub producer: String,

    /// Number of nodes
    #[pyo3(get)]
    pub node_count: usize,

    /// Number of initializers (weights)
    #[pyo3(get)]
    pub initializer_count: usize,

    /// Number of inputs
    #[pyo3(get)]
    pub input_count: usize,

    /// Number of outputs
    #[pyo3(get)]
    pub output_count: usize,

    /// Node type distribution
    #[pyo3(get)]
    pub op_counts: HashMap<String, usize>,

    /// Whether model is valid
    #[pyo3(get)]
    pub is_valid: bool,
}

#[pymethods]
impl PyModelInfo {
    fn __repr__(&self) -> String {
        format!(
            "ModelInfo(nodes={}, opset={}, inputs={}, outputs={}, valid={})",
            self.node_count, self.opset_version, self.input_count, self.output_count, self.is_valid
        )
    }

    /// Get list of unique operation types.
    fn op_types(&self) -> Vec<String> {
        self.op_counts.keys().cloned().collect()
    }

    /// Get top N most common operations.
    fn top_ops(&self, n: usize) -> Vec<(String, usize)> {
        let mut ops: Vec<_> = self
            .op_counts
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        ops.sort_by(|a, b| b.1.cmp(&a.1));
        ops.truncate(n);
        ops
    }
}

// ============================================================================
// Python-exposed functions
// ============================================================================

/// Optimize an ONNX model and save the result.
///
/// Args:
///     input_path: Path to input ONNX model
///     output_path: Path to save optimized model
///     config: Optional optimization configuration
///
/// Returns:
///     OptimizationResult with statistics
///
/// Raises:
///     IOError: If file cannot be read/written
///     ValueError: If model is invalid
#[pyfunction]
#[pyo3(signature = (input_path, output_path, config = None))]
fn optimize_model(
    input_path: &str,
    output_path: &str,
    config: Option<PyOptimizationConfig>,
) -> PyResult<PyOptimizationResult> {
    let config = config.unwrap_or_default();
    let input = Path::new(input_path);
    let output = Path::new(output_path);

    // Load model
    let mut model = load_model(input)
        .map_err(|e| PyIOError::new_err(format!("Failed to load model: {}", e)))?;

    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| PyValueError::new_err("Model has no graph"))?;

    let original_nodes = graph.node.len();

    // Create context and run pipeline
    let mut ctx = GraphContext::new(graph);

    let start = std::time::Instant::now();

    // Build pipeline based on config
    let pipeline = OptimizationPipeline {
        infer: config.infer_axes,
        fuse: config.fuse_conv_bn || config.fuse_gemm_add || config.fuse_matmul_add,
        eliminate: config.eliminate_identity || config.eliminate_dropout,
        optimize_shape: config.optimize_reshape || config.optimize_transpose,
        constant_fold: true,
        iterations: config.iterations,
    };

    pipeline
        .run(&mut ctx)
        .map_err(|e| PyRuntimeError::new_err(format!("Optimization failed: {}", e)))?;

    let optimize_time = start.elapsed();

    // Build optimized model
    let optimized_graph = build_graph_from_context(&ctx);
    let optimized_nodes = optimized_graph.node.len();

    model.graph = Some(optimized_graph);

    // Auto-add com.microsoft domain for Gelu/FastGelu nodes
    if let Some(graph) = &model.graph {
        if graph
            .node
            .iter()
            .any(|n| n.op_type == "Gelu" || n.op_type == "FastGelu")
        {
            if !model
                .opset_import
                .iter()
                .any(|opset| opset.domain == "com.microsoft")
            {
                model
                    .opset_import
                    .push(crate::proto::onnx::OperatorSetIdProto {
                        domain: "com.microsoft".to_string(),
                        version: 1,
                    });
            }
        }
    }

    // Validate before saving
    let is_valid = validate_onnx(&model).is_valid;

    // Save model
    save_model(&model, output)
        .map_err(|e| PyIOError::new_err(format!("Failed to save model: {}", e)))?;

    let nodes_removed = original_nodes.saturating_sub(optimized_nodes);
    let reduction_percent = if original_nodes > 0 {
        (nodes_removed as f64 / original_nodes as f64) * 100.0
    } else {
        0.0
    };

    Ok(PyOptimizationResult {
        original_nodes,
        optimized_nodes,
        nodes_removed,
        reduction_percent,
        optimize_time_ms: optimize_time.as_secs_f64() * 1000.0,
        is_valid,
        output_path: output_path.to_string(),
    })
}

/// Analyze an ONNX model without modifying it.
///
/// Args:
///     model_path: Path to ONNX model
///
/// Returns:
///     ModelInfo with analysis results
#[pyfunction]
fn analyze_model(model_path: &str) -> PyResult<PyModelInfo> {
    let path = Path::new(model_path);

    let model =
        load_model(path).map_err(|e| PyIOError::new_err(format!("Failed to load model: {}", e)))?;

    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| PyValueError::new_err("Model has no graph"))?;

    // Get opset version
    let opset_version = crate::opset::get_opset_version(&model);

    // Count operations
    let mut op_counts: HashMap<String, usize> = HashMap::new();
    for node in &graph.node {
        *op_counts.entry(node.op_type.clone()).or_insert(0) += 1;
    }

    // Validate
    let is_valid = validate_onnx(&model).is_valid;

    Ok(PyModelInfo {
        ir_version: model.ir_version,
        opset_version,
        producer: format!("{} {}", model.producer_name, model.producer_version),
        node_count: graph.node.len(),
        initializer_count: graph.initializer.len(),
        input_count: graph.input.len(),
        output_count: graph.output.len(),
        op_counts,
        is_valid,
    })
}

/// Validate an ONNX model.
///
/// Args:
///     model_path: Path to ONNX model
///
/// Returns:
///     True if valid, False otherwise
#[pyfunction]
fn validate(model_path: &str) -> PyResult<bool> {
    let path = Path::new(model_path);

    let model =
        load_model(path).map_err(|e| PyIOError::new_err(format!("Failed to load model: {}", e)))?;

    let result = validate_onnx(&model);
    Ok(result.is_valid)
}

/// Get the version of this library.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// ============================================================================
// Module registration
// ============================================================================

/// Python module for ONNX model optimization.
///
/// This module provides high-performance ONNX model optimization
/// implemented in Rust for the Furiosa NPU.
#[pymodule]
#[pyo3(name = "furiosa_optimizer")]
fn furiosa_optimizer_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<PyOptimizationConfig>()?;
    m.add_class::<PyOptimizationResult>()?;
    m.add_class::<PyModelInfo>()?;

    // Functions
    m.add_function(wrap_pyfunction!(optimize_model, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_model, m)?)?;
    m.add_function(wrap_pyfunction!(validate, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;

    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "Jung Wook Yang")?;

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = PyOptimizationConfig::default();
        assert!(config.fuse_conv_bn);
        assert!(config.fuse_gemm_add);
        assert_eq!(config.iterations, 3);
    }

    #[test]
    fn test_config_minimal() {
        let config = PyOptimizationConfig::minimal();
        assert!(config.fuse_conv_bn);
        assert!(!config.fuse_gemm_add);
        assert_eq!(config.iterations, 1);
    }

    #[test]
    fn test_config_aggressive() {
        let config = PyOptimizationConfig::aggressive();
        assert!(config.fuse_conv_bn);
        assert!(config.fuse_gemm_add);
        assert_eq!(config.iterations, 5);
    }
}
