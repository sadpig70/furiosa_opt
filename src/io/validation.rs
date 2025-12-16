//! ONNX model validation
//!
//! Validate ONNX models for correctness and compatibility.

use std::collections::HashSet;

use crate::error::{OnnxResult, TransformError};
use crate::proto::{GraphProto, ModelProto};

/// Validation result with detailed issues
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the model is valid
    pub is_valid: bool,
    /// List of errors (critical issues)
    pub errors: Vec<String>,
    /// List of warnings (non-critical issues)
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Create a valid result
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Add an error
    pub fn add_error(&mut self, msg: impl Into<String>) {
        self.errors.push(msg.into());
        self.is_valid = false;
    }

    /// Add a warning
    pub fn add_warning(&mut self, msg: impl Into<String>) {
        self.warnings.push(msg.into());
    }

    /// Merge with another result
    pub fn merge(&mut self, other: ValidationResult) {
        if !other.is_valid {
            self.is_valid = false;
        }
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
    }
}

/// Validate an ONNX model
///
/// Performs comprehensive validation including:
/// - Graph structure
/// - Node connectivity
/// - Tensor references
/// - Opset compatibility
pub fn validate_model(model: &ModelProto) -> ValidationResult {
    let mut result = ValidationResult::valid();

    // Check IR version
    if model.ir_version < 3 {
        result.add_warning(format!(
            "IR version {} is very old, consider upgrading",
            model.ir_version
        ));
    }

    // Check opset imports
    if model.opset_import.is_empty() {
        result.add_warning("No opset imports specified");
    }

    // Check graph
    match &model.graph {
        Some(graph) => {
            result.merge(validate_graph(graph));
        }
        None => {
            result.add_error("Model does not contain a graph");
        }
    }

    result
}

/// Validate a graph
pub fn validate_graph(graph: &GraphProto) -> ValidationResult {
    let mut result = ValidationResult::valid();

    // Collect all known tensor names
    let mut known_tensors: HashSet<&str> = HashSet::new();

    // Graph inputs are known
    for input in &graph.input {
        if input.name.is_empty() {
            result.add_error("Graph input has empty name");
        } else {
            known_tensors.insert(&input.name);
        }
    }

    // Initializers are known
    for init in &graph.initializer {
        if init.name.is_empty() {
            result.add_warning("Initializer has empty name");
        } else {
            known_tensors.insert(&init.name);
        }
    }

    // Validate nodes
    let mut node_outputs: HashSet<&str> = HashSet::new();

    for (idx, node) in graph.node.iter().enumerate() {
        // Check op_type
        if node.op_type.is_empty() {
            result.add_error(format!("Node {} has empty op_type", idx));
        }

        // Check inputs exist
        for input in &node.input {
            if !input.is_empty() && !known_tensors.contains(input.as_str()) {
                result.add_error(format!(
                    "Node '{}' ({}): input '{}' not found",
                    node.name, node.op_type, input
                ));
            }
        }

        // Check outputs
        if node.output.is_empty() {
            result.add_warning(format!(
                "Node '{}' ({}) has no outputs",
                node.name, node.op_type
            ));
        }

        for output in &node.output {
            if !output.is_empty() {
                // Check for duplicate outputs
                if node_outputs.contains(output.as_str()) {
                    result.add_error(format!(
                        "Duplicate output '{}' in node '{}'",
                        output, node.name
                    ));
                }
                node_outputs.insert(output);
                known_tensors.insert(output);
            }
        }
    }

    // Check graph outputs exist
    for output in &graph.output {
        if output.name.is_empty() {
            result.add_error("Graph output has empty name");
        } else if !known_tensors.contains(output.name.as_str()) {
            result.add_error(format!(
                "Graph output '{}' not produced by any node",
                output.name
            ));
        }
    }

    // Check for empty graph
    if graph.node.is_empty() && graph.output.is_empty() {
        result.add_warning("Graph is empty (no nodes or outputs)");
    }

    result
}

/// Quick validation that returns an error if invalid
pub fn check_model(model: &ModelProto) -> OnnxResult<()> {
    let result = validate_model(model);
    if result.is_valid {
        Ok(())
    } else {
        Err(TransformError::InvalidModel(result.errors.join("; ")))
    }
}

/// Supported opset versions
pub const MIN_OPSET_VERSION: i64 = 7;
/// Maximum supported opset version
pub const MAX_OPSET_VERSION: i64 = 21;

/// Check if opset version is supported
pub fn is_opset_supported(version: i64) -> bool {
    (MIN_OPSET_VERSION..=MAX_OPSET_VERSION).contains(&version)
}

/// Get the default opset version from a model
pub fn get_opset_version(model: &ModelProto) -> Option<i64> {
    model
        .opset_import
        .iter()
        .find(|op| op.domain.is_empty() || op.domain == "ai.onnx")
        .map(|op| op.version)
}

/// Validation options
#[derive(Debug, Clone, Default)]
pub struct ValidationOptions {
    /// Check for unused initializers
    pub check_unused_initializers: bool,
    /// Check for unused nodes
    pub check_unused_nodes: bool,
    /// Strict mode (warnings become errors)
    pub strict: bool,
}

/// Validate with options
pub fn validate_model_with_options(
    model: &ModelProto,
    options: &ValidationOptions,
) -> ValidationResult {
    let mut result = validate_model(model);

    if let Some(graph) = &model.graph {
        if options.check_unused_initializers {
            let used = collect_used_tensors(graph);
            for init in &graph.initializer {
                if !used.contains(init.name.as_str()) {
                    let msg = format!("Unused initializer: {}", init.name);
                    if options.strict {
                        result.add_error(msg);
                    } else {
                        result.add_warning(msg);
                    }
                }
            }
        }
    }

    if options.strict {
        // Convert warnings to errors
        for warning in std::mem::take(&mut result.warnings) {
            result.add_error(warning);
        }
    }

    result
}

fn collect_used_tensors(graph: &GraphProto) -> HashSet<&str> {
    let mut used = HashSet::new();

    for node in &graph.node {
        for input in &node.input {
            if !input.is_empty() {
                used.insert(input.as_str());
            }
        }
    }

    // Graph outputs are used
    for output in &graph.output {
        used.insert(output.name.as_str());
    }

    used
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::{NodeProto, ValueInfoProto};

    fn make_valid_graph() -> GraphProto {
        GraphProto {
            name: "test".to_string(),
            node: vec![NodeProto {
                op_type: "Relu".to_string(),
                name: "relu_0".to_string(),
                input: vec!["X".to_string()],
                output: vec!["Y".to_string()],
                ..Default::default()
            }],
            input: vec![ValueInfoProto {
                name: "X".to_string(),
                ..Default::default()
            }],
            output: vec![ValueInfoProto {
                name: "Y".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn test_validate_valid_model() {
        let model = ModelProto {
            ir_version: 8,
            graph: Some(make_valid_graph()),
            ..Default::default()
        };

        let result = validate_model(&model);
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_validate_missing_graph() {
        let model = ModelProto {
            ir_version: 8,
            ..Default::default()
        };

        let result = validate_model(&model);
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| e.contains("graph")));
    }

    #[test]
    fn test_validate_missing_input() {
        let graph = GraphProto {
            node: vec![NodeProto {
                op_type: "Relu".to_string(),
                name: "relu_0".to_string(),
                input: vec!["missing".to_string()],
                output: vec!["Y".to_string()],
                ..Default::default()
            }],
            output: vec![ValueInfoProto {
                name: "Y".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };

        let result = validate_graph(&graph);
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| e.contains("missing")));
    }

    #[test]
    fn test_validate_duplicate_output() {
        let graph = GraphProto {
            node: vec![
                NodeProto {
                    op_type: "Relu".to_string(),
                    name: "relu_0".to_string(),
                    input: vec!["X".to_string()],
                    output: vec!["dup".to_string()],
                    ..Default::default()
                },
                NodeProto {
                    op_type: "Relu".to_string(),
                    name: "relu_1".to_string(),
                    input: vec!["X".to_string()],
                    output: vec!["dup".to_string()], // Duplicate!
                    ..Default::default()
                },
            ],
            input: vec![ValueInfoProto {
                name: "X".to_string(),
                ..Default::default()
            }],
            output: vec![ValueInfoProto {
                name: "dup".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };

        let result = validate_graph(&graph);
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| e.contains("Duplicate")));
    }

    #[test]
    fn test_check_model() {
        let model = ModelProto {
            ir_version: 8,
            graph: Some(make_valid_graph()),
            ..Default::default()
        };

        assert!(check_model(&model).is_ok());
    }

    #[test]
    fn test_opset_supported() {
        assert!(is_opset_supported(13));
        assert!(is_opset_supported(17));
        assert!(!is_opset_supported(1));
        assert!(!is_opset_supported(100));
    }
}
