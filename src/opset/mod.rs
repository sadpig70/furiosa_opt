//! ONNX Opset Version Upgrader
//!
//! Upgrades ONNX models from older opset versions to newer ones,
//! handling operator signature changes between versions.
//!
//! # Supported Upgrades
//!
//! | From | To | Changes |
//! |------|-----|---------|
//! | 9-12 | 13+ | Squeeze/Unsqueeze axes to input |
//! | 9-12 | 13+ | Split sizes to input |
//! | Any | 17+ | Enables LayerNormalization |
//!
//! # Example
//!
//! ```ignore
//! use furiosa_optimizer::opset::{OpsetUpgrader, upgrade_model};
//!
//! // Upgrade model to opset 17
//! let upgraded = upgrade_model(&model, 17)?;
//! ```

use crate::error::OnnxResult;
use crate::proto::onnx::{AttributeProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto};

/// ONNX domain identifier
const ONNX_DOMAIN: &str = "ai.onnx";

/// Opset version upgrader
pub struct OpsetUpgrader {
    /// Target opset version
    target_version: i64,
}

impl OpsetUpgrader {
    /// Create a new upgrader targeting the specified opset version
    pub fn new(target_version: i64) -> Self {
        Self { target_version }
    }

    /// Create upgrader for opset 13 (Squeeze/Unsqueeze changes)
    pub fn to_opset_13() -> Self {
        Self::new(13)
    }

    /// Create upgrader for opset 17 (LayerNormalization support)
    pub fn to_opset_17() -> Self {
        Self::new(17)
    }

    /// Get the current opset version of a model
    pub fn get_opset_version(model: &ModelProto) -> i64 {
        for opset in &model.opset_import {
            let domain = opset.domain.as_str();
            if domain.is_empty() || domain == ONNX_DOMAIN {
                return opset.version;
            }
        }
        // Default to opset 1 if not specified
        1
    }

    /// Upgrade a model to the target opset version
    pub fn upgrade(&self, model: &ModelProto) -> OnnxResult<ModelProto> {
        let current_version = Self::get_opset_version(model);

        if current_version >= self.target_version {
            // Already at or above target version
            return Ok(model.clone());
        }

        let mut upgraded = model.clone();

        // Upgrade opset imports
        self.upgrade_opset_imports(&mut upgraded);

        // Upgrade graph nodes
        if let Some(ref mut graph) = upgraded.graph {
            self.upgrade_nodes(&mut graph.node, current_version)?;

            // Add new initializers for converted nodes
            self.add_axes_initializers(graph)?;
        }

        Ok(upgraded)
    }

    /// Upgrade opset import declarations
    fn upgrade_opset_imports(&self, model: &mut ModelProto) {
        let mut found = false;

        for opset in &mut model.opset_import {
            let domain = opset.domain.as_str();
            if domain.is_empty() || domain == ONNX_DOMAIN {
                opset.version = self.target_version;
                found = true;
            }
        }

        // Add default opset if not found
        if !found {
            model.opset_import.push(OperatorSetIdProto {
                domain: String::new(),
                version: self.target_version,
            });
        }
    }

    /// Upgrade nodes based on version differences
    fn upgrade_nodes(&self, nodes: &mut [NodeProto], from_version: i64) -> OnnxResult<()> {
        for node in nodes.iter_mut() {
            // Opset 13: Squeeze/Unsqueeze axes becomes input
            if from_version < 13 && self.target_version >= 13 {
                match node.op_type.as_str() {
                    "Squeeze" => self.upgrade_squeeze_to_13(node)?,
                    "Unsqueeze" => self.upgrade_unsqueeze_to_13(node)?,
                    "Split" => self.upgrade_split_to_13(node)?,
                    _ => {}
                }
            }

            // Opset 14: Reshape allowzero
            if from_version < 14 && self.target_version >= 14 && node.op_type == "Reshape" {
                self.upgrade_reshape_to_14(node)?;
            }
        }

        Ok(())
    }

    /// Upgrade Squeeze node from opset <13 to >=13
    /// Changes: axes attribute -> axes input
    fn upgrade_squeeze_to_13(&self, node: &mut NodeProto) -> OnnxResult<()> {
        // Find axes attribute
        let axes_attr_idx = node.attribute.iter().position(|a| a.name == "axes");

        if let Some(idx) = axes_attr_idx {
            let axes = node.attribute[idx].ints.clone();

            // Remove the attribute
            node.attribute.remove(idx);

            // Create a unique name for the axes tensor
            let axes_tensor_name = format!("{}_axes", node.name);

            // Add axes as second input
            if node.input.len() == 1 {
                node.input.push(axes_tensor_name.clone());
            } else if node.input.len() > 1 {
                node.input[1] = axes_tensor_name.clone();
            }

            // Store axes info for later initializer creation
            // We'll use a special attribute to mark this
            node.attribute.push(AttributeProto {
                name: "_upgrade_axes_data".to_string(),
                r#type: 7, // INTS
                ints: axes,
                ..Default::default()
            });
        }

        Ok(())
    }

    /// Upgrade Unsqueeze node from opset <13 to >=13
    fn upgrade_unsqueeze_to_13(&self, node: &mut NodeProto) -> OnnxResult<()> {
        // Find axes attribute
        let axes_attr_idx = node.attribute.iter().position(|a| a.name == "axes");

        if let Some(idx) = axes_attr_idx {
            let axes = node.attribute[idx].ints.clone();

            // Remove the attribute
            node.attribute.remove(idx);

            // Create a unique name for the axes tensor
            let axes_tensor_name = format!("{}_axes", node.name);

            // Add axes as second input
            if node.input.len() == 1 {
                node.input.push(axes_tensor_name.clone());
            } else if node.input.len() > 1 {
                node.input[1] = axes_tensor_name.clone();
            }

            // Store axes info for later initializer creation
            node.attribute.push(AttributeProto {
                name: "_upgrade_axes_data".to_string(),
                r#type: 7, // INTS
                ints: axes,
                ..Default::default()
            });
        }

        Ok(())
    }

    /// Upgrade Split node from opset <13 to >=13
    fn upgrade_split_to_13(&self, node: &mut NodeProto) -> OnnxResult<()> {
        // Find split attribute
        let split_attr_idx = node.attribute.iter().position(|a| a.name == "split");

        if let Some(idx) = split_attr_idx {
            let split_sizes = node.attribute[idx].ints.clone();

            // Remove the attribute
            node.attribute.remove(idx);

            // Create a unique name for the split tensor
            let split_tensor_name = format!("{}_split", node.name);

            // Add split as second input
            if node.input.len() == 1 {
                node.input.push(split_tensor_name.clone());
            } else if node.input.len() > 1 {
                node.input[1] = split_tensor_name.clone();
            }

            // Store split info for later initializer creation
            node.attribute.push(AttributeProto {
                name: "_upgrade_split_data".to_string(),
                r#type: 7, // INTS
                ints: split_sizes,
                ..Default::default()
            });
        }

        Ok(())
    }

    /// Upgrade Reshape node from opset <14 to >=14
    fn upgrade_reshape_to_14(&self, node: &mut NodeProto) -> OnnxResult<()> {
        // Add allowzero attribute if not present (default 0)
        let has_allowzero = node.attribute.iter().any(|a| a.name == "allowzero");

        if !has_allowzero {
            node.attribute.push(AttributeProto {
                name: "allowzero".to_string(),
                r#type: 2, // INT
                i: 0,
                ..Default::default()
            });
        }

        Ok(())
    }

    /// Add initializers for axes/split tensors created during upgrade
    fn add_axes_initializers(&self, graph: &mut crate::proto::onnx::GraphProto) -> OnnxResult<()> {
        let mut new_initializers = Vec::new();

        for node in &mut graph.node {
            // Process _upgrade_axes_data
            let axes_attr_idx = node
                .attribute
                .iter()
                .position(|a| a.name == "_upgrade_axes_data");

            if let Some(idx) = axes_attr_idx {
                let axes = node.attribute[idx].ints.clone();
                let axes_name = format!("{}_axes", node.name);

                // Create INT64 tensor for axes
                let tensor = TensorProto {
                    name: axes_name,
                    dims: vec![axes.len() as i64],
                    data_type: 7, // INT64
                    int64_data: axes,
                    ..Default::default()
                };
                new_initializers.push(tensor);

                // Remove the temporary attribute
                node.attribute.remove(idx);
            }

            // Process _upgrade_split_data
            let split_attr_idx = node
                .attribute
                .iter()
                .position(|a| a.name == "_upgrade_split_data");

            if let Some(idx) = split_attr_idx {
                let split_sizes = node.attribute[idx].ints.clone();
                let split_name = format!("{}_split", node.name);

                // Create INT64 tensor for split sizes
                let tensor = TensorProto {
                    name: split_name,
                    dims: vec![split_sizes.len() as i64],
                    data_type: 7, // INT64
                    int64_data: split_sizes,
                    ..Default::default()
                };
                new_initializers.push(tensor);

                // Remove the temporary attribute
                node.attribute.remove(idx);
            }
        }

        // Add new initializers to graph
        graph.initializer.extend(new_initializers);

        Ok(())
    }
}

/// Upgrade a model to a specific opset version
pub fn upgrade_model(model: &ModelProto, target_version: i64) -> OnnxResult<ModelProto> {
    OpsetUpgrader::new(target_version).upgrade(model)
}

/// Upgrade a model to opset 17 (for LayerNormalization support)
pub fn upgrade_to_opset_17(model: &ModelProto) -> OnnxResult<ModelProto> {
    OpsetUpgrader::to_opset_17().upgrade(model)
}

/// Get the current opset version of a model
pub fn get_opset_version(model: &ModelProto) -> i64 {
    OpsetUpgrader::get_opset_version(model)
}

/// Check if a model supports a specific operator
pub fn supports_operator(model: &ModelProto, op_type: &str) -> bool {
    let version = get_opset_version(model);

    match op_type {
        "LayerNormalization" => version >= 17,
        "GroupNormalization" => version >= 18,
        "Bernoulli" => version >= 15,
        "Optional" | "OptionalHasElement" | "OptionalGetElement" => version >= 15,
        _ => true, // Most operators available from early versions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::onnx::GraphProto;

    fn make_test_model(opset_version: i64) -> ModelProto {
        ModelProto {
            ir_version: 8,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: opset_version,
            }],
            graph: Some(GraphProto::default()),
            ..Default::default()
        }
    }

    #[test]
    fn test_get_opset_version() {
        let model = make_test_model(13);
        assert_eq!(get_opset_version(&model), 13);
    }

    #[test]
    fn test_upgrade_opset_imports() {
        let model = make_test_model(9);
        let upgraded = upgrade_model(&model, 17).unwrap();
        assert_eq!(get_opset_version(&upgraded), 17);
    }

    #[test]
    fn test_no_upgrade_needed() {
        let model = make_test_model(17);
        let upgraded = upgrade_model(&model, 17).unwrap();
        assert_eq!(get_opset_version(&upgraded), 17);
    }

    #[test]
    fn test_upgrade_squeeze() {
        let mut model = make_test_model(11);

        // Add Squeeze node with axes attribute
        if let Some(ref mut graph) = model.graph {
            graph.node.push(NodeProto {
                op_type: "Squeeze".to_string(),
                name: "squeeze_0".to_string(),
                input: vec!["input".to_string()],
                output: vec!["output".to_string()],
                attribute: vec![AttributeProto {
                    name: "axes".to_string(),
                    r#type: 7,
                    ints: vec![0, 2],
                    ..Default::default()
                }],
                ..Default::default()
            });
        }

        let upgraded = upgrade_model(&model, 13).unwrap();

        let graph = upgraded.graph.as_ref().unwrap();
        let squeeze = &graph.node[0];

        // Check axes attribute is removed
        assert!(!squeeze.attribute.iter().any(|a| a.name == "axes"));

        // Check axes is now an input
        assert_eq!(squeeze.input.len(), 2);
        assert_eq!(squeeze.input[1], "squeeze_0_axes");

        // Check initializer was created
        let init = graph
            .initializer
            .iter()
            .find(|t| t.name == "squeeze_0_axes");
        assert!(init.is_some());
        assert_eq!(init.unwrap().int64_data, vec![0, 2]);
    }

    #[test]
    fn test_upgrade_unsqueeze() {
        let mut model = make_test_model(11);

        if let Some(ref mut graph) = model.graph {
            graph.node.push(NodeProto {
                op_type: "Unsqueeze".to_string(),
                name: "unsqueeze_0".to_string(),
                input: vec!["input".to_string()],
                output: vec!["output".to_string()],
                attribute: vec![AttributeProto {
                    name: "axes".to_string(),
                    r#type: 7,
                    ints: vec![1],
                    ..Default::default()
                }],
                ..Default::default()
            });
        }

        let upgraded = upgrade_model(&model, 13).unwrap();

        let graph = upgraded.graph.as_ref().unwrap();
        let unsqueeze = &graph.node[0];

        assert!(!unsqueeze.attribute.iter().any(|a| a.name == "axes"));
        assert_eq!(unsqueeze.input.len(), 2);
        assert_eq!(unsqueeze.input[1], "unsqueeze_0_axes");
    }

    #[test]
    fn test_supports_operator() {
        let model_9 = make_test_model(9);
        let model_17 = make_test_model(17);

        assert!(!supports_operator(&model_9, "LayerNormalization"));
        assert!(supports_operator(&model_17, "LayerNormalization"));
    }
}
