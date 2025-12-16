//! Inference transformers
//!
//! Transformers that infer missing attributes from context.

#![allow(missing_docs)]

use crate::error::OnnxResult;
use crate::graph::GraphContext;
use crate::tensor::convert::tensor_to_array_i64;

use super::common::{set_attr_ints, OnnxTransformer, TransformResult};

/// Infer axes for Squeeze operations
///
/// ONNX opset 13+ requires axes as input or attribute.
/// This transformer infers axes from shape information when missing.
#[derive(Debug, Default)]
pub struct InferSqueezeAxes;

impl InferSqueezeAxes {
    pub fn new() -> Self {
        Self
    }
}

impl OnnxTransformer for InferSqueezeAxes {
    fn name(&self) -> &'static str {
        "InferSqueezeAxes"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        // Find Squeeze nodes without axes
        let squeeze_nodes: Vec<String> = ctx
            .nodes()
            .filter(|n| n.op_type == "Squeeze")
            .filter(|n| {
                // Check if axes is missing (no attribute and no second input)
                let has_attr = n.attribute.iter().any(|a| a.name == "axes");
                let has_input = n.input.len() > 1 && !n.input[1].is_empty();
                !has_attr && !has_input
            })
            .map(|n| n.name.clone())
            .collect();

        result.patterns_matched = squeeze_nodes.len();

        for name in squeeze_nodes {
            let node = match ctx.get_node(&name) {
                Some(n) => n.clone(),
                None => continue,
            };

            // Get input shape
            let input_name = match node.input.first() {
                Some(inp) => inp,
                None => continue,
            };

            let input_shape = match ctx.get_tensor_shape(input_name) {
                Some(s) => s,
                None => continue,
            };

            // Find axes with dimension 1
            let axes: Vec<i64> = input_shape
                .iter()
                .enumerate()
                .filter(|(_, &dim)| dim == 1)
                .map(|(i, _)| i as i64)
                .collect();

            if !axes.is_empty() {
                if let Some(entry) = ctx.get_entry_mut(&name) {
                    set_attr_ints(&mut entry.node, "axes", axes);
                    result.record(&name);
                }
            }
        }

        Ok(result)
    }
}

/// Infer axes for Unsqueeze operations
#[derive(Debug, Default)]
pub struct InferUnsqueezeAxes;

impl InferUnsqueezeAxes {
    pub fn new() -> Self {
        Self
    }
}

impl OnnxTransformer for InferUnsqueezeAxes {
    fn name(&self) -> &'static str {
        "InferUnsqueezeAxes"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        // Find Unsqueeze nodes where axes is provided as constant input
        let unsqueeze_nodes: Vec<String> = ctx
            .nodes()
            .filter(|n| n.op_type == "Unsqueeze")
            .filter(|n| {
                // Has axes as input (not attribute)
                n.input.len() > 1 && !n.input[1].is_empty()
            })
            .map(|n| n.name.clone())
            .collect();

        result.patterns_matched = unsqueeze_nodes.len();

        for name in unsqueeze_nodes {
            let node = match ctx.get_node(&name) {
                Some(n) => n.clone(),
                None => continue,
            };

            // Get axes from initializer
            let axes_name = match node.input.get(1) {
                Some(a) if !a.is_empty() => a,
                _ => continue,
            };

            let axes_tensor = match ctx.get_initializer(axes_name) {
                Some(t) => t.clone(),
                None => continue,
            };

            // Convert to attribute
            if let Ok(axes_arr) = tensor_to_array_i64(&axes_tensor) {
                let axes: Vec<i64> = axes_arr.iter().copied().collect();

                if let Some(entry) = ctx.get_entry_mut(&name) {
                    set_attr_ints(&mut entry.node, "axes", axes);
                    // Remove axes input
                    if entry.node.input.len() > 1 {
                        entry.node.input[1] = String::new();
                    }
                    result.record(&name);
                }
            }
        }

        Ok(result)
    }
}

/// Infer reduction axes for Reduce operations
#[derive(Debug, Default)]
pub struct InferReduceAxes;

impl InferReduceAxes {
    pub fn new() -> Self {
        Self
    }

    const REDUCE_OPS: &'static [&'static str] = &[
        "ReduceSum",
        "ReduceMean",
        "ReduceMax",
        "ReduceMin",
        "ReduceProd",
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSum",
        "ReduceLogSumExp",
        "ReduceSumSquare",
    ];
}

impl OnnxTransformer for InferReduceAxes {
    fn name(&self) -> &'static str {
        "InferReduceAxes"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        // Find Reduce nodes where axes is provided as constant input
        let reduce_nodes: Vec<String> = ctx
            .nodes()
            .filter(|n| Self::REDUCE_OPS.contains(&n.op_type.as_str()))
            .filter(|n| {
                // Has axes as input (opset 18+)
                n.input.len() > 1 && !n.input[1].is_empty()
            })
            .map(|n| n.name.clone())
            .collect();

        result.patterns_matched = reduce_nodes.len();

        for name in reduce_nodes {
            let node = match ctx.get_node(&name) {
                Some(n) => n.clone(),
                None => continue,
            };

            // Get axes from initializer
            let axes_name = match node.input.get(1) {
                Some(a) if !a.is_empty() => a,
                _ => continue,
            };

            let axes_tensor = match ctx.get_initializer(axes_name) {
                Some(t) => t.clone(),
                None => continue,
            };

            // Convert to attribute
            if let Ok(axes_arr) = tensor_to_array_i64(&axes_tensor) {
                let axes: Vec<i64> = axes_arr.iter().copied().collect();

                if let Some(entry) = ctx.get_entry_mut(&name) {
                    set_attr_ints(&mut entry.node, "axes", axes);
                    // Remove axes input
                    if entry.node.input.len() > 1 {
                        entry.node.input[1] = String::new();
                    }
                    result.record(&name);
                }
            }
        }

        Ok(result)
    }
}

/// Combined inference transformer
#[derive(Debug, Default)]
pub struct InferAll;

impl InferAll {
    pub fn new() -> Self {
        Self
    }
}

impl OnnxTransformer for InferAll {
    fn name(&self) -> &'static str {
        "InferAll"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        result.merge(InferSqueezeAxes::new().transform(ctx)?);
        result.merge(InferUnsqueezeAxes::new().transform(ctx)?);
        result.merge(InferReduceAxes::new().transform(ctx)?);

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::{
        tensor_shape_proto, type_proto, GraphProto, TensorShapeProto, TypeProto, ValueInfoProto,
    };
    use crate::tensor::convert::vec_to_tensor_i64;

    fn make_vi_with_shape(name: &str, dims: &[i64]) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: 1, // FLOAT
                    shape: Some(TensorShapeProto {
                        dim: dims
                            .iter()
                            .map(|&d| tensor_shape_proto::Dimension {
                                value: Some(tensor_shape_proto::dimension::Value::DimValue(d)),
                                ..Default::default()
                            })
                            .collect(),
                    }),
                    ..Default::default()
                })),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn test_infer_squeeze_axes() {
        let graph = GraphProto {
            node: vec![make_node("Squeeze", &["X"], &["Y"], "squeeze_0")],
            input: vec![make_vi_with_shape("X", &[1, 3, 1, 4])],
            output: vec![ValueInfoProto {
                name: "Y".to_string(),
                ..Default::default()
            }],
            value_info: vec![make_vi_with_shape("X", &[1, 3, 1, 4])],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let transformer = InferSqueezeAxes::new();
        let result = transformer.transform(&mut ctx).unwrap();

        assert_eq!(result.transforms_applied, 1);

        let squeeze = ctx.get_node("squeeze_0").unwrap();
        let axes_attr = squeeze.attribute.iter().find(|a| a.name == "axes");
        assert!(axes_attr.is_some());
        assert_eq!(axes_attr.unwrap().ints, vec![0, 2]); // dims 0 and 2 are 1
    }

    #[test]
    fn test_infer_unsqueeze_axes() {
        let axes_tensor = vec_to_tensor_i64(&[0, 2], "axes");

        let graph = GraphProto {
            node: vec![make_node(
                "Unsqueeze",
                &["X", "axes"],
                &["Y"],
                "unsqueeze_0",
            )],
            input: vec![ValueInfoProto {
                name: "X".to_string(),
                ..Default::default()
            }],
            output: vec![ValueInfoProto {
                name: "Y".to_string(),
                ..Default::default()
            }],
            initializer: vec![axes_tensor],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let transformer = InferUnsqueezeAxes::new();
        let result = transformer.transform(&mut ctx).unwrap();

        assert_eq!(result.transforms_applied, 1);

        let unsqueeze = ctx.get_node("unsqueeze_0").unwrap();
        let axes_attr = unsqueeze.attribute.iter().find(|a| a.name == "axes");
        assert!(axes_attr.is_some());
        assert_eq!(axes_attr.unwrap().ints, vec![0, 2]);

        // Second input should be cleared
        assert!(unsqueeze.input[1].is_empty());
    }

    #[test]
    fn test_infer_reduce_axes() {
        let axes_tensor = vec_to_tensor_i64(&[1], "axes");

        let graph = GraphProto {
            node: vec![make_node("ReduceSum", &["X", "axes"], &["Y"], "reduce_0")],
            input: vec![ValueInfoProto {
                name: "X".to_string(),
                ..Default::default()
            }],
            output: vec![ValueInfoProto {
                name: "Y".to_string(),
                ..Default::default()
            }],
            initializer: vec![axes_tensor],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let transformer = InferReduceAxes::new();
        let result = transformer.transform(&mut ctx).unwrap();

        assert_eq!(result.transforms_applied, 1);
    }
}
