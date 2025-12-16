//! Elimination transformers
//!
//! Transformers that remove unnecessary nodes from the graph.

#![allow(missing_docs)]

use crate::error::OnnxResult;
use crate::graph::GraphContext;
use crate::transform::{can_eliminate, eliminate_node};

use super::common::{OnnxTransformer, TransformResult};

/// Eliminate Identity nodes
///
/// Identity nodes simply pass their input to output without modification.
/// They can be safely removed by connecting predecessors directly to successors.
#[derive(Debug, Default)]
pub struct EliminateIdentity;

impl EliminateIdentity {
    pub fn new() -> Self {
        Self
    }
}

impl OnnxTransformer for EliminateIdentity {
    fn name(&self) -> &'static str {
        "EliminateIdentity"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        // Collect Identity nodes
        let identity_nodes: Vec<String> = ctx
            .nodes()
            .filter(|n| n.op_type == "Identity")
            .filter(|n| can_eliminate(ctx, &n.name))
            .map(|n| n.name.clone())
            .collect();

        result.patterns_matched = identity_nodes.len();

        for name in identity_nodes {
            if eliminate_node(ctx, &name, 0).is_some() {
                result.record_elimination(&name);
            }
        }

        Ok(result)
    }
}

/// Eliminate Dropout nodes (inference mode)
///
/// During inference, Dropout nodes with ratio < 1.0 act as Identity.
/// They can be safely removed.
#[derive(Debug, Default)]
pub struct EliminateDropout;

impl EliminateDropout {
    pub fn new() -> Self {
        Self
    }
}

impl OnnxTransformer for EliminateDropout {
    fn name(&self) -> &'static str {
        "EliminateDropout"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        let dropout_nodes: Vec<String> = ctx
            .nodes()
            .filter(|n| n.op_type == "Dropout")
            .filter(|n| can_eliminate(ctx, &n.name))
            .map(|n| n.name.clone())
            .collect();

        result.patterns_matched = dropout_nodes.len();

        for name in dropout_nodes {
            if eliminate_node(ctx, &name, 0).is_some() {
                result.record_elimination(&name);
            }
        }

        Ok(result)
    }
}

/// Eliminate Cast nodes where input and output types are the same
#[derive(Debug, Default)]
pub struct EliminateNopCast;

impl EliminateNopCast {
    pub fn new() -> Self {
        Self
    }

    fn is_nop_cast(&self, ctx: &GraphContext, node_name: &str) -> bool {
        let node = match ctx.get_node(node_name) {
            Some(n) => n,
            None => return false,
        };

        if node.op_type != "Cast" {
            return false;
        }

        // Get input type
        let input_type = node
            .input
            .first()
            .and_then(|inp| ctx.get_tensor_elem_type(inp));

        // Get target type from attribute
        let target_type = node
            .attribute
            .iter()
            .find(|a| a.name == "to")
            .map(|a| a.i as i32);

        match (input_type, target_type) {
            (Some(inp), Some(out)) => inp == out,
            _ => false,
        }
    }
}

impl OnnxTransformer for EliminateNopCast {
    fn name(&self) -> &'static str {
        "EliminateNopCast"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        let cast_nodes: Vec<String> = ctx
            .nodes()
            .filter(|n| n.op_type == "Cast")
            .map(|n| n.name.clone())
            .collect();

        for name in cast_nodes {
            if self.is_nop_cast(ctx, &name) && can_eliminate(ctx, &name) {
                result.patterns_matched += 1;
                if eliminate_node(ctx, &name, 0).is_some() {
                    result.record_elimination(&name);
                }
            }
        }

        Ok(result)
    }
}

/// Eliminate Reshape nodes where shape doesn't change
#[derive(Debug, Default)]
pub struct EliminateNopReshape;

impl EliminateNopReshape {
    pub fn new() -> Self {
        Self
    }

    fn is_nop_reshape(&self, ctx: &GraphContext, node_name: &str) -> bool {
        let node = match ctx.get_node(node_name) {
            Some(n) => n,
            None => return false,
        };

        if node.op_type != "Reshape" {
            return false;
        }

        // Get input and output shapes
        let input_shape = node.input.first().and_then(|inp| ctx.get_tensor_shape(inp));
        let output_shape = node
            .output
            .first()
            .and_then(|out| ctx.get_tensor_shape(out));

        match (input_shape, output_shape) {
            (Some(inp), Some(out)) => inp == out,
            _ => false,
        }
    }
}

impl OnnxTransformer for EliminateNopReshape {
    fn name(&self) -> &'static str {
        "EliminateNopReshape"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        let reshape_nodes: Vec<String> = ctx
            .nodes()
            .filter(|n| n.op_type == "Reshape")
            .map(|n| n.name.clone())
            .collect();

        for name in reshape_nodes {
            if self.is_nop_reshape(ctx, &name) && can_eliminate(ctx, &name) {
                result.patterns_matched += 1;
                if eliminate_node(ctx, &name, 0).is_some() {
                    result.record_elimination(&name);
                }
            }
        }

        Ok(result)
    }
}

/// Eliminate Transpose nodes that are identity permutation
#[derive(Debug, Default)]
pub struct EliminateNopTranspose;

impl EliminateNopTranspose {
    pub fn new() -> Self {
        Self
    }

    fn is_identity_perm(perm: &[i64]) -> bool {
        perm.iter().enumerate().all(|(i, &p)| p == i as i64)
    }
}

impl OnnxTransformer for EliminateNopTranspose {
    fn name(&self) -> &'static str {
        "EliminateNopTranspose"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        let transpose_nodes: Vec<(String, bool)> = ctx
            .nodes()
            .filter(|n| n.op_type == "Transpose")
            .map(|n| {
                let is_identity = n
                    .attribute
                    .iter()
                    .find(|a| a.name == "perm")
                    .map(|a| Self::is_identity_perm(&a.ints))
                    .unwrap_or(false);
                (n.name.clone(), is_identity)
            })
            .collect();

        for (name, is_identity) in transpose_nodes {
            if is_identity && can_eliminate(ctx, &name) {
                result.patterns_matched += 1;
                if eliminate_node(ctx, &name, 0).is_some() {
                    result.record_elimination(&name);
                }
            }
        }

        Ok(result)
    }
}

/// Combined eliminator that runs all elimination passes
#[derive(Debug, Default)]
pub struct EliminateAll;

impl EliminateAll {
    pub fn new() -> Self {
        Self
    }
}

impl OnnxTransformer for EliminateAll {
    fn name(&self) -> &'static str {
        "EliminateAll"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        // Run each eliminator
        result.merge(EliminateIdentity::new().transform(ctx)?);
        result.merge(EliminateDropout::new().transform(ctx)?);
        result.merge(EliminateNopCast::new().transform(ctx)?);
        result.merge(EliminateNopReshape::new().transform(ctx)?);
        result.merge(EliminateNopTranspose::new().transform(ctx)?);

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::{GraphProto, ValueInfoProto};

    fn make_identity_graph() -> GraphProto {
        GraphProto {
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
            ..Default::default()
        }
    }

    #[test]
    fn test_eliminate_identity() {
        let graph = make_identity_graph();
        let mut ctx = GraphContext::new(&graph);

        let transformer = EliminateIdentity::new();
        let result = transformer.transform(&mut ctx).unwrap();

        assert_eq!(result.patterns_matched, 1);
        assert_eq!(result.nodes_eliminated, 1);
        assert_eq!(ctx.active_node_count(), 2);
    }

    #[test]
    fn test_eliminate_dropout() {
        let graph = GraphProto {
            node: vec![
                make_node("Conv", &["X", "W"], &["conv_out"], "conv_0"),
                make_node("Dropout", &["conv_out"], &["drop_out"], "dropout_0"),
                make_node("Relu", &["drop_out"], &["Y"], "relu_0"),
            ],
            input: vec![ValueInfoProto {
                name: "X".to_string(),
                ..Default::default()
            }],
            output: vec![ValueInfoProto {
                name: "Y".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let transformer = EliminateDropout::new();
        let result = transformer.transform(&mut ctx).unwrap();

        assert_eq!(result.nodes_eliminated, 1);
    }

    #[test]
    fn test_eliminate_nop_transpose() {
        let mut transpose_node = make_node("Transpose", &["X"], &["mid"], "transpose_0");
        transpose_node.attribute.push(crate::proto::AttributeProto {
            name: "perm".to_string(),
            ints: vec![0, 1, 2], // Identity permutation
            r#type: 7,
            ..Default::default()
        });

        let graph = GraphProto {
            node: vec![
                transpose_node,
                make_node("Relu", &["mid"], &["Y"], "relu_0"),
            ],
            input: vec![ValueInfoProto {
                name: "X".to_string(),
                ..Default::default()
            }],
            output: vec![ValueInfoProto {
                name: "Y".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let transformer = EliminateNopTranspose::new();
        let result = transformer.transform(&mut ctx).unwrap();

        assert_eq!(result.patterns_matched, 1);
        assert_eq!(result.nodes_eliminated, 1);
    }

    #[test]
    fn test_eliminate_all() {
        let graph = make_identity_graph();
        let mut ctx = GraphContext::new(&graph);

        let transformer = EliminateAll::new();
        let result = transformer.transform(&mut ctx).unwrap();

        assert!(result.nodes_eliminated >= 1);
    }
}
