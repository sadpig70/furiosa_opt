//! Common utilities for transformers
//!
//! Shared helper functions and types used across multiple transformers.

use crate::error::OnnxResult;
use crate::graph::GraphContext;
use crate::proto::{AttributeProto, NodeProto};

/// Get attribute value as i64
pub fn get_attr_i(node: &NodeProto, name: &str) -> Option<i64> {
    node.attribute.iter().find(|a| a.name == name).map(|a| a.i)
}

/// Get attribute value as f32
pub fn get_attr_f(node: &NodeProto, name: &str) -> Option<f32> {
    node.attribute.iter().find(|a| a.name == name).map(|a| a.f)
}

/// Get attribute value as string
pub fn get_attr_s<'a>(node: &'a NodeProto, name: &str) -> Option<&'a str> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| std::str::from_utf8(&a.s).unwrap_or(""))
}

/// Get attribute value as i64 list
pub fn get_attr_ints<'a>(node: &'a NodeProto, name: &str) -> Option<&'a [i64]> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.ints.as_slice())
}

/// Get attribute value as f32 list
pub fn get_attr_floats<'a>(node: &'a NodeProto, name: &str) -> Option<&'a [f32]> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.floats.as_slice())
}

/// Set or update an attribute
pub fn set_attr_i(node: &mut NodeProto, name: &str, value: i64) {
    // Find existing attribute
    for attr in &mut node.attribute {
        if attr.name == name {
            attr.i = value;
            return;
        }
    }
    // Add new attribute
    node.attribute.push(AttributeProto {
        name: name.to_string(),
        i: value,
        r#type: 2, // INT
        ..Default::default()
    });
}

/// Set or update a float attribute
pub fn set_attr_f(node: &mut NodeProto, name: &str, value: f32) {
    for attr in &mut node.attribute {
        if attr.name == name {
            attr.f = value;
            return;
        }
    }
    node.attribute.push(AttributeProto {
        name: name.to_string(),
        f: value,
        r#type: 1, // FLOAT
        ..Default::default()
    });
}

/// Set or update an ints attribute
pub fn set_attr_ints(node: &mut NodeProto, name: &str, values: Vec<i64>) {
    for attr in &mut node.attribute {
        if attr.name == name {
            attr.ints = values;
            return;
        }
    }
    node.attribute.push(AttributeProto {
        name: name.to_string(),
        ints: values,
        r#type: 7, // INTS
        ..Default::default()
    });
}

/// Remove an attribute by name
pub fn remove_attr(node: &mut NodeProto, name: &str) -> Option<AttributeProto> {
    let pos = node.attribute.iter().position(|a| a.name == name)?;
    Some(node.attribute.remove(pos))
}

/// Check if node has attribute
pub fn has_attr(node: &NodeProto, name: &str) -> bool {
    node.attribute.iter().any(|a| a.name == name)
}

/// Get constant tensor from input name (checks Initializers and Constant nodes)
pub fn get_constant_tensor<'a>(
    ctx: &'a GraphContext,
    name: &str,
) -> Option<&'a crate::proto::TensorProto> {
    // 1. Check Initializers
    if let Some(init) = ctx.get_initializer(name) {
        return Some(init);
    }

    // 2. Check Constant nodes
    if let Some(producer) = ctx.get_producer(name) {
        if producer.op_type == "Constant" {
            for attr in &producer.attribute {
                if attr.name == "value" {
                    return attr.t.as_ref();
                }
            }
        }
    }

    None
}

/// Transformation result for statistics
#[derive(Debug, Default, Clone)]
pub struct TransformResult {
    /// Number of patterns matched
    pub patterns_matched: usize,
    /// Number of transformations applied
    pub transforms_applied: usize,
    /// Number of nodes eliminated
    pub nodes_eliminated: usize,
    /// Names of transformed nodes
    pub transformed_nodes: Vec<String>,
}

impl TransformResult {
    /// Create empty result
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful transformation
    pub fn record(&mut self, node_name: &str) {
        self.transforms_applied += 1;
        self.transformed_nodes.push(node_name.to_string());
    }

    /// Record elimination
    pub fn record_elimination(&mut self, node_name: &str) {
        self.nodes_eliminated += 1;
        self.transformed_nodes.push(node_name.to_string());
    }

    /// Merge with another result
    pub fn merge(&mut self, other: TransformResult) {
        self.patterns_matched += other.patterns_matched;
        self.transforms_applied += other.transforms_applied;
        self.nodes_eliminated += other.nodes_eliminated;
        self.transformed_nodes.extend(other.transformed_nodes);
    }
}

/// Trait for individual transformers
pub trait OnnxTransformer {
    /// Name of the transformer
    fn name(&self) -> &'static str;

    /// Apply the transformation
    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult>;

    /// Check if this transformer is applicable to the graph
    fn is_applicable(&self, _ctx: &GraphContext) -> bool {
        true
    }
}

/// Run multiple transformers in sequence
pub fn run_transformers(
    ctx: &mut GraphContext,
    transformers: &[&dyn OnnxTransformer],
) -> OnnxResult<TransformResult> {
    let mut total = TransformResult::new();

    for transformer in transformers {
        if transformer.is_applicable(ctx) {
            let result = transformer.transform(ctx)?;
            total.merge(result);
        }
    }

    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;

    #[test]
    fn test_get_attr_i() {
        let mut node = make_node("Conv", &["X"], &["Y"], "conv");
        node.attribute.push(AttributeProto {
            name: "group".to_string(),
            i: 4,
            r#type: 1,
            ..Default::default()
        });

        assert_eq!(get_attr_i(&node, "group"), Some(4));
        assert_eq!(get_attr_i(&node, "missing"), None);
    }

    #[test]
    fn test_set_attr_i() {
        let mut node = make_node("Conv", &["X"], &["Y"], "conv");

        set_attr_i(&mut node, "group", 4);
        assert_eq!(get_attr_i(&node, "group"), Some(4));

        // Update existing
        set_attr_i(&mut node, "group", 8);
        assert_eq!(get_attr_i(&node, "group"), Some(8));
    }

    #[test]
    fn test_remove_attr() {
        let mut node = make_node("Conv", &["X"], &["Y"], "conv");
        set_attr_i(&mut node, "group", 4);

        let removed = remove_attr(&mut node, "group");
        assert!(removed.is_some());
        assert_eq!(get_attr_i(&node, "group"), None);
    }

    #[test]
    fn test_transform_result() {
        let mut result = TransformResult::new();

        result.record("node_1");
        result.record_elimination("node_2");

        assert_eq!(result.transforms_applied, 1);
        assert_eq!(result.nodes_eliminated, 1);
        assert_eq!(result.transformed_nodes.len(), 2);
    }
}
