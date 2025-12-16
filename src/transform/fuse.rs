//! Node fusion transformations
//!
//! Handles the fusion of multiple nodes into a single optimized node.

use crate::error::{OnnxResult, TransformError};
use crate::graph::GraphContext;
use crate::proto::NodeProto;

use super::bridge::bridge_disconnected_nodes;

/// Result of a fusion operation
#[derive(Debug, Clone)]
pub struct FusionResult {
    /// Name of the fused (new) node
    pub fused_node_name: String,
    /// Names of nodes that were eliminated
    pub eliminated_nodes: Vec<String>,
    /// Number of connections bridged
    pub bridged_connections: usize,
}

/// Fuse two adjacent nodes
///
/// This is equivalent to Python's `transform_to_fuse`.
/// The first node (consumer) is updated to take inputs from the second node's predecessor,
/// and the second node is eliminated.
///
/// # Arguments
/// * `ctx` - The graph context
/// * `consumer_name` - Name of the consumer node (will be updated)
/// * `producer_name` - Name of the producer node (will be eliminated)
/// * `fuse_inputs` - Function to compute new inputs for the fused node
///
/// # Returns
/// * `Ok(FusionResult)` on success
/// * `Err` if nodes not found or fusion not possible
pub fn fuse_nodes<F>(
    ctx: &mut GraphContext,
    consumer_name: &str,
    producer_name: &str,
    fuse_inputs: F,
) -> OnnxResult<FusionResult>
where
    F: FnOnce(&NodeProto, &NodeProto) -> Vec<String>,
{
    // Get both nodes
    let consumer = ctx
        .get_node(consumer_name)
        .ok_or_else(|| TransformError::InvalidNode(consumer_name.to_string()))?
        .clone();

    let producer = ctx
        .get_node(producer_name)
        .ok_or_else(|| TransformError::InvalidNode(producer_name.to_string()))?
        .clone();

    // Verify they are adjacent
    if !ctx.are_adjacent(producer_name, consumer_name) {
        return Err(TransformError::PatternNotMatched(format!(
            "{} and {} are not adjacent",
            producer_name, consumer_name
        )));
    }

    // Compute new inputs
    let new_inputs = fuse_inputs(&consumer, &producer);

    // Update consumer's inputs
    if let Some(entry) = ctx.get_entry_mut(consumer_name) {
        entry.node.input = new_inputs;
    }

    // Mark producer as eliminated
    ctx.mark_eliminated(producer_name);

    // Bridge any remaining connections
    let bridged = bridge_disconnected_nodes(ctx, &producer, 0);

    Ok(FusionResult {
        fused_node_name: consumer_name.to_string(),
        eliminated_nodes: vec![producer_name.to_string()],
        bridged_connections: bridged,
    })
}

/// Fuse nodes with additional node modification
pub fn fuse_nodes_with_update<F, U>(
    ctx: &mut GraphContext,
    consumer_name: &str,
    producer_name: &str,
    fuse_inputs: F,
    update_node: U,
) -> OnnxResult<FusionResult>
where
    F: FnOnce(&NodeProto, &NodeProto) -> Vec<String>,
    U: FnOnce(&mut NodeProto, &NodeProto),
{
    let consumer = ctx
        .get_node(consumer_name)
        .ok_or_else(|| TransformError::InvalidNode(consumer_name.to_string()))?
        .clone();

    let producer = ctx
        .get_node(producer_name)
        .ok_or_else(|| TransformError::InvalidNode(producer_name.to_string()))?
        .clone();

    if !ctx.are_adjacent(producer_name, consumer_name) {
        return Err(TransformError::PatternNotMatched(format!(
            "{} and {} are not adjacent",
            producer_name, consumer_name
        )));
    }

    let new_inputs = fuse_inputs(&consumer, &producer);

    if let Some(entry) = ctx.get_entry_mut(consumer_name) {
        entry.node.input = new_inputs;
        update_node(&mut entry.node, &producer);
    }

    ctx.mark_eliminated(producer_name);
    let bridged = bridge_disconnected_nodes(ctx, &producer, 0);

    Ok(FusionResult {
        fused_node_name: consumer_name.to_string(),
        eliminated_nodes: vec![producer_name.to_string()],
        bridged_connections: bridged,
    })
}

/// Fuse a pattern of multiple nodes
///
/// Given a matched pattern (from PatternMatcher), fuse all nodes into the anchor.
///
/// # Arguments
/// * `ctx` - The graph context
/// * `anchor_name` - Name of the anchor node (first in pattern, will be kept)
/// * `pattern_names` - Names of all nodes in the pattern (including anchor)
/// * `compute_fused` - Function to compute the final fused node
pub fn fuse_pattern<F>(
    ctx: &mut GraphContext,
    anchor_name: &str,
    pattern_names: &[&str],
    compute_fused: F,
) -> OnnxResult<FusionResult>
where
    F: FnOnce(&[&NodeProto]) -> OnnxResult<NodeProto>,
{
    // Collect all nodes in the pattern
    let mut nodes: Vec<&NodeProto> = Vec::new();
    for name in pattern_names {
        let node = ctx
            .get_node(name)
            .ok_or_else(|| TransformError::InvalidNode((*name).to_string()))?;
        nodes.push(node);
    }

    // Compute the fused node
    let fused = compute_fused(&nodes)?;

    // Replace the anchor with the fused node
    ctx.replace_node(fused);

    // Eliminate other nodes in the pattern
    let eliminated: Vec<String> = pattern_names
        .iter()
        .skip(1) // Skip anchor
        .map(|s| s.to_string())
        .collect();

    let mut total_bridged = 0;
    for name in &eliminated {
        if let Some(node) = ctx.get_node(name).cloned() {
            ctx.mark_eliminated(name);
            total_bridged += bridge_disconnected_nodes(ctx, &node, 0);
        }
    }

    Ok(FusionResult {
        fused_node_name: anchor_name.to_string(),
        eliminated_nodes: eliminated,
        bridged_connections: total_bridged,
    })
}

/// Check if two nodes can be fused
///
/// Conditions:
/// 1. Nodes must be adjacent
/// 2. The intermediate tensor must be single-use
pub fn can_fuse(ctx: &GraphContext, producer_name: &str, consumer_name: &str) -> bool {
    // Check adjacency
    if !ctx.are_adjacent(producer_name, consumer_name) {
        return false;
    }

    // Check single-use condition
    let producer = match ctx.get_node(producer_name) {
        Some(n) => n,
        None => return false,
    };

    // The connecting tensor should be single-use
    for output in &producer.output {
        if ctx.get_consumer_names(output).map(|c| c.len()).unwrap_or(0) > 1 {
            return false;
        }
    }

    true
}

/// Batch fusion result
#[derive(Debug, Default)]
pub struct BatchFusionResult {
    /// Individual fusion results
    pub fusions: Vec<FusionResult>,
    /// Total fusions performed
    pub total_fused: usize,
    /// Total nodes eliminated
    pub total_eliminated: usize,
}

impl BatchFusionResult {
    /// Create from a list of fusion results
    pub fn from_results(results: Vec<FusionResult>) -> Self {
        let total_fused = results.len();
        let total_eliminated: usize = results.iter().map(|r| r.eliminated_nodes.len()).sum();

        Self {
            fusions: results,
            total_fused,
            total_eliminated,
        }
    }
}

/// Standard input fusion: take producer's first input
pub fn standard_fuse_inputs(consumer: &NodeProto, producer: &NodeProto) -> Vec<String> {
    let mut inputs = consumer.input.clone();

    // Find which consumer input comes from producer's output
    for output in &producer.output {
        for input in &mut inputs {
            if input == output {
                // Replace with producer's first input
                if let Some(prod_input) = producer.input.first() {
                    *input = prod_input.clone();
                }
            }
        }
    }

    inputs
}

/// Merge inputs: combine inputs from both nodes (removing the connection)
pub fn merge_inputs(consumer: &NodeProto, producer: &NodeProto) -> Vec<String> {
    let mut inputs: Vec<String> = Vec::new();

    // First, add producer's inputs
    inputs.extend(producer.input.iter().cloned());

    // Then add consumer's inputs that don't come from producer
    for input in &consumer.input {
        let from_producer = producer.output.contains(input);
        if !from_producer {
            inputs.push(input.clone());
        }
    }

    inputs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::{GraphProto, ValueInfoProto};

    fn make_fusible_graph() -> GraphProto {
        GraphProto {
            node: vec![
                make_node("Conv", &["X", "W"], &["conv_out"], "conv_0"),
                make_node(
                    "BatchNormalization",
                    &["conv_out", "scale", "B", "mean", "var"],
                    &["bn_out"],
                    "bn_0",
                ),
                make_node("Relu", &["bn_out"], &["Y"], "relu_0"),
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
    fn test_can_fuse() {
        let graph = make_fusible_graph();
        let ctx = GraphContext::new(&graph);

        assert!(can_fuse(&ctx, "conv_0", "bn_0"));
        assert!(can_fuse(&ctx, "bn_0", "relu_0"));
        assert!(!can_fuse(&ctx, "conv_0", "relu_0")); // Not adjacent
    }

    #[test]
    fn test_fuse_nodes() {
        let graph = make_fusible_graph();
        let mut ctx = GraphContext::new(&graph);

        let result = fuse_nodes(&mut ctx, "bn_0", "conv_0", |bn, conv| {
            // BN takes conv's input (X) instead of conv_out
            let mut inputs = bn.input.clone();
            inputs[0] = conv.input[0].clone();
            inputs
        });

        assert!(result.is_ok());
        let r = result.unwrap();
        assert_eq!(r.fused_node_name, "bn_0");
        assert_eq!(r.eliminated_nodes, vec!["conv_0"]);

        // Verify bn now takes X as first input
        let bn = ctx.get_node("bn_0").unwrap();
        assert_eq!(bn.input[0], "X");
    }

    #[test]
    fn test_fuse_nodes_not_adjacent() {
        let graph = make_fusible_graph();
        let mut ctx = GraphContext::new(&graph);

        let result = fuse_nodes(&mut ctx, "relu_0", "conv_0", |_, _| vec![]);

        assert!(result.is_err());
    }

    #[test]
    fn test_standard_fuse_inputs() {
        let consumer = make_node(
            "BatchNormalization",
            &["conv_out", "scale", "B"],
            &["bn_out"],
            "bn",
        );
        let producer = make_node("Conv", &["X", "W"], &["conv_out"], "conv");

        let fused = standard_fuse_inputs(&consumer, &producer);

        assert_eq!(fused[0], "X"); // Replaced conv_out with X
        assert_eq!(fused[1], "scale");
        assert_eq!(fused[2], "B");
    }

    #[test]
    fn test_merge_inputs() {
        let consumer = make_node("Add", &["conv_out", "bias"], &["add_out"], "add");
        let producer = make_node("Conv", &["X", "W"], &["conv_out"], "conv");

        let merged = merge_inputs(&consumer, &producer);

        // Should have: X, W (from producer), bias (from consumer, excluding conv_out)
        assert_eq!(merged.len(), 3);
        assert!(merged.contains(&"X".to_string()));
        assert!(merged.contains(&"W".to_string()));
        assert!(merged.contains(&"bias".to_string()));
    }
}
