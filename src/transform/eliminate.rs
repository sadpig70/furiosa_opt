//! Node elimination transformations
//!
//! Handles the elimination of nodes from the graph while maintaining connectivity.

use crate::error::OnnxResult;
use crate::graph::GraphContext;
use crate::proto::NodeProto;

use super::bridge::bridge_disconnected_nodes;

/// Result of an elimination operation
#[derive(Debug, Clone)]
pub struct EliminationResult {
    /// Name of the eliminated node
    pub node_name: String,
    /// Number of connections bridged
    pub bridged_connections: usize,
}

/// Eliminate a single node and bridge connections
///
/// This is equivalent to Python's `transform_to_eliminate`.
///
/// # Arguments
/// * `ctx` - The graph context
/// * `node_name` - Name of the node to eliminate
/// * `bridge_input_idx` - Which input to use for bridging (default 0)
///
/// # Returns
/// * `Some(EliminationResult)` if successful
/// * `None` if node not found
pub fn eliminate_node(
    ctx: &mut GraphContext,
    node_name: &str,
    bridge_input_idx: usize,
) -> Option<EliminationResult> {
    // Get the node before marking eliminated
    let node = ctx.get_node(node_name)?.clone();

    // Mark as eliminated
    ctx.mark_eliminated(node_name);

    // Bridge disconnected nodes
    let bridged = bridge_disconnected_nodes(ctx, &node, bridge_input_idx);

    Some(EliminationResult {
        node_name: node_name.to_string(),
        bridged_connections: bridged,
    })
}

/// Eliminate a node only if conditions are met
pub fn eliminate_node_if<F>(
    ctx: &mut GraphContext,
    node_name: &str,
    bridge_input_idx: usize,
    condition: F,
) -> Option<EliminationResult>
where
    F: FnOnce(&NodeProto, &GraphContext) -> bool,
{
    let node = ctx.get_node(node_name)?;
    if !condition(node, ctx) {
        return None;
    }

    eliminate_node(ctx, node_name, bridge_input_idx)
}

/// Eliminate multiple nodes in sequence
pub fn eliminate_nodes(
    ctx: &mut GraphContext,
    node_names: &[&str],
    bridge_input_idx: usize,
) -> Vec<EliminationResult> {
    node_names
        .iter()
        .filter_map(|name| eliminate_node(ctx, name, bridge_input_idx))
        .collect()
}

/// Eliminate a chain of nodes (in reverse order to maintain correct bridging)
///
/// Useful when eliminating a matched pattern where multiple nodes should be removed.
pub fn eliminate_chain(
    ctx: &mut GraphContext,
    chain: &[&str],
    keep_first: bool,
) -> Vec<EliminationResult> {
    let to_eliminate: Vec<&str> = if keep_first {
        chain.iter().skip(1).copied().collect()
    } else {
        chain.to_vec()
    };

    // Eliminate in reverse order (from output to input)
    let mut results = Vec::new();
    for name in to_eliminate.iter().rev() {
        if let Some(result) = eliminate_node(ctx, name, 0) {
            results.push(result);
        }
    }

    results
}

/// Check if a node can be safely eliminated
///
/// A node can be safely eliminated if:
/// 1. It exists and is not already eliminated
/// 2. Its output is single-use OR is not a graph output
pub fn can_eliminate(ctx: &GraphContext, node_name: &str) -> bool {
    let node = match ctx.get_node(node_name) {
        Some(n) => n,
        None => return false,
    };

    if ctx.is_eliminated(node_name) {
        return false;
    }

    // Check if any output is a graph output that's multi-use
    for output in &node.output {
        if ctx.is_graph_output(output) {
            // Graph outputs need special handling
            return false;
        }
    }

    true
}

/// Eliminate identity-like operations
///
/// Removes nodes that don't change the data (like Identity, Dropout in inference mode).
pub fn eliminate_identity_ops(ctx: &mut GraphContext) -> Vec<EliminationResult> {
    let identity_ops = ["Identity", "Dropout"];

    let to_eliminate: Vec<String> = ctx
        .nodes()
        .filter(|n| identity_ops.contains(&n.op_type.as_str()))
        .filter(|n| can_eliminate(ctx, &n.name))
        .map(|n| n.name.clone())
        .collect();

    to_eliminate
        .iter()
        .filter_map(|name| eliminate_node(ctx, name, 0))
        .collect()
}

/// Eliminate dead nodes (outputs not used by any other node or graph output)
pub fn eliminate_dead_nodes(ctx: &mut GraphContext) -> Vec<EliminationResult> {
    let mut results = Vec::new();
    let mut changed = true;

    while changed {
        changed = false;

        let dead_nodes: Vec<String> = ctx
            .nodes()
            .filter(|n| !ctx.is_eliminated(&n.name))
            .filter(|n| {
                // Check if all outputs are unused
                n.output
                    .iter()
                    .all(|out| !ctx.is_graph_output(out) && ctx.get_input_count(out) == 0)
            })
            .map(|n| n.name.clone())
            .collect();

        for name in dead_nodes {
            if let Some(result) = eliminate_node(ctx, &name, 0) {
                results.push(result);
                changed = true;
            }
        }
    }

    results
}

/// Transform result containing all eliminations
#[derive(Debug, Default)]
pub struct BatchEliminationResult {
    /// Individual elimination results
    pub eliminations: Vec<EliminationResult>,
    /// Total nodes eliminated
    pub total_eliminated: usize,
    /// Total connections bridged
    pub total_bridged: usize,
}

impl BatchEliminationResult {
    /// Create from a list of elimination results
    pub fn from_results(results: Vec<EliminationResult>) -> Self {
        let total_eliminated = results.len();
        let total_bridged = results.iter().map(|r| r.bridged_connections).sum();

        Self {
            eliminations: results,
            total_eliminated,
            total_bridged,
        }
    }
}

/// Batch elimination with condition
pub fn eliminate_nodes_where<F>(
    ctx: &mut GraphContext,
    condition: F,
) -> OnnxResult<BatchEliminationResult>
where
    F: Fn(&NodeProto, &GraphContext) -> bool,
{
    let to_eliminate: Vec<String> = ctx
        .nodes()
        .filter(|n| condition(n, ctx))
        .map(|n| n.name.clone())
        .collect();

    let results: Vec<EliminationResult> = to_eliminate
        .iter()
        .filter_map(|name| eliminate_node(ctx, name, 0))
        .collect();

    Ok(BatchEliminationResult::from_results(results))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::{GraphProto, ValueInfoProto};

    fn make_test_graph() -> GraphProto {
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
    fn test_eliminate_node() {
        let graph = make_test_graph();
        let mut ctx = GraphContext::new(&graph);

        let result = eliminate_node(&mut ctx, "identity_0", 0);
        assert!(result.is_some());

        let r = result.unwrap();
        assert_eq!(r.node_name, "identity_0");
        assert_eq!(r.bridged_connections, 1);

        // Verify relu now takes conv_out as input
        let relu = ctx.get_node("relu_0").unwrap();
        assert_eq!(relu.input[0], "conv_out");
    }

    #[test]
    fn test_can_eliminate() {
        let graph = make_test_graph();
        let ctx = GraphContext::new(&graph);

        assert!(can_eliminate(&ctx, "identity_0"));
        assert!(can_eliminate(&ctx, "conv_0"));
        // relu_0 outputs to graph output Y
        assert!(!can_eliminate(&ctx, "relu_0"));
    }

    #[test]
    fn test_eliminate_identity_ops() {
        let graph = make_test_graph();
        let mut ctx = GraphContext::new(&graph);

        let results = eliminate_identity_ops(&mut ctx);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_name, "identity_0");
    }

    #[test]
    fn test_eliminate_chain() {
        let graph = GraphProto {
            node: vec![
                make_node("Conv", &["X"], &["a"], "conv_0"),
                make_node("Relu", &["a"], &["b"], "relu_0"),
                make_node("Identity", &["b"], &["Y"], "identity_0"),
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

        // Eliminate relu and identity, keep conv
        let results = eliminate_chain(&mut ctx, &["conv_0", "relu_0", "identity_0"], true);

        // Should eliminate identity_0 first (reverse order), then relu_0
        // But identity_0 outputs to graph output, might be skipped
        assert!(!results.is_empty());
    }

    #[test]
    fn test_eliminate_node_if() {
        let graph = make_test_graph();
        let mut ctx = GraphContext::new(&graph);

        // Condition that passes
        let result = eliminate_node_if(&mut ctx, "identity_0", 0, |node, _ctx| {
            node.op_type == "Identity"
        });
        assert!(result.is_some());

        // Reset for next test
        let graph = make_test_graph();
        let mut ctx = GraphContext::new(&graph);

        // Condition that fails
        let result = eliminate_node_if(&mut ctx, "conv_0", 0, |node, _ctx| {
            node.op_type == "Identity"
        });
        assert!(result.is_none());
    }
}
