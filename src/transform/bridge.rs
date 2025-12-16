//! Bridge disconnected nodes
//!
//! When a node is eliminated, its inputs and outputs need to be reconnected.
//! This module provides utilities for bridging these disconnections.

use crate::graph::GraphContext;
use crate::proto::NodeProto;

/// Bridge disconnected nodes after elimination
///
/// When a node is eliminated, this function reconnects:
/// - The eliminated node's predecessor outputs to
/// - The eliminated node's successor inputs
///
/// # Arguments
/// * `ctx` - The graph context
/// * `eliminated` - The eliminated node
/// * `input_idx` - Which input of the eliminated node to use as the bridge source (default 0)
///
/// # Returns
/// * Number of connections bridged
pub fn bridge_disconnected_nodes(
    ctx: &mut GraphContext,
    eliminated: &NodeProto,
    input_idx: usize,
) -> usize {
    let mut bridged = 0;

    // Get the input tensor that should replace the eliminated node's output
    let bridge_input = match eliminated.input.get(input_idx) {
        Some(inp) if !inp.is_empty() => inp.clone(),
        _ => return 0,
    };

    // Get the output tensor that will be replaced
    let bridge_output = match eliminated.output.first() {
        Some(out) if !out.is_empty() => out.clone(),
        _ => return 0,
    };

    // Find all consumers of the eliminated node's output
    let consumers: Vec<String> = ctx
        .get_consumer_names(&bridge_output)
        .map(|names| names.to_vec())
        .unwrap_or_default();

    // Update each consumer to use the bridge input instead
    for consumer_name in consumers {
        if let Some(consumer_entry) = ctx.get_entry_mut(&consumer_name) {
            for input in &mut consumer_entry.node.input {
                if input == &bridge_output {
                    *input = bridge_input.clone();
                    bridged += 1;
                }
            }
        }
    }

    // Update the consumer map
    if bridged > 0 {
        // Remove old output from producer map
        ctx.remove_producer(&bridge_output);

        // Update consumer map: remove old entries
        if let Some(old_consumers) = ctx.consumer_map.get(&bridge_output).cloned() {
            for consumer in &old_consumers {
                ctx.add_consumer(&bridge_input, consumer);
            }
        }
        ctx.consumer_map.remove(&bridge_output);
    }

    bridged
}

/// Bridge with output preservation
///
/// Similar to `bridge_disconnected_nodes` but also handles cases where
/// the eliminated node's output is a graph output.
pub fn bridge_with_output_preservation(
    ctx: &mut GraphContext,
    eliminated: &NodeProto,
    input_idx: usize,
) -> usize {
    let bridge_output = match eliminated.output.first() {
        Some(out) => out.clone(),
        None => return 0,
    };

    // If the output is a graph output, we need to update the predecessor's output
    if ctx.is_graph_output(&bridge_output) {
        let bridge_input = match eliminated.input.get(input_idx) {
            Some(inp) if !inp.is_empty() => inp.clone(),
            _ => return 0,
        };

        // Find the producer of the input and update its output
        if let Some(producer_name) = ctx.get_producer_name(&bridge_input).cloned() {
            // Update producer's output to match the graph output name
            if let Some(producer_entry) = ctx.get_entry_mut(&producer_name) {
                for output in &mut producer_entry.node.output {
                    if output == &bridge_input {
                        *output = bridge_output.clone();
                        // Update producer map
                        ctx.update_producer(&bridge_output, &producer_name);
                        ctx.remove_producer(&bridge_input);
                        return 1;
                    }
                }
            }
        }
        return 0;
    }

    // Standard bridging for non-graph-output cases
    bridge_disconnected_nodes(ctx, eliminated, input_idx)
}

/// Batch bridge multiple eliminated nodes
pub fn bridge_all_eliminated(ctx: &mut GraphContext) -> usize {
    let eliminated_names: Vec<String> = ctx
        .optimizer_map
        .iter()
        .filter(|(_, entry)| entry.eliminated)
        .map(|(name, _)| name.clone())
        .collect();

    let mut total_bridged = 0;

    for name in eliminated_names {
        if let Some(entry) = ctx.optimizer_map.get(&name) {
            let node = entry.node.clone();
            total_bridged += bridge_disconnected_nodes(ctx, &node, 0);
        }
    }

    total_bridged
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::{GraphProto, ValueInfoProto};

    fn make_chain_graph() -> GraphProto {
        GraphProto {
            node: vec![
                make_node("Conv", &["X", "W"], &["conv_out"], "conv_0"),
                make_node("Relu", &["conv_out"], &["relu_out"], "relu_0"),
                make_node("Sigmoid", &["relu_out"], &["Y"], "sigmoid_0"),
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
    fn test_bridge_disconnected_nodes() {
        let graph = make_chain_graph();
        let mut ctx = GraphContext::new(&graph);

        // Eliminate relu_0, should bridge conv_out -> sigmoid_0
        let relu = ctx.get_node("relu_0").unwrap().clone();
        ctx.mark_eliminated("relu_0");

        let bridged = bridge_disconnected_nodes(&mut ctx, &relu, 0);
        assert_eq!(bridged, 1);

        // Verify sigmoid now takes conv_out as input
        let sigmoid = ctx.get_node("sigmoid_0").unwrap();
        assert_eq!(sigmoid.input[0], "conv_out");
    }

    #[test]
    fn test_bridge_with_graph_output() {
        let graph = GraphProto {
            node: vec![
                make_node("Conv", &["X", "W"], &["conv_out"], "conv_0"),
                make_node("Relu", &["conv_out"], &["Y"], "relu_0"), // Y is graph output
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
        let relu = ctx.get_node("relu_0").unwrap().clone();

        let bridged = bridge_with_output_preservation(&mut ctx, &relu, 0);
        assert_eq!(bridged, 1);

        // Conv should now output Y directly
        let conv = ctx.get_node("conv_0").unwrap();
        assert_eq!(conv.output[0], "Y");
    }

    #[test]
    fn test_bridge_no_input() {
        let graph = make_chain_graph();
        let mut ctx = GraphContext::new(&graph);

        let empty_node = NodeProto {
            name: "empty".to_string(),
            input: vec![],
            output: vec!["out".to_string()],
            ..Default::default()
        };

        let bridged = bridge_disconnected_nodes(&mut ctx, &empty_node, 0);
        assert_eq!(bridged, 0);
    }
}
