//! Graph cleanup utilities
//!
//! Functions for removing unused elements from ONNX graphs.

use std::collections::HashSet;

use crate::graph::GraphContext;
use crate::proto::{GraphProto, TensorProto, ValueInfoProto};

/// Collect all tensor names that are actually used in the graph
pub fn collect_used_tensors(ctx: &GraphContext) -> HashSet<String> {
    let mut used = HashSet::new();

    // Collect from active nodes
    for node in ctx.active_nodes() {
        for input in &node.input {
            if !input.is_empty() {
                used.insert(input.clone());
            }
        }
        for output in &node.output {
            if !output.is_empty() {
                used.insert(output.clone());
            }
        }
    }

    // Graph inputs and outputs are always used
    for name in ctx.graph_input_map.keys() {
        used.insert(name.clone());
    }
    for name in ctx.graph_output_map.keys() {
        used.insert(name.clone());
    }

    used
}

/// Filter initializers to keep only used ones
pub fn filter_initializers(
    initializers: &[TensorProto],
    used_tensors: &HashSet<String>,
) -> Vec<TensorProto> {
    initializers
        .iter()
        .filter(|t| used_tensors.contains(&t.name))
        .cloned()
        .collect()
}

/// Filter value_info to keep only used ones (excluding graph inputs/outputs)
pub fn filter_value_info(
    value_info: &[ValueInfoProto],
    used_tensors: &HashSet<String>,
    graph_inputs: &HashSet<String>,
    graph_outputs: &HashSet<String>,
) -> Vec<ValueInfoProto> {
    value_info
        .iter()
        .filter(|vi| {
            used_tensors.contains(&vi.name)
                && !graph_inputs.contains(&vi.name)
                && !graph_outputs.contains(&vi.name)
        })
        .cloned()
        .collect()
}

/// Remove unused initializers from a graph
pub fn remove_unused_initializers(graph: &mut GraphProto) {
    let used: HashSet<String> = graph
        .node
        .iter()
        .flat_map(|n| n.input.iter())
        .filter(|s| !s.is_empty())
        .cloned()
        .collect();

    graph.initializer.retain(|t| used.contains(&t.name));
}

/// Remove unused value_info from a graph
pub fn remove_unused_value_info(graph: &mut GraphProto) {
    let mut used: HashSet<String> = HashSet::new();

    // Collect from nodes
    for node in &graph.node {
        for input in &node.input {
            used.insert(input.clone());
        }
        for output in &node.output {
            used.insert(output.clone());
        }
    }

    // Graph inputs and outputs
    let inputs: HashSet<String> = graph.input.iter().map(|vi| vi.name.clone()).collect();
    let outputs: HashSet<String> = graph.output.iter().map(|vi| vi.name.clone()).collect();

    graph.value_info.retain(|vi| {
        used.contains(&vi.name) && !inputs.contains(&vi.name) && !outputs.contains(&vi.name)
    });
}

/// Clean up all unused elements in a graph
pub fn cleanup_graph(graph: &mut GraphProto) {
    remove_unused_initializers(graph);
    remove_unused_value_info(graph);
}

/// Statistics from cleanup operation
#[derive(Debug, Default, Clone)]
pub struct CleanupStats {
    /// Number of initializers removed
    pub initializers_removed: usize,
    /// Number of value_info entries removed
    pub value_info_removed: usize,
    /// Number of nodes removed (eliminated)
    pub nodes_removed: usize,
}

/// Perform cleanup and return statistics
pub fn cleanup_with_stats(graph: &mut GraphProto) -> CleanupStats {
    let init_before = graph.initializer.len();
    let vi_before = graph.value_info.len();

    cleanup_graph(graph);

    CleanupStats {
        initializers_removed: init_before.saturating_sub(graph.initializer.len()),
        value_info_removed: vi_before.saturating_sub(graph.value_info.len()),
        nodes_removed: 0, // Nodes are removed via transform, not cleanup
    }
}

/// Remove duplicate initializers (keep first occurrence)
pub fn deduplicate_initializers(graph: &mut GraphProto) {
    let mut seen = HashSet::new();
    graph.initializer.retain(|t| seen.insert(t.name.clone()));
}

/// Remove duplicate value_info (keep first occurrence)
pub fn deduplicate_value_info(graph: &mut GraphProto) {
    let mut seen = HashSet::new();
    graph.value_info.retain(|vi| seen.insert(vi.name.clone()));
}

/// Full deduplication pass
pub fn deduplicate_graph(graph: &mut GraphProto) {
    deduplicate_initializers(graph);
    deduplicate_value_info(graph);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;

    fn make_test_graph() -> GraphProto {
        GraphProto {
            node: vec![
                make_node("Conv", &["X", "W"], &["conv_out"], "conv_0"),
                make_node("Relu", &["conv_out"], &["Y"], "relu_0"),
            ],
            input: vec![ValueInfoProto {
                name: "X".to_string(),
                ..Default::default()
            }],
            output: vec![ValueInfoProto {
                name: "Y".to_string(),
                ..Default::default()
            }],
            initializer: vec![
                TensorProto {
                    name: "W".to_string(),
                    ..Default::default()
                },
                TensorProto {
                    name: "unused_weight".to_string(),
                    ..Default::default()
                },
            ],
            value_info: vec![
                ValueInfoProto {
                    name: "conv_out".to_string(),
                    ..Default::default()
                },
                ValueInfoProto {
                    name: "unused_tensor".to_string(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    #[test]
    fn test_collect_used_tensors() {
        let graph = make_test_graph();
        let ctx = GraphContext::new(&graph);

        let used = collect_used_tensors(&ctx);

        assert!(used.contains("X"));
        assert!(used.contains("W"));
        assert!(used.contains("conv_out"));
        assert!(used.contains("Y"));
        assert!(!used.contains("unused_weight"));
        assert!(!used.contains("unused_tensor"));
    }

    #[test]
    fn test_remove_unused_initializers() {
        let mut graph = make_test_graph();

        assert_eq!(graph.initializer.len(), 2);
        remove_unused_initializers(&mut graph);
        assert_eq!(graph.initializer.len(), 1);
        assert_eq!(graph.initializer[0].name, "W");
    }

    #[test]
    fn test_remove_unused_value_info() {
        let mut graph = make_test_graph();

        assert_eq!(graph.value_info.len(), 2);
        remove_unused_value_info(&mut graph);
        assert_eq!(graph.value_info.len(), 1);
        assert_eq!(graph.value_info[0].name, "conv_out");
    }

    #[test]
    fn test_cleanup_graph() {
        let mut graph = make_test_graph();

        cleanup_graph(&mut graph);

        assert_eq!(graph.initializer.len(), 1);
        assert_eq!(graph.value_info.len(), 1);
    }

    #[test]
    fn test_cleanup_with_stats() {
        let mut graph = make_test_graph();

        let stats = cleanup_with_stats(&mut graph);

        assert_eq!(stats.initializers_removed, 1);
        assert_eq!(stats.value_info_removed, 1);
    }

    #[test]
    fn test_deduplicate_initializers() {
        let mut graph = GraphProto {
            initializer: vec![
                TensorProto {
                    name: "W".to_string(),
                    ..Default::default()
                },
                TensorProto {
                    name: "W".to_string(),
                    ..Default::default()
                }, // duplicate
                TensorProto {
                    name: "B".to_string(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        deduplicate_initializers(&mut graph);

        assert_eq!(graph.initializer.len(), 2);
    }
}
