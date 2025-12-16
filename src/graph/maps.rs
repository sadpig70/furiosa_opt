//! Graph map types and builders
//!
//! Defines the core data structures for efficient graph traversal.

use indexmap::IndexMap;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::proto::{GraphProto, NodeProto, TensorProto, ValueInfoProto};

/// Entry in the optimizer map - tracks node and its state
#[derive(Debug, Clone)]
pub struct OpEntry {
    /// The node itself
    pub node: NodeProto,
    /// Whether this node should be eliminated
    pub eliminated: bool,
}

impl OpEntry {
    /// Create a new entry with the given node
    pub fn new(node: NodeProto) -> Self {
        Self {
            node,
            eliminated: false,
        }
    }

    /// Mark this node for elimination
    pub fn mark_eliminated(&mut self) {
        self.eliminated = true;
    }
}

/// Type alias for producer map: output_name → node_name
pub type ProducerMap = FxHashMap<String, String>;

/// Type alias for consumer map: tensor_name → [consumer_node_names]
/// SmallVec optimized for common case of 1-4 consumers
pub type ConsumerMap = FxHashMap<String, SmallVec<[String; 4]>>;

/// Type alias for optimizer map: node_name → OpEntry (order preserved)
pub type OptimizerMap = IndexMap<String, OpEntry>;

/// Type alias for initializer map: name → TensorProto
pub type InitializerMap = FxHashMap<String, TensorProto>;

/// Type alias for value info map: name → ValueInfoProto
pub type ValueInfoMap = FxHashMap<String, ValueInfoProto>;

/// Type alias for input count map: tensor_name → reference count
pub type InputCountMap = FxHashMap<String, usize>;

/// Build producer map from graph nodes
///
/// Maps each output tensor name to the node that produces it.
pub fn build_producer_map(graph: &GraphProto) -> ProducerMap {
    let mut map = FxHashMap::default();

    for node in &graph.node {
        for output in &node.output {
            if !output.is_empty() {
                map.insert(output.clone(), node.name.clone());
            }
        }
    }

    map
}

/// Build consumer map from graph nodes
///
/// Maps each tensor name to the list of nodes that consume it.
pub fn build_consumer_map(graph: &GraphProto) -> ConsumerMap {
    let mut map: ConsumerMap = FxHashMap::default();

    for node in &graph.node {
        for input in &node.input {
            if !input.is_empty() {
                map.entry(input.clone())
                    .or_default()
                    .push(node.name.clone());
            }
        }
    }

    map
}

/// Build optimizer map from graph nodes
///
/// Preserves node order using IndexMap.
pub fn build_optimizer_map(graph: &GraphProto) -> OptimizerMap {
    let mut map = IndexMap::new();

    for node in &graph.node {
        map.insert(node.name.clone(), OpEntry::new(node.clone()));
    }

    map
}

/// Build initializer map from graph
pub fn build_initializer_map(graph: &GraphProto) -> InitializerMap {
    graph
        .initializer
        .iter()
        .map(|t| (t.name.clone(), t.clone()))
        .collect()
}

/// Build value info map from graph
///
/// Combines graph inputs, outputs, and intermediate value_info.
pub fn build_value_info_map(graph: &GraphProto) -> ValueInfoMap {
    let mut map = FxHashMap::default();

    // Graph inputs
    for vi in &graph.input {
        map.insert(vi.name.clone(), vi.clone());
    }

    // Graph outputs
    for vi in &graph.output {
        map.insert(vi.name.clone(), vi.clone());
    }

    // Intermediate value_info
    for vi in &graph.value_info {
        map.insert(vi.name.clone(), vi.clone());
    }

    map
}

/// Build input count map (reference counting for tensors)
pub fn build_input_count_map(graph: &GraphProto) -> InputCountMap {
    let mut map: InputCountMap = FxHashMap::default();

    for node in &graph.node {
        for input in &node.input {
            if !input.is_empty() {
                *map.entry(input.clone()).or_insert(0) += 1;
            }
        }
    }

    map
}

/// Build graph input map
pub fn build_graph_input_map(graph: &GraphProto) -> ValueInfoMap {
    graph
        .input
        .iter()
        .map(|vi| (vi.name.clone(), vi.clone()))
        .collect()
}

/// Build graph output map
pub fn build_graph_output_map(graph: &GraphProto) -> ValueInfoMap {
    graph
        .output
        .iter()
        .map(|vi| (vi.name.clone(), vi.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_graph() -> GraphProto {
        use crate::proto::extensions::make_node;

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
            initializer: vec![
                TensorProto {
                    name: "W".to_string(),
                    ..Default::default()
                },
                TensorProto {
                    name: "scale".to_string(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    #[test]
    fn test_build_producer_map() {
        let graph = make_test_graph();
        let map = build_producer_map(&graph);

        assert_eq!(map.get("conv_out"), Some(&"conv_0".to_string()));
        assert_eq!(map.get("bn_out"), Some(&"bn_0".to_string()));
        assert_eq!(map.get("Y"), Some(&"relu_0".to_string()));
        assert!(map.get("X").is_none()); // input, not produced by node
    }

    #[test]
    fn test_build_consumer_map() {
        let graph = make_test_graph();
        let map = build_consumer_map(&graph);

        assert_eq!(
            map.get("conv_out").map(|v| v.as_slice()),
            Some(&["bn_0".to_string()][..])
        );
        assert_eq!(
            map.get("bn_out").map(|v| v.as_slice()),
            Some(&["relu_0".to_string()][..])
        );
    }

    #[test]
    fn test_build_optimizer_map_preserves_order() {
        let graph = make_test_graph();
        let map = build_optimizer_map(&graph);

        let keys: Vec<_> = map.keys().collect();
        assert_eq!(keys, vec!["conv_0", "bn_0", "relu_0"]);
    }

    #[test]
    fn test_build_initializer_map() {
        let graph = make_test_graph();
        let map = build_initializer_map(&graph);

        assert!(map.contains_key("W"));
        assert!(map.contains_key("scale"));
        assert!(!map.contains_key("X"));
    }

    #[test]
    fn test_build_input_count_map() {
        let graph = make_test_graph();
        let map = build_input_count_map(&graph);

        assert_eq!(map.get("conv_out"), Some(&1)); // consumed by bn_0
        assert_eq!(map.get("bn_out"), Some(&1)); // consumed by relu_0
        assert_eq!(map.get("X"), Some(&1)); // consumed by conv_0
    }
}
