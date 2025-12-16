//! Graph context for ONNX model manipulation
//!
//! `GraphContext` is the central structure for working with ONNX graphs.
//! It maintains efficient maps for node lookup, traversal, and manipulation.

use crate::error::{OnnxResult, TransformError};
use crate::proto::{GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto};

use super::maps::{
    build_consumer_map, build_graph_input_map, build_graph_output_map, build_initializer_map,
    build_input_count_map, build_optimizer_map, build_producer_map, build_value_info_map,
    ConsumerMap, InitializerMap, InputCountMap, OpEntry, OptimizerMap, ProducerMap, ValueInfoMap,
};

/// Graph context for efficient graph operations
///
/// This structure mirrors Python's ONNXTransformer internal state,
/// providing O(1) lookups for nodes, tensors, and their relationships.
#[derive(Debug)]
pub struct GraphContext {
    /// Maps output tensor name → producer node name
    pub producer_map: ProducerMap,

    /// Maps tensor name → consumer node names
    pub consumer_map: ConsumerMap,

    /// Maps node name → OpEntry (preserves insertion order)
    pub optimizer_map: OptimizerMap,

    /// Maps initializer name → TensorProto
    pub initializer_map: InitializerMap,

    /// Maps tensor name → ValueInfoProto (inputs + outputs + value_info)
    pub value_info_map: ValueInfoMap,

    /// Maps graph input name → ValueInfoProto
    pub graph_input_map: ValueInfoMap,

    /// Maps graph output name → ValueInfoProto
    pub graph_output_map: ValueInfoMap,

    /// Maps tensor name → reference count
    pub input_count_map: InputCountMap,
}

impl GraphContext {
    /// Create a new GraphContext from a GraphProto
    pub fn new(graph: &GraphProto) -> Self {
        Self {
            producer_map: build_producer_map(graph),
            consumer_map: build_consumer_map(graph),
            optimizer_map: build_optimizer_map(graph),
            initializer_map: build_initializer_map(graph),
            value_info_map: build_value_info_map(graph),
            graph_input_map: build_graph_input_map(graph),
            graph_output_map: build_graph_output_map(graph),
            input_count_map: build_input_count_map(graph),
        }
    }

    /// Create from a ModelProto
    pub fn from_model(model: &ModelProto) -> OnnxResult<Self> {
        let graph = model
            .graph
            .as_ref()
            .ok_or_else(|| TransformError::MissingField("model.graph".to_string()))?;

        Ok(Self::new(graph))
    }

    // ========================================================================
    // Node accessors
    // ========================================================================

    /// Get a node by name
    pub fn get_node(&self, name: &str) -> Option<&NodeProto> {
        self.optimizer_map.get(name).map(|e| &e.node)
    }

    /// Get a mutable node by name
    pub fn get_node_mut(&mut self, name: &str) -> Option<&mut NodeProto> {
        self.optimizer_map.get_mut(name).map(|e| &mut e.node)
    }

    /// Get OpEntry by name
    pub fn get_entry(&self, name: &str) -> Option<&OpEntry> {
        self.optimizer_map.get(name)
    }

    /// Get mutable OpEntry by name
    pub fn get_entry_mut(&mut self, name: &str) -> Option<&mut OpEntry> {
        self.optimizer_map.get_mut(name)
    }

    /// Check if a node exists
    pub fn has_node(&self, name: &str) -> bool {
        self.optimizer_map.contains_key(name)
    }

    /// Get the number of nodes
    pub fn node_count(&self) -> usize {
        self.optimizer_map.len()
    }

    /// Iterate over all nodes in order
    pub fn nodes(&self) -> impl Iterator<Item = &NodeProto> {
        self.optimizer_map.values().map(|e| &e.node)
    }

    /// Iterate over node names in order
    pub fn node_names(&self) -> impl Iterator<Item = &String> {
        self.optimizer_map.keys()
    }

    // ========================================================================
    // Graph traversal
    // ========================================================================

    /// Get the producer node for a tensor
    pub fn get_producer(&self, tensor_name: &str) -> Option<&NodeProto> {
        self.producer_map
            .get(tensor_name)
            .and_then(|name| self.get_node(name))
    }

    /// Get the producer node name for a tensor
    pub fn get_producer_name(&self, tensor_name: &str) -> Option<&String> {
        self.producer_map.get(tensor_name)
    }

    /// Get consumer nodes for a tensor
    pub fn get_consumers(&self, tensor_name: &str) -> Vec<&NodeProto> {
        self.consumer_map
            .get(tensor_name)
            .map(|names| {
                names
                    .iter()
                    .filter_map(|name| self.get_node(name))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get consumer node names for a tensor
    pub fn get_consumer_names(&self, tensor_name: &str) -> Option<&[String]> {
        self.consumer_map.get(tensor_name).map(|v| v.as_slice())
    }

    /// Get the previous node (producer of first input)
    pub fn get_prev_node(&self, node: &NodeProto) -> Option<&NodeProto> {
        node.input.first().and_then(|inp| self.get_producer(inp))
    }

    /// Get the next nodes (consumers of first output)
    pub fn get_next_nodes(&self, node: &NodeProto) -> Vec<&NodeProto> {
        node.output
            .first()
            .map(|out| self.get_consumers(out))
            .unwrap_or_default()
    }

    /// Check if a tensor is a graph input
    pub fn is_graph_input(&self, name: &str) -> bool {
        self.graph_input_map.contains_key(name)
    }

    /// Check if a tensor is a graph output
    pub fn is_graph_output(&self, name: &str) -> bool {
        self.graph_output_map.contains_key(name)
    }

    /// Check if a tensor is an initializer
    pub fn is_initializer(&self, name: &str) -> bool {
        self.initializer_map.contains_key(name)
    }

    // ========================================================================
    // Value info and initializer accessors
    // ========================================================================

    /// Get value info for a tensor
    pub fn get_value_info(&self, name: &str) -> Option<&ValueInfoProto> {
        self.value_info_map.get(name)
    }

    /// Get initializer by name
    pub fn get_initializer(&self, name: &str) -> Option<&TensorProto> {
        self.initializer_map.get(name)
    }

    /// Get reference count for a tensor
    pub fn get_input_count(&self, name: &str) -> usize {
        self.input_count_map.get(name).copied().unwrap_or(0)
    }

    // ========================================================================
    // Node state management
    // ========================================================================

    /// Mark a node for elimination
    pub fn mark_eliminated(&mut self, name: &str) -> bool {
        if let Some(entry) = self.optimizer_map.get_mut(name) {
            entry.mark_eliminated();
            true
        } else {
            false
        }
    }

    /// Check if a node is marked for elimination
    pub fn is_eliminated(&self, name: &str) -> bool {
        self.optimizer_map
            .get(name)
            .map(|e| e.eliminated)
            .unwrap_or(false)
    }

    /// Get nodes that are not eliminated
    pub fn active_nodes(&self) -> impl Iterator<Item = &NodeProto> {
        self.optimizer_map
            .values()
            .filter(|e| !e.eliminated)
            .map(|e| &e.node)
    }

    /// Count active (non-eliminated) nodes
    pub fn active_node_count(&self) -> usize {
        self.optimizer_map
            .values()
            .filter(|e| !e.eliminated)
            .count()
    }
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
            initializer: vec![TensorProto {
                name: "W".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn test_context_creation() {
        let graph = make_test_graph();
        let ctx = GraphContext::new(&graph);

        assert_eq!(ctx.node_count(), 2);
        assert!(ctx.has_node("conv_0"));
        assert!(ctx.has_node("relu_0"));
    }

    #[test]
    fn test_get_node() {
        let graph = make_test_graph();
        let ctx = GraphContext::new(&graph);

        let conv = ctx.get_node("conv_0").unwrap();
        assert_eq!(conv.op_type, "Conv");

        assert!(ctx.get_node("nonexistent").is_none());
    }

    #[test]
    fn test_get_producer() {
        let graph = make_test_graph();
        let ctx = GraphContext::new(&graph);

        let producer = ctx.get_producer("conv_out").unwrap();
        assert_eq!(producer.name, "conv_0");

        assert!(ctx.get_producer("X").is_none()); // graph input
    }

    #[test]
    fn test_get_consumers() {
        let graph = make_test_graph();
        let ctx = GraphContext::new(&graph);

        let consumers = ctx.get_consumers("conv_out");
        assert_eq!(consumers.len(), 1);
        assert_eq!(consumers[0].name, "relu_0");
    }

    #[test]
    fn test_prev_next_nodes() {
        let graph = make_test_graph();
        let ctx = GraphContext::new(&graph);

        let relu = ctx.get_node("relu_0").unwrap();
        let prev = ctx.get_prev_node(relu).unwrap();
        assert_eq!(prev.name, "conv_0");

        let conv = ctx.get_node("conv_0").unwrap();
        let next = ctx.get_next_nodes(conv);
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].name, "relu_0");
    }

    #[test]
    fn test_is_graph_input_output() {
        let graph = make_test_graph();
        let ctx = GraphContext::new(&graph);

        assert!(ctx.is_graph_input("X"));
        assert!(!ctx.is_graph_input("conv_out"));
        assert!(ctx.is_graph_output("Y"));
        assert!(!ctx.is_graph_output("conv_out"));
    }

    #[test]
    fn test_is_initializer() {
        let graph = make_test_graph();
        let ctx = GraphContext::new(&graph);

        assert!(ctx.is_initializer("W"));
        assert!(!ctx.is_initializer("X"));
    }

    #[test]
    fn test_mark_eliminated() {
        let graph = make_test_graph();
        let mut ctx = GraphContext::new(&graph);

        assert!(!ctx.is_eliminated("conv_0"));
        assert!(ctx.mark_eliminated("conv_0"));
        assert!(ctx.is_eliminated("conv_0"));

        assert_eq!(ctx.active_node_count(), 1);
    }

    #[test]
    fn test_node_iteration_order() {
        let graph = make_test_graph();
        let ctx = GraphContext::new(&graph);

        let names: Vec<_> = ctx.node_names().collect();
        assert_eq!(names, vec!["conv_0", "relu_0"]);
    }
}
