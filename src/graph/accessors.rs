//! Advanced graph accessor methods
//!
//! High-level methods for pattern matching and graph analysis.

use crate::proto::NodeProto;
use crate::tensor::shape_from_value_info;

use super::context::GraphContext;

impl GraphContext {
    // ========================================================================
    // Pattern matching helpers
    // ========================================================================

    /// Find nodes by op type
    pub fn find_nodes_by_op(&self, op_type: &str) -> Vec<&NodeProto> {
        self.nodes().filter(|n| n.op_type == op_type).collect()
    }

    /// Find nodes matching any of the given op types
    pub fn find_nodes_by_ops(&self, op_types: &[&str]) -> Vec<&NodeProto> {
        self.nodes()
            .filter(|n| op_types.contains(&n.op_type.as_str()))
            .collect()
    }

    /// Check if a node matches an op type pattern
    pub fn node_matches_op(&self, node_name: &str, op_type: &str) -> bool {
        self.get_node(node_name)
            .map(|n| n.op_type == op_type)
            .unwrap_or(false)
    }

    /// Get all predecessor nodes (recursive)
    pub fn get_predecessors(&self, node: &NodeProto) -> Vec<&NodeProto> {
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();
        self.collect_predecessors(node, &mut result, &mut visited);
        result
    }

    fn collect_predecessors<'a>(
        &'a self,
        node: &NodeProto,
        result: &mut Vec<&'a NodeProto>,
        visited: &mut std::collections::HashSet<String>,
    ) {
        for input in &node.input {
            if let Some(producer_name) = self.producer_map.get(input) {
                if visited.insert(producer_name.clone()) {
                    if let Some(producer) = self.get_node(producer_name) {
                        result.push(producer);
                        self.collect_predecessors(producer, result, visited);
                    }
                }
            }
        }
    }

    /// Get all successor nodes (recursive)
    pub fn get_successors(&self, node: &NodeProto) -> Vec<&NodeProto> {
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();
        self.collect_successors(node, &mut result, &mut visited);
        result
    }

    fn collect_successors<'a>(
        &'a self,
        node: &NodeProto,
        result: &mut Vec<&'a NodeProto>,
        visited: &mut std::collections::HashSet<String>,
    ) {
        for output in &node.output {
            if let Some(consumer_names) = self.consumer_map.get(output) {
                for name in consumer_names {
                    if visited.insert(name.clone()) {
                        if let Some(consumer) = self.get_node(name) {
                            result.push(consumer);
                            self.collect_successors(consumer, result, visited);
                        }
                    }
                }
            }
        }
    }

    // ========================================================================
    // Shape and type queries
    // ========================================================================

    /// Get the shape of a tensor
    pub fn get_tensor_shape(&self, name: &str) -> Option<Vec<i64>> {
        // Check value_info first
        if let Some(vi) = self.value_info_map.get(name) {
            if let Some(shape) = shape_from_value_info(vi) {
                return Some(shape);
            }
        }

        // Check initializers
        if let Some(init) = self.initializer_map.get(name) {
            return Some(init.dims.clone());
        }

        None
    }

    /// Get the element type of a tensor
    pub fn get_tensor_elem_type(&self, name: &str) -> Option<i32> {
        // Check value_info
        if let Some(vi) = self.value_info_map.get(name) {
            if let Some(elem_type) = vi.get_elem_type() {
                return Some(elem_type);
            }
        }

        // Check initializers
        if let Some(init) = self.initializer_map.get(name) {
            return Some(init.data_type);
        }

        None
    }

    // ========================================================================
    // Connectivity analysis
    // ========================================================================

    /// Check if a tensor is used only once
    pub fn is_single_use(&self, tensor_name: &str) -> bool {
        self.get_input_count(tensor_name) == 1
    }

    /// Check if a tensor is unused (dead code)
    pub fn is_unused(&self, tensor_name: &str) -> bool {
        !self.is_graph_output(tensor_name) && self.get_input_count(tensor_name) == 0
    }

    /// Get all unused tensors (for cleanup)
    pub fn find_unused_tensors(&self) -> Vec<&str> {
        let mut unused = Vec::new();

        // Check outputs of all nodes
        for node in self.nodes() {
            for output in &node.output {
                if self.is_unused(output) {
                    unused.push(output.as_str());
                }
            }
        }

        // Check initializers
        for name in self.initializer_map.keys() {
            if self.is_unused(name) && !self.is_graph_input(name) {
                unused.push(name.as_str());
            }
        }

        unused
    }

    /// Check if two nodes are adjacent (output of first connects to input of second)
    pub fn are_adjacent(&self, first_name: &str, second_name: &str) -> bool {
        let first = match self.get_node(first_name) {
            Some(n) => n,
            None => return false,
        };

        for output in &first.output {
            if let Some(consumers) = self.consumer_map.get(output) {
                if consumers.iter().any(|n| n == second_name) {
                    return true;
                }
            }
        }

        false
    }

    /// Get the topological order of nodes (using Kahn's algorithm)
    pub fn topological_order(&self) -> Vec<&str> {
        use std::collections::VecDeque;

        let mut in_degree: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        let mut result = Vec::new();

        // Initialize in-degrees
        for name in self.optimizer_map.keys() {
            in_degree.insert(name.as_str(), 0);
        }

        // Count in-degrees (number of predecessor nodes)
        for (name, entry) in &self.optimizer_map {
            let mut pred_count = 0;
            for input in &entry.node.input {
                if self.producer_map.contains_key(input) {
                    pred_count += 1;
                }
            }
            in_degree.insert(name.as_str(), pred_count);
        }

        // Queue of nodes with no dependencies
        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, &count)| count == 0)
            .map(|(&name, _)| name)
            .collect();

        while let Some(name) = queue.pop_front() {
            result.push(name);

            if let Some(entry) = self.optimizer_map.get(name) {
                for output in &entry.node.output {
                    if let Some(consumers) = self.consumer_map.get(output) {
                        for consumer in consumers {
                            if let Some(count) = in_degree.get_mut(consumer.as_str()) {
                                *count = count.saturating_sub(1);
                                if *count == 0 {
                                    queue.push_back(consumer.as_str());
                                }
                            }
                        }
                    }
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::{GraphProto, TensorProto, ValueInfoProto};

    fn make_test_graph() -> GraphProto {
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
            initializer: vec![TensorProto {
                name: "W".to_string(),
                dims: vec![64, 3, 3, 3],
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn test_find_nodes_by_op() {
        let graph = make_test_graph();
        let ctx = GraphContext::new(&graph);

        let convs = ctx.find_nodes_by_op("Conv");
        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].name, "conv_0");

        let empty = ctx.find_nodes_by_op("Softmax");
        assert!(empty.is_empty());
    }

    #[test]
    fn test_find_nodes_by_ops() {
        let graph = make_test_graph();
        let ctx = GraphContext::new(&graph);

        let nodes = ctx.find_nodes_by_ops(&["Conv", "Relu"]);
        assert_eq!(nodes.len(), 2);
    }

    #[test]
    fn test_get_predecessors() {
        let graph = make_test_graph();
        let ctx = GraphContext::new(&graph);

        let relu = ctx.get_node("relu_0").unwrap();
        let preds = ctx.get_predecessors(relu);

        assert_eq!(preds.len(), 2); // bn_0, conv_0
        let names: Vec<_> = preds.iter().map(|n| n.name.as_str()).collect();
        assert!(names.contains(&"bn_0"));
        assert!(names.contains(&"conv_0"));
    }

    #[test]
    fn test_get_successors() {
        let graph = make_test_graph();
        let ctx = GraphContext::new(&graph);

        let conv = ctx.get_node("conv_0").unwrap();
        let succs = ctx.get_successors(conv);

        assert_eq!(succs.len(), 2); // bn_0, relu_0
    }

    #[test]
    fn test_get_tensor_shape() {
        let graph = make_test_graph();
        let ctx = GraphContext::new(&graph);

        let shape = ctx.get_tensor_shape("W").unwrap();
        assert_eq!(shape, vec![64, 3, 3, 3]);
    }

    #[test]
    fn test_is_single_use() {
        let graph = make_test_graph();
        let ctx = GraphContext::new(&graph);

        assert!(ctx.is_single_use("conv_out"));
        assert!(ctx.is_single_use("bn_out"));
    }

    #[test]
    fn test_are_adjacent() {
        let graph = make_test_graph();
        let ctx = GraphContext::new(&graph);

        assert!(ctx.are_adjacent("conv_0", "bn_0"));
        assert!(ctx.are_adjacent("bn_0", "relu_0"));
        assert!(!ctx.are_adjacent("conv_0", "relu_0"));
    }

    #[test]
    fn test_topological_order() {
        let graph = make_test_graph();
        let ctx = GraphContext::new(&graph);

        let order = ctx.topological_order();

        // conv_0 must come before bn_0, bn_0 must come before relu_0
        let conv_pos = order.iter().position(|&n| n == "conv_0").unwrap();
        let bn_pos = order.iter().position(|&n| n == "bn_0").unwrap();
        let relu_pos = order.iter().position(|&n| n == "relu_0").unwrap();

        assert!(conv_pos < bn_pos);
        assert!(bn_pos < relu_pos);
    }
}
