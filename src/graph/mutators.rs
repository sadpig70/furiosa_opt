//! Graph mutation operations
//!
//! Methods for modifying the graph structure: adding, removing, and replacing nodes.

use crate::proto::{NodeProto, TensorProto, ValueInfoProto};

use super::context::GraphContext;
use super::maps::OpEntry;

impl GraphContext {
    // ========================================================================
    // Node mutation
    // ========================================================================

    /// Insert a new node into the graph
    ///
    /// Updates all relevant maps.
    pub fn insert_node(&mut self, node: NodeProto) {
        let name = node.name.clone();

        // Update producer_map for outputs
        for output in &node.output {
            if !output.is_empty() {
                self.producer_map.insert(output.clone(), name.clone());
            }
        }

        // Update consumer_map for inputs
        for input in &node.input {
            if !input.is_empty() {
                self.consumer_map
                    .entry(input.clone())
                    .or_default()
                    .push(name.clone());
            }
        }

        // Update input_count_map
        for input in &node.input {
            if !input.is_empty() {
                *self.input_count_map.entry(input.clone()).or_insert(0) += 1;
            }
        }

        // Insert into optimizer_map
        self.optimizer_map.insert(name, OpEntry::new(node));
    }

    /// Remove a node from the graph
    ///
    /// Does NOT update consumer links - use `bridge_disconnected_nodes` for that.
    pub fn remove_node(&mut self, name: &str) -> Option<NodeProto> {
        let entry = self.optimizer_map.swap_remove(name)?;
        let node = entry.node;

        // Remove from producer_map
        for output in &node.output {
            self.producer_map.remove(output);
        }

        // Remove from consumer_map
        for input in &node.input {
            if let Some(consumers) = self.consumer_map.get_mut(input) {
                consumers.retain(|n| n != name);
            }
        }

        // Update input_count_map
        for input in &node.input {
            if let Some(count) = self.input_count_map.get_mut(input) {
                *count = count.saturating_sub(1);
            }
        }

        Some(node)
    }

    /// Replace a node with a new one
    ///
    /// The replacement must have the same name.
    pub fn replace_node(&mut self, node: NodeProto) -> Option<NodeProto> {
        let name = &node.name;
        let old_entry = self.optimizer_map.get_mut(name)?;
        let old_node = std::mem::replace(&mut old_entry.node, node.clone());

        // Update producer_map if outputs changed
        for output in &old_node.output {
            self.producer_map.remove(output);
        }
        for output in &node.output {
            if !output.is_empty() {
                self.producer_map.insert(output.clone(), name.clone());
            }
        }

        // Update consumer_map if inputs changed
        for input in &old_node.input {
            if let Some(consumers) = self.consumer_map.get_mut(input) {
                consumers.retain(|n| n != name);
            }
        }
        for input in &node.input {
            if !input.is_empty() {
                self.consumer_map
                    .entry(input.clone())
                    .or_default()
                    .push(name.clone());
            }
        }

        Some(old_node)
    }

    // ========================================================================
    // Initializer mutation
    // ========================================================================

    /// Add or update an initializer
    pub fn set_initializer(&mut self, tensor: TensorProto) {
        let name = tensor.name.clone();
        self.initializer_map.insert(name, tensor);
    }

    /// Remove an initializer
    pub fn remove_initializer(&mut self, name: &str) -> Option<TensorProto> {
        self.initializer_map.remove(name)
    }

    // ========================================================================
    // Value info mutation
    // ========================================================================

    /// Add or update value info
    pub fn set_value_info(&mut self, vi: ValueInfoProto) {
        let name = vi.name.clone();
        self.value_info_map.insert(name, vi);
    }

    /// Remove value info
    pub fn remove_value_info(&mut self, name: &str) -> Option<ValueInfoProto> {
        self.value_info_map.remove(name)
    }

    // ========================================================================
    // Map update helpers
    // ========================================================================

    /// Update producer map entry
    pub fn update_producer(&mut self, tensor_name: &str, node_name: &str) {
        self.producer_map
            .insert(tensor_name.to_string(), node_name.to_string());
    }

    /// Remove entry from producer map
    pub fn remove_producer(&mut self, tensor_name: &str) {
        self.producer_map.remove(tensor_name);
    }

    /// Add a consumer to a tensor
    pub fn add_consumer(&mut self, tensor_name: &str, node_name: &str) {
        self.consumer_map
            .entry(tensor_name.to_string())
            .or_default()
            .push(node_name.to_string());
    }

    /// Remove a consumer from a tensor
    pub fn remove_consumer(&mut self, tensor_name: &str, node_name: &str) {
        if let Some(consumers) = self.consumer_map.get_mut(tensor_name) {
            consumers.retain(|n| n != node_name);
        }
    }

    /// Decrement input count for a tensor
    pub fn decrement_input_count(&mut self, tensor_name: &str) {
        if let Some(count) = self.input_count_map.get_mut(tensor_name) {
            *count = count.saturating_sub(1);
        }
    }

    /// Increment input count for a tensor
    pub fn increment_input_count(&mut self, tensor_name: &str) {
        *self
            .input_count_map
            .entry(tensor_name.to_string())
            .or_insert(0) += 1;
    }

    // ========================================================================
    // Node input/output manipulation
    // ========================================================================

    /// Update a node's input at the given index
    pub fn update_node_input(&mut self, node_name: &str, index: usize, new_input: &str) -> bool {
        if let Some(entry) = self.optimizer_map.get_mut(node_name) {
            if index < entry.node.input.len() {
                let old_input =
                    std::mem::replace(&mut entry.node.input[index], new_input.to_string());

                // Update consumer maps
                if !old_input.is_empty() {
                    self.remove_consumer(&old_input, node_name);
                    self.decrement_input_count(&old_input);
                }
                if !new_input.is_empty() {
                    self.add_consumer(new_input, node_name);
                    self.increment_input_count(new_input);
                }

                return true;
            }
        }
        false
    }

    /// Update a node's output at the given index
    pub fn update_node_output(&mut self, node_name: &str, index: usize, new_output: &str) -> bool {
        if let Some(entry) = self.optimizer_map.get_mut(node_name) {
            if index < entry.node.output.len() {
                let old_output =
                    std::mem::replace(&mut entry.node.output[index], new_output.to_string());

                // Update producer map
                self.producer_map.remove(&old_output);
                if !new_output.is_empty() {
                    self.producer_map
                        .insert(new_output.to_string(), node_name.to_string());
                }

                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::GraphProto;

    fn make_test_graph() -> GraphProto {
        GraphProto {
            node: vec![
                make_node("Conv", &["X", "W"], &["conv_out"], "conv_0"),
                make_node("Relu", &["conv_out"], &["Y"], "relu_0"),
            ],
            ..Default::default()
        }
    }

    #[test]
    fn test_insert_node() {
        let graph = make_test_graph();
        let mut ctx = GraphContext::new(&graph);

        let new_node = make_node("Sigmoid", &["Y"], &["Z"], "sigmoid_0");
        ctx.insert_node(new_node);

        assert!(ctx.has_node("sigmoid_0"));
        assert_eq!(ctx.get_producer_name("Z"), Some(&"sigmoid_0".to_string()));
        assert!(ctx
            .get_consumer_names("Y")
            .unwrap()
            .contains(&"sigmoid_0".to_string()));
    }

    #[test]
    fn test_remove_node() {
        let graph = make_test_graph();
        let mut ctx = GraphContext::new(&graph);

        let removed = ctx.remove_node("relu_0").unwrap();
        assert_eq!(removed.op_type, "Relu");
        assert!(!ctx.has_node("relu_0"));
        assert!(ctx.get_producer_name("Y").is_none());
    }

    #[test]
    fn test_replace_node() {
        let graph = make_test_graph();
        let mut ctx = GraphContext::new(&graph);

        let new_relu = make_node("LeakyRelu", &["conv_out"], &["Y"], "relu_0");
        let old = ctx.replace_node(new_relu).unwrap();

        assert_eq!(old.op_type, "Relu");
        let node = ctx.get_node("relu_0").unwrap();
        assert_eq!(node.op_type, "LeakyRelu");
    }

    #[test]
    fn test_set_initializer() {
        let graph = make_test_graph();
        let mut ctx = GraphContext::new(&graph);

        let tensor = TensorProto {
            name: "new_weight".to_string(),
            dims: vec![3, 3],
            ..Default::default()
        };
        ctx.set_initializer(tensor);

        assert!(ctx.is_initializer("new_weight"));
    }

    #[test]
    fn test_update_node_input() {
        let graph = make_test_graph();
        let mut ctx = GraphContext::new(&graph);

        assert!(ctx.update_node_input("relu_0", 0, "new_input"));

        let relu = ctx.get_node("relu_0").unwrap();
        assert_eq!(relu.input[0], "new_input");

        // Check consumer maps updated
        assert!(
            ctx.get_consumer_names("conv_out").unwrap().is_empty()
                || !ctx
                    .get_consumer_names("conv_out")
                    .unwrap()
                    .contains(&"relu_0".to_string())
        );
        assert!(ctx
            .get_consumer_names("new_input")
            .unwrap()
            .contains(&"relu_0".to_string()));
    }

    #[test]
    fn test_update_node_output() {
        let graph = make_test_graph();
        let mut ctx = GraphContext::new(&graph);

        assert!(ctx.update_node_output("conv_0", 0, "new_output"));

        let conv = ctx.get_node("conv_0").unwrap();
        assert_eq!(conv.output[0], "new_output");

        assert_eq!(
            ctx.get_producer_name("new_output"),
            Some(&"conv_0".to_string())
        );
        assert!(ctx.get_producer_name("conv_out").is_none());
    }
}
