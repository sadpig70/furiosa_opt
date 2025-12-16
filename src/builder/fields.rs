//! Field update utilities for ONNX models
//!
//! Functions for updating and manipulating model/graph fields.

use crate::proto::{
    GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto, ValueInfoProto,
};

/// Update model metadata
pub fn update_model_metadata(
    model: &mut ModelProto,
    producer_name: Option<&str>,
    producer_version: Option<&str>,
    doc_string: Option<&str>,
) {
    if let Some(name) = producer_name {
        model.producer_name = name.to_string();
    }
    if let Some(version) = producer_version {
        model.producer_version = version.to_string();
    }
    if let Some(doc) = doc_string {
        model.doc_string = doc.to_string();
    }
}

/// Set IR version
pub fn set_ir_version(model: &mut ModelProto, version: i64) {
    model.ir_version = version;
}

/// Add or update opset import
pub fn set_opset_version(model: &mut ModelProto, domain: &str, version: i64) {
    // Check if domain already exists
    for opset in &mut model.opset_import {
        if opset.domain == domain {
            opset.version = version;
            return;
        }
    }

    // Add new opset
    model.opset_import.push(OperatorSetIdProto {
        domain: domain.to_string(),
        version,
    });
}

/// Get opset version for a domain
pub fn get_opset_version(model: &ModelProto, domain: &str) -> Option<i64> {
    model
        .opset_import
        .iter()
        .find(|op| op.domain == domain)
        .map(|op| op.version)
}

/// Update graph name
pub fn set_graph_name(graph: &mut GraphProto, name: &str) {
    graph.name = name.to_string();
}

/// Add initializer to graph
pub fn add_initializer(graph: &mut GraphProto, tensor: TensorProto) {
    // Remove existing if present
    graph.initializer.retain(|t| t.name != tensor.name);
    graph.initializer.push(tensor);
}

/// Add multiple initializers
pub fn add_initializers(graph: &mut GraphProto, tensors: Vec<TensorProto>) {
    for tensor in tensors {
        add_initializer(graph, tensor);
    }
}

/// Remove initializer by name
pub fn remove_initializer(graph: &mut GraphProto, name: &str) -> Option<TensorProto> {
    let pos = graph.initializer.iter().position(|t| t.name == name)?;
    Some(graph.initializer.remove(pos))
}

/// Add value_info to graph
pub fn add_value_info(graph: &mut GraphProto, vi: ValueInfoProto) {
    // Remove existing if present
    graph.value_info.retain(|v| v.name != vi.name);
    graph.value_info.push(vi);
}

/// Remove value_info by name
pub fn remove_value_info(graph: &mut GraphProto, name: &str) -> Option<ValueInfoProto> {
    let pos = graph.value_info.iter().position(|vi| vi.name == name)?;
    Some(graph.value_info.remove(pos))
}

/// Add node to graph
pub fn add_node(graph: &mut GraphProto, node: NodeProto) {
    graph.node.push(node);
}

/// Insert node at specific position
pub fn insert_node(graph: &mut GraphProto, index: usize, node: NodeProto) {
    if index <= graph.node.len() {
        graph.node.insert(index, node);
    } else {
        graph.node.push(node);
    }
}

/// Remove node by name
pub fn remove_node(graph: &mut GraphProto, name: &str) -> Option<NodeProto> {
    let pos = graph.node.iter().position(|n| n.name == name)?;
    Some(graph.node.remove(pos))
}

/// Replace node by name
pub fn replace_node(graph: &mut GraphProto, name: &str, new_node: NodeProto) -> Option<NodeProto> {
    let pos = graph.node.iter().position(|n| n.name == name)?;
    Some(std::mem::replace(&mut graph.node[pos], new_node))
}

/// Update graph input
pub fn set_graph_input(graph: &mut GraphProto, input: ValueInfoProto) {
    // Remove existing if present
    graph.input.retain(|vi| vi.name != input.name);
    graph.input.push(input);
}

/// Update graph output
pub fn set_graph_output(graph: &mut GraphProto, output: ValueInfoProto) {
    // Remove existing if present
    graph.output.retain(|vi| vi.name != output.name);
    graph.output.push(output);
}

/// Rename tensor throughout the graph
///
/// Updates all references to `old_name` with `new_name` in:
/// - Node inputs/outputs
/// - Initializers
/// - Value info
/// - Graph inputs/outputs
pub fn rename_tensor(graph: &mut GraphProto, old_name: &str, new_name: &str) {
    // Node inputs and outputs
    for node in &mut graph.node {
        for input in &mut node.input {
            if input == old_name {
                *input = new_name.to_string();
            }
        }
        for output in &mut node.output {
            if output == old_name {
                *output = new_name.to_string();
            }
        }
    }

    // Initializers
    for init in &mut graph.initializer {
        if init.name == old_name {
            init.name = new_name.to_string();
        }
    }

    // Value info
    for vi in &mut graph.value_info {
        if vi.name == old_name {
            vi.name = new_name.to_string();
        }
    }

    // Graph inputs
    for input in &mut graph.input {
        if input.name == old_name {
            input.name = new_name.to_string();
        }
    }

    // Graph outputs
    for output in &mut graph.output {
        if output.name == old_name {
            output.name = new_name.to_string();
        }
    }
}

/// Rename node
pub fn rename_node(graph: &mut GraphProto, old_name: &str, new_name: &str) {
    for node in &mut graph.node {
        if node.name == old_name {
            node.name = new_name.to_string();
            return;
        }
    }
}

/// Generate unique tensor name
pub fn unique_tensor_name(graph: &GraphProto, prefix: &str) -> String {
    let mut counter = 0;
    loop {
        let name = format!("{}_{}", prefix, counter);

        // Check if name exists in any tensor collection
        let exists = graph
            .node
            .iter()
            .any(|n| n.input.contains(&name) || n.output.contains(&name))
            || graph.initializer.iter().any(|t| t.name == name)
            || graph.value_info.iter().any(|vi| vi.name == name)
            || graph.input.iter().any(|vi| vi.name == name)
            || graph.output.iter().any(|vi| vi.name == name);

        if !exists {
            return name;
        }
        counter += 1;
    }
}

/// Generate unique node name
pub fn unique_node_name(graph: &GraphProto, prefix: &str) -> String {
    let mut counter = 0;
    loop {
        let name = format!("{}_{}", prefix, counter);
        if !graph.node.iter().any(|n| n.name == name) {
            return name;
        }
        counter += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;

    fn make_test_model() -> ModelProto {
        ModelProto {
            ir_version: 7,
            producer_name: "test".to_string(),
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 13,
            }],
            graph: Some(GraphProto {
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
            }),
            ..Default::default()
        }
    }

    #[test]
    fn test_update_model_metadata() {
        let mut model = make_test_model();

        update_model_metadata(
            &mut model,
            Some("furiosa-optimizer"),
            Some("0.1.0"),
            Some("Optimized model"),
        );

        assert_eq!(model.producer_name, "furiosa-optimizer");
        assert_eq!(model.producer_version, "0.1.0");
        assert_eq!(model.doc_string, "Optimized model");
    }

    #[test]
    fn test_set_opset_version() {
        let mut model = make_test_model();

        // Update existing
        set_opset_version(&mut model, "", 14);
        assert_eq!(get_opset_version(&model, ""), Some(14));

        // Add new
        set_opset_version(&mut model, "ai.onnx.ml", 2);
        assert_eq!(get_opset_version(&model, "ai.onnx.ml"), Some(2));
    }

    #[test]
    fn test_add_initializer() {
        let mut model = make_test_model();
        let graph = model.graph.as_mut().unwrap();

        let tensor = TensorProto {
            name: "new_weight".to_string(),
            ..Default::default()
        };

        add_initializer(graph, tensor);

        assert_eq!(graph.initializer.len(), 2);
        assert!(graph.initializer.iter().any(|t| t.name == "new_weight"));
    }

    #[test]
    fn test_rename_tensor() {
        let mut model = make_test_model();
        let graph = model.graph.as_mut().unwrap();

        rename_tensor(graph, "conv_out", "new_name");

        // Check node outputs
        assert_eq!(graph.node[0].output[0], "new_name");
        // Check node inputs
        assert_eq!(graph.node[1].input[0], "new_name");
    }

    #[test]
    fn test_unique_tensor_name() {
        let model = make_test_model();
        let graph = model.graph.as_ref().unwrap();

        let name = unique_tensor_name(graph, "tensor");
        assert!(name.starts_with("tensor_"));

        // Should not conflict with existing names
        assert_ne!(name, "X");
        assert_ne!(name, "W");
        assert_ne!(name, "conv_out");
        assert_ne!(name, "Y");
    }

    #[test]
    fn test_unique_node_name() {
        let model = make_test_model();
        let graph = model.graph.as_ref().unwrap();

        let name = unique_node_name(graph, "node");
        assert!(name.starts_with("node_"));
        assert_ne!(name, "conv_0");
        assert_ne!(name, "relu_0");
    }

    #[test]
    fn test_remove_node() {
        let mut model = make_test_model();
        let graph = model.graph.as_mut().unwrap();

        let removed = remove_node(graph, "relu_0");

        assert!(removed.is_some());
        assert_eq!(removed.unwrap().name, "relu_0");
        assert_eq!(graph.node.len(), 1);
    }
}
