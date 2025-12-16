//! Model builder for ONNX models
//!
//! Provides utilities for building and assembling optimized ONNX models.

use crate::error::{OnnxResult, TransformError};
use crate::graph::GraphContext;
use crate::proto::{
    GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto, ValueInfoProto,
};

use super::cleanup::{cleanup_graph, CleanupStats};

/// Model builder for constructing optimized models
#[derive(Debug)]
pub struct ModelBuilder {
    /// Original model (for metadata)
    original: ModelProto,
    /// Working graph context
    ctx: Option<GraphContext>,
    /// Built graph (after finalization)
    graph: Option<GraphProto>,
    /// Producer name override
    producer_name: Option<String>,
    /// Producer version override
    producer_version: Option<String>,
    /// Whether to run cleanup
    cleanup: bool,
}

impl ModelBuilder {
    /// Create a new builder from an original model
    pub fn new(model: ModelProto) -> Self {
        Self {
            original: model,
            ctx: None,
            graph: None,
            producer_name: None,
            producer_version: None,
            cleanup: true,
        }
    }

    /// Set the graph context
    pub fn with_context(mut self, ctx: GraphContext) -> Self {
        self.ctx = Some(ctx);
        self
    }

    /// Set producer name
    pub fn producer_name(mut self, name: &str) -> Self {
        self.producer_name = Some(name.to_string());
        self
    }

    /// Set producer version
    pub fn producer_version(mut self, version: &str) -> Self {
        self.producer_version = Some(version.to_string());
        self
    }

    /// Enable or disable cleanup
    pub fn cleanup(mut self, enable: bool) -> Self {
        self.cleanup = enable;
        self
    }

    /// Build the graph from context
    pub fn build_graph(&mut self) -> OnnxResult<&GraphProto> {
        if self.graph.is_some() {
            return Ok(self.graph.as_ref().unwrap());
        }

        let ctx = self
            .ctx
            .as_ref()
            .ok_or_else(|| TransformError::MissingField("context".to_string()))?;

        let mut graph = build_graph_from_context(ctx);

        if self.cleanup {
            cleanup_graph(&mut graph);
        }

        self.graph = Some(graph);
        Ok(self.graph.as_ref().unwrap())
    }

    /// Build the final model
    pub fn build(mut self) -> OnnxResult<ModelProto> {
        self.build_graph()?;

        let mut model = self.original.clone();

        // Set graph
        model.graph = self.graph;

        // Update metadata
        if let Some(name) = self.producer_name {
            model.producer_name = name;
        }
        if let Some(version) = self.producer_version {
            model.producer_version = version;
        }

        Ok(model)
    }
}

/// Build a GraphProto from a GraphContext
///
/// This is the core function that assembles the final graph from the
/// transformation context.
pub fn build_graph_from_context(ctx: &GraphContext) -> GraphProto {
    // Collect active nodes in order
    let nodes: Vec<NodeProto> = ctx.active_nodes().cloned().collect();

    // Collect used tensor names
    let mut used_tensors = std::collections::HashSet::new();
    for node in &nodes {
        for input in &node.input {
            if !input.is_empty() {
                used_tensors.insert(input.clone());
            }
        }
        for output in &node.output {
            if !output.is_empty() {
                used_tensors.insert(output.clone());
            }
        }
    }

    // Add graph inputs/outputs
    for name in ctx.graph_input_map.keys() {
        used_tensors.insert(name.clone());
    }
    for name in ctx.graph_output_map.keys() {
        used_tensors.insert(name.clone());
    }

    // Filter initializers
    let initializers: Vec<TensorProto> = ctx
        .initializer_map
        .values()
        .filter(|t| used_tensors.contains(&t.name))
        .cloned()
        .collect();

    // Get graph inputs/outputs
    let inputs: Vec<ValueInfoProto> = ctx.graph_input_map.values().cloned().collect();
    let outputs: Vec<ValueInfoProto> = ctx.graph_output_map.values().cloned().collect();

    // Filter value_info (exclude graph inputs/outputs)
    let graph_inputs: std::collections::HashSet<_> = inputs.iter().map(|vi| &vi.name).collect();
    let graph_outputs: std::collections::HashSet<_> = outputs.iter().map(|vi| &vi.name).collect();

    let value_info: Vec<ValueInfoProto> = ctx
        .value_info_map
        .values()
        .filter(|vi| {
            used_tensors.contains(&vi.name)
                && !graph_inputs.contains(&vi.name)
                && !graph_outputs.contains(&vi.name)
        })
        .cloned()
        .collect();

    GraphProto {
        node: nodes,
        initializer: initializers,
        input: inputs,
        output: outputs,
        value_info,
        name: String::new(),
        doc_string: String::new(),
        ..Default::default()
    }
}

/// Build optimized model from context and original model
///
/// This is the main entry point for building optimized models.
pub fn build_optimized_model(ctx: &GraphContext, original: &ModelProto) -> ModelProto {
    let mut model = original.clone();
    model.graph = Some(build_graph_from_context(ctx));

    // Cleanup
    if let Some(graph) = model.graph.as_mut() {
        cleanup_graph(graph);
    }

    // Auto-add com.microsoft domain for Gelu/FastGelu nodes
    if let Some(graph) = &model.graph {
        if graph
            .node
            .iter()
            .any(|n| n.op_type == "Gelu" || n.op_type == "FastGelu")
            && !model
                .opset_import
                .iter()
                .any(|opset| opset.domain == "com.microsoft")
            {
                model.opset_import.push(OperatorSetIdProto {
                    domain: "com.microsoft".to_string(),
                    version: 1,
                });
            }
    }

    model
}

/// Build optimized model with statistics
pub fn build_optimized_model_with_stats(
    ctx: &GraphContext,
    original: &ModelProto,
) -> (ModelProto, CleanupStats) {
    let mut model = original.clone();
    model.graph = Some(build_graph_from_context(ctx));

    let stats = if let Some(graph) = model.graph.as_mut() {
        super::cleanup::cleanup_with_stats(graph)
    } else {
        CleanupStats::default()
    };

    // Auto-add com.microsoft domain for Gelu/FastGelu nodes
    if let Some(graph) = &model.graph {
        if graph
            .node
            .iter()
            .any(|n| n.op_type == "Gelu" || n.op_type == "FastGelu")
            && !model
                .opset_import
                .iter()
                .any(|opset| opset.domain == "com.microsoft")
            {
                model.opset_import.push(OperatorSetIdProto {
                    domain: "com.microsoft".to_string(),
                    version: 1,
                });
            }
    }

    (model, stats)
}

/// Quick optimization helper - creates context, applies transform, builds model
pub fn optimize_model<F>(model: ModelProto, transform: F) -> OnnxResult<ModelProto>
where
    F: FnOnce(&mut GraphContext) -> OnnxResult<()>,
{
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| TransformError::MissingField("model.graph".to_string()))?;

    let mut ctx = GraphContext::new(graph);
    transform(&mut ctx)?;

    Ok(build_optimized_model(&ctx, &model))
}

/// Validate model structure
pub fn validate_model(model: &ModelProto) -> OnnxResult<()> {
    // Check for graph
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| TransformError::MissingField("model.graph".to_string()))?;

    // Check for at least one output
    if graph.output.is_empty() {
        return Err(TransformError::ValidationFailed(
            "Graph has no outputs".to_string(),
        ));
    }

    // Check that all node outputs are unique
    let mut outputs = std::collections::HashSet::new();
    for node in &graph.node {
        for output in &node.output {
            if !output.is_empty() && !outputs.insert(output.clone()) {
                return Err(TransformError::ValidationFailed(format!(
                    "Duplicate output tensor: {}",
                    output
                )));
            }
        }
    }

    // Check that all node inputs exist (as graph input, initializer, or node output)
    let mut available: std::collections::HashSet<String> = graph
        .input
        .iter()
        .map(|vi| vi.name.clone())
        .chain(graph.initializer.iter().map(|t| t.name.clone()))
        .collect();

    for node in &graph.node {
        // Check inputs
        for input in &node.input {
            if !input.is_empty() && !available.contains(input) {
                return Err(TransformError::ValidationFailed(format!(
                    "Missing input tensor: {} (required by node {})",
                    input, node.name
                )));
            }
        }
        // Add outputs to available
        for output in &node.output {
            if !output.is_empty() {
                available.insert(output.clone());
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::OperatorSetIdProto;

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
    fn test_model_builder() {
        let model = make_test_model();
        let graph = model.graph.as_ref().unwrap();
        let ctx = GraphContext::new(graph);

        let result = ModelBuilder::new(model.clone())
            .with_context(ctx)
            .producer_name("furiosa-optimizer")
            .producer_version("0.1.0")
            .build();

        assert!(result.is_ok());
        let built = result.unwrap();
        assert_eq!(built.producer_name, "furiosa-optimizer");
        assert_eq!(built.producer_version, "0.1.0");
    }

    #[test]
    fn test_build_graph_from_context() {
        let model = make_test_model();
        let graph = model.graph.as_ref().unwrap();
        let ctx = GraphContext::new(graph);

        let built_graph = build_graph_from_context(&ctx);

        assert_eq!(built_graph.node.len(), 2);
        assert_eq!(built_graph.initializer.len(), 1);
        assert_eq!(built_graph.input.len(), 1);
        assert_eq!(built_graph.output.len(), 1);
    }

    #[test]
    fn test_build_optimized_model() {
        let model = make_test_model();
        let graph = model.graph.as_ref().unwrap();
        let ctx = GraphContext::new(graph);

        let optimized = build_optimized_model(&ctx, &model);

        assert!(optimized.graph.is_some());
        assert_eq!(optimized.graph.as_ref().unwrap().node.len(), 2);
    }

    #[test]
    fn test_validate_model_valid() {
        let model = make_test_model();
        assert!(validate_model(&model).is_ok());
    }

    #[test]
    fn test_validate_model_missing_input() {
        let mut model = make_test_model();
        let graph = model.graph.as_mut().unwrap();
        graph.node[0].input.push("missing_tensor".to_string());

        let result = validate_model(&model);
        assert!(result.is_err());
    }

    #[test]
    fn test_optimize_model() {
        let model = make_test_model();

        let result = optimize_model(model, |_ctx| {
            // No-op transform
            Ok(())
        });

        assert!(result.is_ok());
    }
}
