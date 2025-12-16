use crate::error::OnnxResult;
use crate::graph::GraphContext;
use crate::transformers::common::{OnnxTransformer, TransformResult};
use crate::transformers::infer::InferAll;

/// Polishes the model by performing general cleanup and inference
///
/// Steps:
/// 1. Extract Constant nodes to Initializers
/// 2. Run type/shape/axis inference
#[derive(Debug, Default)]
pub struct PolishModel;

impl PolishModel {
    /// Create a new PolishModel transformer
    pub fn new() -> Self {
        Self
    }

    fn extract_constants(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();
        let mut nodes_to_remove = Vec::new();
        let mut new_initializers = Vec::new();

        for node in ctx.nodes() {
            if node.op_type == "Constant" {
                // Extract value from attribute
                let mut tensor_value = None;
                for attr in &node.attribute {
                    if attr.name == "value" {
                        if let Some(t) = &attr.t {
                            tensor_value = Some(t.clone());
                        }
                    }
                }

                if let Some(mut tensor) = tensor_value {
                    // The constant node's output name should be the initializer name
                    if let Some(output_name) = node.output.first() {
                        tensor.name = output_name.clone();
                        new_initializers.push(tensor);
                        nodes_to_remove.push(node.name.clone());
                        result.record_elimination(&node.name);
                    }
                }
            }
        }

        for tensor in new_initializers {
            ctx.set_initializer(tensor);
        }

        for name in nodes_to_remove {
            ctx.remove_node(&name);
        }

        Ok(result)
    }
}

impl OnnxTransformer for PolishModel {
    fn name(&self) -> &'static str {
        "PolishModel"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut total = TransformResult::new();

        // 1. Extract Constants to Initializers
        total.merge(self.extract_constants(ctx)?);

        // 2. Run Inference (Shape, Type, Axes)
        // This corresponds to the fixed_point loop in Python which runs InferenceShape and InferSqueezeAxes
        // InferAll includes InferSqueezeAxes and other inference passes.
        // We might want to run it in a loop if needed, but InferAll usually handles dependencies.
        total.merge(InferAll::new().transform(ctx)?);

        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::{AttributeProto, GraphProto, NodeProto, TensorProto};

    fn make_constant_node(name: &str, output: &str, data: Vec<i64>) -> NodeProto {
        let tensor = TensorProto {
            dims: vec![data.len() as i64],
            data_type: 7,               // INT64
            name: "unused".to_string(), // Name inside attribute doesn't matter much
            int64_data: data,
            ..Default::default()
        };

        NodeProto {
            op_type: "Constant".to_string(),
            name: name.to_string(),
            output: vec![output.to_string()],
            attribute: vec![AttributeProto {
                name: "value".to_string(),
                r#type: 4, // TENSOR
                t: Some(tensor),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn test_polish_model_extract_constants() {
        let graph = GraphProto {
            node: vec![
                make_constant_node("const_0", "X", vec![1, 2, 3]),
                make_node("Identity", &["X"], &["Y"], "id_0"),
            ],
            output: vec![],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let result = PolishModel::new().transform(&mut ctx).unwrap();

        // Constant node should be removed
        assert!(!ctx.has_node("const_0"));
        // X should be an initializer
        assert!(ctx.get_initializer("X").is_some());

        let init = ctx.get_initializer("X").unwrap();
        assert_eq!(init.int64_data, vec![1, 2, 3]);
    }
}
