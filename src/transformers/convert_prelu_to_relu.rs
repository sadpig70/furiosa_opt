use crate::error::{OnnxResult, TransformError};
use crate::graph::GraphContext;
use crate::proto::extensions::make_node;
use crate::transformers::common::{OnnxTransformer, TransformResult};

/// Converts PRelu to Relu
///
/// Replaces PRelu with Relu if the slope is 0.
#[derive(Debug, Default)]
pub struct ConvertPReluToRelu;

impl ConvertPReluToRelu {
    /// Create a new ConvertPReluToRelu transformer
    pub fn new() -> Self {
        Self
    }
}

impl OnnxTransformer for ConvertPReluToRelu {
    fn name(&self) -> &'static str {
        "ConvertPReluToRelu"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        loop {
            let mut nodes_to_process = Vec::new();
            for node in ctx.nodes() {
                if node.op_type == "PRelu" {
                    nodes_to_process.push(node.name.clone());
                }
            }

            if nodes_to_process.is_empty() {
                break;
            }

            let mut any_changed = false;
            for node_name in nodes_to_process {
                if ctx.is_eliminated(&node_name) {
                    continue;
                }

                if self.try_convert(ctx, &node_name)? {
                    result.transforms_applied += 1;
                    any_changed = true;
                }
            }

            if !any_changed {
                break;
            }
        }

        Ok(result)
    }

    fn is_applicable(&self, ctx: &GraphContext) -> bool {
        ctx.nodes().any(|n| n.op_type == "PRelu")
    }
}

impl ConvertPReluToRelu {
    fn try_convert(&self, ctx: &mut GraphContext, node_name: &str) -> OnnxResult<bool> {
        // Scope to restrict borrow of ctx
        let (input_x, input_slope, output_name) = {
            let node = ctx
                .get_node(node_name)
                .ok_or_else(|| TransformError::InvalidNode(node_name.to_string()))?;

            if node.input.len() < 2 {
                return Ok(false);
            }
            (
                node.input[0].clone(),
                node.input[1].clone(),
                node.output[0].clone(),
            )
        };

        // Check if slope is initializer
        let slope_initializer = ctx.get_initializer(&input_slope).cloned();

        // Names for new nodes/tensors
        let relu_out = format!("{}_relu_out", output_name);
        let slope_fused = format!("{}_slope_fused", output_name);
        let mul_0_out = format!("{}_mul_0_out", output_name);
        let mul_1_out = format!("{}_mul_1_out", output_name);

        let relu_node_name = format!("{}_relu", node_name);
        let sub_node_name = format!("{}_sub", node_name); // Only used if dynamic
        let mul_0_node_name = format!("{}_mul_0", node_name);
        let mul_1_node_name = format!("{}_mul_1", node_name);
        let add_node_name = format!("{}_add", node_name);

        // 1. Create Relu(x)
        let relu_node = make_node("Relu", &[&input_x], &[&relu_out], &relu_node_name);
        ctx.insert_node(relu_node);

        // 2. Prepare (1 - slope)
        if let Some(slope_tensor) = slope_initializer {
            // Pattern 1: Slope is initializer
            // Calculate 1 - slope
            let slope_data = crate::tensor::convert::tensor_to_array_f32(&slope_tensor)?;
            let one_minus_slope = 1.0 - slope_data;

            // Create new initializer
            let new_init =
                crate::tensor::convert::array_to_tensor_f32(&one_minus_slope, &slope_fused);
            ctx.set_initializer(new_init);
        } else {
            // Pattern 2: Slope is dynamic
            // Create Sub(1, slope)
            // Need a "1" constant tensor with same shape as slope?
            // Or just scalar 1 if broadcasting works?
            // Python code uses `np.ones_like(slope)`.
            // We need to know the shape/rank of slope to create ones_like.
            // Or we can use `Sub` with a scalar 1 if slope allows broadcasting from scalar.
            // But ONNX `Sub` supports broadcasting.
            // Let's try to create a scalar 1 initializer.

            let ones_name = format!("{}_ones", node_name);
            let one_tensor = crate::tensor::convert::scalar_to_tensor_f32(1.0, &ones_name);
            ctx.set_initializer(one_tensor);

            let sub_node = make_node(
                "Sub",
                &[&ones_name, &input_slope],
                &[&slope_fused],
                &sub_node_name,
            );
            ctx.insert_node(sub_node);
        }

        // 3. Create Mul(Relu(x), 1-slope)
        let mul_0_node = make_node(
            "Mul",
            &[&relu_out, &slope_fused],
            &[&mul_0_out],
            &mul_0_node_name,
        );
        ctx.insert_node(mul_0_node);

        // 4. Create Mul(x, slope)
        let mul_1_node = make_node(
            "Mul",
            &[&input_x, &input_slope],
            &[&mul_1_out],
            &mul_1_node_name,
        );
        ctx.insert_node(mul_1_node);

        // 5. Create Add(...)
        let add_node = make_node(
            "Add",
            &[&mul_0_out, &mul_1_out],
            &[&output_name], // Reuse original output name
            &add_node_name,
        );
        ctx.insert_node(add_node);

        // Remove original PRelu node
        ctx.remove_node(node_name);

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::{make_node, make_tensor_value_info};
    use crate::proto::{GraphProto, TensorProto};

    #[test]
    fn test_convert_prelu_static_slope() {
        // PRelu with static slope
        let slope_data = vec![0.25];
        let slope = TensorProto {
            name: "slope".to_string(),
            dims: vec![1],
            data_type: 1, // FLOAT
            float_data: slope_data,
            ..Default::default()
        };

        let graph = GraphProto {
            node: vec![make_node("PRelu", &["X", "slope"], &["Y"], "prelu_0")],
            input: vec![make_tensor_value_info("X", 1, &[1, 3, 224, 224])],
            output: vec![make_tensor_value_info("Y", 1, &[1, 3, 224, 224])],
            initializer: vec![slope],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let result = ConvertPReluToRelu::new().transform(&mut ctx).unwrap();

        assert_eq!(result.transforms_applied, 1);
        assert!(!ctx.has_node("prelu_0"));
        assert!(ctx.nodes().any(|n| n.op_type == "Relu"));
        assert!(ctx.nodes().any(|n| n.op_type == "Mul"));
        assert!(ctx.nodes().any(|n| n.op_type == "Add"));

        // Check if 1-slope initializer exists (1 - 0.25 = 0.75)
        let init = ctx.get_initializer("Y_slope_fused").unwrap();
        let data = crate::tensor::convert::tensor_to_array_f32(init).unwrap();
        assert!((data[[0]] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_convert_prelu_dynamic_slope() {
        // PRelu with dynamic slope (no initializer for slope)
        let graph = GraphProto {
            node: vec![make_node("PRelu", &["X", "slope"], &["Y"], "prelu_0")],
            input: vec![
                make_tensor_value_info("X", 1, &[1, 3, 224, 224]),
                make_tensor_value_info("slope", 1, &[1]),
            ],
            output: vec![make_tensor_value_info("Y", 1, &[1, 3, 224, 224])],
            initializer: vec![],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let result = ConvertPReluToRelu::new().transform(&mut ctx).unwrap();

        assert_eq!(result.transforms_applied, 1);
        assert!(!ctx.has_node("prelu_0"));
        assert!(ctx.nodes().any(|n| n.op_type == "Relu"));
        assert!(ctx.nodes().any(|n| n.op_type == "Sub")); // Should have Sub for dynamic slope
        assert!(ctx.nodes().any(|n| n.op_type == "Mul"));
        assert!(ctx.nodes().any(|n| n.op_type == "Add"));
    }
}
