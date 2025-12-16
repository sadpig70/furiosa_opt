//! BatchNormalization decomposition transformer
//!
//! Decomposes BatchNormalization into Mul and Add operations.
//! This is typically applied when BN cannot be fused into a preceding Conv.

use crate::error::{OnnxResult, TransformError};
use crate::graph::GraphContext;

use crate::proto::extensions::make_node;
use crate::tensor::convert::{array_to_tensor_f32, tensor_to_array_f32};

use super::common::{OnnxTransformer, TransformResult};

/// Decompose BatchNormalization
///
/// Transforms:
///   y = BN(x, scale, B, mean, var)
///
/// Into:
///   y = (x * multiplier) + shifter
///   where multiplier = scale / sqrt(var + eps)
///         shifter = -mean * multiplier + B
#[derive(Debug, Default)]
pub struct DecomposeBN;

impl DecomposeBN {
    /// Create a new DecomposeBN transformer
    pub fn new() -> Self {
        Self
    }

    fn decompose(&self, ctx: &mut GraphContext, bn_name: &str) -> OnnxResult<bool> {
        let bn = ctx
            .get_node(bn_name)
            .ok_or_else(|| TransformError::InvalidNode(bn_name.to_string()))?
            .clone();

        // Check predecessor
        // If predecessor is Conv or ConvTranspose, we skip because FuseConvBN should handle it.
        // However, FuseConvBN might have failed or not run yet.
        // Ideally, we should check if FuseConvBN *could* apply.
        // For now, following the Python logic: "if prev.op_type != Conv"

        if let Some(producer) = ctx.get_producer(&bn.input[0]) {
            if producer.op_type == "Conv" || producer.op_type == "ConvTranspose" {
                return Ok(false);
            }
        }

        // Get inputs
        if bn.input.len() < 5 {
            return Ok(false);
        }
        let input_name = &bn.input[0];
        let scale_name = &bn.input[1];
        let b_name = &bn.input[2];
        let mean_name = &bn.input[3];
        let var_name = &bn.input[4];

        // Get initializers
        let scale = ctx
            .get_initializer(scale_name)
            .ok_or_else(|| TransformError::ValueInfoNotFound(scale_name.to_string()))?;
        let b = ctx
            .get_initializer(b_name)
            .ok_or_else(|| TransformError::ValueInfoNotFound(b_name.to_string()))?;
        let mean = ctx
            .get_initializer(mean_name)
            .ok_or_else(|| TransformError::ValueInfoNotFound(mean_name.to_string()))?;
        let var = ctx
            .get_initializer(var_name)
            .ok_or_else(|| TransformError::ValueInfoNotFound(var_name.to_string()))?;

        // Get epsilon
        let epsilon = bn
            .attribute
            .iter()
            .find(|a| a.name == "epsilon")
            .map(|a| a.f)
            .unwrap_or(1e-5);

        // Convert to arrays
        let scale_arr = tensor_to_array_f32(scale)?;
        let b_arr = tensor_to_array_f32(b)?;
        let mean_arr = tensor_to_array_f32(mean)?;
        let var_arr = tensor_to_array_f32(var)?;

        // Calculate multiplier and shifter
        // std = sqrt(var + eps)
        let std_arr = var_arr.mapv(|v| (v + epsilon).sqrt());

        // multiplier = scale / std
        let multiplier_arr = &scale_arr / &std_arr;

        // shifter = -mean * multiplier + B
        let shifter_arr = &b_arr - (&mean_arr * &multiplier_arr);

        // Create new initializers
        // We need to reshape them to match input rank if needed, but usually BN params are 1D [C].
        // ONNX Mul/Add supports broadcasting, so 1D [C] should work if input is [N, C, H, W].
        // However, for [N, C, H, W], we need [1, C, 1, 1] for correct broadcasting in some backends,
        // or rely on numpy-style broadcasting where it aligns to the right?
        // ONNX broadcasting is unidirectional (right-aligned) unless axis is specified?
        // Actually, for [N, C, H, W] and [C], standard broadcasting might fail if not aligned.
        // Python implementation reshapes:
        // shape = [dim if i == 1 else 1 for (i, dim) in enumerate(self.get_value_info_shape(batch_norm.input[0]))]

        // We need input shape to reshape correctly.
        let input_shape = if let Some(vi) = ctx.get_value_info(input_name) {
            vi.get_shape().unwrap_or_default()
        } else {
            vec![]
        };

        let mut target_shape = vec![1; input_shape.len()];
        if input_shape.len() >= 2 {
            target_shape[1] = multiplier_arr.len() as i64;
        } else {
            // Fallback: keep as 1D
            target_shape = vec![multiplier_arr.len() as i64];
        }

        let target_dims: Vec<usize> = target_shape.iter().map(|&x| x as usize).collect();

        let multiplier_reshaped = multiplier_arr.into_shape(target_dims.clone()).unwrap();
        let shifter_reshaped = shifter_arr.into_shape(target_dims).unwrap();

        let multiplier_name = format!("{}_multiplier", bn_name);
        let shifter_name = format!("{}_shifter", bn_name);

        let multiplier_tensor = array_to_tensor_f32(&multiplier_reshaped, &multiplier_name);
        let shifter_tensor = array_to_tensor_f32(&shifter_reshaped, &shifter_name);

        ctx.set_initializer(multiplier_tensor);
        ctx.set_initializer(shifter_tensor);

        // Create Mul node
        let mul_output = format!("{}_mul", bn_name);
        let mul_node = make_node(
            "Mul",
            &[input_name, &multiplier_name],
            &[&mul_output],
            &format!("{}_mul", bn_name),
        );

        // Create Add node
        let add_node = make_node(
            "Add",
            &[&mul_output, &shifter_name],
            &[&bn.output[0]], // Connect to original output
            &format!("{}_add", bn_name),
        );

        // Insert nodes
        // We can't easily insert "in place" with current API, but we can append and rewire.
        // Since we are replacing BN, we can just remove BN and add these two.
        // But we need to ensure topological order if we just append?
        // GraphContext doesn't enforce order on `add_node` but `insert_node` does?
        // `insert_node` inserts into `graph.node`.

        // We will remove BN and append Mul and Add.
        // To preserve order somewhat, we could find BN index, but `GraphContext` abstracts that.
        // Appending is fine as long as we sort later or if the execution engine handles it.
        // But `GraphContext` has `insert_node` which appends.

        let add_node_name = add_node.name.clone();
        ctx.remove_node(bn_name);
        ctx.insert_node(mul_node);
        ctx.insert_node(add_node);

        // Update producer for output
        ctx.update_producer(&bn.output[0], &add_node_name);

        Ok(true)
    }
}

impl OnnxTransformer for DecomposeBN {
    fn name(&self) -> &'static str {
        "DecomposeBN"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        let bn_nodes: Vec<String> = ctx
            .nodes()
            .filter(|n| n.op_type == "BatchNormalization")
            .map(|n| n.name.clone())
            .collect();

        for bn_name in bn_nodes {
            if ctx.is_eliminated(&bn_name) {
                continue;
            }

            match self.decompose(ctx, &bn_name) {
                Ok(true) => {
                    result.record(&bn_name);
                }
                Ok(false) => {}
                Err(_) => {}
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::extensions::make_tensor_value_info;
    use crate::proto::{GraphProto, ValueInfoProto};
    use crate::tensor::convert::vec_to_tensor_f32;

    #[test]
    fn test_decompose_bn() {
        // Relu -> BN
        // Scale=2, B=1, Mean=0, Var=1, Eps=0 -> Std=1
        // Multiplier = 2/1 = 2
        // Shifter = -0*2 + 1 = 1

        let scale = vec_to_tensor_f32(&[2.0], "scale");
        let b = vec_to_tensor_f32(&[1.0], "B");
        let mean = vec_to_tensor_f32(&[0.0], "mean");
        let var = vec_to_tensor_f32(&[1.0], "var");

        let graph = GraphProto {
            node: vec![
                make_node("Relu", &["X"], &["relu_out"], "relu_0"),
                make_node(
                    "BatchNormalization",
                    &["relu_out", "scale", "B", "mean", "var"],
                    &["Y"],
                    "bn_0",
                ),
            ],
            input: vec![make_tensor_value_info("X", 1, &[1, 1, 1, 1])],
            output: vec![ValueInfoProto {
                name: "Y".to_string(),
                ..Default::default()
            }],
            initializer: vec![scale, b, mean, var],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let transformer = DecomposeBN::new();
        let result = transformer.transform(&mut ctx).unwrap();

        assert_eq!(result.transforms_applied, 1);

        // BN should be gone
        assert!(ctx.get_node("bn_0").is_none());

        // Mul and Add should exist
        assert!(ctx.nodes().any(|n| n.op_type == "Mul"));
        assert!(ctx.nodes().any(|n| n.op_type == "Add"));
    }
}
