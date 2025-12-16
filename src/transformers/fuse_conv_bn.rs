//! Conv + BatchNormalization fusion transformer
//!
//! Fuses Conv/ConvTranspose followed by BatchNormalization into a single Conv
//! with modified weights and biases.

#![allow(missing_docs)]

use crate::error::{OnnxResult, TransformError};
use crate::graph::GraphContext;
use crate::pattern::{PatternMatcher, CONV_BN, CONV_TRANSPOSE_BN};
use crate::tensor::convert::{array_to_tensor_f32, tensor_to_array_f32};
use crate::transform::can_fuse;

use super::common::{OnnxTransformer, TransformResult};

/// Fuse Conv + BatchNormalization
///
/// BatchNorm can be folded into Conv by modifying weights and biases:
///
/// Conv: y = W * x + b
/// BN:   z = gamma * (y - mean) / sqrt(var + eps) + beta
///
/// Fused: z = W' * x + b'
/// where:
///   W' = W * gamma / sqrt(var + eps)
///   b' = (b - mean) * gamma / sqrt(var + eps) + beta
#[derive(Debug)]
pub struct FuseConvBN {
    /// Epsilon for numerical stability (default 1e-5)
    pub epsilon: f32,
}

impl Default for FuseConvBN {
    fn default() -> Self {
        Self { epsilon: 1e-5 }
    }
}

impl FuseConvBN {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_epsilon(mut self, eps: f32) -> Self {
        self.epsilon = eps;
        self
    }

    /// Fuse a single Conv+BN pair
    fn fuse_pair(
        &self,
        ctx: &mut GraphContext,
        conv_name: &str,
        bn_name: &str,
    ) -> OnnxResult<bool> {
        // Get nodes
        let conv = ctx
            .get_node(conv_name)
            .ok_or_else(|| TransformError::InvalidNode(conv_name.to_string()))?
            .clone();
        let bn = ctx
            .get_node(bn_name)
            .ok_or_else(|| TransformError::InvalidNode(bn_name.to_string()))?
            .clone();

        // Check fusibility
        if !can_fuse(ctx, conv_name, bn_name) {
            return Ok(false);
        }

        // Get Conv weight tensor name
        let weight_name = conv.input.get(1).ok_or_else(|| {
            TransformError::MissingField(format!("{}.input[1] (weight)", conv_name))
        })?;

        // Get optional Conv bias
        let conv_bias_name = conv.input.get(2).cloned();

        // Get BN parameters: scale, B, mean, var
        if bn.input.len() < 5 {
            return Err(TransformError::MissingField(format!(
                "{} requires 5 inputs, got {}",
                bn_name,
                bn.input.len()
            )));
        }
        let scale_name = &bn.input[1];
        let bn_bias_name = &bn.input[2];
        let mean_name = &bn.input[3];
        let var_name = &bn.input[4];

        // Load tensors
        let weight = ctx
            .get_initializer(weight_name)
            .ok_or_else(|| TransformError::MissingField(format!("initializer: {}", weight_name)))?
            .clone();

        let scale = ctx
            .get_initializer(scale_name)
            .ok_or_else(|| TransformError::MissingField(format!("initializer: {}", scale_name)))?
            .clone();

        let bn_bias = ctx
            .get_initializer(bn_bias_name)
            .ok_or_else(|| TransformError::MissingField(format!("initializer: {}", bn_bias_name)))?
            .clone();

        let mean = ctx
            .get_initializer(mean_name)
            .ok_or_else(|| TransformError::MissingField(format!("initializer: {}", mean_name)))?
            .clone();

        let var = ctx
            .get_initializer(var_name)
            .ok_or_else(|| TransformError::MissingField(format!("initializer: {}", var_name)))?
            .clone();

        // Get epsilon from BN attributes (default 1e-5)
        let epsilon = bn
            .attribute
            .iter()
            .find(|a| a.name == "epsilon")
            .map(|a| a.f)
            .unwrap_or(self.epsilon);

        // Convert to arrays
        let weight_arr = tensor_to_array_f32(&weight)?;
        let scale_arr = tensor_to_array_f32(&scale)?;
        let bn_bias_arr = tensor_to_array_f32(&bn_bias)?;
        let mean_arr = tensor_to_array_f32(&mean)?;
        let var_arr = tensor_to_array_f32(&var)?;

        // Load or create conv bias
        let conv_bias_arr = if let Some(ref bias_name) = conv_bias_name {
            if let Some(bias_tensor) = ctx.get_initializer(bias_name) {
                tensor_to_array_f32(bias_tensor)?
            } else {
                // Create zero bias with same shape as scale
                ndarray::ArrayD::zeros(scale_arr.shape().to_vec())
            }
        } else {
            ndarray::ArrayD::zeros(scale_arr.shape().to_vec())
        };

        // Compute fused parameters
        // std = sqrt(var + eps)
        let std_arr = var_arr.mapv(|v| (v + epsilon).sqrt());

        // scale_factor = gamma / std
        let scale_factor = &scale_arr / &std_arr;

        // Fused weight: W' = W * scale_factor (broadcasted over output channels)
        let weight_shape = weight_arr.shape().to_vec();
        let out_channels = weight_shape[0];

        let mut fused_weight = weight_arr.clone();
        for oc in 0..out_channels {
            let sf = scale_factor[oc];
            fused_weight
                .slice_mut(ndarray::s![oc, .., .., ..])
                .mapv_inplace(|w| w * sf);
        }

        // Fused bias: b' = (b - mean) * gamma / std + beta
        let fused_bias = (&conv_bias_arr - &mean_arr) * &scale_factor + &bn_bias_arr;

        // Create new tensors
        let new_weight = array_to_tensor_f32(&fused_weight, weight_name);

        let new_bias_name = format!("{}_fused_bias", conv_name);
        let new_bias = array_to_tensor_f32(&fused_bias, &new_bias_name);

        // Update initializers
        ctx.set_initializer(new_weight);
        ctx.set_initializer(new_bias);

        // Update Conv node inputs to include fused bias
        if let Some(entry) = ctx.get_entry_mut(conv_name) {
            // Ensure Conv has 3 inputs (X, W, B)
            while entry.node.input.len() < 3 {
                entry.node.input.push(String::new());
            }
            entry.node.input[2] = new_bias_name;

            // Update Conv output to BN output
            if let Some(bn_output) = bn.output.first() {
                entry.node.output[0] = bn_output.clone();
                // Update producer map
                ctx.update_producer(bn_output, conv_name);
            }
        }

        // Eliminate BN node
        ctx.mark_eliminated(bn_name);

        // Remove BN-specific initializers if they're no longer used
        // (Scale, B, mean, var are typically only used by this BN)

        Ok(true)
    }
}

impl OnnxTransformer for FuseConvBN {
    fn name(&self) -> &'static str {
        "FuseConvBN"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        // Find Conv+BN patterns
        let patterns = [CONV_BN, CONV_TRANSPOSE_BN];

        for pattern in patterns {
            loop {
                // Find matches (need to re-search after each fusion)
                let matches: Vec<(String, String)> = {
                    let matcher = PatternMatcher::new(ctx);
                    matcher
                        .find_all_matches(pattern)
                        .into_iter()
                        .filter(|m| {
                            !ctx.is_eliminated(&m.nodes[0].name)
                                && !ctx.is_eliminated(&m.nodes[1].name)
                        })
                        .map(|m| (m.nodes[1].name.clone(), m.nodes[0].name.clone()))
                        .collect()
                };

                if matches.is_empty() {
                    break;
                }

                result.patterns_matched += matches.len();

                let mut any_fused = false;
                for (conv_name, bn_name) in matches {
                    if ctx.is_eliminated(&conv_name) || ctx.is_eliminated(&bn_name) {
                        continue;
                    }

                    match self.fuse_pair(ctx, &conv_name, &bn_name) {
                        Ok(true) => {
                            result.record(&conv_name);
                            result.record_elimination(&bn_name);
                            any_fused = true;
                        }
                        Ok(false) => {}
                        Err(_) => {
                            // Skip this pair, continue with others
                        }
                    }
                }

                if !any_fused {
                    break;
                }
            }
        }

        Ok(result)
    }

    fn is_applicable(&self, ctx: &GraphContext) -> bool {
        // Check if there are any BatchNormalization nodes
        ctx.nodes().any(|n| n.op_type == "BatchNormalization")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::{GraphProto, ValueInfoProto};
    use crate::tensor::convert::vec_to_tensor_f32;

    fn make_conv_bn_graph() -> GraphProto {
        // Create simple Conv + BN graph
        // Conv: 1 input channel, 2 output channels, 3x3 kernel
        let weight_data: Vec<f32> = (0..18).map(|i| i as f32 * 0.1).collect();
        let mut weight = vec_to_tensor_f32(&weight_data, "W");
        weight.dims = vec![2, 1, 3, 3]; // [out_ch, in_ch, kH, kW]

        let scale = vec_to_tensor_f32(&[1.0, 1.0], "scale");
        let bn_bias = vec_to_tensor_f32(&[0.0, 0.0], "B");
        let mean = vec_to_tensor_f32(&[0.0, 0.0], "mean");
        let var = vec_to_tensor_f32(&[1.0, 1.0], "var");

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
            initializer: vec![weight, scale, bn_bias, mean, var],
            ..Default::default()
        }
    }

    #[test]
    fn test_fuse_conv_bn_applicable() {
        let graph = make_conv_bn_graph();
        let ctx = GraphContext::new(&graph);

        let transformer = FuseConvBN::new();
        assert!(transformer.is_applicable(&ctx));
    }

    #[test]
    fn test_fuse_conv_bn() {
        let graph = make_conv_bn_graph();
        let mut ctx = GraphContext::new(&graph);

        let transformer = FuseConvBN::new();
        let result = transformer.transform(&mut ctx).unwrap();

        // Should have fused 1 Conv+BN pair
        assert_eq!(result.transforms_applied, 1);
        assert_eq!(result.nodes_eliminated, 1);

        // BN should be eliminated
        assert!(ctx.is_eliminated("bn_0"));

        // Conv should now output bn_out
        let conv = ctx.get_node("conv_0").unwrap();
        assert_eq!(conv.output[0], "bn_out");

        // Conv should have fused bias
        assert_eq!(conv.input.len(), 3);
    }

    #[test]
    fn test_fuse_conv_bn_no_bn() {
        let graph = GraphProto {
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
            ..Default::default()
        };

        let ctx = GraphContext::new(&graph);
        let transformer = FuseConvBN::new();

        // Should not be applicable (no BN)
        assert!(!transformer.is_applicable(&ctx));
    }
}
