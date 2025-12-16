//! Conv + Mul + Add fusion transformer
//!
//! Fuses Conv followed by Mul and Add into a single Conv.
//! This pattern often appears when BatchNormalization is decomposed or
//! when explicit scaling/shifting is used.

use crate::error::{OnnxResult, TransformError};
use crate::graph::GraphContext;
use crate::pattern::{PatternMatcher, CONV_MUL_ADD};
use crate::tensor::convert::{array_to_tensor_f32, tensor_to_array_f32};
use crate::transform::can_fuse;

use super::common::{get_constant_tensor, OnnxTransformer, TransformResult};

/// Fuse Conv + Mul + Add
///
/// Transforms:
///   y = Conv(x)
///   z = y * scale + shift
///
/// Into:
///   z = Conv'(x)
///   where weights' = weights * scale
///         bias' = bias * scale + shift
#[derive(Debug, Default)]
pub struct FuseConvMulAdd;

impl FuseConvMulAdd {
    /// Create a new FuseConvMulAdd transformer
    pub fn new() -> Self {
        Self
    }

    fn fuse_triplet(
        &self,
        ctx: &mut GraphContext,
        conv_name: &str,
        mul_name: &str,
        add_name: &str,
    ) -> OnnxResult<bool> {
        // Get nodes
        let conv = ctx
            .get_node(conv_name)
            .ok_or_else(|| TransformError::InvalidNode(conv_name.to_string()))?
            .clone();
        let mul = ctx
            .get_node(mul_name)
            .ok_or_else(|| TransformError::InvalidNode(mul_name.to_string()))?
            .clone();
        let add = ctx
            .get_node(add_name)
            .ok_or_else(|| TransformError::InvalidNode(add_name.to_string()))?
            .clone();

        // Check fusibility
        if !can_fuse(ctx, conv_name, mul_name) || !can_fuse(ctx, mul_name, add_name) {
            return Ok(false);
        }

        // Identify scale and shift inputs
        // Mul inputs: [conv_output, scale] or [scale, conv_output]
        let scale_input = if mul.input[0] == conv.output[0] {
            &mul.input[1]
        } else {
            &mul.input[0]
        };

        // Add inputs: [mul_output, shift] or [shift, mul_output]
        let shift_input = if add.input[0] == mul.output[0] {
            &add.input[1]
        } else {
            &add.input[0]
        };

        // Get constant tensors
        let scale_tensor = get_constant_tensor(ctx, scale_input)
            .ok_or_else(|| TransformError::ValueInfoNotFound(scale_input.to_string()))?;
        let shift_tensor = get_constant_tensor(ctx, shift_input)
            .ok_or_else(|| TransformError::ValueInfoNotFound(shift_input.to_string()))?;

        // Get Conv weight
        let weight_name = &conv.input[1];
        let weight_tensor = ctx
            .get_initializer(weight_name)
            .ok_or_else(|| TransformError::MissingField(format!("initializer: {}", weight_name)))?;

        // Get Conv bias (optional)
        let conv_bias_name = conv.input.get(2);

        // Convert to arrays
        let scale_arr = tensor_to_array_f32(scale_tensor)?;
        let shift_arr = tensor_to_array_f32(shift_tensor)?;
        let weight_arr = tensor_to_array_f32(weight_tensor)?;

        // Check shapes
        // Scale and shift should be 1D or broadcastable to [OutCh, 1, 1, 1]
        // For simplicity, we assume they are 1D [OutCh] or scalar, or [OutCh, 1, 1] etc.
        // We will flatten them to 1D for channel-wise operation if possible, or rely on broadcasting.
        // However, Conv weights are [OutCh, InCh, KH, KW].
        // We need to broadcast scale to [OutCh, 1, 1, 1].

        let out_channels = weight_arr.shape()[0];

        // Flatten scale and shift if they match out_channels
        let scale_flat = if scale_arr.len() == out_channels {
            scale_arr.into_shape((out_channels,)).unwrap()
        } else if scale_arr.len() == 1 {
            // Scalar broadcast
            let val = scale_arr.first().unwrap();
            ndarray::Array1::from_elem(out_channels, *val)
        } else {
            // Complex broadcasting not fully supported in this simplified implementation
            // Fallback or error? Python implementation flattens.
            // Let's try to flatten and check length.
            let flat = scale_arr.iter().cloned().collect::<Vec<f32>>();
            if flat.len() == out_channels {
                ndarray::Array1::from_vec(flat)
            } else {
                return Ok(false); // Shape mismatch
            }
        };

        let shift_flat = if shift_arr.len() == out_channels {
            shift_arr.into_shape((out_channels,)).unwrap()
        } else if shift_arr.len() == 1 {
            let val = shift_arr.first().unwrap();
            ndarray::Array1::from_elem(out_channels, *val)
        } else {
            let flat = shift_arr.iter().cloned().collect::<Vec<f32>>();
            if flat.len() == out_channels {
                ndarray::Array1::from_vec(flat)
            } else {
                return Ok(false);
            }
        };

        // Compute new weights
        let mut new_weight_arr = weight_arr.clone();
        for oc in 0..out_channels {
            let s = scale_flat[oc];
            new_weight_arr
                .slice_mut(ndarray::s![oc, .., .., ..])
                .mapv_inplace(|x| x * s);
        }

        // Compute new bias
        let conv_bias_arr = if let Some(name) = conv_bias_name {
            if let Some(t) = ctx.get_initializer(name) {
                tensor_to_array_f32(t)?
            } else {
                ndarray::ArrayD::zeros(vec![out_channels])
            }
        } else {
            ndarray::ArrayD::zeros(vec![out_channels])
        };

        // Flatten conv bias if needed (it should be 1D usually)
        let conv_bias_flat = if conv_bias_arr.len() == out_channels {
            conv_bias_arr.into_shape((out_channels,)).unwrap()
        } else {
            // Should not happen for valid Conv
            return Ok(false);
        };

        let new_bias_arr = (conv_bias_flat * scale_flat) + shift_flat;

        // Create new tensors
        let new_weight = array_to_tensor_f32(&new_weight_arr, weight_name);
        let new_bias_name = format!("{}_fused_bias", conv_name);
        let new_bias = array_to_tensor_f32(&new_bias_arr.into_dyn(), &new_bias_name);

        // Update initializers
        ctx.set_initializer(new_weight);
        ctx.set_initializer(new_bias);

        // Update Conv node
        if let Some(entry) = ctx.get_entry_mut(conv_name) {
            // Ensure 3 inputs
            while entry.node.input.len() < 3 {
                entry.node.input.push(String::new());
            }
            entry.node.input[2] = new_bias_name;

            // Update output to Add's output
            entry.node.output[0] = add.output[0].clone();
            ctx.update_producer(&add.output[0], conv_name);
        }

        // Eliminate Mul and Add
        ctx.mark_eliminated(mul_name);
        ctx.mark_eliminated(add_name);

        Ok(true)
    }
}

impl OnnxTransformer for FuseConvMulAdd {
    fn name(&self) -> &'static str {
        "FuseConvMulAdd"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        loop {
            let matches: Vec<(String, String, String)> = {
                let matcher = PatternMatcher::new(ctx);
                matcher
                    .find_all_matches(CONV_MUL_ADD)
                    .into_iter()
                    .filter(|m| {
                        !ctx.is_eliminated(&m.nodes[0].name) && // Add
                        !ctx.is_eliminated(&m.nodes[1].name) && // Mul
                        !ctx.is_eliminated(&m.nodes[2].name) // Conv
                    })
                    .map(|m| {
                        (
                            m.nodes[2].name.clone(),
                            m.nodes[1].name.clone(),
                            m.nodes[0].name.clone(),
                        )
                    })
                    .collect()
            };

            if matches.is_empty() {
                break;
            }

            result.patterns_matched += matches.len();
            let mut any_fused = false;

            for (conv, mul, add) in matches {
                if ctx.is_eliminated(&conv) || ctx.is_eliminated(&mul) || ctx.is_eliminated(&add) {
                    continue;
                }

                match self.fuse_triplet(ctx, &conv, &mul, &add) {
                    Ok(true) => {
                        result.record(&conv);
                        result.record_elimination(&mul);
                        result.record_elimination(&add);
                        any_fused = true;
                    }
                    Ok(false) => {}
                    Err(_) => {}
                }
            }

            if !any_fused {
                break;
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::{GraphProto, ValueInfoProto};
    use crate::tensor::convert::vec_to_tensor_f32;

    #[test]
    fn test_fuse_conv_mul_add() {
        // Conv (1x1, 1 in, 1 out) -> Mul (2.0) -> Add (1.0)
        // W = 1.0, B = 0.0
        // Expected: W' = 2.0, B' = 1.0

        let weight = vec_to_tensor_f32(&[1.0], "W");
        let mut weight = weight;
        weight.dims = vec![1, 1, 1, 1];

        let scale = vec_to_tensor_f32(&[2.0], "scale");
        let shift = vec_to_tensor_f32(&[1.0], "shift");

        let graph = GraphProto {
            node: vec![
                make_node("Conv", &["X", "W"], &["conv_out"], "conv_0"),
                make_node("Mul", &["conv_out", "scale"], &["mul_out"], "mul_0"),
                make_node("Add", &["mul_out", "shift"], &["Y"], "add_0"),
            ],
            input: vec![ValueInfoProto {
                name: "X".to_string(),
                ..Default::default()
            }],
            output: vec![ValueInfoProto {
                name: "Y".to_string(),
                ..Default::default()
            }],
            initializer: vec![weight, scale, shift],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let transformer = FuseConvMulAdd::new();
        let result = transformer.transform(&mut ctx).unwrap();

        assert_eq!(result.transforms_applied, 1);
        assert_eq!(result.nodes_eliminated, 2); // Mul, Add

        let conv = ctx.get_node("conv_0").unwrap();
        assert_eq!(conv.output[0], "Y");

        // Check new bias
        let bias_name = &conv.input[2];
        let bias = ctx.get_initializer(bias_name).unwrap();
        let bias_arr = tensor_to_array_f32(bias).unwrap();
        assert_eq!(bias_arr.first().unwrap(), &1.0);

        // Check new weight
        let weight = ctx.get_initializer("W").unwrap();
        let weight_arr = tensor_to_array_f32(weight).unwrap();
        assert_eq!(weight_arr.first().unwrap(), &2.0);
    }
}
