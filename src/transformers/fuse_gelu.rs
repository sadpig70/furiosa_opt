//! GELU activation fusion
//!
//! Fuses GELU pattern into single node for Transformer models.
//!
//! GELU can be represented in multiple ways in ONNX:
//! 1. Exact: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
//! 2. Approximation: x * sigmoid(1.702 * x)
//! 3. Error function: 0.5 * x * (1 + erf(x / sqrt(2)))

#![allow(missing_docs)]

use crate::error::OnnxResult;
use crate::graph::GraphContext;

use super::common::{OnnxTransformer, TransformResult};

/// Fuse GELU pattern (Erf-based) into single GELU node
///
/// Pattern: x -> Div(sqrt(2)) -> Erf -> Add(1) -> Mul(0.5) -> Mul(x) -> GELU
#[derive(Debug, Default)]
pub struct FuseGeluErf;

impl FuseGeluErf {
    pub fn new() -> Self {
        Self
    }

    /// Try to match and fuse GELU pattern
    fn try_fuse(&self, ctx: &mut GraphContext, div_name: &str) -> OnnxResult<bool> {
        // Pattern:
        // x -> Div -> Erf -> Add -> Mul -> Mul
        //  \______________________________/

        let div = match ctx.get_node(div_name) {
            Some(n) => n.clone(),
            None => return Ok(false),
        };

        if div.op_type != "Div" || div.input.len() < 2 {
            return Ok(false);
        }

        // Check if divisor is sqrt(2)
        let divisor_name = &div.input[1];
        if let Some(divisor_tensor) = super::common::get_constant_tensor(ctx, divisor_name) {
            if let Some(value) = get_scalar_float(divisor_tensor) {
                if (value - std::f32::consts::SQRT_2).abs() > 0.001 {
                    // println!("Divisor mismatch: {} != 1.414", value);
                    return Ok(false);
                }
            } else {
                // println!("Divisor not a scalar float");
                return Ok(false);
            }
        } else {
            // println!("Divisor constant not found: {}", divisor_name);
            return Ok(false);
        }

        // Find Erf node
        let div_output = &div.output[0];
        let consumers = match ctx.get_consumer_names(div_output) {
            Some(c) => c,
            None => return Ok(false),
        };
        let erf_name = match consumers.first() {
            Some(name) => name.clone(),
            None => return Ok(false),
        };

        let erf = match ctx.get_node(&erf_name) {
            Some(n) if n.op_type == "Erf" => n.clone(),
            _ => return Ok(false),
        };

        // Find Add(1) node
        let erf_output = &erf.output[0];
        let consumers = match ctx.get_consumer_names(erf_output) {
            Some(c) => c,
            None => return Ok(false),
        };
        let add_name = match consumers.first() {
            Some(name) => name.clone(),
            None => return Ok(false),
        };

        let add = match ctx.get_node(&add_name) {
            Some(n) if n.op_type == "Add" => n.clone(),
            _ => return Ok(false),
        };

        // Check if adding 1.0
        let add_const_name = if add.input[0] == *erf_output {
            &add.input[1]
        } else if add.input[1] == *erf_output {
            &add.input[0]
        } else {
            return Ok(false);
        };

        if let Some(add_tensor) = super::common::get_constant_tensor(ctx, add_const_name) {
            if let Some(value) = get_scalar_float(add_tensor) {
                if (value - 1.0).abs() > 0.001 {
                    // println!("Add constant mismatch: {} != 1.0", value);
                    return Ok(false);
                }
            } else {
                // println!("Add constant not a scalar float");
                return Ok(false);
            }
        } else {
            // println!("Add constant not found: {}", add_const_name);
            return Ok(false);
        }

        // Find first Mul node
        let add_output = &add.output[0];
        let consumers = match ctx.get_consumer_names(add_output) {
            Some(c) => c,
            None => return Ok(false),
        };
        let mul1_name = match consumers.first() {
            Some(name) => name.clone(),
            None => return Ok(false),
        };

        let mul1 = match ctx.get_node(&mul1_name) {
            Some(n) if n.op_type == "Mul" => n.clone(),
            _ => return Ok(false),
        };

        // Check first Mul inputs to decide pattern
        // Pattern 1: ... -> Mul(0.5) -> Mul(x)
        // Pattern 2: ... -> Mul(x) -> Mul(0.5)

        let original_input = &div.input[0];
        let mut is_pattern_2 = false;

        // Check if Mul1 is multiplying by 0.5
        let mul1_is_half = if let Some(other_input) = get_other_input(&mul1, add_output) {
            if let Some(tensor) = super::common::get_constant_tensor(ctx, other_input) {
                if let Some(value) = get_scalar_float(tensor) {
                    (value - 0.5).abs() <= 0.001
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        };

        // Check if Mul1 is multiplying by x
        let mul1_is_x = if let Some(other_input) = get_other_input(&mul1, add_output) {
            other_input == original_input
        } else {
            false
        };

        if !mul1_is_half && !mul1_is_x {
            return Ok(false);
        }

        if mul1_is_x {
            is_pattern_2 = true;
        }

        // Find second Mul node
        let mul1_output = &mul1.output[0];
        let consumers = match ctx.get_consumer_names(mul1_output) {
            Some(c) => c,
            None => return Ok(false),
        };
        let mul2_name = match consumers.first() {
            Some(name) => name.clone(),
            None => return Ok(false),
        };

        let mul2 = match ctx.get_node(&mul2_name) {
            Some(n) if n.op_type == "Mul" => n.clone(),
            _ => return Ok(false),
        };

        if is_pattern_2 {
            // Pattern 2: Mul1 was Mul(x), so Mul2 must be Mul(0.5)
            let mul2_is_half = if let Some(other_input) = get_other_input(&mul2, mul1_output) {
                if let Some(tensor) = super::common::get_constant_tensor(ctx, other_input) {
                    if let Some(value) = get_scalar_float(tensor) {
                        (value - 0.5).abs() <= 0.001
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            };

            if !mul2_is_half {
                return Ok(false);
            }
        } else {
            // Pattern 1: Mul1 was Mul(0.5), so Mul2 must be Mul(x)
            let mul2_is_x = if let Some(other_input) = get_other_input(&mul2, mul1_output) {
                other_input == original_input
            } else {
                false
            };

            if !mul2_is_x {
                return Ok(false);
            }
        }

        // All checks passed - fuse into GELU

        // Mark intermediate nodes for elimination
        ctx.mark_eliminated(&erf_name);
        ctx.mark_eliminated(&add_name);
        ctx.mark_eliminated(&mul1_name);
        ctx.mark_eliminated(&mul2_name);

        // Update div node to become GELU
        if let Some(entry) = ctx.get_entry_mut(div_name) {
            entry.node.op_type = "Gelu".to_string();
            entry.node.domain = "com.microsoft".to_string();
            entry.node.input = vec![original_input.clone()];
            entry.node.output = mul2.output.clone();

            // Update consumers
            if let Some(output) = mul2.output.first() {
                ctx.update_producer(output, div_name);
            }
        }

        Ok(true)
    }
}

fn get_other_input<'a>(node: &'a crate::proto::NodeProto, input: &str) -> Option<&'a String> {
    if node.input.len() != 2 {
        return None;
    }
    if node.input[0] == input {
        Some(&node.input[1])
    } else if node.input[1] == input {
        Some(&node.input[0])
    } else {
        None
    }
}

impl OnnxTransformer for FuseGeluErf {
    fn name(&self) -> &'static str {
        "FuseGeluErf"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        let mut iteration = 0;
        loop {
            iteration += 1;
            if iteration > 100 {
                println!("DEBUG: Breaking infinite loop in FuseGeluErf");
                break;
            }

            // Find Div nodes that could be start of GELU pattern
            let patterns: Vec<String> = ctx
                .nodes()
                .filter(|n| n.op_type == "Div" && !ctx.is_eliminated(&n.name))
                .map(|n| n.name.clone())
                .collect();

            if patterns.is_empty() {
                break;
            }

            let mut any_fused = false;
            for div_name in patterns {
                if ctx.is_eliminated(&div_name) {
                    continue;
                }

                match self.try_fuse(ctx, &div_name) {
                    Ok(true) => {
                        println!("DEBUG: Successfully fused Gelu at {}", div_name);
                        result.record(&div_name);
                        result.patterns_matched += 1;
                        result.transforms_applied += 1;
                        result.nodes_eliminated += 4; // Erf, Add, Mul, Mul
                        any_fused = true;
                    }
                    Ok(false) => {}
                    Err(e) => {
                        println!("DEBUG: Error fusing Gelu at {}: {:?}", div_name, e);
                    }
                }
            }

            if !any_fused {
                break;
            }
        }

        Ok(result)
    }

    fn is_applicable(&self, ctx: &GraphContext) -> bool {
        ctx.nodes().any(|n| n.op_type == "Erf")
    }
}

/// Helper to get scalar float value from tensor
fn get_scalar_float(tensor: &crate::proto::TensorProto) -> Option<f32> {
    use crate::proto::onnx::tensor_proto::DataType;

    let dtype = DataType::try_from(tensor.data_type).ok()?;

    match dtype {
        DataType::Float => {
            if !tensor.float_data.is_empty() {
                Some(tensor.float_data[0])
            } else if !tensor.raw_data.is_empty() && tensor.raw_data.len() >= 4 {
                Some(f32::from_le_bytes([
                    tensor.raw_data[0],
                    tensor.raw_data[1],
                    tensor.raw_data[2],
                    tensor.raw_data[3],
                ]))
            } else {
                None
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::{GraphProto, TensorProto, ValueInfoProto};

    #[test]
    fn test_gelu_pattern_detection() {
        // Create GELU pattern: x -> Div(sqrt(2)) -> Erf -> Add(1) -> Mul(0.5) -> Mul(x)
        let sqrt2 = TensorProto {
            name: "sqrt2".to_string(),
            dims: vec![],
            data_type: 1, // FLOAT
            float_data: vec![std::f32::consts::SQRT_2],
            ..Default::default()
        };

        let one = TensorProto {
            name: "one".to_string(),
            dims: vec![],
            data_type: 1,
            float_data: vec![1.0],
            ..Default::default()
        };

        let half = TensorProto {
            name: "half".to_string(),
            dims: vec![],
            data_type: 1,
            float_data: vec![0.5],
            ..Default::default()
        };

        let graph = GraphProto {
            node: vec![
                make_node("Div", &["x", "sqrt2"], &["div_out"], "div"),
                make_node("Erf", &["div_out"], &["erf_out"], "erf"),
                make_node("Add", &["erf_out", "one"], &["add_out"], "add"),
                make_node("Mul", &["add_out", "half"], &["mul1_out"], "mul1"),
                make_node("Mul", &["mul1_out", "x"], &["y"], "mul2"),
            ],
            input: vec![ValueInfoProto {
                name: "x".to_string(),
                ..Default::default()
            }],
            output: vec![ValueInfoProto {
                name: "y".to_string(),
                ..Default::default()
            }],
            initializer: vec![sqrt2, one, half],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let result = FuseGeluErf::new().transform(&mut ctx).unwrap();

        assert_eq!(result.patterns_matched, 1);
        assert_eq!(result.transforms_applied, 1);
        assert_eq!(result.nodes_eliminated, 4);

        // Check that Gelu node was created
        let gelu = ctx.get_node("div").unwrap();
        assert_eq!(gelu.op_type, "Gelu");
    }
}
