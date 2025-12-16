//! LayerNorm Fusion Transformer
//!
//! Fuses the decomposed LayerNorm pattern into a single LayerNormalization op.
//!
//! Pattern detected:
//! ```text
//!   input
//!     ├─────────────────────────┐
//!     ↓                         │
//!   ReduceMean (mean)           │
//!     ↓                         │
//!   Sub ←────────────────────────┘
//!     ├─────────────────────────┐
//!     ↓                         │
//!   Pow (square)                │
//!     ↓                         │
//!   ReduceMean (variance)       │
//!     ↓                         │
//!   Add (epsilon)               │
//!     ↓                         │
//!   Sqrt (std)                  │
//!     ↓                         │
//!   Div ←────────────────────────┘
//!     ↓
//!   Mul (scale/gamma)
//!     ↓
//!   Add (bias/beta)
//!     ↓
//!   output
//! ```
//!
//! Fused into:
//! ```text
//!   LayerNormalization(input, scale, bias, axis, epsilon)
//! ```

use crate::error::OnnxResult;
use crate::graph::GraphContext;
use crate::proto::onnx::{AttributeProto, NodeProto};
use crate::transformers::common::{OnnxTransformer, TransformResult};

/// Pattern for LayerNorm detection
#[derive(Debug, Clone)]
struct LayerNormPattern {
    /// First ReduceMean node (computes mean)
    reduce_mean1: String,
    /// Sub node (input - mean)
    sub: String,
    /// Pow node (squared difference)
    pow: String,
    /// Second ReduceMean node (computes variance)
    reduce_mean2: String,
    /// Add node (variance + epsilon)
    add_eps: String,
    /// Sqrt node (standard deviation)
    sqrt: String,
    /// Div node (normalized)
    div: String,
    /// Mul node (scale/gamma) - optional
    mul: Option<String>,
    /// Add node (bias/beta) - optional
    add_bias: Option<String>,

    /// Input tensor name
    input: String,
    /// Output tensor name
    output: String,
    /// Scale tensor name (gamma)
    scale: Option<String>,
    /// Bias tensor name (beta)
    bias: Option<String>,
    /// Epsilon value
    epsilon: f32,
    /// Axis for normalization
    axis: i64,
}

/// Fuse decomposed LayerNorm patterns into LayerNormalization op
pub struct FuseLayerNorm {
    /// Minimum opset version that supports LayerNormalization
    #[allow(dead_code)]
    min_opset: i64,
}

impl FuseLayerNorm {
    /// Create new FuseLayerNorm transformer
    pub fn new() -> Self {
        Self { min_opset: 17 } // LayerNormalization added in opset 17
    }

    /// Find LayerNorm patterns in the graph
    fn find_patterns(&self, ctx: &GraphContext) -> Vec<LayerNormPattern> {
        let mut patterns = Vec::new();

        // Find all ReduceMean nodes that could be the start of LayerNorm
        for node in ctx.nodes() {
            if node.op_type != "ReduceMean" {
                continue;
            }

            // Check if this ReduceMean is consumed by Sub
            let rm_output = match node.output.first() {
                Some(o) => o.as_str(),
                None => continue,
            };

            // Find Sub node that uses this ReduceMean
            let sub_node = ctx.nodes().find(|n| {
                n.op_type == "Sub" && n.input.get(1).map(|s| s.as_str()) == Some(rm_output)
            });

            let sub_node = match sub_node {
                Some(n) => n,
                None => continue,
            };

            // Verify Sub's first input matches ReduceMean's input (same tensor)
            let rm_input = match node.input.first() {
                Some(i) => i.as_str(),
                None => continue,
            };

            if sub_node.input.first().map(|s| s.as_str()) != Some(rm_input) {
                continue;
            }

            // Continue tracing the pattern
            if let Some(pattern) = self.trace_pattern(ctx, node, sub_node, rm_input) {
                patterns.push(pattern);
            }
        }

        patterns
    }

    /// Trace the complete LayerNorm pattern starting from ReduceMean and Sub
    fn trace_pattern(
        &self,
        ctx: &GraphContext,
        reduce_mean1: &NodeProto,
        sub_node: &NodeProto,
        input: &str,
    ) -> Option<LayerNormPattern> {
        let sub_output = sub_node.output.first()?.as_str();

        // Find Pow node: Pow(sub_output, 2)
        let pow_node = ctx.nodes().find(|n| {
            n.op_type == "Pow" && n.input.first().map(|s| s.as_str()) == Some(sub_output)
        })?;
        let pow_output = pow_node.output.first()?.as_str();

        // Find second ReduceMean: ReduceMean(pow_output)
        let reduce_mean2 = ctx.nodes().find(|n| {
            n.op_type == "ReduceMean" && n.input.first().map(|s| s.as_str()) == Some(pow_output)
        })?;
        let rm2_output = reduce_mean2.output.first()?.as_str();

        // Find Add (epsilon): Add(rm2_output, epsilon)
        let add_eps = ctx
            .nodes()
            .find(|n| n.op_type == "Add" && n.input.contains(&rm2_output.to_string()))?;
        let add_eps_output = add_eps.output.first()?.as_str();

        // Extract epsilon value
        let epsilon = self.extract_epsilon(ctx, add_eps)?;

        // Find Sqrt: Sqrt(add_eps_output)
        let sqrt_node = ctx.nodes().find(|n| {
            n.op_type == "Sqrt" && n.input.first().map(|s| s.as_str()) == Some(add_eps_output)
        })?;
        let sqrt_output = sqrt_node.output.first()?.as_str();

        // Find Div: Div(sub_output, sqrt_output)
        let div_node = ctx.nodes().find(|n| {
            n.op_type == "Div"
                && n.input.first().map(|s| s.as_str()) == Some(sub_output)
                && n.input.get(1).map(|s| s.as_str()) == Some(sqrt_output)
        })?;
        let div_output = div_node.output.first()?.as_str();

        // Extract axis from ReduceMean
        let axis = self.extract_axis(reduce_mean1)?;

        // Find optional Mul (scale): Mul(div_output, scale)
        let mul_node = ctx.nodes().find(|n| {
            n.op_type == "Mul" && n.input.first().map(|s| s.as_str()) == Some(div_output)
        });

        let (mul_name, mul_output, scale) = if let Some(mul) = mul_node {
            let scale_input = mul.input.get(1)?.clone();
            let mul_out = mul.output.first()?.as_str();
            (Some(mul.name.clone()), mul_out, Some(scale_input))
        } else {
            (None, div_output, None)
        };

        // Find optional Add (bias): Add(mul_output, bias)
        let add_bias_node = ctx.nodes().find(|n| {
            n.op_type == "Add" && n.input.first().map(|s| s.as_str()) == Some(mul_output)
        });

        let (add_bias_name, final_output, bias) = if let Some(add_bias) = add_bias_node {
            let bias_input = add_bias.input.get(1)?.clone();
            let add_out = add_bias.output.first()?.as_str();
            (
                Some(add_bias.name.clone()),
                add_out.to_string(),
                Some(bias_input),
            )
        } else {
            (None, mul_output.to_string(), None)
        };

        Some(LayerNormPattern {
            reduce_mean1: reduce_mean1.name.clone(),
            sub: sub_node.name.clone(),
            pow: pow_node.name.clone(),
            reduce_mean2: reduce_mean2.name.clone(),
            add_eps: add_eps.name.clone(),
            sqrt: sqrt_node.name.clone(),
            div: div_node.name.clone(),
            mul: mul_name,
            add_bias: add_bias_name,
            input: input.to_string(),
            output: final_output,
            scale,
            bias,
            epsilon,
            axis,
        })
    }

    /// Extract epsilon value from Add node
    fn extract_epsilon(&self, ctx: &GraphContext, add_node: &NodeProto) -> Option<f32> {
        // Check second input (epsilon is usually a small constant)
        let eps_input = add_node.input.get(1)?;

        // Try to get from initializer
        if let Some(tensor) = ctx.get_initializer(eps_input) {
            if !tensor.float_data.is_empty() {
                return Some(tensor.float_data[0]);
            }
            if !tensor.raw_data.is_empty() && tensor.raw_data.len() >= 4 {
                let bytes = [
                    tensor.raw_data[0],
                    tensor.raw_data[1],
                    tensor.raw_data[2],
                    tensor.raw_data[3],
                ];
                return Some(f32::from_le_bytes(bytes));
            }
        }

        // Try Constant node
        if let Some(producer) = ctx.get_producer(eps_input) {
            if producer.op_type == "Constant" {
                for attr in &producer.attribute {
                    if attr.name == "value" {
                        if let Some(tensor) = &attr.t {
                            if !tensor.float_data.is_empty() {
                                return Some(tensor.float_data[0]);
                            }
                            if !tensor.raw_data.is_empty() && tensor.raw_data.len() >= 4 {
                                let bytes = [
                                    tensor.raw_data[0],
                                    tensor.raw_data[1],
                                    tensor.raw_data[2],
                                    tensor.raw_data[3],
                                ];
                                return Some(f32::from_le_bytes(bytes));
                            }
                        }
                    }
                }
            }
        }

        // Default epsilon
        Some(1e-5)
    }

    /// Extract axis from ReduceMean node
    fn extract_axis(&self, node: &NodeProto) -> Option<i64> {
        for attr in &node.attribute {
            if attr.name == "axes" && !attr.ints.is_empty() {
                // Return last axis (typically -1 for LayerNorm)
                return Some(*attr.ints.last()?);
            }
        }
        // Default to last axis
        Some(-1)
    }

    /// Create LayerNormalization node from pattern
    fn create_layer_norm_node(&self, pattern: &LayerNormPattern, name: &str) -> NodeProto {
        let mut inputs = vec![pattern.input.clone()];

        // Add scale (required for opset 17+)
        if let Some(ref scale) = pattern.scale {
            inputs.push(scale.clone());
        }

        // Add bias (optional)
        if let Some(ref bias) = pattern.bias {
            inputs.push(bias.clone());
        }

        let mut attributes = vec![
            AttributeProto {
                name: "axis".to_string(),
                r#type: 2, // INT
                i: pattern.axis,
                ..Default::default()
            },
            AttributeProto {
                name: "epsilon".to_string(),
                r#type: 1, // FLOAT
                f: pattern.epsilon,
                ..Default::default()
            },
        ];

        // stash_type attribute (default 1 = FLOAT)
        attributes.push(AttributeProto {
            name: "stash_type".to_string(),
            r#type: 2, // INT
            i: 1,
            ..Default::default()
        });

        NodeProto {
            op_type: "LayerNormalization".to_string(),
            name: name.to_string(),
            input: inputs,
            output: vec![pattern.output.clone()],
            attribute: attributes,
            ..Default::default()
        }
    }

    /// Get all node names in the pattern
    fn pattern_nodes(&self, pattern: &LayerNormPattern) -> Vec<String> {
        let mut nodes = vec![
            pattern.reduce_mean1.clone(),
            pattern.sub.clone(),
            pattern.pow.clone(),
            pattern.reduce_mean2.clone(),
            pattern.add_eps.clone(),
            pattern.sqrt.clone(),
            pattern.div.clone(),
        ];

        if let Some(ref mul) = pattern.mul {
            nodes.push(mul.clone());
        }
        if let Some(ref add_bias) = pattern.add_bias {
            nodes.push(add_bias.clone());
        }

        nodes
    }

    /// Get all intermediate tensor names in the pattern
    fn pattern_intermediate_tensors(
        &self,
        ctx: &GraphContext,
        pattern: &LayerNormPattern,
    ) -> Vec<String> {
        let mut tensors = Vec::new();

        // Get outputs of each node in the pattern (except the final output)
        let node_names = vec![
            &pattern.reduce_mean1,
            &pattern.sub,
            &pattern.pow,
            &pattern.reduce_mean2,
            &pattern.add_eps,
            &pattern.sqrt,
            &pattern.div,
        ];

        for name in node_names {
            if let Some(node) = ctx.get_node(name) {
                for output in &node.output {
                    if output != &pattern.output {
                        tensors.push(output.clone());
                    }
                }
            }
        }

        // Include mul output if it exists and is not the final output
        if let Some(ref mul) = pattern.mul {
            if let Some(node) = ctx.get_node(mul) {
                for output in &node.output {
                    if output != &pattern.output {
                        tensors.push(output.clone());
                    }
                }
            }
        }

        tensors
    }

    /// Check if any intermediate tensor is used outside the pattern
    fn has_external_consumers(&self, ctx: &GraphContext, pattern: &LayerNormPattern) -> bool {
        let pattern_nodes: std::collections::HashSet<String> =
            self.pattern_nodes(pattern).into_iter().collect();
        let intermediate_tensors = self.pattern_intermediate_tensors(ctx, pattern);

        for tensor in &intermediate_tensors {
            // Find all consumers of this tensor
            for node in ctx.nodes() {
                if pattern_nodes.contains(&node.name) {
                    continue; // Skip nodes in the pattern
                }

                if node.input.contains(tensor) {
                    // External node uses this intermediate tensor
                    return true;
                }
            }
        }

        false
    }
}

impl Default for FuseLayerNorm {
    fn default() -> Self {
        Self::new()
    }
}

impl OnnxTransformer for FuseLayerNorm {
    fn name(&self) -> &'static str {
        "FuseLayerNorm"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        // Find all LayerNorm patterns
        let patterns = self.find_patterns(ctx);

        if patterns.is_empty() {
            return Ok(result);
        }

        // Process each pattern
        for (i, pattern) in patterns.iter().enumerate() {
            // Check if any intermediate tensor is used outside the pattern
            if self.has_external_consumers(ctx, pattern) {
                // Skip this pattern - intermediate outputs are used elsewhere
                continue;
            }

            // Create LayerNormalization node
            let ln_name = format!("LayerNorm_{}", i);
            let ln_node = self.create_layer_norm_node(pattern, &ln_name);

            // Get nodes to remove
            let nodes_to_remove = self.pattern_nodes(pattern);

            // Add the new node
            ctx.insert_node(ln_node);

            // Remove old nodes
            for node_name in &nodes_to_remove {
                ctx.remove_node(node_name);
                result.record_elimination(node_name);
            }

            result.transforms_applied += 1;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::onnx::{GraphProto, TensorProto};

    fn make_reduce_mean(name: &str, input: &str, output: &str, axes: Vec<i64>) -> NodeProto {
        NodeProto {
            op_type: "ReduceMean".to_string(),
            name: name.to_string(),
            input: vec![input.to_string()],
            output: vec![output.to_string()],
            attribute: vec![AttributeProto {
                name: "axes".to_string(),
                r#type: 7, // INTS
                ints: axes,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn make_binary_op(op: &str, name: &str, a: &str, b: &str, out: &str) -> NodeProto {
        NodeProto {
            op_type: op.to_string(),
            name: name.to_string(),
            input: vec![a.to_string(), b.to_string()],
            output: vec![out.to_string()],
            ..Default::default()
        }
    }

    fn make_unary_op(op: &str, name: &str, input: &str, out: &str) -> NodeProto {
        NodeProto {
            op_type: op.to_string(),
            name: name.to_string(),
            input: vec![input.to_string()],
            output: vec![out.to_string()],
            ..Default::default()
        }
    }

    fn make_epsilon_init() -> TensorProto {
        TensorProto {
            name: "epsilon".to_string(),
            dims: vec![],
            data_type: 1, // FLOAT
            float_data: vec![1e-5],
            ..Default::default()
        }
    }

    fn make_pow_exp_init() -> TensorProto {
        TensorProto {
            name: "pow_exp".to_string(),
            dims: vec![],
            data_type: 1, // FLOAT
            float_data: vec![2.0],
            ..Default::default()
        }
    }

    #[test]
    fn test_fuse_layer_norm_basic() {
        // Build LayerNorm pattern:
        // input -> ReduceMean -> Sub -> Pow -> ReduceMean -> Add -> Sqrt -> Div -> output
        let graph = GraphProto {
            name: "test".to_string(),
            node: vec![
                make_reduce_mean("rm1", "input", "mean", vec![-1]),
                make_binary_op("Sub", "sub", "input", "mean", "centered"),
                make_binary_op("Pow", "pow", "centered", "pow_exp", "squared"),
                make_reduce_mean("rm2", "squared", "variance", vec![-1]),
                make_binary_op("Add", "add_eps", "variance", "epsilon", "var_eps"),
                make_unary_op("Sqrt", "sqrt", "var_eps", "std"),
                make_binary_op("Div", "div", "centered", "std", "output"),
            ],
            initializer: vec![make_epsilon_init(), make_pow_exp_init()],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let result = FuseLayerNorm::new().transform(&mut ctx).unwrap();

        // Should have found and fused 1 pattern
        assert!(
            result.transforms_applied > 0,
            "Expected LayerNorm pattern to be fused"
        );

        // Check that old nodes are removed
        assert!(
            ctx.get_node("rm1").is_none(),
            "ReduceMean1 should be removed"
        );
        assert!(ctx.get_node("sub").is_none(), "Sub should be removed");
        assert!(ctx.get_node("div").is_none(), "Div should be removed");

        // Check that LayerNormalization node exists
        let ln_node = ctx.get_node("LayerNorm_0");
        assert!(ln_node.is_some(), "LayerNormalization node should exist");

        let ln = ln_node.unwrap();
        assert_eq!(ln.op_type, "LayerNormalization");
        assert_eq!(ln.input.first().map(|s| s.as_str()), Some("input"));
        assert_eq!(ln.output.first().map(|s| s.as_str()), Some("output"));
    }

    #[test]
    fn test_fuse_layer_norm_with_scale_bias() {
        // Build complete LayerNorm pattern with scale and bias
        let graph = GraphProto {
            name: "test".to_string(),
            node: vec![
                make_reduce_mean("rm1", "input", "mean", vec![-1]),
                make_binary_op("Sub", "sub", "input", "mean", "centered"),
                make_binary_op("Pow", "pow", "centered", "pow_exp", "squared"),
                make_reduce_mean("rm2", "squared", "variance", vec![-1]),
                make_binary_op("Add", "add_eps", "variance", "epsilon", "var_eps"),
                make_unary_op("Sqrt", "sqrt", "var_eps", "std"),
                make_binary_op("Div", "div", "centered", "std", "normalized"),
                make_binary_op("Mul", "mul", "normalized", "gamma", "scaled"),
                make_binary_op("Add", "add_bias", "scaled", "beta", "output"),
            ],
            initializer: vec![
                make_epsilon_init(),
                make_pow_exp_init(),
                TensorProto {
                    name: "gamma".to_string(),
                    dims: vec![768],
                    data_type: 1,
                    ..Default::default()
                },
                TensorProto {
                    name: "beta".to_string(),
                    dims: vec![768],
                    data_type: 1,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let result = FuseLayerNorm::new().transform(&mut ctx).unwrap();

        assert!(
            result.transforms_applied > 0,
            "Expected LayerNorm pattern to be fused"
        );

        // Check LayerNormalization node has scale and bias
        let ln_node = ctx.get_node("LayerNorm_0").unwrap();
        assert_eq!(ln_node.input.len(), 3); // input, scale, bias
        assert_eq!(ln_node.input[0], "input");
        assert_eq!(ln_node.input[1], "gamma");
        assert_eq!(ln_node.input[2], "beta");

        // Check output is correct
        assert_eq!(ln_node.output.first().map(|s| s.as_str()), Some("output"));
    }

    #[test]
    fn test_pattern_detection() {
        let fuser = FuseLayerNorm::new();

        // Build a graph with LayerNorm pattern
        let graph = GraphProto {
            name: "test".to_string(),
            node: vec![
                make_reduce_mean("rm1", "X", "mean", vec![-1]),
                make_binary_op("Sub", "sub", "X", "mean", "centered"),
                make_binary_op("Pow", "pow", "centered", "two", "squared"),
                make_reduce_mean("rm2", "squared", "var", vec![-1]),
                make_binary_op("Add", "add_e", "var", "eps", "var_eps"),
                make_unary_op("Sqrt", "sqrt", "var_eps", "std"),
                make_binary_op("Div", "div", "centered", "std", "norm"),
                make_binary_op("Mul", "mul", "norm", "g", "scaled"),
                make_binary_op("Add", "add_b", "scaled", "b", "Y"),
            ],
            initializer: vec![
                TensorProto {
                    name: "eps".to_string(),
                    dims: vec![],
                    data_type: 1,
                    float_data: vec![1e-5],
                    ..Default::default()
                },
                TensorProto {
                    name: "two".to_string(),
                    dims: vec![],
                    data_type: 1,
                    float_data: vec![2.0],
                    ..Default::default()
                },
                TensorProto {
                    name: "g".to_string(),
                    dims: vec![768],
                    data_type: 1,
                    ..Default::default()
                },
                TensorProto {
                    name: "b".to_string(),
                    dims: vec![768],
                    data_type: 1,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let ctx = GraphContext::new(&graph);
        let patterns = fuser.find_patterns(&ctx);

        assert_eq!(patterns.len(), 1, "Should find exactly 1 LayerNorm pattern");

        let p = &patterns[0];
        assert_eq!(p.input, "X");
        assert_eq!(p.output, "Y");
        assert_eq!(p.scale, Some("g".to_string()));
        assert_eq!(p.bias, Some("b".to_string()));
    }
}
