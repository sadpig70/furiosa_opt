//! Bias-GELU Fusion Transformer
//!
//! Implements the PPR/Gantree design for fusing Add (Bias) + GELU patterns into FastGelu.
//!
//! # Gantree Structure
//! FuseBiasGelu
//!     PerceiveBiasGeluPattern
//!         FindGeluNodes
//!         TraceProducerAdd
//!     ProcessValidation
//!         ValidateTopology (Multi-consumer check)
//!         ValidateBiasShape
//!     ResponseFusion
//!         CreateFastGeluNode
//!         UpdateConnectivity

use super::common::{OnnxTransformer, TransformResult};
use crate::error::OnnxResult;
use crate::graph::GraphContext;
use crate::proto::onnx::NodeProto;

/// Represents a detected Bias-GELU pattern
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BiasGeluPattern {
    // The Gelu node
    gelu_node: String,
    // The Add node (producer of Gelu input)
    add_node: String,
    // Inputs to the Add node (Input, Bias)
    input_x: String,
    input_bias: String,
}

/// Fuses Bias + Gelu
///
/// Transforms:
///   Add(bias) -> Gelu
/// Into:
///   FastGelu(bias)
#[derive(Debug, Default)]
pub struct FuseBiasGelu;

impl FuseBiasGelu {
    /// Create a new FuseBiasGelu transformer
    pub fn new() -> Self {
        Self
    }

    // ========================================================================
    // Perceive Phase: 인지
    // ========================================================================

    fn perceive_patterns(&self, ctx: &GraphContext) -> Vec<BiasGeluPattern> {
        let mut patterns = Vec::new();

        // Gantree: FindGeluNodes
        let gelu_nodes: Vec<&NodeProto> = ctx
            .nodes()
            .filter(|n| n.op_type == "Gelu" && !ctx.is_eliminated(&n.name))
            .collect();

        for gelu in gelu_nodes {
            // Gantree: TraceProducerAdd
            // Gelu input[0] should come from Add
            if let Some(producer) = ctx.get_producer(&gelu.input[0]) {
                if producer.op_type == "Add" {
                    // Found Add -> Gelu

                    // Check inputs of Add
                    // One should be the main input, the other should be bias (initializer)
                    // We don't strictly enforce initializer here, but usually bias is static.
                    // FastGelu supports dynamic bias too, so we just need 2 inputs.

                    if producer.input.len() == 2 {
                        // Heuristic: assume input[1] is bias if it's an initializer,
                        // or if input[0] is the main flow.
                        // For simplicity, we take them as is. FastGelu(X, Bias).
                        // If input[0] is bias, we might need to swap.
                        // Let's check if one is initializer.

                        let input0 = &producer.input[0];
                        let input1 = &producer.input[1];

                        let (input_x, input_bias) = if ctx.is_initializer(input1) {
                            (input0.clone(), input1.clone())
                        } else if ctx.is_initializer(input0) {
                            (input1.clone(), input0.clone())
                        } else {
                            // Both dynamic or both static. Assume 0 is X, 1 is Bias.
                            (input0.clone(), input1.clone())
                        };

                        println!(
                            "Found Bias-GELU candidate: Gelu={}, Add={}",
                            gelu.name, producer.name
                        );

                        patterns.push(BiasGeluPattern {
                            gelu_node: gelu.name.clone(),
                            add_node: producer.name.clone(),
                            input_x,
                            input_bias,
                        });
                    }
                }
            }
        }

        patterns
    }

    // ========================================================================
    // Process Phase: 처리
    // ========================================================================

    fn process_validation(&self, ctx: &GraphContext, pattern: &BiasGeluPattern) -> bool {
        // Gantree: ValidateTopology (Multi-consumer check)
        // The Add node's output must ONLY be consumed by the Gelu node.
        if ctx.get_input_count(&pattern.add_node) > 1 {
            // Actually we need to check if the OUTPUT of add_node is consumed by others.
            // pattern.add_node is the name of the node.
            // We need the output tensor name.
            if let Some(add_node) = ctx.get_node(&pattern.add_node) {
                if let Some(out) = add_node.output.first() {
                    if ctx.get_consumers(out).len() > 1 {
                        return false;
                    }
                }
            }
        }

        // Gantree: ValidateBiasShape
        // FastGelu bias requires 1D tensor.
        if let Some(bias_tensor) = ctx.get_initializer(&pattern.input_bias) {
            if bias_tensor.dims.len() != 1 {
                return false;
            }
        } else {
            // If bias is not initializer, we can't check shape easily (unless we check value info)
            // But for now we assume bias MUST be initializer for FastGelu fusion to be safe.
            return false;
        }

        true
    }

    // ========================================================================
    // Response Phase: 반응
    // ========================================================================

    fn response_fusion(
        &self,
        ctx: &mut GraphContext,
        pattern: &BiasGeluPattern,
    ) -> OnnxResult<bool> {
        // Gantree: CreateFastGeluNode
        // We will create a new node "FastGelu" in com.microsoft domain.

        // 1. Get Gelu output name (do this before mutable borrow of ctx)
        let gelu_output = if let Some(gelu) = ctx.get_node(&pattern.gelu_node) {
            gelu.output[0].clone()
        } else {
            return Ok(false);
        };

        // Check if we need to swap inputs (FastGelu expects X, Bias)
        // In perceive_patterns we already identified input_x and input_bias.

        if let Some(entry) = ctx.get_entry_mut(&pattern.add_node) {
            entry.node.op_type = "FastGelu".to_string();
            entry.node.domain = "com.microsoft".to_string();
            entry.node.input = vec![pattern.input_x.clone(), pattern.input_bias.clone()];

            // 2. Update Add node to FastGelu
            entry.node.output = vec![gelu_output.clone()];

            // 3. Update producers map: "gelu_out" is now produced by "add_node" (which is now FastGelu)
            ctx.update_producer(&gelu_output, &pattern.add_node);
        }

        // Mark Gelu as eliminated.
        // Note: We reused Add node, so we don't eliminate it.
        // We only eliminate Gelu.
        ctx.mark_eliminated(&pattern.gelu_node);

        Ok(true)
    }
}

impl OnnxTransformer for FuseBiasGelu {
    fn name(&self) -> &'static str {
        "FuseBiasGelu"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        loop {
            let patterns = self.perceive_patterns(ctx);
            if patterns.is_empty() {
                break;
            }

            let mut any_fused = false;
            for pattern in patterns {
                if self.process_validation(ctx, &pattern)
                    && self.response_fusion(ctx, &pattern)? {
                        result.patterns_matched += 1;
                        result.transforms_applied += 1;
                        result.nodes_eliminated += 1; // Add + Gelu (2) -> FastGelu (1) = Net -1? No, 2 removed, 1 added.
                        any_fused = true;
                    }
            }

            if !any_fused {
                break;
            }
        }

        Ok(result)
    }

    fn is_applicable(&self, ctx: &GraphContext) -> bool {
        ctx.nodes().any(|n| n.op_type == "Gelu") && ctx.nodes().any(|n| n.op_type == "Add")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::{GraphProto, TensorProto, ValueInfoProto};

    #[test]
    fn test_fuse_bias_gelu_invalid_bias_shape() {
        let input = ValueInfoProto {
            name: "X".to_string(),
            ..Default::default()
        };
        let bias = TensorProto {
            name: "B".to_string(),
            dims: vec![1, 1, 10], // 3D 편향 (FastGelu에 대해 유효하지 않음)
            data_type: 1,
            float_data: vec![0.0; 10],
            ..Default::default()
        };
        let graph = GraphProto {
            node: vec![
                make_node("Add", &["X", "B"], &["add_out"], "add_0"),
                make_node("Gelu", &["add_out"], &["Y"], "gelu_0"),
            ],
            input: vec![input],
            output: vec![ValueInfoProto {
                name: "Y".to_string(),
                ..Default::default()
            }],
            initializer: vec![bias],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let result = FuseBiasGelu::new().transform(&mut ctx).unwrap();

        // 편향이 3D이므로 융합되지 않아야 함
        assert_eq!(result.transforms_applied, 0);
        assert!(ctx.has_node("gelu_0"));
        assert_eq!(ctx.get_node("gelu_0").unwrap().op_type, "Gelu");
    }
}
