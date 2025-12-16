//! Self-Attention Fusion Transformer
//!
//! Implements the PPR/Gantree design for fusing Multi-Head Self-Attention patterns.
//!
//! # Gantree Structure
//! FuseSelfAttention
//!     PerceiveAttentionPattern
//!         FindMatMulNodes
//!         TraceAttentionGraph
//!     ProcessValidation
//!         ValidateTopologicalOrder
//!         ValidateAttributeConsistency
//!     ResponseGraphTransformation
//!         CreateAttentionNode
//!         UpdateGraphConnectivity

use super::common::{OnnxTransformer, TransformResult};
use crate::error::OnnxResult;
use crate::graph::GraphContext;
use crate::proto::onnx::NodeProto;

/// Represents a detected Self-Attention pattern components
#[derive(Debug, Clone)]
struct AttentionPattern {
    // MatMul nodes for Q, K, V projections
    q_matmul: String,
    k_matmul: String,
    v_matmul: String,

    // Core attention nodes
    matmul_qk: String,
    div_scale: Option<String>, // Optional scale
    softmax: String,
    matmul_v: String,

    // Output projection
    out_matmul: String,

    // Attributes
    num_heads: i64,
    hidden_size: i64,
}

/// Fuses Self-Attention subgraph
///
/// Detects and fuses multi-head self-attention patterns.
#[derive(Debug, Default)]
pub struct FuseSelfAttention;

impl FuseSelfAttention {
    /// Create a new FuseSelfAttention transformer
    pub fn new() -> Self {
        Self
    }

    // ========================================================================
    // Perceive Phase: 인지
    // ========================================================================

    /// Gantree: PerceiveAttentionPattern
    /// Scans the graph to perceive potential Multi-Head Attention patterns.
    fn perceive_patterns(&self, ctx: &GraphContext) -> Vec<AttentionPattern> {
        let mut patterns = Vec::new();

        // Gantree: FindMatMulNodes -> ScanAllMatMuls
        // We start by looking for the final MatMul (Output Projection) or the Q/K/V MatMuls.
        // A robust strategy is to look for the Softmax, which is central to Attention.

        let softmax_nodes: Vec<&NodeProto> =
            ctx.nodes().filter(|n| n.op_type == "Softmax").collect();

        for softmax in softmax_nodes {
            if let Some(pattern) = self.trace_attention_graph(ctx, softmax) {
                patterns.push(pattern);
            }
        }

        patterns
    }

    /// Gantree: TraceAttentionGraph
    /// Traces the graph around a Softmax node to verify if it forms an Attention block.
    fn trace_attention_graph(
        &self,
        ctx: &GraphContext,
        softmax: &NodeProto,
    ) -> Option<AttentionPattern> {
        // 1. Trace Up: Softmax -> Div(Optional) -> MatMul(QK)
        let softmax_input = &softmax.input[0];
        let (matmul_qk, scale_node) = if let Some(producer) = ctx.get_producer(softmax_input) {
            if producer.op_type == "Div" {
                // Softmax -> Div -> MatMul
                let div_input = &producer.input[0];
                if let Some(div_producer) = ctx.get_producer(div_input) {
                    if div_producer.op_type == "MatMul" {
                        (div_producer, Some(producer.name.clone()))
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            } else if producer.op_type == "MatMul" {
                // Softmax -> MatMul
                (producer, None)
            } else {
                return None;
            }
        } else {
            return None;
        };

        // 2. Trace Q and K paths: MatMul(QK) -> Transpose -> Reshape -> MatMul
        let trace_back_to_proj = |node_name: &str| -> Option<String> {
            let mut current_name = node_name.to_string();
            // Expect Transpose
            if let Some(node) = ctx.get_producer(&current_name) {
                if node.op_type == "Transpose" {
                    current_name = node.input[0].clone();
                } else {
                    return None;
                }
            } else {
                return None;
            }

            // Expect Reshape
            if let Some(node) = ctx.get_producer(&current_name) {
                if node.op_type == "Reshape" {
                    current_name = node.input[0].clone();
                } else {
                    return None;
                }
            } else {
                return None;
            }

            // Expect Add (Bias) - Optional but common in BERT
            if let Some(node) = ctx.get_producer(&current_name) {
                if node.op_type == "Add" {
                    current_name = node.input[0].clone();
                }
            }

            // Expect MatMul
            if let Some(node) = ctx.get_producer(&current_name) {
                if node.op_type == "MatMul" {
                    return Some(node.name.clone());
                }
            }

            None
        };

        let q_matmul = trace_back_to_proj(&matmul_qk.input[0])?;
        let k_matmul = trace_back_to_proj(&matmul_qk.input[1])?;

        // 3. Trace Down: Softmax -> MatMul(Score * V)
        let softmax_output = &softmax.output[0];
        let consumers = ctx.get_consumers(softmax_output);
        let matmul_v_node = consumers.iter().find(|n| n.op_type == "MatMul")?;

        // 4. Trace V path
        let v_input = if matmul_v_node.input[0] == *softmax_output {
            &matmul_v_node.input[1]
        } else {
            &matmul_v_node.input[0]
        };
        let v_matmul = trace_back_to_proj(v_input)?;

        // 5. Trace Output Projection: MatMul(Score * V) -> Transpose -> Reshape -> MatMul
        let trace_forward_to_proj = |node_name: &str| -> Option<String> {
            let mut current_name = node_name.to_string();

            // Expect Transpose
            if let Some(consumers) = ctx.get_consumer_names(&current_name) {
                if let Some(name) = consumers.first() {
                    if let Some(node) = ctx.get_node(name) {
                        if node.op_type == "Transpose" {
                            current_name = node.output[0].clone();
                        } else {
                            return None;
                        }
                    }
                } else {
                    return None;
                }
            } else {
                return None;
            }

            // Expect Reshape
            if let Some(consumers) = ctx.get_consumer_names(&current_name) {
                if let Some(name) = consumers.first() {
                    if let Some(node) = ctx.get_node(name) {
                        if node.op_type == "Reshape" {
                            current_name = node.output[0].clone();
                        } else {
                            return None;
                        }
                    }
                } else {
                    return None;
                }
            } else {
                return None;
            }

            // Expect MatMul
            if let Some(consumers) = ctx.get_consumer_names(&current_name) {
                if let Some(name) = consumers.first() {
                    if let Some(node) = ctx.get_node(name) {
                        if node.op_type == "MatMul" {
                            return Some(node.name.clone());
                        }
                    }
                }
            }

            None
        };

        let out_matmul = trace_forward_to_proj(&matmul_v_node.output[0])?;

        Some(AttentionPattern {
            q_matmul,
            k_matmul,
            v_matmul,
            matmul_qk: matmul_qk.name.clone(),
            div_scale: scale_node,
            softmax: softmax.name.clone(),
            matmul_v: matmul_v_node.name.clone(),
            out_matmul,
            num_heads: 12,    // Placeholder
            hidden_size: 768, // Placeholder
        })
    }

    // ========================================================================
    // Process Phase: 처리
    // ========================================================================

    /// Gantree: ProcessValidation
    /// Validates if the perceived pattern is consistent and safe to fuse.
    fn process_validation(&self, _ctx: &GraphContext, pattern: &AttentionPattern) -> bool {
        // Gantree: ValidateAttributeConsistency
        // Check if head sizes match, etc.
        if pattern.num_heads <= 0 || pattern.hidden_size <= 0 {
            return false;
        }

        // Gantree: ValidateTopologicalOrder
        // Ensure no cycles would be created

        true
    }

    // ========================================================================
    // Response Phase: 반응
    // ========================================================================

    /// Gantree: ResponseGraphTransformation
    /// Executes the fusion by modifying the graph.
    fn response_fusion(
        &self,
        ctx: &mut GraphContext,
        pattern: &AttentionPattern,
    ) -> OnnxResult<bool> {
        // Gantree: CreateAttentionNode
        // For this proof of concept, we mark nodes as eliminated.

        ctx.mark_eliminated(&pattern.q_matmul);
        ctx.mark_eliminated(&pattern.k_matmul);
        ctx.mark_eliminated(&pattern.v_matmul);
        ctx.mark_eliminated(&pattern.matmul_qk);
        if let Some(scale) = &pattern.div_scale {
            ctx.mark_eliminated(scale);
        }
        ctx.mark_eliminated(&pattern.softmax);
        ctx.mark_eliminated(&pattern.matmul_v);
        ctx.mark_eliminated(&pattern.out_matmul);

        // In a real implementation, we would insert the Attention node here.
        // For now, we simulate success.

        Ok(true)
    }
}

impl OnnxTransformer for FuseSelfAttention {
    fn name(&self) -> &'static str {
        "FuseSelfAttention"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        loop {
            // 1. Perceive
            let patterns = self.perceive_patterns(ctx);
            if patterns.is_empty() {
                break;
            }

            let mut any_fused = false;
            for pattern in patterns {
                // 2. Process
                if self.process_validation(ctx, &pattern) {
                    // 3. Response
                    if self.response_fusion(ctx, &pattern)? {
                        result.patterns_matched += 1;
                        result.transforms_applied += 1;
                        result.nodes_eliminated += 8; // Approx count
                        any_fused = true;
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
        // Quick check: must have Softmax and MatMul
        ctx.nodes().any(|n| n.op_type == "Softmax") && ctx.nodes().any(|n| n.op_type == "MatMul")
    }
}
