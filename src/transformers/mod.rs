//! ONNX transformers module
//!
//! This module provides ready-to-use transformers for common ONNX optimizations:
//!
//! - **Elimination**: Remove unnecessary nodes (Identity, Dropout, etc.)
//! - **Fusion**: Combine adjacent nodes (Conv+BN, Gemm+Add, etc.)
//! - **Inference**: Infer missing attributes (axes, shapes, etc.)
//! - **Shape**: Optimize Reshape, Transpose chains
//!
//! # Overview
//!
//! Each transformer implements the [`OnnxTransformer`] trait and can be
//! applied individually or combined.
//!
//! # Example
//!
//! ```ignore
//! use furiosa_optimizer::transformers::{
//!     EliminateIdentity, FuseConvBN, FuseGemmAdd, OnnxTransformer
//! };
//! use furiosa_optimizer::graph::GraphContext;
//!
//! let mut ctx = GraphContext::new(&graph);
//!
//! // Apply individual transformers
//! let result1 = EliminateIdentity::new().transform(&mut ctx)?;
//! let result2 = FuseConvBN::new().transform(&mut ctx)?;
//! let result3 = FuseGemmAdd::new().transform(&mut ctx)?;
//!
//! println!("Eliminated {} nodes", result1.nodes_eliminated);
//! println!("Fused {} Conv+BN pairs", result2.transforms_applied);
//! ```
//!
//! # Using OptimizationPipeline
//!
//! For convenience, use [`OptimizationPipeline`] to run multiple transformers:
//!
//! ```ignore
//! use furiosa_optimizer::transformers::OptimizationPipeline;
//!
//! let pipeline = OptimizationPipeline::default();
//! let result = pipeline.run(&mut ctx)?;
//! ```

/// Common utilities and types
pub mod common;
/// Constant folding transformers
pub mod constant_fold;
/// Convert negative pads to slice
pub mod convert_negative_pads_to_slice;
/// Convert PRelu to Relu
pub mod convert_prelu_to_relu;
/// Decompose BatchNormalization
pub mod decompose_bn;
/// Elimination transformers
pub mod eliminate;
/// Experimental transformers
pub mod experimental;
/// Fuse Self Attention
pub mod fuse_attention;
/// Fuse Bias + Gelu
pub mod fuse_bias_gelu;
/// Fuse Convolution
pub mod fuse_conv;
/// Fuse Conv + BN
pub mod fuse_conv_bn;
/// Fuse Conv + Mul + Add
pub mod fuse_conv_mul_add;
/// Fuse Gather + MatMul
pub mod fuse_gather_matmul;
/// Fuse Gelu
pub mod fuse_gelu;
/// Fuse Gemm
pub mod fuse_gemm;
/// Fuse Layer Normalization
pub mod fuse_layer_norm;
/// Fuse Pad
pub mod fuse_pad;
/// Fuse Skip + Layer Normalization
pub mod fuse_skip_layer_norm;
/// Inference transformers
pub mod infer;
/// Optimize Reshape chains
pub mod optimize_reshape;
/// Optimize Transpose chains
pub mod optimize_transpose;
/// Polish model
pub mod polish_model;

// Re-export common types
pub use common::{
    get_attr_f, get_attr_floats, get_attr_i, get_attr_ints, get_attr_s, has_attr, remove_attr,
    run_transformers, set_attr_f, set_attr_i, set_attr_ints, OnnxTransformer, TransformResult,
};

// Re-export elimination transformers
pub use eliminate::{
    EliminateAll, EliminateDropout, EliminateIdentity, EliminateNopCast, EliminateNopReshape,
    EliminateNopTranspose,
};

// Re-export fusion transformers
pub use convert_negative_pads_to_slice::ConvertNegativePadsToSlice;
pub use convert_prelu_to_relu::ConvertPReluToRelu;
pub use decompose_bn::DecomposeBN;
pub use experimental::{
    EliminateDetectionPostprocess, EmbeddingBagPorting, FuseDivForBert, ReifyConvForBert,
};
pub use fuse_attention::FuseSelfAttention;
pub use fuse_bias_gelu::FuseBiasGelu;
pub use fuse_conv::FuseConv;
pub use fuse_conv_bn::FuseConvBN;
pub use fuse_conv_mul_add::FuseConvMulAdd;
pub use fuse_gather_matmul::FuseGatherMatMul;
pub use fuse_gelu::FuseGeluErf;
pub use fuse_gemm::{FuseGemmAdd, FuseGemmAll, FuseMatMulAdd};
pub use fuse_layer_norm::FuseLayerNorm;
pub use fuse_pad::FusePad;
pub use fuse_skip_layer_norm::FuseSkipLayerNorm;

// Re-export inference transformers
pub use infer::{InferAll, InferReduceAxes, InferSqueezeAxes, InferUnsqueezeAxes};

// Re-export shape optimization transformers
pub use optimize_reshape::{
    MergeReshape, MergeSqueeze, MergeUnsqueeze, OptimizeReshapeAll, SimplifyFlattenReshape,
};
pub use optimize_transpose::{
    CancelInverseTranspose, MergeTranspose, OptimizeTransposeAll, SinkTranspose,
};

// Re-export constant folding transformers
pub use constant_fold::{ConstantFold, ConstantOptimizeAll, EliminateUnusedConstants};

// Re-export polish model
pub use polish_model::PolishModel;

use crate::error::OnnxResult;
use crate::graph::GraphContext;

/// Optimization pipeline that runs multiple transformers in sequence
#[derive(Debug)]
pub struct OptimizationPipeline {
    /// Enable inference passes
    pub infer: bool,
    /// Enable fusion passes
    pub fuse: bool,
    /// Enable elimination passes
    pub eliminate: bool,
    /// Enable shape optimization passes
    pub optimize_shape: bool,
    /// Enable constant folding passes
    pub constant_fold: bool,
    /// Number of iterations
    pub iterations: usize,
}

impl Default for OptimizationPipeline {
    fn default() -> Self {
        Self {
            infer: true,
            fuse: true,
            eliminate: true,
            optimize_shape: true,
            constant_fold: true,
            iterations: 3,
        }
    }
}

impl OptimizationPipeline {
    /// Create a new pipeline with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable only elimination passes
    pub fn eliminate_only() -> Self {
        Self {
            infer: false,
            fuse: false,
            eliminate: true,
            optimize_shape: false,
            constant_fold: false,
            iterations: 1,
        }
    }

    /// Enable only fusion passes
    pub fn fuse_only() -> Self {
        Self {
            infer: false,
            fuse: true,
            eliminate: false,
            optimize_shape: false,
            constant_fold: false,
            iterations: 1,
        }
    }

    /// Full optimization (all passes enabled)
    pub fn full() -> Self {
        Self {
            infer: true,
            fuse: true,
            eliminate: true,
            optimize_shape: true,
            constant_fold: true,
            iterations: 5,
        }
    }

    /// Run the optimization pipeline
    pub fn run(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut total = TransformResult::new();

        for _ in 0..self.iterations {
            let before = total.transforms_applied + total.nodes_eliminated;

            // Inference passes (run first to prepare for other passes)
            if self.infer {
                total.merge(InferAll::new().transform(ctx)?);
            }

            // Constant folding (early to simplify graph)
            if self.constant_fold {
                total.merge(ConstantOptimizeAll::new().transform(ctx)?);
            }

            // Shape optimization passes (before fusion)
            if self.optimize_shape {
                total.merge(OptimizeTransposeAll::new().transform(ctx)?);
                total.merge(OptimizeReshapeAll::new().transform(ctx)?);
            }

            // Fusion passes
            if self.fuse {
                total.merge(FuseConvBN::new().transform(ctx)?);
                total.merge(FuseConv::new().transform(ctx)?);
                total.merge(FuseDivForBert::new().transform(ctx)?);
                total.merge(ReifyConvForBert::new().transform(ctx)?);
                total.merge(FuseGatherMatMul::new().transform(ctx)?);
                total.merge(FuseGemmAll::new().transform(ctx)?);
                total.merge(FuseGeluErf::new().transform(ctx)?);
                total.merge(FuseBiasGelu::new().transform(ctx)?);
                total.merge(FuseSelfAttention::new().transform(ctx)?);
                total.merge(FuseSkipLayerNorm::new().transform(ctx)?);
                total.merge(FuseConvMulAdd::new().transform(ctx)?); // Added
                total.merge(DecomposeBN::new().transform(ctx)?); // Added
                                                                 // Note: FuseLayerNorm is not included in the default pipeline
                                                                 // because LayerNormalization requires opset >= 17 and may
                                                                 // conflict with other transformations. Use it separately.
            }

            // Elimination passes (run last)
            if self.eliminate {
                total.merge(EliminateAll::new().transform(ctx)?);
            }

            let after = total.transforms_applied + total.nodes_eliminated;

            // Stop if no progress
            if after == before {
                break;
            }
        }

        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::{GraphProto, ValueInfoProto};

    fn make_test_graph() -> GraphProto {
        GraphProto {
            node: vec![
                make_node("Conv", &["X", "W"], &["conv_out"], "conv_0"),
                make_node("Identity", &["conv_out"], &["id_out"], "identity_0"),
                make_node("Relu", &["id_out"], &["Y"], "relu_0"),
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
        }
    }

    #[test]
    fn test_optimization_pipeline() {
        let graph = make_test_graph();
        let mut ctx = GraphContext::new(&graph);

        let pipeline = OptimizationPipeline::default();
        let result = pipeline.run(&mut ctx).unwrap();

        // Should eliminate Identity node
        assert!(result.nodes_eliminated >= 1);
        assert_eq!(ctx.active_node_count(), 2);
    }

    #[test]
    fn test_eliminate_only_pipeline() {
        let graph = make_test_graph();
        let mut ctx = GraphContext::new(&graph);

        let pipeline = OptimizationPipeline::eliminate_only();
        let result = pipeline.run(&mut ctx).unwrap();

        assert!(result.nodes_eliminated >= 1);
    }
}
