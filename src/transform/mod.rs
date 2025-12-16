//! Graph transformation module
//!
//! This module provides the core transformation infrastructure for ONNX graphs:
//!
//! - [`TransformEngine`]: Main transformation loop
//! - [`fuse`]: Node fusion operations
//! - [`eliminate`]: Node elimination operations
//! - [`bridge`]: Connection bridging after elimination
//!
//! # Overview
//!
//! Transformations work on a `GraphContext` and modify it in-place.
//! The typical workflow is:
//!
//! 1. Create a `TransformEngine` from a graph
//! 2. Apply patterns using `apply_pattern`
//! 3. Build the optimized graph using `build_graph`
//!
//! # Example
//!
//! ```ignore
//! use furiosa_optimizer::transform::{TransformEngine, eliminate::eliminate_node};
//! use furiosa_optimizer::pattern::ops::CONV_BN;
//!
//! let mut engine = TransformEngine::new(&graph);
//!
//! // Apply Conv+BN fusion
//! engine.apply_pattern(CONV_BN, |ctx, m| {
//!     // m.nodes[0] = BatchNorm (anchor)
//!     // m.nodes[1] = Conv (predecessor)
//!     fuse_conv_bn(ctx, &m)?;
//!     Ok(true)
//! })?;
//!
//! // Eliminate Identity nodes
//! engine.apply_pattern(&["Identity"], |ctx, m| {
//!     eliminate_node(ctx, &m.nodes[0].name, 0);
//!     Ok(true)
//! })?;
//!
//! let optimized = engine.build_graph();
//! ```
//!
//! # Python Equivalence
//!
//! | Python Method | Rust Equivalent |
//! |---------------|-----------------|
//! | `transform()` | `TransformEngine::apply_pattern()` |
//! | `transform_to_fuse()` | `fuse::fuse_nodes()` |
//! | `transform_to_eliminate()` | `eliminate::eliminate_node()` |
//! | `bridge_disconnected_nodes()` | `bridge::bridge_disconnected_nodes()` |
//! | `build_optimized_model()` | `TransformEngine::build_model()` |

pub mod bridge;
pub mod core;
pub mod eliminate;
pub mod fuse;

// Re-export main types and functions
pub use bridge::{
    bridge_all_eliminated, bridge_disconnected_nodes, bridge_with_output_preservation,
};

pub use core::{
    build_optimized_graph, transform_once, transform_until_fixed_point, MatchedNodes,
    TransformConfig, TransformEngine, TransformStats,
};

pub use eliminate::{
    can_eliminate, eliminate_chain, eliminate_dead_nodes, eliminate_identity_ops, eliminate_node,
    eliminate_node_if, eliminate_nodes, eliminate_nodes_where, BatchEliminationResult,
    EliminationResult,
};

pub use fuse::{
    can_fuse, fuse_nodes, fuse_nodes_with_update, fuse_pattern, merge_inputs, standard_fuse_inputs,
    BatchFusionResult, FusionResult,
};
