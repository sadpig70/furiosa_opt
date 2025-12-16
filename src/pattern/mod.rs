//! Pattern matching module for ONNX graph optimization
//!
//! This module provides tools for identifying and matching patterns in ONNX graphs,
//! which is essential for graph transformations like node fusion.
//!
//! # Overview
//!
//! The pattern matching system works by:
//! 1. Defining patterns as sequences of op_types
//! 2. Matching patterns in reverse order (output → input)
//! 3. Optionally applying additional conditions
//!
//! # Example
//!
//! ```ignore
//! use furiosa_optimizer::pattern::{PatternMatcher, ops};
//!
//! let matcher = PatternMatcher::new(&ctx);
//!
//! // Find all Conv+BatchNorm patterns for fusion
//! let matches = matcher.find_all_matches(ops::CONV_BN);
//!
//! for m in matches {
//!     // m.nodes[0] = BatchNorm (anchor)
//!     // m.nodes[1] = Conv (predecessor)
//!     println!("Found fusible pair: {} <- {}", m.nodes[0].name, m.nodes[1].name);
//! }
//! ```
//!
//! # Traversal
//!
//! The module also provides graph traversal utilities:
//!
//! ```ignore
//! use furiosa_optimizer::pattern::traversal::{BfsIterator, Direction};
//!
//! // Forward BFS from a node
//! for node in BfsIterator::forward(&ctx, "conv_0") {
//!     println!("Visiting: {}", node.name);
//! }
//!
//! // Find all predecessors
//! let preds = traversal::predecessors(&ctx, "output_node");
//! ```

pub mod matcher;
pub mod ops;
pub mod traversal;

// Re-export main types
pub use matcher::{matcher, MatchResult, PatternMatcher};
pub use ops::{
    categorize_op, is_activation, is_binary_op, is_conv_like, is_norm_op, is_pool_op, is_reduce_op,
    is_shape_op, OpCategory, PatternBuilder,
};
pub use traversal::{
    find_path, has_path, predecessors, reachable_nodes, successors, BfsIterator, DfsIterator,
    Direction,
};

// Re-export common patterns
pub use ops::{
    ACTIVATIONS, BINARY_OPS, CONV_ADD, CONV_BN, CONV_LIKE, CONV_MUL_ADD, CONV_TRANSPOSE_BN,
    GATHER_MATMUL, NORM_OPS, PAD, PAD_CONV, POOL_OPS, PRELU, REDUCE_OPS, RELU_BN_CONV, SHAPE_OPS,
    SQUEEZE,
};
