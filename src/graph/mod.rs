//! Graph manipulation module for ONNX models
//!
//! This module provides the core infrastructure for working with ONNX graphs:
//!
//! - [`GraphContext`]: Central structure for graph operations with O(1) lookups
//! - [`maps`]: Type definitions and builders for graph maps
//!
//! # Overview
//!
//! The `GraphContext` mirrors Python's `ONNXTransformer` internal state,
//! providing efficient access to nodes, tensors, and their relationships.
//!
//! # Example
//!
//! ```ignore
//! use furiosa_optimizer::graph::GraphContext;
//!
//! // Create context from graph
//! let ctx = GraphContext::new(&graph);
//!
//! // Find nodes
//! let conv_nodes = ctx.find_nodes_by_op("Conv");
//!
//! // Traverse graph
//! let producer = ctx.get_producer("conv_out");
//! let consumers = ctx.get_consumers("conv_out");
//!
//! // Check relationships
//! if ctx.is_single_use("tensor_name") {
//!     // Safe to eliminate...
//! }
//! ```
//!
//! # Maps
//!
//! The context maintains several maps for O(1) lookups:
//!
//! | Map | Description |
//! |-----|-------------|
//! | `producer_map` | output_name → producer node name |
//! | `consumer_map` | tensor_name → consumer node names |
//! | `optimizer_map` | node_name → OpEntry (order preserved) |
//! | `initializer_map` | name → TensorProto |
//! | `value_info_map` | name → ValueInfoProto |

pub mod accessors;
pub mod context;
pub mod maps;
pub mod mutators;

// Re-export main types
pub use context::GraphContext;
pub use maps::{
    ConsumerMap, InitializerMap, InputCountMap, OpEntry, OptimizerMap, ProducerMap, ValueInfoMap,
};
