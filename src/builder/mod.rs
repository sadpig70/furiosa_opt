//! Model builder module for ONNX optimization
//!
//! This module provides utilities for building and assembling optimized ONNX models:
//!
//! - [`ModelBuilder`]: Fluent builder for constructing models
//! - [`cleanup`]: Graph cleanup and deduplication
//! - [`fields`]: Field manipulation utilities
//!
//! # Overview
//!
//! After transformations are applied to a `GraphContext`, the builder module
//! assembles the final optimized model.
//!
//! # Example
//!
//! ```ignore
//! use furiosa_optimizer::builder::{ModelBuilder, build_optimized_model};
//!
//! // Using ModelBuilder (fluent API)
//! let model = ModelBuilder::new(original_model)
//!     .with_context(ctx)
//!     .producer_name("furiosa-optimizer")
//!     .producer_version("0.1.0")
//!     .cleanup(true)
//!     .build()?;
//!
//! // Or using the simple function
//! let model = build_optimized_model(&ctx, &original_model);
//! ```
//!
//! # Cleanup
//!
//! The cleanup submodule handles removal of unused elements:
//!
//! ```ignore
//! use furiosa_optimizer::builder::cleanup;
//!
//! // Remove unused initializers and value_info
//! cleanup::cleanup_graph(&mut graph);
//!
//! // With statistics
//! let stats = cleanup::cleanup_with_stats(&mut graph);
//! println!("Removed {} initializers", stats.initializers_removed);
//! ```
//!
//! # Field Manipulation
//!
//! The fields submodule provides utilities for modifying model components:
//!
//! ```ignore
//! use furiosa_optimizer::builder::fields;
//!
//! // Rename a tensor throughout the graph
//! fields::rename_tensor(&mut graph, "old_name", "new_name");
//!
//! // Add an initializer
//! fields::add_initializer(&mut graph, tensor);
//!
//! // Generate unique names
//! let name = fields::unique_tensor_name(&graph, "tensor");
//! ```

pub mod cleanup;
pub mod fields;
pub mod model;

// Re-export main types and functions
pub use cleanup::{
    cleanup_graph, cleanup_with_stats, collect_used_tensors, deduplicate_graph,
    deduplicate_initializers, deduplicate_value_info, filter_initializers, filter_value_info,
    remove_unused_initializers, remove_unused_value_info, CleanupStats,
};

pub use fields::{
    add_initializer, add_initializers, add_node, add_value_info, get_opset_version, insert_node,
    remove_initializer, remove_node, remove_value_info, rename_node, rename_tensor, replace_node,
    set_graph_input, set_graph_name, set_graph_output, set_ir_version, set_opset_version,
    unique_node_name, unique_tensor_name, update_model_metadata,
};

pub use model::{
    build_graph_from_context, build_optimized_model, build_optimized_model_with_stats,
    optimize_model, validate_model, ModelBuilder,
};
