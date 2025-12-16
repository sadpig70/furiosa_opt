//! # Furiosa Optimizer
//!
//! ONNX model optimizer for Furiosa NPU - Rust implementation.
//!
//! This crate provides graph-level optimizations for ONNX models,
//! including pattern matching, node fusion, and model transformations.
//!
//! ## Features
//!
//! - **Pattern Matching**: Identify common subgraph patterns for optimization
//! - **Node Fusion**: Fuse BatchNorm into Conv, PReLU decomposition, etc.
//! - **Graph Cleanup**: Remove unused nodes, initializers, and value_info
//!
//! ## Example
//!
//! ```ignore
//! use furiosa_optimizer::prelude::*;
//!
//! let model = load_model("model.onnx")?;
//! let optimized = optimize_model(model, None)?;
//! ```

#![warn(missing_docs)]
#![deny(unsafe_code)]

// ============================================================================
// Module declarations (to be implemented)
// ============================================================================

pub mod builder;
pub mod error;
pub mod graph;
pub mod io;
pub mod opset;
pub mod pattern;
pub mod proto;
pub mod tensor;
pub mod traits;
pub mod transform;
pub mod transformers;

// Python bindings (only with python feature)
#[cfg(feature = "python")]
pub mod python;

// Future modules (commented out until implemented)
// pub mod graph;
// pub mod pattern;
// pub mod transform;
// pub mod builder;
// pub mod transformers;
// pub mod utils;

// ============================================================================
// Prelude module for convenient imports
// ============================================================================

/// Prelude module - import commonly used types with `use furiosa_optimizer::prelude::*`
pub mod prelude {
    pub use crate::builder::{build_optimized_model, ModelBuilder};
    pub use crate::error::{OnnxResult, TransformError};
    pub use crate::graph::GraphContext;
    pub use crate::io::{load_model, optimize_file, save_model, OptimizeOptions, OptimizeStats};
    pub use crate::opset::{get_opset_version, upgrade_model, upgrade_to_opset_17, OpsetUpgrader};
    pub use crate::pattern::{matcher, MatchResult, PatternMatcher};
    pub use crate::proto::onnx::*;
    pub use crate::traits::Transformer;
    pub use crate::transform::{TransformConfig, TransformEngine};
    pub use crate::transformers::{OnnxTransformer, OptimizationPipeline};
}

// ============================================================================
// Crate-level re-exports
// ============================================================================

pub use error::{OnnxResult, TransformError};
pub use traits::Transformer;

// ============================================================================
// Version information
// ============================================================================

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Supported ONNX opset version range
pub const SUPPORTED_OPSET_MIN: i64 = 9;
/// Maximum supported ONNX opset version
pub const SUPPORTED_OPSET_MAX: i64 = 17;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_opset_range() {
        // Verify opset range is valid
        assert_eq!(SUPPORTED_OPSET_MIN, 9);
        assert_eq!(SUPPORTED_OPSET_MAX, 17);
    }
}
