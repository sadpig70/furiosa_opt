//! Error types for furiosa-optimizer
//!
//! This module defines all error types used throughout the crate.

use thiserror::Error;

/// Main error type for ONNX transformation operations
#[derive(Error, Debug)]
pub enum TransformError {
    /// Pattern matching failed
    #[error("Pattern matching failed: {0}")]
    PatternNotMatched(String),

    /// Invalid node configuration
    #[error("Invalid node: {0}")]
    InvalidNode(String),

    /// Invalid model
    #[error("Invalid model: {0}")]
    InvalidModel(String),

    /// Missing required field
    #[error("Missing required field: {0}")]
    MissingField(String),

    /// Value info not found
    #[error("Value info not found for tensor: {0}")]
    ValueInfoNotFound(String),

    /// Initializer not found
    #[error("Initializer not found: {0}")]
    InitializerNotFound(String),

    /// Shape inference failed
    #[error("Shape inference failed: {0}")]
    ShapeInferenceFailed(String),

    /// Unsupported opset version
    #[error("Unsupported opset version: {version}, expected {min}..={max}")]
    UnsupportedOpset {
        /// Actual version
        version: i64,
        /// Minimum supported
        min: i64,
        /// Maximum supported
        max: i64,
    },

    /// Invalid tensor data type
    #[error("Invalid data type: {0}")]
    InvalidDataType(i32),

    /// Model validation failed
    #[error("Model validation failed: {0}")]
    ValidationFailed(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Protobuf decode error
    #[error("Protobuf decode error: {0}")]
    ProtoDecode(#[from] prost::DecodeError),

    /// Protobuf encode error
    #[error("Protobuf encode error: {0}")]
    ProtoEncode(#[from] prost::EncodeError),

    /// Generic internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type alias for ONNX operations
pub type OnnxResult<T> = Result<T, TransformError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = TransformError::PatternNotMatched("Conv->BN".to_string());
        assert!(err.to_string().contains("Conv->BN"));
    }

    #[test]
    fn test_unsupported_opset() {
        let err = TransformError::UnsupportedOpset {
            version: 11,
            min: 12,
            max: 13,
        };
        assert!(err.to_string().contains("11"));
    }
}
