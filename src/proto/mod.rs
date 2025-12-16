//! ONNX Protocol Buffer types
//!
//! This module re-exports the generated protobuf types from `prost-build`.
//! Additional extension methods are provided in the `extensions` submodule.

/// Generated ONNX protobuf types
#[allow(missing_docs)]
#[allow(clippy::all)]
pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

// Re-export commonly used types at module level
pub use onnx::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TypeProto, ValueInfoProto,
};

// Re-export submodules for nested types
pub use onnx::tensor_shape_proto;
pub use onnx::type_proto;

/// Extension methods for ONNX protobuf types
pub mod extensions;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_proto_default() {
        let model = ModelProto::default();
        assert_eq!(model.ir_version, 0);
    }

    #[test]
    fn test_node_proto_default() {
        let node = NodeProto::default();
        assert!(node.input.is_empty());
        assert!(node.output.is_empty());
    }
}
