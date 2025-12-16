//! Tensor utilities for ONNX models
//!
//! This module provides utilities for working with ONNX tensors:
//! - Data type mappings (`dtype`)
//! - Shape utilities (`shape`)
//! - Conversion between TensorProto and ndarray (`convert`)
//!
//! # Example
//!
//! ```ignore
//! use furiosa_optimizer::tensor::{tensor_to_array_f32, array_to_tensor_f32};
//!
//! // Convert TensorProto to ndarray
//! let array = tensor_to_array_f32(&tensor)?;
//!
//! // Perform operations...
//! let result = &array * 2.0;
//!
//! // Convert back to TensorProto
//! let output = array_to_tensor_f32(&result, "output");
//! ```

pub mod convert;
pub mod dtype;
pub mod shape;

// Re-export commonly used items
pub use convert::{
    array_to_tensor_f32, array_to_tensor_i64, scalar_to_tensor_f32, tensor_to_array_f32,
    tensor_to_array_i64, vec_to_tensor_f32, vec_to_tensor_i64,
};
pub use dtype::{dtype_size, i32_to_dtype, is_float_type, is_int_type, is_signed};
pub use shape::{
    broadcast_shape, is_broadcastable, is_dynamic, normalize_axes, normalize_axis, numel,
    shape_from_value_info,
};
