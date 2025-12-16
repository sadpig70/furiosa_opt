//! ONNX data type mappings
//!
//! Maps between ONNX TensorProto data types and Rust types.

use crate::error::{OnnxResult, TransformError};
use crate::proto::onnx::tensor_proto::DataType;

/// Size in bytes for each ONNX data type
pub fn dtype_size(dtype: DataType) -> OnnxResult<usize> {
    match dtype {
        DataType::Float => Ok(4),
        DataType::Uint8 => Ok(1),
        DataType::Int8 => Ok(1),
        DataType::Uint16 => Ok(2),
        DataType::Int16 => Ok(2),
        DataType::Int32 => Ok(4),
        DataType::Int64 => Ok(8),
        DataType::Bool => Ok(1),
        DataType::Float16 => Ok(2),
        DataType::Double => Ok(8),
        DataType::Uint32 => Ok(4),
        DataType::Uint64 => Ok(8),
        DataType::Bfloat16 => Ok(2),
        DataType::Undefined => Err(TransformError::InvalidDataType(0)),
        _ => Err(TransformError::InvalidDataType(dtype as i32)),
    }
}

/// Convert i32 to DataType enum
pub fn i32_to_dtype(value: i32) -> OnnxResult<DataType> {
    DataType::try_from(value).map_err(|_| TransformError::InvalidDataType(value))
}

/// Check if data type is floating point
pub fn is_float_type(dtype: DataType) -> bool {
    matches!(
        dtype,
        DataType::Float | DataType::Double | DataType::Float16 | DataType::Bfloat16
    )
}

/// Check if data type is integer
pub fn is_int_type(dtype: DataType) -> bool {
    matches!(
        dtype,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Uint8
            | DataType::Uint16
            | DataType::Uint32
            | DataType::Uint64
    )
}

/// Check if data type is signed
pub fn is_signed(dtype: DataType) -> bool {
    matches!(
        dtype,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Float
            | DataType::Double
            | DataType::Float16
            | DataType::Bfloat16
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(dtype_size(DataType::Float).unwrap(), 4);
        assert_eq!(dtype_size(DataType::Int64).unwrap(), 8);
        assert_eq!(dtype_size(DataType::Uint8).unwrap(), 1);
        assert_eq!(dtype_size(DataType::Double).unwrap(), 8);
    }

    #[test]
    fn test_i32_to_dtype() {
        assert_eq!(i32_to_dtype(1).unwrap(), DataType::Float);
        assert_eq!(i32_to_dtype(7).unwrap(), DataType::Int64);
        assert!(i32_to_dtype(999).is_err());
    }

    #[test]
    fn test_is_float_type() {
        assert!(is_float_type(DataType::Float));
        assert!(is_float_type(DataType::Double));
        assert!(!is_float_type(DataType::Int32));
    }

    #[test]
    fn test_is_int_type() {
        assert!(is_int_type(DataType::Int32));
        assert!(is_int_type(DataType::Uint8));
        assert!(!is_int_type(DataType::Float));
    }
}
