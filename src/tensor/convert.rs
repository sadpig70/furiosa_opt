//! Tensor conversion utilities
//!
//! Convert between ONNX TensorProto and ndarray types.

use ndarray::{Array, ArrayD, IxDyn};

use crate::error::{OnnxResult, TransformError};
use crate::proto::onnx::tensor_proto::DataType;
use crate::proto::TensorProto;

use super::dtype::{dtype_size, i32_to_dtype};
use super::shape::numel;

/// Convert TensorProto to f32 ndarray
///
/// This handles both raw_data and float_data formats.
pub fn tensor_to_array_f32(tensor: &TensorProto) -> OnnxResult<ArrayD<f32>> {
    let dtype = i32_to_dtype(tensor.data_type)?;
    let shape: Vec<usize> = tensor.dims.iter().map(|&d| d as usize).collect();
    let expected_len = numel(&tensor.dims);

    let data: Vec<f32> = if !tensor.raw_data.is_empty() {
        // Decode from raw_data based on dtype
        decode_raw_to_f32(&tensor.raw_data, dtype, expected_len)?
    } else {
        // Use typed data fields
        match dtype {
            DataType::Float => tensor.float_data.clone(),
            DataType::Double => tensor.double_data.iter().map(|&v| v as f32).collect(),
            DataType::Int32 => tensor.int32_data.iter().map(|&v| v as f32).collect(),
            DataType::Int64 => tensor.int64_data.iter().map(|&v| v as f32).collect(),
            DataType::Uint64 => tensor.uint64_data.iter().map(|&v| v as f32).collect(),
            _ => {
                return Err(TransformError::InvalidDataType(tensor.data_type));
            }
        }
    };

    if data.len() != expected_len {
        return Err(TransformError::ShapeInferenceFailed(format!(
            "Data length {} does not match shape {:?} (expected {})",
            data.len(),
            tensor.dims,
            expected_len
        )));
    }

    let ix = IxDyn(&shape);
    Array::from_shape_vec(ix, data).map_err(|e| TransformError::Internal(e.to_string()))
}

/// Convert TensorProto to i64 ndarray
pub fn tensor_to_array_i64(tensor: &TensorProto) -> OnnxResult<ArrayD<i64>> {
    let dtype = i32_to_dtype(tensor.data_type)?;
    let shape: Vec<usize> = tensor.dims.iter().map(|&d| d as usize).collect();
    let expected_len = numel(&tensor.dims);

    let data: Vec<i64> = if !tensor.raw_data.is_empty() {
        decode_raw_to_i64(&tensor.raw_data, dtype, expected_len)?
    } else {
        match dtype {
            DataType::Int64 => tensor.int64_data.clone(),
            DataType::Int32 => tensor.int32_data.iter().map(|&v| v as i64).collect(),
            DataType::Uint64 => tensor.uint64_data.iter().map(|&v| v as i64).collect(),
            _ => {
                return Err(TransformError::InvalidDataType(tensor.data_type));
            }
        }
    };

    if data.len() != expected_len {
        return Err(TransformError::ShapeInferenceFailed(format!(
            "Data length {} does not match expected {}",
            data.len(),
            expected_len
        )));
    }

    let ix = IxDyn(&shape);
    Array::from_shape_vec(ix, data).map_err(|e| TransformError::Internal(e.to_string()))
}

/// Create TensorProto from f32 array
pub fn array_to_tensor_f32(array: &ArrayD<f32>, name: &str) -> TensorProto {
    let dims: Vec<i64> = array.shape().iter().map(|&d| d as i64).collect();

    TensorProto {
        dims,
        data_type: DataType::Float as i32,
        float_data: array.iter().copied().collect(),
        name: name.to_string(),
        ..Default::default()
    }
}

/// Create TensorProto from i64 array
pub fn array_to_tensor_i64(array: &ArrayD<i64>, name: &str) -> TensorProto {
    let dims: Vec<i64> = array.shape().iter().map(|&d| d as i64).collect();

    TensorProto {
        dims,
        data_type: DataType::Int64 as i32,
        int64_data: array.iter().copied().collect(),
        name: name.to_string(),
        ..Default::default()
    }
}

/// Create scalar TensorProto from f32
pub fn scalar_to_tensor_f32(value: f32, name: &str) -> TensorProto {
    TensorProto {
        dims: vec![],
        data_type: DataType::Float as i32,
        float_data: vec![value],
        name: name.to_string(),
        ..Default::default()
    }
}

/// Create 1D TensorProto from f32 slice
pub fn vec_to_tensor_f32(data: &[f32], name: &str) -> TensorProto {
    TensorProto {
        dims: vec![data.len() as i64],
        data_type: DataType::Float as i32,
        float_data: data.to_vec(),
        name: name.to_string(),
        ..Default::default()
    }
}

/// Create 1D TensorProto from i64 slice
pub fn vec_to_tensor_i64(data: &[i64], name: &str) -> TensorProto {
    TensorProto {
        dims: vec![data.len() as i64],
        data_type: DataType::Int64 as i32,
        int64_data: data.to_vec(),
        name: name.to_string(),
        ..Default::default()
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

fn decode_raw_to_f32(raw: &[u8], dtype: DataType, expected: usize) -> OnnxResult<Vec<f32>> {
    let elem_size = dtype_size(dtype)?;
    if raw.len() != expected * elem_size {
        return Err(TransformError::ShapeInferenceFailed(format!(
            "Raw data size {} does not match expected {} * {}",
            raw.len(),
            expected,
            elem_size
        )));
    }

    match dtype {
        DataType::Float => Ok(raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()),
        DataType::Double => Ok(raw
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32)
            .collect()),
        DataType::Int32 => Ok(raw
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32)
            .collect()),
        DataType::Int64 => Ok(raw
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32)
            .collect()),
        DataType::Uint8 => Ok(raw.iter().map(|&b| b as f32).collect()),
        DataType::Int8 => Ok(raw.iter().map(|&b| b as i8 as f32).collect()),
        _ => Err(TransformError::InvalidDataType(dtype as i32)),
    }
}

fn decode_raw_to_i64(raw: &[u8], dtype: DataType, expected: usize) -> OnnxResult<Vec<i64>> {
    let elem_size = dtype_size(dtype)?;
    if raw.len() != expected * elem_size {
        return Err(TransformError::ShapeInferenceFailed(format!(
            "Raw data size {} does not match expected {} * {}",
            raw.len(),
            expected,
            elem_size
        )));
    }

    match dtype {
        DataType::Int64 => Ok(raw
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
            .collect()),
        DataType::Int32 => Ok(raw
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64)
            .collect()),
        DataType::Uint64 => Ok(raw
            .chunks_exact(8)
            .map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as i64)
            .collect()),
        DataType::Uint8 => Ok(raw.iter().map(|&b| b as i64).collect()),
        DataType::Int8 => Ok(raw.iter().map(|&b| b as i8 as i64).collect()),
        _ => Err(TransformError::InvalidDataType(dtype as i32)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_to_array_f32_float_data() {
        let tensor = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float as i32,
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ..Default::default()
        };

        let array = tensor_to_array_f32(&tensor).unwrap();
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array[[0, 0]], 1.0);
        assert_eq!(array[[1, 2]], 6.0);
    }

    #[test]
    fn test_tensor_to_array_f32_raw_data() {
        let raw: Vec<u8> = vec![1.0f32, 2.0f32]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let tensor = TensorProto {
            dims: vec![2],
            data_type: DataType::Float as i32,
            raw_data: raw,
            ..Default::default()
        };

        let array = tensor_to_array_f32(&tensor).unwrap();
        assert_eq!(array.shape(), &[2]);
        assert_eq!(array[0], 1.0);
        assert_eq!(array[1], 2.0);
    }

    #[test]
    fn test_array_to_tensor_f32() {
        let array = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let tensor = array_to_tensor_f32(&array, "test");

        assert_eq!(tensor.dims, vec![2, 2]);
        assert_eq!(tensor.float_data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(tensor.name, "test");
    }

    #[test]
    fn test_scalar_to_tensor() {
        let tensor = scalar_to_tensor_f32(3.14, "pi");
        assert!(tensor.dims.is_empty());
        assert_eq!(tensor.float_data, vec![3.14]);
    }

    #[test]
    fn test_vec_to_tensor() {
        let tensor = vec_to_tensor_i64(&[1, 2, 3], "axes");
        assert_eq!(tensor.dims, vec![3]);
        assert_eq!(tensor.int64_data, vec![1, 2, 3]);
    }
}
