//! Shape utilities for ONNX tensors
//!
//! Functions for working with tensor shapes and dimensions.

use crate::error::{OnnxResult, TransformError};
use crate::proto::{TensorShapeProto, ValueInfoProto};

/// Calculate total number of elements from shape
pub fn numel(shape: &[i64]) -> usize {
    if shape.is_empty() {
        1 // scalar
    } else {
        shape.iter().map(|&d| d.max(0) as usize).product()
    }
}

/// Check if shape contains dynamic dimensions (negative values)
pub fn is_dynamic(shape: &[i64]) -> bool {
    shape.iter().any(|&d| d < 0)
}

/// Check if two shapes are broadcastable
pub fn is_broadcastable(shape_a: &[i64], shape_b: &[i64]) -> bool {
    let len_a = shape_a.len();
    let len_b = shape_b.len();
    let max_len = len_a.max(len_b);

    for i in 0..max_len {
        // Index from the right (broadcasting aligns from trailing dimensions)
        let idx_a = if i < len_a { len_a - 1 - i } else { usize::MAX };
        let idx_b = if i < len_b { len_b - 1 - i } else { usize::MAX };

        let dim_a = if idx_a != usize::MAX {
            shape_a[idx_a]
        } else {
            1
        };
        let dim_b = if idx_b != usize::MAX {
            shape_b[idx_b]
        } else {
            1
        };

        // Dynamic dims are considered broadcastable
        if dim_a < 0 || dim_b < 0 {
            continue;
        }
        if dim_a != dim_b && dim_a != 1 && dim_b != 1 {
            return false;
        }
    }
    true
}

/// Compute broadcast output shape
pub fn broadcast_shape(shape_a: &[i64], shape_b: &[i64]) -> OnnxResult<Vec<i64>> {
    if !is_broadcastable(shape_a, shape_b) {
        return Err(TransformError::ShapeInferenceFailed(format!(
            "Shapes {:?} and {:?} are not broadcastable",
            shape_a, shape_b
        )));
    }

    let len_a = shape_a.len();
    let len_b = shape_b.len();
    let max_len = len_a.max(len_b);
    let mut result = vec![0i64; max_len];

    for i in 0..max_len {
        // Index from the right (broadcasting aligns from trailing dimensions)
        let idx_a = if i < len_a { len_a - 1 - i } else { usize::MAX };
        let idx_b = if i < len_b { len_b - 1 - i } else { usize::MAX };

        let dim_a = if idx_a != usize::MAX {
            shape_a[idx_a]
        } else {
            1
        };
        let dim_b = if idx_b != usize::MAX {
            shape_b[idx_b]
        } else {
            1
        };

        // Handle dynamic dimensions
        let out_dim = if dim_a < 0 {
            dim_b
        } else if dim_b < 0 {
            dim_a
        } else {
            dim_a.max(dim_b)
        };

        result[max_len - 1 - i] = out_dim;
    }

    Ok(result)
}

/// Extract shape from ValueInfoProto
pub fn shape_from_value_info(vi: &ValueInfoProto) -> Option<Vec<i64>> {
    vi.get_shape()
}

/// Extract shape from TensorShapeProto
pub fn shape_from_proto(shape_proto: &TensorShapeProto) -> Vec<i64> {
    use crate::proto::onnx::tensor_shape_proto::dimension::Value;

    shape_proto
        .dim
        .iter()
        .map(|d| match &d.value {
            Some(Value::DimValue(v)) => *v,
            Some(Value::DimParam(_)) => -1, // symbolic dimension
            None => -1,
        })
        .collect()
}

/// Normalize axis to positive index
pub fn normalize_axis(axis: i64, ndim: usize) -> OnnxResult<usize> {
    let ndim_i64 = ndim as i64;
    let normalized = if axis < 0 { axis + ndim_i64 } else { axis };

    if normalized < 0 || normalized >= ndim_i64 {
        return Err(TransformError::InvalidNode(format!(
            "Axis {} out of bounds for ndim {}",
            axis, ndim
        )));
    }

    Ok(normalized as usize)
}

/// Normalize multiple axes
pub fn normalize_axes(axes: &[i64], ndim: usize) -> OnnxResult<Vec<usize>> {
    axes.iter().map(|&a| normalize_axis(a, ndim)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numel() {
        assert_eq!(numel(&[2, 3, 4]), 24);
        assert_eq!(numel(&[1, 1, 1]), 1);
        assert_eq!(numel(&[]), 1); // scalar
    }

    #[test]
    fn test_is_dynamic() {
        assert!(!is_dynamic(&[1, 3, 224, 224]));
        assert!(is_dynamic(&[-1, 3, 224, 224]));
        assert!(is_dynamic(&[1, -1]));
    }

    #[test]
    fn test_is_broadcastable() {
        assert!(is_broadcastable(&[3, 4], &[4]));
        assert!(is_broadcastable(&[1, 3, 1], &[2, 1, 4]));
        assert!(!is_broadcastable(&[3, 4], &[5]));
    }

    #[test]
    fn test_broadcast_shape() {
        assert_eq!(broadcast_shape(&[3, 4], &[4]).unwrap(), vec![3, 4]);
        assert_eq!(
            broadcast_shape(&[1, 3, 1], &[2, 1, 4]).unwrap(),
            vec![2, 3, 4]
        );
        assert!(broadcast_shape(&[3, 4], &[5]).is_err());
    }

    #[test]
    fn test_normalize_axis() {
        assert_eq!(normalize_axis(0, 4).unwrap(), 0);
        assert_eq!(normalize_axis(-1, 4).unwrap(), 3);
        assert_eq!(normalize_axis(-2, 4).unwrap(), 2);
        assert!(normalize_axis(4, 4).is_err());
        assert!(normalize_axis(-5, 4).is_err());
    }
}
