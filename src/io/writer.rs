//! ONNX model writer
//!
//! Save ONNX models to files or bytes.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use prost::Message;

use crate::error::{OnnxResult, TransformError};
use crate::proto::ModelProto;

/// Save an ONNX model to a file
///
/// # Example
///
/// ```ignore
/// use furiosa_optimizer::io::save_model;
///
/// save_model(&model, "optimized.onnx")?;
/// ```
pub fn save_model<P: AsRef<Path>>(model: &ModelProto, path: P) -> OnnxResult<()> {
    let path = path.as_ref();

    let file = File::create(path).map_err(|e| {
        TransformError::InvalidModel(format!("Failed to create file '{}': {}", path.display(), e))
    })?;

    let mut writer = BufWriter::new(file);
    let bytes = model.encode_to_vec();

    writer.write_all(&bytes).map_err(|e| {
        TransformError::InvalidModel(format!("Failed to write file '{}': {}", path.display(), e))
    })?;

    writer.flush().map_err(|e| {
        TransformError::InvalidModel(format!("Failed to flush file '{}': {}", path.display(), e))
    })?;

    Ok(())
}

/// Encode an ONNX model to bytes
///
/// # Example
///
/// ```ignore
/// use furiosa_optimizer::io::model_to_bytes;
///
/// let bytes = model_to_bytes(&model);
/// ```
pub fn model_to_bytes(model: &ModelProto) -> Vec<u8> {
    model.encode_to_vec()
}

/// Calculate the size of an encoded model in bytes
pub fn model_size(model: &ModelProto) -> usize {
    model.encoded_len()
}

/// Save model with compression statistics
#[derive(Debug, Clone)]
pub struct SaveStats {
    /// Size in bytes
    pub size_bytes: usize,
    /// Number of nodes
    pub node_count: usize,
    /// Number of initializers
    pub initializer_count: usize,
}

/// Save model and return statistics
pub fn save_model_with_stats<P: AsRef<Path>>(model: &ModelProto, path: P) -> OnnxResult<SaveStats> {
    let bytes = model.encode_to_vec();
    let stats = SaveStats {
        size_bytes: bytes.len(),
        node_count: model.graph.as_ref().map(|g| g.node.len()).unwrap_or(0),
        initializer_count: model
            .graph
            .as_ref()
            .map(|g| g.initializer.len())
            .unwrap_or(0),
    };

    let path = path.as_ref();
    let file = File::create(path).map_err(|e| {
        TransformError::InvalidModel(format!("Failed to create file '{}': {}", path.display(), e))
    })?;

    let mut writer = BufWriter::new(file);
    writer.write_all(&bytes).map_err(|e| {
        TransformError::InvalidModel(format!("Failed to write file '{}': {}", path.display(), e))
    })?;

    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::{GraphProto, NodeProto, ValueInfoProto};
    use std::io::Read;

    fn create_test_model() -> ModelProto {
        ModelProto {
            ir_version: 8,
            producer_name: "test".to_string(),
            graph: Some(GraphProto {
                name: "test_graph".to_string(),
                node: vec![NodeProto {
                    op_type: "Relu".to_string(),
                    name: "relu_0".to_string(),
                    input: vec!["X".to_string()],
                    output: vec!["Y".to_string()],
                    ..Default::default()
                }],
                input: vec![ValueInfoProto {
                    name: "X".to_string(),
                    ..Default::default()
                }],
                output: vec![ValueInfoProto {
                    name: "Y".to_string(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn test_model_to_bytes() {
        let model = create_test_model();
        let bytes = model_to_bytes(&model);

        assert!(!bytes.is_empty());

        // Verify we can decode it back
        let decoded = ModelProto::decode(bytes.as_slice()).unwrap();
        assert_eq!(decoded.ir_version, 8);
    }

    #[test]
    fn test_model_size() {
        let model = create_test_model();
        let size = model_size(&model);
        let bytes = model_to_bytes(&model);

        assert_eq!(size, bytes.len());
    }

    #[test]
    fn test_save_and_load() {
        let model = create_test_model();
        let path = "/tmp/test_model.onnx";

        // Save
        save_model(&model, path).unwrap();

        // Load and verify
        let mut file = File::open(path).unwrap();
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();

        let loaded = ModelProto::decode(bytes.as_slice()).unwrap();
        assert_eq!(loaded.ir_version, 8);
        assert_eq!(loaded.producer_name, "test");

        // Cleanup
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_save_with_stats() {
        let model = create_test_model();
        let path = "/tmp/test_model_stats.onnx";

        let stats = save_model_with_stats(&model, path).unwrap();

        assert!(stats.size_bytes > 0);
        assert_eq!(stats.node_count, 1);

        // Cleanup
        std::fs::remove_file(path).ok();
    }
}
