//! ONNX model reader
//!
//! Load ONNX models from files or bytes.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use prost::Message;

use crate::error::{OnnxResult, TransformError};
use crate::proto::ModelProto;

/// Load an ONNX model from a file path
///
/// # Example
///
/// ```ignore
/// use furiosa_optimizer::io::load_model;
///
/// let model = load_model("model.onnx")?;
/// println!("Model IR version: {}", model.ir_version);
/// ```
pub fn load_model<P: AsRef<Path>>(path: P) -> OnnxResult<ModelProto> {
    let path = path.as_ref();

    let file = File::open(path).map_err(|e| {
        TransformError::InvalidModel(format!("Failed to open file '{}': {}", path.display(), e))
    })?;

    let mut reader = BufReader::new(file);
    let mut buffer = Vec::new();

    reader.read_to_end(&mut buffer).map_err(|e| {
        TransformError::InvalidModel(format!("Failed to read file '{}': {}", path.display(), e))
    })?;

    load_model_from_bytes(&buffer)
}

/// Load an ONNX model from bytes
///
/// # Example
///
/// ```ignore
/// use furiosa_optimizer::io::load_model_from_bytes;
///
/// let bytes = std::fs::read("model.onnx")?;
/// let model = load_model_from_bytes(&bytes)?;
/// ```
pub fn load_model_from_bytes(bytes: &[u8]) -> OnnxResult<ModelProto> {
    ModelProto::decode(bytes)
        .map_err(|e| TransformError::InvalidModel(format!("Failed to decode ONNX model: {}", e)))
}

/// Load only the graph from an ONNX model file
///
/// This is useful when you only need the graph structure and not the full model metadata.
pub fn load_graph<P: AsRef<Path>>(path: P) -> OnnxResult<crate::proto::GraphProto> {
    let model = load_model(path)?;
    model
        .graph
        .ok_or_else(|| TransformError::InvalidModel("Model does not contain a graph".to_string()))
}

/// Model metadata extracted from ONNX file
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// IR version
    pub ir_version: i64,
    /// Producer name
    pub producer_name: String,
    /// Producer version
    pub producer_version: String,
    /// Model version
    pub model_version: i64,
    /// Domain
    pub domain: String,
    /// Doc string
    pub doc_string: String,
    /// Opset imports
    pub opsets: Vec<(String, i64)>,
    /// Graph name
    pub graph_name: String,
    /// Number of nodes
    pub node_count: usize,
    /// Number of initializers
    pub initializer_count: usize,
    /// Input names
    pub inputs: Vec<String>,
    /// Output names
    pub outputs: Vec<String>,
}

impl ModelInfo {
    /// Extract metadata from a model
    pub fn from_model(model: &ModelProto) -> Self {
        let graph = model.graph.as_ref();

        Self {
            ir_version: model.ir_version,
            producer_name: model.producer_name.clone(),
            producer_version: model.producer_version.clone(),
            model_version: model.model_version,
            domain: model.domain.clone(),
            doc_string: model.doc_string.clone(),
            opsets: model
                .opset_import
                .iter()
                .map(|op| (op.domain.clone(), op.version))
                .collect(),
            graph_name: graph.map(|g| g.name.clone()).unwrap_or_default(),
            node_count: graph.map(|g| g.node.len()).unwrap_or(0),
            initializer_count: graph.map(|g| g.initializer.len()).unwrap_or(0),
            inputs: graph
                .map(|g| g.input.iter().map(|i| i.name.clone()).collect())
                .unwrap_or_default(),
            outputs: graph
                .map(|g| g.output.iter().map(|o| o.name.clone()).collect())
                .unwrap_or_default(),
        }
    }
}

/// Get model information without fully parsing
pub fn get_model_info<P: AsRef<Path>>(path: P) -> OnnxResult<ModelInfo> {
    let model = load_model(path)?;
    Ok(ModelInfo::from_model(&model))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::{GraphProto, NodeProto, ValueInfoProto};

    fn create_test_model() -> ModelProto {
        ModelProto {
            ir_version: 8,
            producer_name: "test".to_string(),
            producer_version: "1.0".to_string(),
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
    fn test_load_from_bytes() {
        let model = create_test_model();
        let bytes = model.encode_to_vec();

        let loaded = load_model_from_bytes(&bytes).unwrap();
        assert_eq!(loaded.ir_version, 8);
        assert_eq!(loaded.producer_name, "test");
    }

    #[test]
    fn test_model_info() {
        let model = create_test_model();
        let info = ModelInfo::from_model(&model);

        assert_eq!(info.ir_version, 8);
        assert_eq!(info.producer_name, "test");
        assert_eq!(info.graph_name, "test_graph");
        assert_eq!(info.node_count, 1);
        assert_eq!(info.inputs, vec!["X"]);
        assert_eq!(info.outputs, vec!["Y"]);
    }

    #[test]
    fn test_load_invalid_bytes() {
        let result = load_model_from_bytes(&[0, 1, 2, 3]);
        assert!(result.is_err());
    }
}
