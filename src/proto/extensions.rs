//! Extension methods for ONNX protobuf types
//!
//! Provides convenient helper methods for working with ONNX protobuf types.

use super::onnx::*;

// ============================================================================
// ModelProto extensions
// ============================================================================

impl ModelProto {
    /// Get the opset version for the default domain
    pub fn get_opset_version(&self) -> Option<i64> {
        self.opset_import
            .iter()
            .find(|op| op.domain.is_empty())
            .map(|op| op.version)
    }

    /// Check if the model has a graph
    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    /// Get mutable reference to graph, creating if necessary
    pub fn graph_mut(&mut self) -> &mut GraphProto {
        self.graph.get_or_insert_with(GraphProto::default)
    }
}

// ============================================================================
// NodeProto extensions
// ============================================================================

impl NodeProto {
    /// Get attribute by name
    pub fn get_attribute(&self, name: &str) -> Option<&AttributeProto> {
        self.attribute.iter().find(|attr| attr.name == name)
    }

    /// Get integer attribute value with default
    pub fn get_attribute_int(&self, name: &str, default: i64) -> i64 {
        self.get_attribute(name).map(|a| a.i).unwrap_or(default)
    }

    /// Get float attribute value with default
    pub fn get_attribute_float(&self, name: &str, default: f32) -> f32 {
        self.get_attribute(name).map(|a| a.f).unwrap_or(default)
    }

    /// Get string attribute value
    pub fn get_attribute_string(&self, name: &str) -> Option<&[u8]> {
        self.get_attribute(name).map(|a| a.s.as_slice())
    }

    /// Get repeated int attribute
    pub fn get_attribute_ints(&self, name: &str) -> Option<&[i64]> {
        self.get_attribute(name).map(|a| a.ints.as_slice())
    }

    /// Check if this node has a specific op type
    pub fn is_op_type(&self, op_type: &str) -> bool {
        self.op_type == op_type
    }

    /// Check if this node's op type is in the given list
    pub fn is_op_type_in(&self, op_types: &[&str]) -> bool {
        op_types.contains(&self.op_type.as_str())
    }
}

// ============================================================================
// ValueInfoProto extensions
// ============================================================================

impl ValueInfoProto {
    /// Get the shape dimensions if available
    pub fn get_shape(&self) -> Option<Vec<i64>> {
        self.r#type.as_ref().and_then(|t| {
            t.value.as_ref().and_then(|v| match v {
                type_proto::Value::TensorType(tensor) => tensor.shape.as_ref().map(|s| {
                    s.dim
                        .iter()
                        .map(|d| match &d.value {
                            Some(tensor_shape_proto::dimension::Value::DimValue(v)) => *v,
                            Some(tensor_shape_proto::dimension::Value::DimParam(_)) => -1,
                            None => -1,
                        })
                        .collect()
                }),
                _ => None,
            })
        })
    }

    /// Get the element type if this is a tensor type
    pub fn get_elem_type(&self) -> Option<i32> {
        self.r#type.as_ref().and_then(|t| {
            t.value.as_ref().and_then(|v| match v {
                type_proto::Value::TensorType(tensor) => Some(tensor.elem_type),
                _ => None,
            })
        })
    }
}

// ============================================================================
// TensorProto extensions
// ============================================================================

impl TensorProto {
    /// Get the total number of elements
    pub fn num_elements(&self) -> usize {
        if self.dims.is_empty() {
            1 // scalar
        } else {
            self.dims.iter().map(|&d| d as usize).product()
        }
    }

    /// Check if this tensor has raw_data
    pub fn has_raw_data(&self) -> bool {
        !self.raw_data.is_empty()
    }

    /// Get data type enum value
    pub fn data_type_enum(&self) -> tensor_proto::DataType {
        tensor_proto::DataType::try_from(self.data_type)
            .unwrap_or(tensor_proto::DataType::Undefined)
    }
}

// ============================================================================
// AttributeProto extensions
// ============================================================================

impl AttributeProto {
    /// Create a new integer attribute
    pub fn new_int(name: &str, value: i64) -> Self {
        Self {
            name: name.to_string(),
            i: value,
            r#type: attribute_proto::AttributeType::Int as i32,
            ..Default::default()
        }
    }

    /// Create a new float attribute
    pub fn new_float(name: &str, value: f32) -> Self {
        Self {
            name: name.to_string(),
            f: value,
            r#type: attribute_proto::AttributeType::Float as i32,
            ..Default::default()
        }
    }

    /// Create a new ints attribute
    pub fn new_ints(name: &str, values: Vec<i64>) -> Self {
        Self {
            name: name.to_string(),
            ints: values,
            r#type: attribute_proto::AttributeType::Ints as i32,
            ..Default::default()
        }
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Create a new ValueInfoProto for a tensor
pub fn make_tensor_value_info(name: &str, elem_type: i32, shape: &[i64]) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type,
                shape: Some(TensorShapeProto {
                    dim: shape
                        .iter()
                        .map(|&d| tensor_shape_proto::Dimension {
                            value: Some(tensor_shape_proto::dimension::Value::DimValue(d)),
                            denotation: String::new(),
                        })
                        .collect(),
                }),
            })),
            denotation: String::new(),
        }),
        doc_string: String::new(),
    }
}

/// Create a new NodeProto
pub fn make_node(op_type: &str, inputs: &[&str], outputs: &[&str], name: &str) -> NodeProto {
    NodeProto {
        op_type: op_type.to_string(),
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: outputs.iter().map(|s| s.to_string()).collect(),
        name: name.to_string(),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_get_attribute() {
        let mut node = NodeProto::default();
        node.attribute.push(AttributeProto::new_int("axis", 1));

        assert_eq!(node.get_attribute_int("axis", 0), 1);
        assert_eq!(node.get_attribute_int("missing", 99), 99);
    }

    #[test]
    fn test_make_tensor_value_info() {
        let vi = make_tensor_value_info("test", 1, &[1, 3, 224, 224]);
        assert_eq!(vi.name, "test");
        assert_eq!(vi.get_shape(), Some(vec![1, 3, 224, 224]));
    }

    #[test]
    fn test_make_node() {
        let node = make_node("Conv", &["X", "W"], &["Y"], "conv_0");
        assert_eq!(node.op_type, "Conv");
        assert_eq!(node.input, vec!["X", "W"]);
        assert_eq!(node.output, vec!["Y"]);
    }
}
