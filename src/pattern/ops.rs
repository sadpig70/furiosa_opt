//! Common ONNX operation patterns
//!
//! Pre-defined patterns for common optimization scenarios.

/// Conv followed by BatchNormalization (for fusion)
pub const CONV_BN: &[&str] = &["BatchNormalization", "Conv"];

/// ConvTranspose followed by BatchNormalization
pub const CONV_TRANSPOSE_BN: &[&str] = &["BatchNormalization", "ConvTranspose"];

/// Relu followed by BatchNormalization followed by Conv
pub const RELU_BN_CONV: &[&str] = &["Relu", "BatchNormalization", "Conv"];

/// PReLU (for conversion to Relu-based)
pub const PRELU: &[&str] = &["PRelu"];

/// Gather followed by MatMul (for fusion)
pub const GATHER_MATMUL: &[&str] = &["MatMul", "Gather"];

/// Squeeze (for axes inference)
pub const SQUEEZE: &[&str] = &["Squeeze"];

/// Pad operations
pub const PAD: &[&str] = &["Pad"];

/// Pad followed by Conv (for fusion)
pub const PAD_CONV: &[&str] = &["Conv", "Pad"];

/// Conv followed by Add (for fusion)
pub const CONV_ADD: &[&str] = &["Add", "Conv"];

/// Conv followed by Mul then Add (for fusion)
pub const CONV_MUL_ADD: &[&str] = &["Add", "Mul", "Conv"];

/// Activation operations
pub const ACTIVATIONS: &[&str] = &[
    "Relu",
    "Sigmoid",
    "Tanh",
    "LeakyRelu",
    "Elu",
    "Selu",
    "Softmax",
];

/// Convolution-like operations
pub const CONV_LIKE: &[&str] = &["Conv", "ConvTranspose"];

/// Normalization operations
pub const NORM_OPS: &[&str] = &[
    "BatchNormalization",
    "InstanceNormalization",
    "LayerNormalization",
];

/// Pooling operations
pub const POOL_OPS: &[&str] = &[
    "MaxPool",
    "AveragePool",
    "GlobalAveragePool",
    "GlobalMaxPool",
];

/// Element-wise binary operations
pub const BINARY_OPS: &[&str] = &["Add", "Sub", "Mul", "Div", "Pow"];

/// Reduction operations
pub const REDUCE_OPS: &[&str] = &[
    "ReduceSum",
    "ReduceMean",
    "ReduceMax",
    "ReduceMin",
    "ReduceProd",
    "ReduceL1",
    "ReduceL2",
];

/// Shape manipulation operations
pub const SHAPE_OPS: &[&str] = &["Reshape", "Transpose", "Squeeze", "Unsqueeze", "Flatten"];

/// Pattern builder for creating custom patterns
#[derive(Debug, Clone)]
pub struct PatternBuilder {
    ops: Vec<String>,
}

impl PatternBuilder {
    /// Create a new pattern builder
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Add an op type to the pattern
    pub fn op(mut self, op_type: &str) -> Self {
        self.ops.push(op_type.to_string());
        self
    }

    /// Add multiple op types
    pub fn ops(mut self, op_types: &[&str]) -> Self {
        self.ops.extend(op_types.iter().map(|s| s.to_string()));
        self
    }

    /// Build the pattern as a Vec<String>
    pub fn build(self) -> Vec<String> {
        self.ops
    }

    /// Get the pattern as references (for temporary use)
    pub fn as_slice(&self) -> Vec<&str> {
        self.ops.iter().map(|s| s.as_str()).collect()
    }
}

impl Default for PatternBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if an op type is a convolution-like operation
pub fn is_conv_like(op_type: &str) -> bool {
    CONV_LIKE.contains(&op_type)
}

/// Check if an op type is a normalization operation
pub fn is_norm_op(op_type: &str) -> bool {
    NORM_OPS.contains(&op_type)
}

/// Check if an op type is an activation function
pub fn is_activation(op_type: &str) -> bool {
    ACTIVATIONS.contains(&op_type)
}

/// Check if an op type is a pooling operation
pub fn is_pool_op(op_type: &str) -> bool {
    POOL_OPS.contains(&op_type)
}

/// Check if an op type is a binary element-wise operation
pub fn is_binary_op(op_type: &str) -> bool {
    BINARY_OPS.contains(&op_type)
}

/// Check if an op type is a reduction operation
pub fn is_reduce_op(op_type: &str) -> bool {
    REDUCE_OPS.contains(&op_type)
}

/// Check if an op type is a shape manipulation operation
pub fn is_shape_op(op_type: &str) -> bool {
    SHAPE_OPS.contains(&op_type)
}

/// Op categories for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpCategory {
    /// Convolution operations
    Convolution,
    /// Normalization operations
    Normalization,
    /// Activation functions
    Activation,
    /// Pooling operations
    Pooling,
    /// Binary element-wise operations
    Binary,
    /// Reduction operations
    Reduction,
    /// Shape manipulation
    Shape,
    /// Unknown/Other
    Other,
}

/// Categorize an op type
pub fn categorize_op(op_type: &str) -> OpCategory {
    if is_conv_like(op_type) {
        OpCategory::Convolution
    } else if is_norm_op(op_type) {
        OpCategory::Normalization
    } else if is_activation(op_type) {
        OpCategory::Activation
    } else if is_pool_op(op_type) {
        OpCategory::Pooling
    } else if is_binary_op(op_type) {
        OpCategory::Binary
    } else if is_reduce_op(op_type) {
        OpCategory::Reduction
    } else if is_shape_op(op_type) {
        OpCategory::Shape
    } else {
        OpCategory::Other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_constants() {
        assert_eq!(CONV_BN.len(), 2);
        assert_eq!(CONV_BN[0], "BatchNormalization");
        assert_eq!(CONV_BN[1], "Conv");
    }

    #[test]
    fn test_pattern_builder() {
        let pattern = PatternBuilder::new()
            .op("Relu")
            .op("BatchNormalization")
            .op("Conv")
            .build();

        assert_eq!(pattern, vec!["Relu", "BatchNormalization", "Conv"]);
    }

    #[test]
    fn test_pattern_builder_ops() {
        let pattern = PatternBuilder::new()
            .ops(&["Relu", "BatchNormalization"])
            .op("Conv")
            .build();

        assert_eq!(pattern.len(), 3);
    }

    #[test]
    fn test_is_conv_like() {
        assert!(is_conv_like("Conv"));
        assert!(is_conv_like("ConvTranspose"));
        assert!(!is_conv_like("Relu"));
    }

    #[test]
    fn test_is_activation() {
        assert!(is_activation("Relu"));
        assert!(is_activation("Sigmoid"));
        assert!(!is_activation("Conv"));
    }

    #[test]
    fn test_categorize_op() {
        assert_eq!(categorize_op("Conv"), OpCategory::Convolution);
        assert_eq!(
            categorize_op("BatchNormalization"),
            OpCategory::Normalization
        );
        assert_eq!(categorize_op("Relu"), OpCategory::Activation);
        assert_eq!(categorize_op("MaxPool"), OpCategory::Pooling);
        assert_eq!(categorize_op("Add"), OpCategory::Binary);
        assert_eq!(categorize_op("ReduceSum"), OpCategory::Reduction);
        assert_eq!(categorize_op("Reshape"), OpCategory::Shape);
        assert_eq!(categorize_op("CustomOp"), OpCategory::Other);
    }
}
