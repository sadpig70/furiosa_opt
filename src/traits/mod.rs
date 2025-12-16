//! Core traits for furiosa-optimizer
//!
//! Defines the fundamental interfaces for model transformation.

use crate::error::OnnxResult;
use crate::proto::ModelProto;

/// Transformer trait for model transformations
///
/// This is the core abstraction for all model-level transformations.
/// Implementations must be able to transform a model and return a new (optimized) model.
///
/// # Example
///
/// ```ignore
/// struct MyTransformer;
///
/// impl Transformer for MyTransformer {
///     fn transform(&self, model: ModelProto) -> OnnxResult<ModelProto> {
///         // Transform the model
///         Ok(model)
///     }
/// }
/// ```
pub trait Transformer {
    /// Transform the given model
    ///
    /// # Arguments
    /// * `model` - The input ONNX model
    ///
    /// # Returns
    /// * `OnnxResult<ModelProto>` - The transformed model or an error
    fn transform(&self, model: ModelProto) -> OnnxResult<ModelProto>;
}

/// Pattern matching trait for ONNX graph patterns
///
/// Implementations define specific graph patterns to match and transform.
pub trait OnnxPattern {
    /// The pattern to match (e.g., ["Conv", "BatchNormalization"])
    fn pattern(&self) -> &[&str];

    /// Check if additional conditions are met after pattern matching
    ///
    /// # Arguments
    /// * `matched_nodes` - The nodes that matched the pattern
    ///
    /// # Returns
    /// * `bool` - True if conditions are satisfied
    fn condition_check(&self, matched_nodes: &[crate::proto::NodeProto]) -> bool {
        // Default: no additional conditions
        !matched_nodes.is_empty()
    }
}

/// Chainable transformer that applies multiple transformers in sequence
pub struct TransformerChain {
    transformers: Vec<Box<dyn Transformer>>,
}

impl TransformerChain {
    /// Create a new empty transformer chain
    pub fn new() -> Self {
        Self {
            transformers: Vec::new(),
        }
    }

    /// Add a transformer to the chain
    #[allow(clippy::should_implement_trait)]
    pub fn add<T: Transformer + 'static>(mut self, transformer: T) -> Self {
        self.transformers.push(Box::new(transformer));
        self
    }
}

impl Default for TransformerChain {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for TransformerChain {
    fn transform(&self, mut model: ModelProto) -> OnnxResult<ModelProto> {
        for transformer in &self.transformers {
            model = transformer.transform(model)?;
        }
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct IdentityTransformer;

    impl Transformer for IdentityTransformer {
        fn transform(&self, model: ModelProto) -> OnnxResult<ModelProto> {
            Ok(model)
        }
    }

    #[test]
    fn test_identity_transformer() {
        let transformer = IdentityTransformer;
        let model = ModelProto::default();
        let result = transformer.transform(model);
        assert!(result.is_ok());
    }

    #[test]
    fn test_transformer_chain() {
        let chain = TransformerChain::new()
            .add(IdentityTransformer)
            .add(IdentityTransformer);

        let model = ModelProto::default();
        let result = chain.transform(model);
        assert!(result.is_ok());
    }
}
