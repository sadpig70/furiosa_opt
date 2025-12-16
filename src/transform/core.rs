//! Core transformation loop
//!
//! Implements the main transformation loop that iterates over the graph
//! and applies pattern-based transformations.

use crate::error::{OnnxResult, TransformError};
use crate::graph::GraphContext;
use crate::pattern::PatternMatcher;
use crate::proto::{GraphProto, ModelProto, NodeProto};

/// Matched nodes by name (to avoid borrowing issues)
#[derive(Debug, Clone)]
pub struct MatchedNodes {
    /// Names of matched nodes in pattern order
    pub names: Vec<String>,
}

impl MatchedNodes {
    /// Get the first (anchor) node name
    pub fn anchor(&self) -> Option<&str> {
        self.names.first().map(|s| s.as_str())
    }

    /// Get all node names
    pub fn as_slice(&self) -> &[String] {
        &self.names
    }

    /// Number of matched nodes
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Get node name at index
    pub fn get(&self, index: usize) -> Option<&str> {
        self.names.get(index).map(|s| s.as_str())
    }
}

/// Transform configuration
#[derive(Debug, Clone)]
pub struct TransformConfig {
    /// Maximum iterations to prevent infinite loops
    pub max_iterations: usize,
    /// Whether to continue on individual transform errors
    pub continue_on_error: bool,
    /// Whether to run cleanup after transformation
    pub cleanup_after: bool,
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            continue_on_error: false,
            cleanup_after: true,
        }
    }
}

/// Statistics from a transform run
#[derive(Debug, Default, Clone)]
pub struct TransformStats {
    /// Number of iterations performed
    pub iterations: usize,
    /// Number of patterns matched
    pub patterns_matched: usize,
    /// Number of transformations applied
    pub transforms_applied: usize,
    /// Number of nodes eliminated
    pub nodes_eliminated: usize,
}

/// Main transformation engine
///
/// Iterates over the graph in reverse order (output → input) and applies
/// pattern-based transformations.
pub struct TransformEngine<'a> {
    ctx: GraphContext,
    config: TransformConfig,
    stats: TransformStats,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> TransformEngine<'a> {
    /// Create a new transform engine from a graph
    pub fn new(graph: &GraphProto) -> Self {
        Self {
            ctx: GraphContext::new(graph),
            config: TransformConfig::default(),
            stats: TransformStats::default(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Create from a model
    pub fn from_model(model: &ModelProto) -> OnnxResult<Self> {
        let graph = model
            .graph
            .as_ref()
            .ok_or_else(|| TransformError::MissingField("model.graph".to_string()))?;

        Ok(Self::new(graph))
    }

    /// Configure the engine
    pub fn with_config(mut self, config: TransformConfig) -> Self {
        self.config = config;
        self
    }

    /// Get the current context
    pub fn context(&self) -> &GraphContext {
        &self.ctx
    }

    /// Get mutable context
    pub fn context_mut(&mut self) -> &mut GraphContext {
        &mut self.ctx
    }

    /// Get statistics
    pub fn stats(&self) -> &TransformStats {
        &self.stats
    }

    /// Apply a single transformation based on pattern matching
    ///
    /// This is the core of Python's `transform` method.
    ///
    /// # Arguments
    /// * `pattern` - The pattern to match
    /// * `transform_fn` - Function to apply when pattern matches
    ///
    /// # Returns
    /// * Number of transformations applied
    pub fn apply_pattern<F>(&mut self, pattern: &[&str], mut transform_fn: F) -> OnnxResult<usize>
    where
        F: FnMut(&mut GraphContext, MatchedNodes) -> OnnxResult<bool>,
    {
        let mut applied = 0;
        let mut iteration = 0;

        loop {
            if iteration >= self.config.max_iterations {
                break;
            }
            iteration += 1;

            // Collect matches as node names (to avoid borrowing issues)
            let matches: Vec<Vec<String>> = {
                let matcher = PatternMatcher::new(&self.ctx);
                matcher
                    .find_all_matches(pattern)
                    .into_iter()
                    .map(|m| m.node_names().into_iter().map(|s| s.to_string()).collect())
                    .collect()
            };

            if matches.is_empty() {
                break;
            }

            self.stats.patterns_matched += matches.len();

            let mut any_applied = false;
            for node_names in matches {
                // Skip already eliminated nodes
                if node_names.iter().any(|n| self.ctx.is_eliminated(n)) {
                    continue;
                }

                let matched = MatchedNodes { names: node_names };

                match transform_fn(&mut self.ctx, matched) {
                    Ok(true) => {
                        applied += 1;
                        any_applied = true;
                        self.stats.transforms_applied += 1;
                    }
                    Ok(false) => {
                        // Transform decided not to apply
                    }
                    Err(e) => {
                        if !self.config.continue_on_error {
                            return Err(e);
                        }
                    }
                }
            }

            if !any_applied {
                break;
            }
        }

        self.stats.iterations = iteration;
        Ok(applied)
    }

    /// Apply transformation with condition
    pub fn apply_pattern_if<F, C>(
        &mut self,
        pattern: &[&str],
        condition: C,
        mut transform_fn: F,
    ) -> OnnxResult<usize>
    where
        F: FnMut(&mut GraphContext, MatchedNodes) -> OnnxResult<bool>,
        C: Fn(&MatchedNodes, &GraphContext) -> bool,
    {
        self.apply_pattern(pattern, |ctx, m| {
            if condition(&m, ctx) {
                transform_fn(ctx, m)
            } else {
                Ok(false)
            }
        })
    }

    /// Build the final optimized graph
    pub fn build_graph(self) -> GraphProto {
        build_optimized_graph(&self.ctx)
    }

    /// Build the final optimized model
    pub fn build_model(self, original: &ModelProto) -> ModelProto {
        let mut model = original.clone();
        model.graph = Some(self.build_graph());
        model
    }
}

/// Build optimized graph from context
///
/// Collects all non-eliminated nodes and creates a new GraphProto.
pub fn build_optimized_graph(ctx: &GraphContext) -> GraphProto {
    let nodes: Vec<NodeProto> = ctx.active_nodes().cloned().collect();

    // Collect used tensors
    let mut used_tensors = std::collections::HashSet::new();
    for node in &nodes {
        for input in &node.input {
            used_tensors.insert(input.as_str());
        }
        for output in &node.output {
            used_tensors.insert(output.as_str());
        }
    }

    // Filter initializers
    let initializers: Vec<_> = ctx
        .initializer_map
        .values()
        .filter(|t| used_tensors.contains(t.name.as_str()))
        .cloned()
        .collect();

    // Keep graph inputs/outputs
    let inputs: Vec<_> = ctx.graph_input_map.values().cloned().collect();
    let outputs: Vec<_> = ctx.graph_output_map.values().cloned().collect();

    // Filter value_info
    let value_info: Vec<_> = ctx
        .value_info_map
        .values()
        .filter(|vi| {
            used_tensors.contains(vi.name.as_str())
                && !ctx.graph_input_map.contains_key(&vi.name)
                && !ctx.graph_output_map.contains_key(&vi.name)
        })
        .cloned()
        .collect();

    GraphProto {
        node: nodes,
        initializer: initializers,
        input: inputs,
        output: outputs,
        value_info,
        ..Default::default()
    }
}

/// Single-shot transform: apply pattern once to all matching locations
pub fn transform_once<F>(
    ctx: &mut GraphContext,
    pattern: &[&str],
    mut transform_fn: F,
) -> OnnxResult<usize>
where
    F: FnMut(&mut GraphContext, MatchedNodes) -> OnnxResult<bool>,
{
    // Collect matches as node names first
    let matches: Vec<Vec<String>> = {
        let matcher = PatternMatcher::new(ctx);
        matcher
            .find_all_matches(pattern)
            .into_iter()
            .map(|m| m.node_names().into_iter().map(|s| s.to_string()).collect())
            .collect()
    };

    let mut applied = 0;
    for node_names in matches {
        if node_names.iter().any(|n| ctx.is_eliminated(n)) {
            continue;
        }

        let matched = MatchedNodes { names: node_names };
        if transform_fn(ctx, matched)? {
            applied += 1;
        }
    }

    Ok(applied)
}

/// Transform all matching patterns until no more matches
pub fn transform_until_fixed_point<F>(
    ctx: &mut GraphContext,
    pattern: &[&str],
    mut transform_fn: F,
    max_iterations: usize,
) -> OnnxResult<usize>
where
    F: FnMut(&mut GraphContext, MatchedNodes) -> OnnxResult<bool>,
{
    let mut total = 0;

    for _ in 0..max_iterations {
        let applied = transform_once(ctx, pattern, &mut transform_fn)?;
        if applied == 0 {
            break;
        }
        total += applied;
    }

    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::ValueInfoProto;
    use crate::transform::eliminate::eliminate_node;

    fn make_test_graph() -> GraphProto {
        GraphProto {
            node: vec![
                make_node("Conv", &["X", "W"], &["conv_out"], "conv_0"),
                make_node("Identity", &["conv_out"], &["id_out"], "identity_0"),
                make_node("Relu", &["id_out"], &["relu_out"], "relu_0"),
                make_node("Identity", &["relu_out"], &["Y"], "identity_1"),
            ],
            input: vec![ValueInfoProto {
                name: "X".to_string(),
                ..Default::default()
            }],
            output: vec![ValueInfoProto {
                name: "Y".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn test_transform_engine_creation() {
        let graph = make_test_graph();
        let engine = TransformEngine::new(&graph);

        assert_eq!(engine.context().node_count(), 4);
    }

    #[test]
    fn test_apply_pattern() {
        let graph = make_test_graph();
        let mut engine = TransformEngine::new(&graph);

        // Remove Identity nodes
        let applied = engine.apply_pattern(&["Identity"], |ctx, m| {
            let node_name = m.get(0).unwrap();
            eliminate_node(ctx, node_name, 0);
            Ok(true)
        });

        assert!(applied.is_ok());
        assert_eq!(applied.unwrap(), 2); // Two Identity nodes

        // Verify only 2 nodes remain
        assert_eq!(engine.context().active_node_count(), 2);
    }

    #[test]
    fn test_build_optimized_graph() {
        let graph = make_test_graph();
        let mut engine = TransformEngine::new(&graph);

        // Eliminate identity nodes
        engine
            .apply_pattern(&["Identity"], |ctx, m| {
                eliminate_node(ctx, m.get(0).unwrap(), 0);
                Ok(true)
            })
            .unwrap();

        let optimized = engine.build_graph();

        // Should have 2 nodes (Conv, Relu)
        assert_eq!(optimized.node.len(), 2);
        assert_eq!(optimized.node[0].op_type, "Conv");
        assert_eq!(optimized.node[1].op_type, "Relu");
    }

    #[test]
    fn test_transform_once() {
        let graph = make_test_graph();
        let mut ctx = GraphContext::new(&graph);

        let applied = transform_once(&mut ctx, &["Identity"], |ctx, m| {
            eliminate_node(ctx, m.get(0).unwrap(), 0);
            Ok(true)
        });

        assert_eq!(applied.unwrap(), 2);
    }

    #[test]
    fn test_transform_stats() {
        let graph = make_test_graph();
        let mut engine = TransformEngine::new(&graph);

        engine
            .apply_pattern(&["Identity"], |ctx, m| {
                eliminate_node(ctx, m.get(0).unwrap(), 0);
                Ok(true)
            })
            .unwrap();

        let stats = engine.stats();
        assert_eq!(stats.transforms_applied, 2);
    }
}
