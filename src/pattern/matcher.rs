//! Pattern matching engine for ONNX graphs
//!
//! Implements reverse (output-to-input) pattern matching as used in
//! Python's ONNXTransformer.pattern_matcher.

use crate::graph::GraphContext;
use crate::proto::NodeProto;

/// Result of a successful pattern match
#[derive(Debug, Clone)]
pub struct MatchResult<'a> {
    /// Matched nodes in pattern order (first = anchor node, last = earliest in graph)
    pub nodes: Vec<&'a NodeProto>,
    /// The anchor node name (where the match started)
    pub anchor: &'a str,
}

impl<'a> MatchResult<'a> {
    /// Get the first matched node (anchor)
    pub fn first(&self) -> Option<&'a NodeProto> {
        self.nodes.first().copied()
    }

    /// Get the last matched node (earliest in graph)
    pub fn last(&self) -> Option<&'a NodeProto> {
        self.nodes.last().copied()
    }

    /// Get node at index
    pub fn get(&self, index: usize) -> Option<&'a NodeProto> {
        self.nodes.get(index).copied()
    }

    /// Number of matched nodes
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get all node names
    pub fn node_names(&self) -> Vec<&'a str> {
        self.nodes.iter().map(|n| n.name.as_str()).collect()
    }
}

/// Pattern matcher for ONNX graphs
///
/// Matches sequences of op_types in reverse order (output → input direction).
/// This mirrors Python's `ONNXTransformer.pattern_matcher`.
pub struct PatternMatcher<'a> {
    ctx: &'a GraphContext,
}

impl<'a> PatternMatcher<'a> {
    /// Create a new pattern matcher
    pub fn new(ctx: &'a GraphContext) -> Self {
        Self { ctx }
    }

    /// Match a pattern starting from the given node
    ///
    /// Pattern is matched in reverse order: first element matches the anchor node,
    /// subsequent elements match predecessor nodes.
    ///
    /// # Arguments
    /// * `node` - The anchor node to start matching from
    /// * `pattern` - Slice of op_types to match (e.g., `["Relu", "BatchNormalization", "Conv"]`)
    ///
    /// # Returns
    /// * `Some(MatchResult)` if pattern matches
    /// * `None` if pattern does not match
    ///
    /// # Example
    /// ```ignore
    /// // Match Relu <- BatchNorm <- Conv pattern
    /// let result = matcher.match_pattern(relu_node, &["Relu", "BatchNormalization", "Conv"]);
    /// ```
    pub fn match_pattern(&self, node: &'a NodeProto, pattern: &[&str]) -> Option<MatchResult<'a>> {
        if pattern.is_empty() {
            return None;
        }

        let mut matched = Vec::with_capacity(pattern.len());
        let mut current: Option<&NodeProto> = Some(node);

        for &op_type in pattern {
            match current {
                Some(n) if n.op_type == op_type => {
                    matched.push(n);
                    // Move to previous node (producer of first input)
                    current = self.ctx.get_prev_node(n);
                }
                _ => return None,
            }
        }

        Some(MatchResult {
            anchor: &node.name,
            nodes: matched,
        })
    }

    /// Match pattern with additional condition check
    ///
    /// # Arguments
    /// * `node` - The anchor node
    /// * `pattern` - Op types to match
    /// * `condition` - Additional condition to check on matched nodes
    pub fn match_pattern_with_condition<F>(
        &self,
        node: &'a NodeProto,
        pattern: &[&str],
        condition: F,
    ) -> Option<MatchResult<'a>>
    where
        F: FnOnce(&[&NodeProto]) -> bool,
    {
        let result = self.match_pattern(node, pattern)?;
        if condition(&result.nodes) {
            Some(result)
        } else {
            None
        }
    }

    /// Find all matches of a pattern in the graph
    ///
    /// Scans all nodes and returns matches where the anchor node matches
    /// the first element of the pattern.
    pub fn find_all_matches(&self, pattern: &[&str]) -> Vec<MatchResult<'a>> {
        if pattern.is_empty() {
            return Vec::new();
        }

        let anchor_op = pattern[0];
        let mut results = Vec::new();

        for node in self.ctx.nodes() {
            if node.op_type == anchor_op {
                if let Some(result) = self.match_pattern(node, pattern) {
                    results.push(result);
                }
            }
        }

        results
    }

    /// Find all matches with condition
    pub fn find_all_matches_with_condition<F>(
        &self,
        pattern: &[&str],
        condition: F,
    ) -> Vec<MatchResult<'a>>
    where
        F: Fn(&[&NodeProto]) -> bool,
    {
        if pattern.is_empty() {
            return Vec::new();
        }

        let anchor_op = pattern[0];
        let mut results = Vec::new();

        for node in self.ctx.nodes() {
            if node.op_type == anchor_op {
                if let Some(result) = self.match_pattern(node, pattern) {
                    if condition(&result.nodes) {
                        results.push(result);
                    }
                }
            }
        }

        results
    }

    /// Match a single op type
    pub fn match_single(&self, node: &'a NodeProto, op_type: &str) -> bool {
        node.op_type == op_type
    }

    /// Match any of the given op types
    pub fn match_any(&self, node: &'a NodeProto, op_types: &[&str]) -> bool {
        op_types.contains(&node.op_type.as_str())
    }

    /// Check if a tensor between two nodes is single-use
    ///
    /// Used to verify fusion safety.
    pub fn is_fusible_connection(&self, producer: &NodeProto, consumer: &NodeProto) -> bool {
        // Get the connecting tensor
        if let Some(output) = producer.output.first() {
            // Check if consumer uses this output
            if consumer.input.contains(output) {
                // Check if it's single-use
                return self.ctx.is_single_use(output);
            }
        }
        false
    }
}

/// Convenience function to create a pattern matcher
pub fn matcher(ctx: &GraphContext) -> PatternMatcher<'_> {
    PatternMatcher::new(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::{GraphProto, TensorProto, ValueInfoProto};

    fn make_conv_bn_relu_graph() -> GraphProto {
        GraphProto {
            node: vec![
                make_node("Conv", &["X", "W"], &["conv_out"], "conv_0"),
                make_node(
                    "BatchNormalization",
                    &["conv_out", "scale", "B", "mean", "var"],
                    &["bn_out"],
                    "bn_0",
                ),
                make_node("Relu", &["bn_out"], &["Y"], "relu_0"),
            ],
            input: vec![ValueInfoProto {
                name: "X".to_string(),
                ..Default::default()
            }],
            output: vec![ValueInfoProto {
                name: "Y".to_string(),
                ..Default::default()
            }],
            initializer: vec![
                TensorProto {
                    name: "W".to_string(),
                    ..Default::default()
                },
                TensorProto {
                    name: "scale".to_string(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    #[test]
    fn test_match_pattern_success() {
        let graph = make_conv_bn_relu_graph();
        let ctx = GraphContext::new(&graph);
        let matcher = PatternMatcher::new(&ctx);

        let relu = ctx.get_node("relu_0").unwrap();
        let result = matcher.match_pattern(relu, &["Relu", "BatchNormalization", "Conv"]);

        assert!(result.is_some());
        let m = result.unwrap();
        assert_eq!(m.len(), 3);
        assert_eq!(m.nodes[0].name, "relu_0");
        assert_eq!(m.nodes[1].name, "bn_0");
        assert_eq!(m.nodes[2].name, "conv_0");
    }

    #[test]
    fn test_match_pattern_partial() {
        let graph = make_conv_bn_relu_graph();
        let ctx = GraphContext::new(&graph);
        let matcher = PatternMatcher::new(&ctx);

        let relu = ctx.get_node("relu_0").unwrap();
        let result = matcher.match_pattern(relu, &["Relu", "BatchNormalization"]);

        assert!(result.is_some());
        let m = result.unwrap();
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn test_match_pattern_failure() {
        let graph = make_conv_bn_relu_graph();
        let ctx = GraphContext::new(&graph);
        let matcher = PatternMatcher::new(&ctx);

        let relu = ctx.get_node("relu_0").unwrap();
        // Wrong pattern
        let result = matcher.match_pattern(relu, &["Relu", "Conv"]);
        assert!(result.is_none());

        // Wrong anchor
        let conv = ctx.get_node("conv_0").unwrap();
        let result = matcher.match_pattern(conv, &["Relu"]);
        assert!(result.is_none());
    }

    #[test]
    fn test_find_all_matches() {
        let graph = make_conv_bn_relu_graph();
        let ctx = GraphContext::new(&graph);
        let matcher = PatternMatcher::new(&ctx);

        let matches = matcher.find_all_matches(&["BatchNormalization", "Conv"]);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].anchor, "bn_0");
    }

    #[test]
    fn test_match_with_condition() {
        let graph = make_conv_bn_relu_graph();
        let ctx = GraphContext::new(&graph);
        let matcher = PatternMatcher::new(&ctx);

        let bn = ctx.get_node("bn_0").unwrap();

        // Condition passes
        let result =
            matcher.match_pattern_with_condition(bn, &["BatchNormalization", "Conv"], |nodes| {
                nodes.len() == 2
            });
        assert!(result.is_some());

        // Condition fails
        let result =
            matcher.match_pattern_with_condition(bn, &["BatchNormalization", "Conv"], |nodes| {
                nodes.len() == 10
            });
        assert!(result.is_none());
    }

    #[test]
    fn test_is_fusible_connection() {
        let graph = make_conv_bn_relu_graph();
        let ctx = GraphContext::new(&graph);
        let matcher = PatternMatcher::new(&ctx);

        let conv = ctx.get_node("conv_0").unwrap();
        let bn = ctx.get_node("bn_0").unwrap();

        assert!(matcher.is_fusible_connection(conv, bn));
    }

    #[test]
    fn test_match_result_accessors() {
        let graph = make_conv_bn_relu_graph();
        let ctx = GraphContext::new(&graph);
        let matcher = PatternMatcher::new(&ctx);

        let relu = ctx.get_node("relu_0").unwrap();
        let result = matcher
            .match_pattern(relu, &["Relu", "BatchNormalization", "Conv"])
            .unwrap();

        assert_eq!(result.first().unwrap().name, "relu_0");
        assert_eq!(result.last().unwrap().name, "conv_0");
        assert_eq!(result.get(1).unwrap().name, "bn_0");
        assert_eq!(result.node_names(), vec!["relu_0", "bn_0", "conv_0"]);
    }
}
