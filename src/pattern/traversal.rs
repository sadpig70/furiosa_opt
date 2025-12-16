//! Graph traversal utilities
//!
//! Provides BFS and DFS traversal for ONNX graphs.

use std::collections::{HashSet, VecDeque};

use crate::graph::GraphContext;
use crate::proto::NodeProto;

/// Direction of traversal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Forward: follow consumer edges (input → output)
    Forward,
    /// Backward: follow producer edges (output → input)
    Backward,
}

/// BFS traversal iterator
pub struct BfsIterator<'a> {
    ctx: &'a GraphContext,
    queue: VecDeque<&'a str>,
    visited: HashSet<&'a str>,
    direction: Direction,
}

impl<'a> BfsIterator<'a> {
    /// Create a new BFS iterator starting from the given node
    pub fn new(ctx: &'a GraphContext, start: &'a str, direction: Direction) -> Self {
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();

        if ctx.has_node(start) {
            queue.push_back(start);
            visited.insert(start);
        }

        Self {
            ctx,
            queue,
            visited,
            direction,
        }
    }

    /// Create forward BFS (follows consumers)
    pub fn forward(ctx: &'a GraphContext, start: &'a str) -> Self {
        Self::new(ctx, start, Direction::Forward)
    }

    /// Create backward BFS (follows producers)
    pub fn backward(ctx: &'a GraphContext, start: &'a str) -> Self {
        Self::new(ctx, start, Direction::Backward)
    }
}

impl<'a> Iterator for BfsIterator<'a> {
    type Item = &'a NodeProto;

    fn next(&mut self) -> Option<Self::Item> {
        let name = self.queue.pop_front()?;
        let node = self.ctx.get_node(name)?;

        // Add neighbors to queue
        match self.direction {
            Direction::Forward => {
                // Get consumers of all outputs
                for output in &node.output {
                    if let Some(consumers) = self.ctx.get_consumer_names(output) {
                        for consumer in consumers {
                            if self.visited.insert(consumer.as_str()) {
                                self.queue.push_back(consumer.as_str());
                            }
                        }
                    }
                }
            }
            Direction::Backward => {
                // Get producers of all inputs
                for input in &node.input {
                    if let Some(producer) = self.ctx.get_producer_name(input) {
                        if self.visited.insert(producer.as_str()) {
                            self.queue.push_back(producer.as_str());
                        }
                    }
                }
            }
        }

        Some(node)
    }
}

/// DFS traversal iterator
pub struct DfsIterator<'a> {
    ctx: &'a GraphContext,
    stack: Vec<&'a str>,
    visited: HashSet<&'a str>,
    direction: Direction,
}

impl<'a> DfsIterator<'a> {
    /// Create a new DFS iterator starting from the given node
    pub fn new(ctx: &'a GraphContext, start: &'a str, direction: Direction) -> Self {
        let mut stack = Vec::new();
        let mut visited = HashSet::new();

        if ctx.has_node(start) {
            stack.push(start);
            visited.insert(start);
        }

        Self {
            ctx,
            stack,
            visited,
            direction,
        }
    }

    /// Create forward DFS
    pub fn forward(ctx: &'a GraphContext, start: &'a str) -> Self {
        Self::new(ctx, start, Direction::Forward)
    }

    /// Create backward DFS
    pub fn backward(ctx: &'a GraphContext, start: &'a str) -> Self {
        Self::new(ctx, start, Direction::Backward)
    }
}

impl<'a> Iterator for DfsIterator<'a> {
    type Item = &'a NodeProto;

    fn next(&mut self) -> Option<Self::Item> {
        let name = self.stack.pop()?;
        let node = self.ctx.get_node(name)?;

        match self.direction {
            Direction::Forward => {
                for output in &node.output {
                    if let Some(consumers) = self.ctx.get_consumer_names(output) {
                        for consumer in consumers {
                            if self.visited.insert(consumer.as_str()) {
                                self.stack.push(consumer.as_str());
                            }
                        }
                    }
                }
            }
            Direction::Backward => {
                for input in &node.input {
                    if let Some(producer) = self.ctx.get_producer_name(input) {
                        if self.visited.insert(producer.as_str()) {
                            self.stack.push(producer.as_str());
                        }
                    }
                }
            }
        }

        Some(node)
    }
}

/// Collect all nodes reachable from start in the given direction
pub fn reachable_nodes<'a>(
    ctx: &'a GraphContext,
    start: &'a str,
    direction: Direction,
) -> Vec<&'a NodeProto> {
    BfsIterator::new(ctx, start, direction).collect()
}

/// Collect all predecessor nodes (backward reachable)
pub fn predecessors<'a>(ctx: &'a GraphContext, start: &'a str) -> Vec<&'a NodeProto> {
    BfsIterator::backward(ctx, start).skip(1).collect() // skip start node
}

/// Collect all successor nodes (forward reachable)
pub fn successors<'a>(ctx: &'a GraphContext, start: &'a str) -> Vec<&'a NodeProto> {
    BfsIterator::forward(ctx, start).skip(1).collect() // skip start node
}

/// Find path between two nodes
pub fn find_path<'a>(ctx: &'a GraphContext, from: &str, to: &str) -> Option<Vec<&'a NodeProto>> {
    use std::collections::HashMap;

    if from == to {
        return ctx.get_node(from).map(|n| vec![n]);
    }

    let mut queue = VecDeque::new();
    let mut parent: HashMap<&str, &str> = HashMap::new();

    queue.push_back(from);
    parent.insert(from, from);

    while let Some(current) = queue.pop_front() {
        let node = ctx.get_node(current)?;

        for output in &node.output {
            if let Some(consumers) = ctx.get_consumer_names(output) {
                for consumer in consumers {
                    if !parent.contains_key(consumer.as_str()) {
                        parent.insert(consumer.as_str(), current);
                        if consumer == to {
                            // Reconstruct path
                            let mut path = Vec::new();
                            let mut curr = to;
                            while curr != from {
                                path.push(ctx.get_node(curr)?);
                                curr = parent.get(curr)?;
                            }
                            path.push(ctx.get_node(from)?);
                            path.reverse();
                            return Some(path);
                        }
                        queue.push_back(consumer.as_str());
                    }
                }
            }
        }
    }

    None
}

/// Check if there's a path between two nodes
pub fn has_path(ctx: &GraphContext, from: &str, to: &str) -> bool {
    if from == to {
        return true;
    }

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(from);
    visited.insert(from);

    while let Some(current) = queue.pop_front() {
        if let Some(node) = ctx.get_node(current) {
            for output in &node.output {
                if let Some(consumers) = ctx.get_consumer_names(output) {
                    for consumer in consumers {
                        if consumer == to {
                            return true;
                        }
                        if visited.insert(consumer.as_str()) {
                            queue.push_back(consumer.as_str());
                        }
                    }
                }
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::{GraphProto, ValueInfoProto};

    fn make_chain_graph() -> GraphProto {
        GraphProto {
            node: vec![
                make_node("Conv", &["X"], &["a"], "node_0"),
                make_node("Relu", &["a"], &["b"], "node_1"),
                make_node("Conv", &["b"], &["c"], "node_2"),
                make_node("Relu", &["c"], &["Y"], "node_3"),
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
    fn test_bfs_forward() {
        let graph = make_chain_graph();
        let ctx = GraphContext::new(&graph);

        let nodes: Vec<_> = BfsIterator::forward(&ctx, "node_0")
            .map(|n| n.name.as_str())
            .collect();

        assert_eq!(nodes, vec!["node_0", "node_1", "node_2", "node_3"]);
    }

    #[test]
    fn test_bfs_backward() {
        let graph = make_chain_graph();
        let ctx = GraphContext::new(&graph);

        let nodes: Vec<_> = BfsIterator::backward(&ctx, "node_3")
            .map(|n| n.name.as_str())
            .collect();

        assert_eq!(nodes, vec!["node_3", "node_2", "node_1", "node_0"]);
    }

    #[test]
    fn test_dfs_forward() {
        let graph = make_chain_graph();
        let ctx = GraphContext::new(&graph);

        let nodes: Vec<_> = DfsIterator::forward(&ctx, "node_0")
            .map(|n| n.name.as_str())
            .collect();

        // DFS on a chain is same as BFS
        assert_eq!(nodes.len(), 4);
        assert_eq!(nodes[0], "node_0");
    }

    #[test]
    fn test_predecessors() {
        let graph = make_chain_graph();
        let ctx = GraphContext::new(&graph);

        let preds = predecessors(&ctx, "node_3");
        let names: Vec<_> = preds.iter().map(|n| n.name.as_str()).collect();

        assert_eq!(names.len(), 3);
        assert!(names.contains(&"node_0"));
        assert!(names.contains(&"node_1"));
        assert!(names.contains(&"node_2"));
    }

    #[test]
    fn test_successors() {
        let graph = make_chain_graph();
        let ctx = GraphContext::new(&graph);

        let succs = successors(&ctx, "node_0");
        let names: Vec<_> = succs.iter().map(|n| n.name.as_str()).collect();

        assert_eq!(names.len(), 3);
        assert!(names.contains(&"node_1"));
        assert!(names.contains(&"node_2"));
        assert!(names.contains(&"node_3"));
    }

    #[test]
    fn test_find_path() {
        let graph = make_chain_graph();
        let ctx = GraphContext::new(&graph);

        let path = find_path(&ctx, "node_0", "node_3").unwrap();
        let names: Vec<_> = path.iter().map(|n| n.name.as_str()).collect();

        assert_eq!(names, vec!["node_0", "node_1", "node_2", "node_3"]);
    }

    #[test]
    fn test_find_path_same_node() {
        let graph = make_chain_graph();
        let ctx = GraphContext::new(&graph);

        let path = find_path(&ctx, "node_1", "node_1").unwrap();
        assert_eq!(path.len(), 1);
        assert_eq!(path[0].name, "node_1");
    }

    #[test]
    fn test_find_path_no_path() {
        let graph = make_chain_graph();
        let ctx = GraphContext::new(&graph);

        // Backward path doesn't exist in forward search
        let path = find_path(&ctx, "node_3", "node_0");
        assert!(path.is_none());
    }

    #[test]
    fn test_has_path() {
        let graph = make_chain_graph();
        let ctx = GraphContext::new(&graph);

        assert!(has_path(&ctx, "node_0", "node_3"));
        assert!(has_path(&ctx, "node_0", "node_1"));
        assert!(has_path(&ctx, "node_1", "node_1")); // same node
        assert!(!has_path(&ctx, "node_3", "node_0")); // backward
    }
}
