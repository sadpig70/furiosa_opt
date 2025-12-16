use std::collections::HashSet;

use crate::error::{OnnxResult, TransformError};
use crate::graph::GraphContext;
use crate::transformers::common::{OnnxTransformer, TransformResult};

/// Eliminates detection post-processing subgraph
///
/// Removes the post-processing part of SSD-like models.
#[derive(Debug, Clone)]
pub struct EliminateDetectionPostprocess {
    ssd_outputs: Vec<String>,
}

impl EliminateDetectionPostprocess {
    /// Create a new EliminateDetectionPostprocess transformer
    pub fn new(ssd_outputs: Vec<String>) -> Self {
        Self { ssd_outputs }
    }

    fn get_postprocess_nodes(&self, ctx: &GraphContext) -> HashSet<String> {
        let mut inputs = HashSet::new();
        for output in &self.ssd_outputs {
            inputs.insert(output.clone());
        }

        let mut postprocess_nodes = HashSet::new();

        // Forward traverse
        for node in ctx.nodes() {
            let mut is_append = false;
            for input in &node.input {
                if inputs.contains(input) {
                    is_append = true;
                    break;
                }
            }

            if is_append {
                postprocess_nodes.insert(node.name.clone());
                for output in &node.output {
                    inputs.insert(output.clone());
                }
            }
        }

        // Backward traverse
        loop {
            let _prev_len = postprocess_nodes.len();

            // We need to collect nodes to add to avoid borrowing issues while iterating
            let mut nodes_to_add = Vec::new();

            for node in ctx.nodes() {
                if postprocess_nodes.contains(&node.name) {
                    continue;
                }

                let mut is_append = false;
                for output in &node.output {
                    // Check if this output is consumed by any node in postprocess_nodes
                    if let Some(consumers) = ctx.get_consumer_names(output) {
                        for consumer in consumers {
                            if postprocess_nodes.contains(consumer)
                                && !self.ssd_outputs.contains(output)
                            {
                                is_append = true;
                                break;
                            }
                        }
                    }
                    if is_append {
                        break;
                    }
                }

                if is_append {
                    nodes_to_add.push(node.name.clone());
                }
            }

            if nodes_to_add.is_empty() {
                break;
            }

            for name in nodes_to_add {
                postprocess_nodes.insert(name);
            }
        }

        postprocess_nodes
    }
}

impl OnnxTransformer for EliminateDetectionPostprocess {
    fn name(&self) -> &'static str {
        "EliminateDetectionPostprocess"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        let nodes_to_remove = self.get_postprocess_nodes(ctx);

        if nodes_to_remove.is_empty() {
            return Ok(result);
        }

        // Remove nodes
        for name in &nodes_to_remove {
            if ctx.remove_node(name).is_some() {
                result.nodes_eliminated += 1;
            }
        }

        // Update graph outputs
        // 1. Remove eliminated outputs from graph_output_map
        let mut outputs_to_remove = Vec::new();
        for output_name in ctx.graph_output_map.keys() {
            if ctx.get_producer(output_name).is_none() {
                outputs_to_remove.push(output_name.clone());
            }
        }

        for output in outputs_to_remove {
            ctx.graph_output_map.remove(&output);
        }

        // Add ssd_outputs to graph outputs
        for output in &self.ssd_outputs {
            if let Some(vi) = ctx.value_info_map.get(output) {
                ctx.graph_output_map.insert(output.clone(), vi.clone());
            } else {
                return Err(TransformError::ValueInfoNotFound(output.clone()));
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::{GraphProto, TensorProto, ValueInfoProto};

    #[test]
    fn test_eliminate_detection_postprocess() {
        // Graph:
        // Input -> Conv -> SSD_Output
        // SSD_Output -> PostProcess1 -> PostProcess2 -> FinalOutput

        let graph = GraphProto {
            node: vec![
                make_node("Conv", &["Input", "W"], &["SSD_Output"], "conv_0"),
                make_node("PostProcess1", &["SSD_Output"], &["PP1_Out"], "pp1"),
                make_node("PostProcess2", &["PP1_Out"], &["FinalOutput"], "pp2"),
            ],
            input: vec![ValueInfoProto {
                name: "Input".to_string(),
                ..Default::default()
            }],
            output: vec![ValueInfoProto {
                name: "FinalOutput".to_string(),
                ..Default::default()
            }],
            initializer: vec![TensorProto {
                name: "W".to_string(),
                ..Default::default()
            }],
            value_info: vec![
                ValueInfoProto {
                    name: "SSD_Output".to_string(),
                    ..Default::default()
                },
                ValueInfoProto {
                    name: "PP1_Out".to_string(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let transformer = EliminateDetectionPostprocess::new(vec!["SSD_Output".to_string()]);
        let result = transformer.transform(&mut ctx).unwrap();

        assert_eq!(result.nodes_eliminated, 2);
        assert!(!ctx.has_node("pp1"));
        assert!(!ctx.has_node("pp2"));
        assert!(ctx.has_node("conv_0"));

        // Check outputs
        assert!(ctx.is_graph_output("SSD_Output"));
        assert!(!ctx.is_graph_output("FinalOutput"));
    }
}
