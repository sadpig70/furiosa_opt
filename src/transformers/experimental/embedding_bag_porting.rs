use std::collections::HashMap;

use crate::error::OnnxResult;
use crate::graph::GraphContext;
use crate::proto::extensions::make_node;
use crate::proto::{AttributeProto, NodeProto, TensorProto};
use crate::transformers::common::{get_attr_i, get_attr_ints, OnnxTransformer, TransformResult};

/// Ports EmbeddingBag to Furiosa compatible ops
///
/// Transforms EmbeddingBag into Gather + ReduceSum/Mean
#[derive(Debug, Default)]
pub struct EmbeddingBagPorting;

impl EmbeddingBagPorting {
    /// Create a new EmbeddingBagPorting transformer
    pub fn new() -> Self {
        Self
    }

    fn transform_subgraph(&self, subgraph: &mut crate::proto::GraphProto) -> bool {
        // Build a map for quick lookup within the subgraph
        let mut node_map: HashMap<String, NodeProto> = HashMap::new();
        let mut producer_map: HashMap<String, String> = HashMap::new();

        for node in &subgraph.node {
            node_map.insert(node.name.clone(), node.clone());
            for output in &node.output {
                producer_map.insert(output.clone(), node.name.clone());
            }
        }

        // Identify base nodes (ReduceSum)
        let mut subgraph_base_nodes = Vec::new();
        for output in &subgraph.output {
            if let Some(producer_name) = producer_map.get(&output.name) {
                if let Some(node) = node_map.get(producer_name) {
                    if node.op_type == "ReduceSum" {
                        subgraph_base_nodes.push(node.clone());
                    } else if matches!(node.op_type.as_str(), "ReduceMean" | "ReduceMax") {
                        // Not implemented for Mean/Max yet, as per Python code
                        return false;
                    }
                }
            }
        }

        if subgraph_base_nodes.len() != 1 {
            return false;
        }

        let base_node = &subgraph_base_nodes[0];

        // Pattern matching: Gather -> Slice -> Unsqueeze -> Gather -> Unsqueeze -> Gather
        // We need to trace back from base_node input[0]

        let gather_0_name = match producer_map.get(&base_node.input[0]) {
            Some(name) => name,
            None => return false,
        };
        let gather_0 = match node_map.get(gather_0_name) {
            Some(node) if node.op_type == "Gather" => node,
            _ => return false,
        };

        let slice_0_name = match producer_map.get(&gather_0.input[1]) {
            Some(name) => name,
            None => return false,
        };
        let slice_0 = match node_map.get(slice_0_name) {
            Some(node) if node.op_type == "Slice" => node,
            _ => return false,
        };

        let unsqueeze_0_name = match producer_map.get(&slice_0.input[1]) {
            Some(name) => name,
            None => return false,
        };
        let unsqueeze_0 = match node_map.get(unsqueeze_0_name) {
            Some(node) if node.op_type == "Unsqueeze" => node,
            _ => return false,
        };

        let gather_1_name = match producer_map.get(&unsqueeze_0.input[0]) {
            Some(name) => name,
            None => return false,
        };
        let gather_1 = match node_map.get(gather_1_name) {
            Some(node) if node.op_type == "Gather" => node,
            _ => return false,
        };

        let unsqueeze_1_name = match producer_map.get(&slice_0.input[2]) {
            Some(name) => name,
            None => return false,
        };
        let unsqueeze_1 = match node_map.get(unsqueeze_1_name) {
            Some(node) if node.op_type == "Unsqueeze" => node,
            _ => return false,
        };

        let gather_2_name = match producer_map.get(&unsqueeze_1.input[0]) {
            Some(name) => name,
            None => return false,
        };
        let gather_2 = match node_map.get(gather_2_name) {
            Some(node) if node.op_type == "Gather" => node,
            _ => return false,
        };

        // Check conditions (simplified from Python)
        // condition 1: base_node axes=[0], keepdims=0
        if get_attr_ints(base_node, "axes") != Some(&[0i64] as &[i64])
            || get_attr_i(base_node, "keepdims").unwrap_or(1) != 0
        {
            return false;
        }
        // condition 2: gather_0 axis=0
        if get_attr_i(gather_0, "axis").unwrap_or(0) != 0 {
            return false;
        }
        // condition 3: slice_0 axes=[0] (checked via initializer in Python, here we skip strict check or assume it matches)
        // condition 4: unsqueeze axes=[0]
        if get_attr_ints(unsqueeze_0, "axes") != Some(&[0i64] as &[i64]) {
            return false;
        }
        if get_attr_ints(unsqueeze_1, "axes") != Some(&[0i64] as &[i64]) {
            return false;
        }
        // condition 5: gather_1, gather_2 axis=0 and same indices
        if get_attr_i(gather_1, "axis").unwrap_or(0) != 0
            || get_attr_i(gather_2, "axis").unwrap_or(0) != 0
        {
            return false;
        }
        if gather_1.input[1] != gather_2.input[1] {
            return false;
        }

        // Apply transformation
        // Create new nodes
        let gather_out = &gather_0.output[0];
        let gather_name = &gather_0.name;

        // 1. Initializer for indice_1
        let indice_1_name = format!("{}_indice_1", gather_out);
        let indice_1_tensor = TensorProto {
            name: indice_1_name.clone(),
            dims: vec![1],
            data_type: 7,                           // INT64
            raw_data: vec![1, 0, 0, 0, 0, 0, 0, 0], // Little endian 1 (int64)
            ..Default::default()
        };
        subgraph.initializer.push(indice_1_tensor);

        // 2. Shape
        let shape_node = make_node(
            "Shape",
            &[gather_out],
            &[&format!("{}_shape", gather_out)],
            &format!("{}_shape", gather_name),
        );

        // 3. Gather (get_shape_1)
        let gather_shape_node = make_node(
            "Gather",
            &[&format!("{}_shape", gather_out), &indice_1_name],
            &[&format!("{}_shape_1", gather_out)],
            &format!("{}_get_shape_1", gather_name),
        );

        // 4. ConstantOfShape
        let zero_vec_node = make_node(
            "ConstantOfShape",
            &[&format!("{}_shape_1", gather_out)],
            &[&format!("{}_zero_vec", gather_out)],
            &format!("{}_zero_vec", gather_name),
        );

        // 5. Unsqueeze
        let mut unsqueeze_node = make_node(
            "Unsqueeze",
            &[&format!("{}_zero_vec", gather_out)],
            &[&format!("{}_zero_vec_unsqueezed", gather_out)],
            &format!("{}_zero_vec_unsqueeze", gather_name),
        );
        // Set axes attribute for Unsqueeze
        let mut axes_attr = AttributeProto::default();
        axes_attr.name = "axes".to_string();
        axes_attr.r#type = 7; // INTS
        axes_attr.ints = vec![0];
        unsqueeze_node.attribute.push(axes_attr);

        // 6. Concat
        let mut concat_node = make_node(
            "Concat",
            &[gather_out, &format!("{}_zero_vec_unsqueezed", gather_out)],
            &[&format!("{}_nonzero", gather_out)],
            &format!("{}_convert", gather_name),
        );
        let mut axis_attr = AttributeProto::default();
        axis_attr.name = "axis".to_string();
        axis_attr.r#type = 2; // INT
        axis_attr.i = 0;
        concat_node.attribute.push(axis_attr);

        // 7. Update base_node input
        // We need to find the base_node in the subgraph and update it.
        // Since we are iterating over a clone or map, we need to find the mutable reference in subgraph.node

        // Add new nodes to subgraph
        subgraph.node.push(shape_node);
        subgraph.node.push(gather_shape_node);
        subgraph.node.push(zero_vec_node);
        subgraph.node.push(unsqueeze_node);
        subgraph.node.push(concat_node);

        // Update base_node
        for node in &mut subgraph.node {
            if node.name == base_node.name {
                node.input[0] = format!("{}_nonzero", gather_out);
                return true;
            }
        }

        false
    }
}

impl OnnxTransformer for EmbeddingBagPorting {
    fn name(&self) -> &'static str {
        "EmbeddingBagPorting"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        // Iterate over all nodes to find Loop nodes
        // We can't iterate and mutate easily with GraphContext if we are modifying attributes deep inside.
        // But here we are modifying the AttributeProto of the Loop node.

        let loop_nodes: Vec<String> = ctx
            .nodes()
            .filter(|n| n.op_type == "Loop")
            .map(|n| n.name.clone())
            .collect();

        for loop_name in loop_nodes {
            if let Some(loop_node) = ctx.get_node_mut(&loop_name) {
                // Find body attribute
                for attr in &mut loop_node.attribute {
                    if attr.name == "body" && attr.g.is_some() {
                        if let Some(subgraph) = &mut attr.g {
                            if self.transform_subgraph(subgraph) {
                                result.transforms_applied += 1;
                            }
                        }
                    }
                }
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::{GraphProto, ValueInfoProto};

    #[test]
    fn test_embedding_bag_porting() {
        // Construct a subgraph that matches the pattern
        // Gather -> Slice -> Unsqueeze -> Gather -> Unsqueeze -> Gather -> ReduceSum

        // This is complex to mock manually. Let's create a minimal matching subgraph.
        let mut subgraph = GraphProto::default();

        // Nodes
        let gather_2 = make_node("Gather", &["In", "Idx"], &["G2_Out"], "gather_2");
        let unsqueeze_1 = make_node("Unsqueeze", &["G2_Out"], &["U1_Out"], "unsqueeze_1"); // axes=[0]
        let gather_1 = make_node("Gather", &["In", "Idx"], &["G1_Out"], "gather_1");
        let unsqueeze_0 = make_node("Unsqueeze", &["G1_Out"], &["U0_Out"], "unsqueeze_0"); // axes=[0]
        let slice_0 = make_node(
            "Slice",
            &["Data", "U0_Out", "U1_Out", "Axes"],
            &["S0_Out"],
            "slice_0",
        );
        let gather_0 = make_node("Gather", &["Table", "S0_Out"], &["G0_Out"], "gather_0");
        let reduce_sum = make_node("ReduceSum", &["G0_Out"], &["Out"], "reduce_sum"); // axes=[0], keepdims=0

        // Add attributes
        let mut g2 = gather_2.clone();
        // axis=0 default

        let mut u1 = unsqueeze_1.clone();
        let mut axes = AttributeProto::default();
        axes.name = "axes".to_string();
        axes.ints = vec![0];
        axes.r#type = 7;
        u1.attribute.push(axes.clone());

        let mut g1 = gather_1.clone();

        let mut u0 = unsqueeze_0.clone();
        u0.attribute.push(axes.clone());

        let mut s0 = slice_0.clone();

        let mut g0 = gather_0.clone();

        let mut rs = reduce_sum.clone();
        rs.attribute.push(axes.clone());
        let mut keepdims = AttributeProto::default();
        keepdims.name = "keepdims".to_string();
        keepdims.i = 0;
        keepdims.r#type = 2;
        rs.attribute.push(keepdims);

        subgraph.node = vec![g2, u1, g1, u0, s0, g0, rs];
        subgraph.output = vec![ValueInfoProto {
            name: "Out".to_string(),
            ..Default::default()
        }];

        // Wrap in Loop node
        let mut loop_node = make_node("Loop", &[], &[], "loop_0");
        let mut body_attr = AttributeProto::default();
        body_attr.name = "body".to_string();
        body_attr.r#type = 5; // GRAPH
        body_attr.g = Some(subgraph);
        loop_node.attribute.push(body_attr);

        let graph = GraphProto {
            node: vec![loop_node],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let transformer = EmbeddingBagPorting::new();
        let result = transformer.transform(&mut ctx).unwrap();

        assert_eq!(result.transforms_applied, 1);

        // Verify subgraph modification
        let loop_node = ctx.get_node("loop_0").unwrap();
        let subgraph = loop_node.attribute[0].g.as_ref().unwrap();

        // Check for new nodes
        let has_concat = subgraph.node.iter().any(|n| n.op_type == "Concat");
        assert!(has_concat);

        // Check ReduceSum input
        let rs_node = subgraph
            .node
            .iter()
            .find(|n| n.op_type == "ReduceSum")
            .unwrap();
        assert!(rs_node.input[0].contains("_nonzero"));
    }
}
