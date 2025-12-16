use crate::error::{OnnxResult, TransformError};
use crate::graph::GraphContext;
use crate::proto::extensions::make_node;
use crate::transformers::common::{OnnxTransformer, TransformResult};

/// Converts Pad nodes with negative pads to Slice nodes
///
/// Negative pads are not supported by some backends, but can be represented as Slicing.
#[derive(Debug, Default)]
pub struct ConvertNegativePadsToSlice;

impl ConvertNegativePadsToSlice {
    /// Create a new ConvertNegativePadsToSlice transformer
    pub fn new() -> Self {
        Self
    }
}

impl OnnxTransformer for ConvertNegativePadsToSlice {
    fn name(&self) -> &'static str {
        "ConvertNegativePadsToSlice"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        loop {
            let mut nodes_to_process = Vec::new();
            for node in ctx.nodes() {
                if node.op_type == "Pad" {
                    nodes_to_process.push(node.name.clone());
                }
            }

            if nodes_to_process.is_empty() {
                break;
            }

            let mut any_changed = false;
            for node_name in nodes_to_process {
                if ctx.is_eliminated(&node_name) {
                    continue;
                }

                if self.try_convert(ctx, &node_name)? {
                    result.transforms_applied += 1;
                    any_changed = true;
                }
            }

            if !any_changed {
                break;
            }
        }

        Ok(result)
    }

    fn is_applicable(&self, ctx: &GraphContext) -> bool {
        ctx.nodes().any(|n| n.op_type == "Pad")
    }
}

impl ConvertNegativePadsToSlice {
    fn try_convert(&self, ctx: &mut GraphContext, node_name: &str) -> OnnxResult<bool> {
        // Scope to restrict borrow of ctx
        let (input_name, output_name, pads_input_name, optional_input_name) = {
            let node = ctx
                .get_node(node_name)
                .ok_or_else(|| TransformError::InvalidNode(node_name.to_string()))?;

            if node.input.len() < 2 {
                return Ok(false);
            }
            (
                node.input[0].clone(),
                node.output[0].clone(),
                node.input[1].clone(),
                node.input.get(2).cloned(),
            )
        };

        let pads_data = {
            let pads_tensor = match ctx.get_initializer(&pads_input_name) {
                Some(t) => t,
                None => return Ok(false), // Pads must be initializer
            };
            crate::tensor::convert::tensor_to_array_i64(pads_tensor)?.into_raw_vec()
        };

        // Check if all zero
        if pads_data.iter().all(|&x| x == 0) {
            // Remove Pad node
            // Update consumers to use input_name instead of output_name
            let consumers = ctx
                .get_consumers(&output_name)
                .iter()
                .map(|n| n.name.clone())
                .collect::<Vec<_>>();
            for consumer_name in consumers {
                if let Some(entry) = ctx.get_entry_mut(&consumer_name) {
                    for input in &mut entry.node.input {
                        if input == &output_name {
                            *input = input_name.clone();
                        }
                    }
                }
            }

            ctx.remove_node(node_name);
            return Ok(true);
        }

        // Check if any negative
        if !pads_data.iter().any(|&x| x < 0) {
            return Ok(false);
        }

        // Calculate Slice params
        let rank = pads_data.len() / 2;
        let mut starts = Vec::new();
        let mut ends = Vec::new();
        let mut axes = Vec::new();
        let mut new_pads = Vec::new();

        let mut needs_slice = false;

        for i in 0..rank {
            let pad_begin = pads_data[i];
            let pad_end = pads_data[i + rank];

            if pad_begin < 0 || pad_end < 0 {
                needs_slice = true;
                starts.push(if pad_begin < 0 { -pad_begin } else { 0 });
                ends.push(if pad_end < 0 { pad_end } else { i64::MAX });
                axes.push(i as i64);
            }

            new_pads.push(if pad_begin > 0 { pad_begin } else { 0 });
        }
        for i in 0..rank {
            let pad_end = pads_data[i + rank];
            new_pads.push(if pad_end > 0 { pad_end } else { 0 });
        }

        if !needs_slice {
            return Ok(false);
        }

        // Create Slice Node
        let slice_name = format!("{}_slice", node_name);
        let slice_out_name = format!("{}_slice_out", output_name);

        // Create initializers for Slice
        let starts_name = format!("{}_starts", slice_name);
        let ends_name = format!("{}_ends", slice_name);
        let axes_name = format!("{}_axes", slice_name);

        ctx.set_initializer(crate::tensor::convert::vec_to_tensor_i64(
            &starts,
            &starts_name,
        ));
        ctx.set_initializer(crate::tensor::convert::vec_to_tensor_i64(&ends, &ends_name));
        ctx.set_initializer(crate::tensor::convert::vec_to_tensor_i64(&axes, &axes_name));

        let slice_node = make_node(
            "Slice",
            &[&input_name, &starts_name, &ends_name, &axes_name],
            &[&slice_out_name],
            &slice_name,
        );

        // Handle remaining positive pads
        let all_new_pads_zero = new_pads.iter().all(|&x| x == 0);

        if all_new_pads_zero {
            // Just Slice, no Pad
            let mut final_slice = slice_node;
            final_slice.output[0] = output_name.clone(); // Hijack the name
            final_slice.name = slice_name;

            // Remove original Pad
            ctx.remove_node(node_name);
            // Insert Slice
            ctx.insert_node(final_slice);
        } else {
            // Slice -> Pad
            // New Pad node
            let new_pad_name = format!("{}_pad", node_name);
            let new_pads_name = format!("{}_pads", new_pad_name);

            ctx.set_initializer(crate::tensor::convert::vec_to_tensor_i64(
                &new_pads,
                &new_pads_name,
            ));

            let mut new_pad_node = make_node(
                "Pad",
                &[&slice_out_name, &new_pads_name],
                &[&output_name], // Reuse original output name
                &new_pad_name,
            );

            // Copy other inputs if present (constant_value)
            if let Some(opt_input) = optional_input_name {
                new_pad_node.input.push(opt_input);
            }

            // Remove original Pad
            ctx.remove_node(node_name);

            // Insert Slice and New Pad
            ctx.insert_node(slice_node);
            ctx.insert_node(new_pad_node);
        }

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::{make_node, make_tensor_value_info};
    use crate::proto::{GraphProto, TensorProto};

    #[test]
    fn test_convert_negative_pads() {
        // Pad with [-1, -1, -1, -1] -> Slice
        let pads_data = vec![-1, -1, -1, -1];
        let pads = TensorProto {
            name: "pads".to_string(),
            dims: vec![4],
            data_type: 7, // INT64
            int64_data: pads_data,
            ..Default::default()
        };

        let graph = GraphProto {
            node: vec![make_node("Pad", &["X", "pads"], &["Y"], "pad_0")],
            input: vec![make_tensor_value_info("X", 1, &[10, 10])],
            output: vec![make_tensor_value_info("Y", 1, &[8, 8])],
            initializer: vec![pads],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let result = ConvertNegativePadsToSlice::new()
            .transform(&mut ctx)
            .unwrap();

        assert_eq!(result.transforms_applied, 1);
        assert!(!ctx.has_node("pad_0"));
        assert!(ctx.nodes().any(|n| n.op_type == "Slice"));
        // Should not have Pad because all pads are handled by Slice
        assert!(!ctx.nodes().any(|n| n.op_type == "Pad"));
    }

    #[test]
    fn test_convert_mixed_pads() {
        // Pad with [-1, 1, -1, 1] -> Slice + Pad
        let pads_data = vec![-1, 1, -1, 1];
        let pads = TensorProto {
            name: "pads".to_string(),
            dims: vec![4],
            data_type: 7, // INT64
            int64_data: pads_data,
            ..Default::default()
        };

        let graph = GraphProto {
            node: vec![make_node("Pad", &["X", "pads"], &["Y"], "pad_0")],
            input: vec![make_tensor_value_info("X", 1, &[10, 10])],
            output: vec![make_tensor_value_info("Y", 1, &[10, 10])],
            initializer: vec![pads],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let result = ConvertNegativePadsToSlice::new()
            .transform(&mut ctx)
            .unwrap();

        assert_eq!(result.transforms_applied, 1);
        assert!(ctx.nodes().any(|n| n.op_type == "Slice"));
        assert!(ctx.nodes().any(|n| n.op_type == "Pad"));
    }
}
