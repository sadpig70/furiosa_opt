use crate::error::{OnnxResult, TransformError};
use crate::graph::GraphContext;
use crate::pattern::{PatternMatcher, PAD_CONV};
use crate::transformers::common::{get_attr_ints, set_attr_ints, OnnxTransformer, TransformResult};

/// Fuses Pad into Conv
///
/// Transforms:
///   Pad(x) -> Conv(x)
/// Into:
///   Conv(x) with updated pads attribute
#[derive(Debug, Default)]
pub struct FusePad;

impl FusePad {
    /// Create a new FusePad transformer
    pub fn new() -> Self {
        Self
    }

    fn fuse_pair(
        &self,
        ctx: &mut GraphContext,
        pad_name: &str,
        conv_name: &str,
    ) -> OnnxResult<bool> {
        let pad_node = ctx
            .get_node(pad_name)
            .ok_or_else(|| TransformError::InvalidNode(pad_name.to_string()))?
            .clone();

        // Check Pad mode (default is constant)
        let mode = pad_node
            .attribute
            .iter()
            .find(|a| a.name == "mode")
            .map(|a| a.s.as_slice())
            .unwrap_or(b"constant");

        if mode != b"constant" {
            return Ok(false);
        }

        // Check Pad value (must be 0)
        // Pad value can be in input[1] (opset 11+) or attribute (opset < 11)
        // For simplicity, let's check if input[1] exists and is a zero initializer
        if pad_node.input.len() > 1 {
            if let Some(constant_value) = ctx.get_initializer(&pad_node.input[1]) {
                // Check if all values are 0
                // This is a simplified check. A robust one would check data type and values.
                // Assuming float 0.0 or int 0
                if !constant_value.float_data.iter().all(|&x| x == 0.0)
                    && !constant_value.int64_data.iter().all(|&x| x == 0)
                    && !constant_value.int32_data.iter().all(|&x| x == 0)
                    && !constant_value.raw_data.iter().all(|&x| x == 0)
                {
                    return Ok(false);
                }
            } else {
                // Dynamic pad value, cannot fuse unless we know it's 0
                return Ok(false);
            }
        }
        // If input[1] is missing, default is 0, so we are good.

        // Get pads
        let pads = if !pad_node.input.is_empty() {
            // Opset 11+: pads are in input[0] (data), input[1] (pads) - wait, no
            // Opset 11: input[0] data, input[1] pads, input[2] constant_value
            if pad_node.input.len() >= 2 {
                if let Some(pads_tensor) = ctx.get_initializer(&pad_node.input[1]) {
                    let array: ndarray::ArrayD<i64> =
                        crate::tensor::convert::tensor_to_array_i64(pads_tensor)?;
                    let vec: Vec<i64> = array.into_raw_vec();
                    vec
                } else {
                    return Ok(false); // Dynamic pads
                }
            } else {
                // Opset < 11: pads in attribute
                get_attr_ints(&pad_node, "pads")
                    .unwrap_or_default()
                    .to_vec()
            }
        } else {
            return Ok(false);
        };

        if pads.is_empty() {
            return Ok(false);
        }

        // Get Conv node
        let conv_node = ctx
            .get_node(conv_name)
            .ok_or_else(|| TransformError::InvalidNode(conv_name.to_string()))?
            .clone();

        // Get Conv pads
        let mut conv_pads = if let Some(p) = get_attr_ints(&conv_node, "pads") {
            p.to_vec()
        } else {
            // If pads attribute is missing, it defaults to 0s.
            // We need to know the rank to create 0s.
            // Assuming 2D Conv (rank 4 input), pads length is 4 (H_begin, W_begin, H_end, W_end).
            // But pads length depends on input rank.
            // Let's assume the pads from Pad node has the correct length for the layout.
            vec![0; pads.len()]
        };

        if conv_pads.len() != pads.len() {
            // Mismatch in dimensions, safer to skip
            return Ok(false);
        }

        // Add pads
        for (c, p) in conv_pads.iter_mut().zip(pads.iter()) {
            *c += *p;
        }

        // Update Conv pads attribute
        if let Some(entry) = ctx.get_entry_mut(conv_name) {
            set_attr_ints(&mut entry.node, "pads", conv_pads);

            // Update Conv input to bypass Pad
            entry.node.input[0] = pad_node.input[0].clone();

            // Update producer map if needed (Pad's input producer now feeds Conv)
            // But GraphContext handles this via update_producer if we change outputs.
            // Here we changed input of Conv.
        }

        // Remove Pad node
        ctx.remove_node(pad_name);

        Ok(true)
    }
}

impl OnnxTransformer for FusePad {
    fn name(&self) -> &'static str {
        "FusePad"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        loop {
            let matches: Vec<(String, String)> = {
                let matcher = PatternMatcher::new(ctx);
                matcher
                    .find_all_matches(PAD_CONV)
                    .into_iter()
                    .filter(|m| {
                        !ctx.is_eliminated(&m.nodes[0].name) && !ctx.is_eliminated(&m.nodes[1].name)
                    })
                    .map(|m| (m.nodes[1].name.clone(), m.nodes[0].name.clone())) // Pad, Conv
                    .collect()
            };

            if matches.is_empty() {
                break;
            }

            let mut any_fused = false;
            for (pad_name, conv_name) in matches {
                if ctx.is_eliminated(&pad_name) || ctx.is_eliminated(&conv_name) {
                    continue;
                }

                if self.fuse_pair(ctx, &pad_name, &conv_name)? {
                    result.transforms_applied += 1;
                    result.record_elimination(&pad_name);
                    any_fused = true;
                }
            }

            if !any_fused {
                break;
            }
        }

        Ok(result)
    }

    fn is_applicable(&self, ctx: &GraphContext) -> bool {
        ctx.nodes().any(|n| n.op_type == "Pad")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::{AttributeProto, GraphProto, NodeProto, ValueInfoProto};

    #[test]
    fn test_fuse_pad_conv() {
        // Pad (pads=[0,0,1,1,0,0,1,1]) -> Conv (pads=[0,0,0,0])
        // Result Conv (pads=[0,0,1,1,0,0,1,1])

        let graph = GraphProto {
            node: vec![
                NodeProto {
                    op_type: "Pad".to_string(),
                    name: "pad_0".to_string(),
                    input: vec!["X".to_string()],
                    output: vec!["pad_out".to_string()],
                    attribute: vec![
                        AttributeProto {
                            name: "pads".to_string(),
                            r#type: 7, // INTS
                            ints: vec![0, 0, 1, 1, 0, 0, 1, 1],
                            ..Default::default()
                        },
                        AttributeProto {
                            name: "mode".to_string(),
                            s: b"constant".to_vec(),
                            r#type: 3, // STRING
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
                make_node("Conv", &["pad_out", "W"], &["Y"], "conv_0"),
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
        };

        let mut ctx = GraphContext::new(&graph);
        let result = FusePad::new().transform(&mut ctx).unwrap();

        assert_eq!(result.transforms_applied, 1);
        assert!(!ctx.has_node("pad_0"));

        let conv = ctx.get_node("conv_0").unwrap();
        let pads = get_attr_ints(conv, "pads").unwrap();
        assert_eq!(pads, vec![0, 0, 1, 1, 0, 0, 1, 1].as_slice());
        assert_eq!(conv.input[0], "X");
    }
}
