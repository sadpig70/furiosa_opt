use crate::error::{OnnxResult, TransformError};
use crate::graph::GraphContext;
use crate::proto::extensions::make_node;
use crate::proto::{NodeProto, TensorProto};
use crate::transformers::common::{get_constant_tensor, OnnxTransformer, TransformResult};

/// Reifies Conv for BERT
///
/// Transforms specific patterns in BERT to use Conv
#[derive(Debug, Default)]
pub struct ReifyConvForBert;

impl ReifyConvForBert {
    /// Create a new ReifyConvForBert transformer
    pub fn new() -> Self {
        Self
    }

    fn reify_conv(&self, ctx: &mut GraphContext, add_name: &str) -> OnnxResult<bool> {
        let add_node = ctx
            .get_node(add_name)
            .ok_or_else(|| TransformError::InvalidNode(add_name.to_string()))?
            .clone();

        // Check pattern: MatMul -> Add
        // Add inputs: [MatMul_Output, Bias] (or swapped)

        let mut matmul_node: Option<NodeProto> = None;
        let mut bias_index = 0;
        let mut _matmul_input_index = 0;

        for (i, input) in add_node.input.iter().enumerate() {
            if let Some(node) = ctx.get_producer(input) {
                if node.op_type == "MatMul" {
                    matmul_node = Some(node.clone());
                    _matmul_input_index = i;
                    bias_index = 1 - i;
                    break;
                }
            }
        }

        let matmul_node = if let Some(n) = matmul_node {
            n
        } else {
            return Ok(false);
        };

        // Check if Bias is constant
        let bias_tensor = if let Some(t) = get_constant_tensor(ctx, &add_node.input[bias_index]) {
            t
        } else {
            return Ok(false);
        };

        // Check if MatMul weight is constant
        let mut weight_index = 0;
        let mut input_index = 0;
        let mut weight_tensor: Option<TensorProto> = None;

        for (i, input) in matmul_node.input.iter().enumerate() {
            if let Some(t) = get_constant_tensor(ctx, input) {
                weight_tensor = Some(t.clone());
                weight_index = i;
                input_index = 1 - i;
                break;
            }
        }

        let weight_tensor = if let Some(t) = weight_tensor {
            t
        } else {
            return Ok(false);
        };

        // Extract data early to release borrows on ctx
        let weight_arr = crate::tensor::convert::tensor_to_array_f32(&weight_tensor)?;
        let bias_arr = crate::tensor::convert::tensor_to_array_f32(bias_tensor)?;

        // Perform Transformation

        // 1. Prepare Conv Weight: (C, N) -> (N, C, 1, 1)
        // Shape should be 2D
        if weight_arr.ndim() != 2 {
            return Ok(false);
        }
        let c = weight_arr.shape()[0];
        let n = weight_arr.shape()[1];

        // Transpose to (N, C) then reshape to (N, C, 1, 1)
        let conv_weight_arr = weight_arr
            .t()
            .to_owned()
            .into_shape((n, c, 1, 1))
            .map_err(|e| TransformError::Internal(e.to_string()))?
            .into_dyn();

        let conv_weight_name = format!("{}_reified", matmul_node.input[weight_index]);
        let conv_weight_tensor =
            crate::tensor::convert::array_to_tensor_f32(&conv_weight_arr, &conv_weight_name);
        ctx.set_initializer(conv_weight_tensor);

        // 2. Prepare Conv Bias
        let bias_name = format!("{}_reified", add_node.input[bias_index]);
        let bias_init = crate::tensor::convert::array_to_tensor_f32(&bias_arr, &bias_name);
        ctx.set_initializer(bias_init);

        // 3. Create Nodes
        let input_name = &matmul_node.input[input_index];
        let matmul_output = &matmul_node.output[0];
        let add_output = &add_node.output[0];

        // Unsqueeze(axes=[1])
        let unsqueeze_out = format!("{}_expanded", matmul_output);
        let unsqueeze_node = make_node(
            "Unsqueeze",
            &[input_name.as_str()],
            &[unsqueeze_out.as_str()],
            &format!("{}_unsqueeze", matmul_output),
        );
        let mut unsqueeze_node = unsqueeze_node;
        crate::transformers::common::set_attr_ints(&mut unsqueeze_node, "axes", vec![1]);

        // Transpose(perm=[0, 3, 1, 2])
        let transpose_out = format!("{}_transposed", matmul_output);
        let mut transpose_node = make_node(
            "Transpose",
            &[unsqueeze_out.as_str()],
            &[transpose_out.as_str()],
            &format!("{}_transpose", matmul_output),
        );
        crate::transformers::common::set_attr_ints(&mut transpose_node, "perm", vec![0, 3, 1, 2]);

        // Conv
        let conv_out = format!("{}_conv_output", matmul_output);
        let mut conv_node = make_node(
            "Conv",
            &[
                transpose_out.as_str(),
                conv_weight_name.as_str(),
                bias_name.as_str(),
            ],
            &[conv_out.as_str()],
            &format!("{}_conv", matmul_output),
        );
        crate::transformers::common::set_attr_ints(&mut conv_node, "dilations", vec![1, 1]);
        crate::transformers::common::set_attr_ints(&mut conv_node, "group", vec![1]);
        crate::transformers::common::set_attr_ints(&mut conv_node, "kernel_shape", vec![1, 1]);
        crate::transformers::common::set_attr_ints(&mut conv_node, "pads", vec![0, 0, 0, 0]);
        crate::transformers::common::set_attr_ints(&mut conv_node, "strides", vec![1, 1]);

        // Squeeze(axes=[2])
        let squeeze_out = format!("{}_squeezed", matmul_output);
        let mut squeeze_node = make_node(
            "Squeeze",
            &[conv_out.as_str()],
            &[squeeze_out.as_str()],
            &format!("{}_squeeze", matmul_output),
        );
        crate::transformers::common::set_attr_ints(&mut squeeze_node, "axes", vec![2]);

        // Transpose(perm=[0, 2, 1])
        let mut transpose_2_node = make_node(
            "Transpose",
            &[squeeze_out.as_str()],
            &[add_output.as_str()], // Final output matches Add output
            &format!("{}_transpose_2", matmul_output),
        );
        crate::transformers::common::set_attr_ints(&mut transpose_2_node, "perm", vec![0, 2, 1]);

        // Remove old nodes first to avoid clearing producer map for reused outputs
        ctx.remove_node(add_name);
        ctx.remove_node(&matmul_node.name);

        // Add new nodes
        ctx.insert_node(unsqueeze_node);
        ctx.insert_node(transpose_node);
        ctx.insert_node(conv_node);
        ctx.insert_node(squeeze_node);
        ctx.insert_node(transpose_2_node);

        Ok(true)
    }
}

impl OnnxTransformer for ReifyConvForBert {
    fn name(&self) -> &'static str {
        "ReifyConvForBert"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        let add_nodes: Vec<String> = ctx
            .nodes()
            .filter(|n| n.op_type == "Add")
            .map(|n| n.name.clone())
            .collect();

        for add_name in add_nodes {
            if ctx.is_eliminated(&add_name) {
                continue;
            }

            if self.reify_conv(ctx, &add_name)? {
                result.transforms_applied += 1;
                result.record_elimination(&add_name);
            }
        }

        Ok(result)
    }

    fn is_applicable(&self, ctx: &GraphContext) -> bool {
        ctx.nodes().any(|n| n.op_type == "Add")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::{GraphProto, NodeProto, ValueInfoProto};
    use crate::tensor::convert::vec_to_tensor_f32;

    #[test]
    fn test_reify_conv_for_bert() {
        // MatMul (W=[C=2, N=2]) -> Add (Bias=[2])
        // Input (B, S, C)

        // Weight (2x2)
        let mut weight = vec_to_tensor_f32(&[1.0, 2.0, 3.0, 4.0], "W");
        weight.dims = vec![2, 2];
        let bias = vec_to_tensor_f32(&[0.1, 0.2], "B");

        let graph = GraphProto {
            node: vec![
                make_node("MatMul", &["X", "W"], &["matmul_out"], "matmul_0"),
                make_node("Add", &["matmul_out", "B"], &["Y"], "add_0"),
            ],
            initializer: vec![weight, bias],
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
        let result = ReifyConvForBert::new().transform(&mut ctx).unwrap();

        assert_eq!(result.transforms_applied, 1);
        assert!(!ctx.has_node("matmul_0"));
        assert!(!ctx.has_node("add_0"));

        // Check for Conv node
        let conv_nodes: Vec<&NodeProto> = ctx.nodes().filter(|n| n.op_type == "Conv").collect();
        assert_eq!(conv_nodes.len(), 1);

        // Check connectivity
        // X -> Unsqueeze -> Transpose -> Conv -> Squeeze -> Transpose -> Y
        let final_node = ctx.get_producer("Y").unwrap();
        assert_eq!(final_node.op_type, "Transpose");
    }
}
