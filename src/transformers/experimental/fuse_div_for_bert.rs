use crate::error::{OnnxResult, TransformError};
use crate::graph::GraphContext;

use crate::transformers::common::{get_constant_tensor, OnnxTransformer, TransformResult};

/// Fuses Div for BERT
///
/// Fuses Div into Add bias or MatMul weights for BERT optimization.
#[derive(Debug, Default)]
pub struct FuseDivForBert;

impl FuseDivForBert {
    /// Create a new FuseDivForBert transformer
    pub fn new() -> Self {
        Self
    }

    fn fuse_div(&self, ctx: &mut GraphContext, div_name: &str) -> OnnxResult<bool> {
        let div_node = ctx
            .get_node(div_name)
            .ok_or_else(|| TransformError::InvalidNode(div_name.to_string()))?
            .clone();

        // Check pattern: Div -> MatMul -> Node -> Node -> Add
        // 1. Div input[1] must be scalar constant
        if div_node.input.len() != 2 {
            return Ok(false);
        }
        let scalar_tensor = if let Some(t) = get_constant_tensor(ctx, &div_node.input[1]) {
            t
        } else {
            return Ok(false);
        };

        // 2. Div input[0] comes from MatMul (prev_node_1)
        let prev_node_1_name = if let Some(n) = ctx.get_producer_name(&div_node.input[0]) {
            n
        } else {
            return Ok(false);
        };
        let prev_node_1 = ctx.get_node(prev_node_1_name).unwrap().clone();
        if prev_node_1.op_type != "MatMul" {
            return Ok(false);
        }

        // 3. MatMul input[0] comes from prev_node_2
        let prev_node_2_name = if let Some(n) = ctx.get_producer_name(&prev_node_1.input[0]) {
            n
        } else {
            return Ok(false);
        };
        let prev_node_2 = ctx.get_node(prev_node_2_name).unwrap().clone();

        // 4. prev_node_2 input[0] comes from prev_node_3
        let prev_node_3_name = if let Some(n) = ctx.get_producer_name(&prev_node_2.input[0]) {
            n
        } else {
            return Ok(false);
        };
        let prev_node_3 = ctx.get_node(prev_node_3_name).unwrap().clone();

        // 5. prev_node_3 input[0] comes from Add (prev_node_4)
        let prev_node_4_name = if let Some(n) = ctx.get_producer_name(&prev_node_3.input[0]) {
            n
        } else {
            return Ok(false);
        };
        let prev_node_4 = ctx.get_node(prev_node_4_name).unwrap().clone(); // Clone to mutate later
        if prev_node_4.op_type != "Add" {
            return Ok(false);
        }

        // 6. Add input[1] must be constant bias
        if prev_node_4.input.len() != 2 {
            return Ok(false);
        }
        let bias_tensor = if let Some(t) = get_constant_tensor(ctx, &prev_node_4.input[1]) {
            t
        } else {
            return Ok(false);
        };

        // Perform Fusion
        // Calculate new bias = bias / scalar
        let scalar_val = crate::tensor::convert::tensor_to_array_f32(scalar_tensor)?;
        let bias_val = crate::tensor::convert::tensor_to_array_f32(bias_tensor)?;

        // Ensure scalar is scalar or size 1
        if scalar_val.len() != 1 {
            return Ok(false);
        }
        let s = scalar_val.iter().next().unwrap();

        let new_bias_val = bias_val.mapv(|x| x / s);

        // Create new bias tensor
        let new_bias_name = format!("{}_div_fused", prev_node_4.input[1]);
        let new_bias_tensor =
            crate::tensor::convert::array_to_tensor_f32(&new_bias_val, &new_bias_name);

        ctx.set_initializer(new_bias_tensor);

        // Update Add node (prev_node_4)
        let mut new_add_node = prev_node_4.clone();
        new_add_node.input[1] = new_bias_name;
        ctx.replace_node(new_add_node);

        // Update MatMul (prev_node_1) output to bypass Div
        // MatMul output -> Div output
        let mut new_matmul_node = prev_node_1.clone();
        let _old_matmul_output = new_matmul_node.output[0].clone();
        let div_output = div_node.output[0].clone();

        new_matmul_node.output[0] = div_output;
        ctx.replace_node(new_matmul_node);

        // Remove Div node
        ctx.remove_node(div_name);

        // Note: We effectively changed the graph structure.
        // MatMul now outputs to what Div used to output.
        // The intermediate nodes (prev_node_2, prev_node_3) are untouched.

        Ok(true)
    }
}

impl OnnxTransformer for FuseDivForBert {
    fn name(&self) -> &'static str {
        "FuseDivForBert"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        // Find all Div nodes
        let div_nodes: Vec<String> = ctx
            .nodes()
            .filter(|n| n.op_type == "Div")
            .map(|n| n.name.clone())
            .collect();

        for div_name in div_nodes {
            if ctx.is_eliminated(&div_name) {
                continue;
            }

            if self.fuse_div(ctx, &div_name)? {
                result.transforms_applied += 1;
                result.record_elimination(&div_name);
            }
        }

        Ok(result)
    }

    fn is_applicable(&self, ctx: &GraphContext) -> bool {
        ctx.nodes().any(|n| n.op_type == "Div")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::{GraphProto, ValueInfoProto};
    use crate::tensor::convert::vec_to_tensor_f32;

    #[test]
    fn test_fuse_div_for_bert() {
        // Add (Bias=10.0) -> Identity -> Identity -> MatMul -> Div (Scalar=2.0)
        // Result: Add (Bias=5.0) -> ... -> MatMul (Output=Div_Out)

        let bias = vec_to_tensor_f32(&[10.0], "Bias");
        let scalar = vec_to_tensor_f32(&[2.0], "Scalar");

        let graph = GraphProto {
            node: vec![
                make_node("Add", &["X", "Bias"], &["add_out"], "add_0"),
                make_node("Identity", &["add_out"], &["id1_out"], "id_1"),
                make_node("Identity", &["id1_out"], &["id2_out"], "id_2"),
                make_node("MatMul", &["id2_out", "W"], &["matmul_out"], "matmul_0"),
                make_node("Div", &["matmul_out", "Scalar"], &["Y"], "div_0"),
            ],
            initializer: vec![bias, scalar],
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
        let result = FuseDivForBert::new().transform(&mut ctx).unwrap();

        assert_eq!(result.transforms_applied, 1);
        assert!(!ctx.has_node("div_0"));

        let add = ctx.get_node("add_0").unwrap();
        let new_bias_name = &add.input[1];
        let new_bias = ctx.get_initializer(new_bias_name).unwrap();
        let new_bias_val = crate::tensor::convert::tensor_to_array_f32(new_bias).unwrap();

        assert_eq!(new_bias_val[0], 5.0); // 10.0 / 2.0

        let matmul = ctx.get_node("matmul_0").unwrap();
        assert_eq!(matmul.output[0], "Y");
    }
}
