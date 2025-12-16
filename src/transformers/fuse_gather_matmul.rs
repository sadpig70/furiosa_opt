use crate::error::{OnnxResult, TransformError};
use crate::graph::GraphContext;
use crate::proto::extensions::make_node;
use crate::transformers::common::{get_attr_i, OnnxTransformer, TransformResult};

/// Fuses Gather + MatMul into a single Gather
///
/// Transforms:
///   Gather(W1, indices) -> MatMul(W2)
#[derive(Debug, Default)]
pub struct FuseGatherMatMul;

impl FuseGatherMatMul {
    /// Create a new FuseGatherMatMul transformer
    pub fn new() -> Self {
        Self
    }
}

struct GatherMatMulPattern {
    gather_node: String,
    matmul_node: String,
    gather_data: String,
    gather_indices: String,
    matmul_weight: String,
    matmul_weight_idx: usize, // 0 or 1
}

impl OnnxTransformer for FuseGatherMatMul {
    fn name(&self) -> &'static str {
        "FuseGatherMatMul"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        loop {
            let patterns = self.perceive_patterns(ctx);
            if patterns.is_empty() {
                break;
            }

            let mut any_fused = false;
            for pattern in patterns {
                if self.process_validation(ctx, &pattern)
                    && self.response_fusion(ctx, &pattern)? {
                        result.patterns_matched += 1;
                        result.transforms_applied += 1;
                        result.nodes_eliminated += 1; // Gather + MatMul -> Gather (1 removed)
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
        ctx.nodes().any(|n| n.op_type == "Gather") && ctx.nodes().any(|n| n.op_type == "MatMul")
    }
}

impl FuseGatherMatMul {
    fn perceive_patterns(&self, ctx: &GraphContext) -> Vec<GatherMatMulPattern> {
        let mut patterns = Vec::new();

        for matmul in ctx.nodes() {
            if matmul.op_type != "MatMul" {
                continue;
            }

            // Check inputs: one must be from Gather, other must be initializer
            let input0 = &matmul.input[0];
            let input1 = &matmul.input[1];

            let (gather_node_name, weight_name, weight_idx) =
                if let Some(node) = ctx.get_producer(input0) {
                    if node.op_type == "Gather" && ctx.is_initializer(input1) {
                        (&node.name, input1, 1)
                    } else {
                        continue;
                    }
                } else if let Some(node) = ctx.get_producer(input1) {
                    if node.op_type == "Gather" && ctx.is_initializer(input0) {
                        (&node.name, input0, 0)
                    } else {
                        continue;
                    }
                } else {
                    continue;
                };

            let gather = ctx.get_node(gather_node_name).unwrap();
            if gather.input.len() < 2 {
                continue;
            }
            let gather_data = &gather.input[0];
            let gather_indices = &gather.input[1];

            // Gather data must be initializer
            if !ctx.is_initializer(gather_data) {
                continue;
            }

            patterns.push(GatherMatMulPattern {
                gather_node: gather_node_name.clone(),
                matmul_node: matmul.name.clone(),
                gather_data: gather_data.clone(),
                gather_indices: gather_indices.clone(),
                matmul_weight: weight_name.clone(),
                matmul_weight_idx: weight_idx,
            });
        }

        patterns
    }

    fn process_validation(&self, ctx: &GraphContext, pattern: &GatherMatMulPattern) -> bool {
        // 1. Check Dtypes (FLOAT)
        let gather_tensor = ctx.get_initializer(&pattern.gather_data).unwrap();
        let weight_tensor = ctx.get_initializer(&pattern.matmul_weight).unwrap();

        if gather_tensor.data_type != 1 || weight_tensor.data_type != 1 {
            return false; // Only FLOAT supported
        }

        // 2. Check Ranks (Must be 2)
        if gather_tensor.dims.len() != 2 || weight_tensor.dims.len() != 2 {
            return false;
        }

        // 3. Check Axis condition
        let gather_node = ctx.get_node(&pattern.gather_node).unwrap();
        let axis = get_attr_i(gather_node, "axis").unwrap_or(0);

        if axis == 0 {
            // Gather(axis=0) -> MatMul(W=input[1])
            // Gather(Table, Indices) @ W
            if pattern.matmul_weight_idx != 1 {
                return false;
            }
        } else if axis == 1 {
            // Gather(axis=1) -> MatMul(W=input[0])
            // W @ Gather(Table, Indices)
            if pattern.matmul_weight_idx != 0 {
                return false;
            }
        } else {
            // Other axes not supported for this fusion
            return false;
        }

        // 4. Check Topology (Gather output used only by MatMul?)
        // If Gather output is used by others, we can still fuse but we duplicate data.
        // Usually fusion implies we remove the intermediate.
        // Let's check if Gather output has other consumers.
        let gather_out = &gather_node.output[0];
        if ctx.get_consumers(gather_out).len() > 1 {
            return false;
        }

        true
    }

    fn response_fusion(
        &self,
        ctx: &mut GraphContext,
        pattern: &GatherMatMulPattern,
    ) -> OnnxResult<bool> {
        // 1. Compute Fused Table
        let table_tensor = ctx.get_initializer(&pattern.gather_data).unwrap();
        let weight_tensor = ctx.get_initializer(&pattern.matmul_weight).unwrap();

        let table_arr = crate::tensor::convert::tensor_to_array_f32(table_tensor)?;
        let weight_arr = crate::tensor::convert::tensor_to_array_f32(weight_tensor)?;

        // Convert to 2D Array
        let table_2d = table_arr
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| TransformError::InvalidNode("Table not 2D".into()))?;
        let weight_2d = weight_arr
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| TransformError::InvalidNode("Weight not 2D".into()))?;

        let fused_table = if pattern.matmul_weight_idx == 1 {
            // Table @ Weight
            table_2d.dot(&weight_2d)
        } else {
            // Weight @ Table
            weight_2d.dot(&table_2d)
        };

        // 2. Create New Initializer
        let fused_name = format!("{}_fused", pattern.gather_data);
        let fused_tensor =
            crate::tensor::convert::array_to_tensor_f32(&fused_table.into_dyn(), &fused_name);
        ctx.set_initializer(fused_tensor);

        // 3. Create New Gather Node
        let matmul_node = ctx.get_node(&pattern.matmul_node).unwrap();
        let output_name = matmul_node.output[0].clone();
        let gather_node = ctx.get_node(&pattern.gather_node).unwrap();

        // Copy attributes from original Gather
        let mut new_gather = make_node(
            "Gather",
            &[&fused_name, &pattern.gather_indices],
            &[&output_name],
            &format!("{}_fused", pattern.gather_node),
        );
        new_gather.attribute = gather_node.attribute.clone();

        // 4. Update Graph
        // Remove MatMul
        ctx.remove_node(&pattern.matmul_node);
        // Remove Gather (old)
        ctx.remove_node(&pattern.gather_node);

        // Add New Gather
        ctx.insert_node(new_gather);

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::{make_node, make_tensor_value_info};
    use crate::proto::{GraphProto, TensorProto};

    #[test]
    fn test_fuse_gather_matmul_axis0() {
        // Gather(axis=0) -> MatMul
        // Table: [2, 2]
        // Indices: [1] (value 0)
        // Weight: [2, 2]

        let table_data = vec![1.0, 2.0, 3.0, 4.0]; // [[1, 2], [3, 4]]
        let weight_data = vec![1.0, 0.0, 0.0, 1.0]; // Identity [[1, 0], [0, 1]]

        let table = TensorProto {
            name: "table".to_string(),
            dims: vec![2i64, 2i64],
            data_type: 1,
            float_data: table_data,
            ..Default::default()
        };

        let weight = TensorProto {
            name: "weight".to_string(),
            dims: vec![2i64, 2i64],
            data_type: 1,
            float_data: weight_data,
            ..Default::default()
        };

        let graph = GraphProto {
            node: vec![
                make_node("Gather", &["table", "indices"], &["gather_out"], "gather_0"),
                make_node("MatMul", &["gather_out", "weight"], &["Y"], "matmul_0"),
            ],
            input: vec![
                make_tensor_value_info("indices", 7, &[1]), // INT64
            ],
            output: vec![
                make_tensor_value_info("Y", 1, &[1, 2]), // FLOAT
            ],
            initializer: vec![table, weight],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let result = FuseGatherMatMul::new().transform(&mut ctx).unwrap();

        assert_eq!(result.transforms_applied, 1);
        assert!(!ctx.has_node("matmul_0"));
        assert!(!ctx.has_node("gather_0"));

        let new_gather = ctx.nodes().find(|n| n.op_type == "Gather").unwrap();
        assert_eq!(new_gather.input[0], "table_fused");

        let fused_tensor = ctx.get_initializer("table_fused").unwrap();
        assert_eq!(fused_tensor.float_data, vec![1.0, 2.0, 3.0, 4.0]); // Identity mult should be same
    }
}
