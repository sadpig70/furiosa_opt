//! Gemm fusion transformers
//!
//! Fuse Gemm with following Add/BiasAdd operations.

#![allow(missing_docs)]

use crate::error::{OnnxResult, TransformError};
use crate::graph::GraphContext;
use crate::transform::can_fuse;

use super::common::{set_attr_f, set_attr_i, OnnxTransformer, TransformResult};

/// Fuse Gemm + Add into single Gemm with bias
#[derive(Debug, Default)]
pub struct FuseGemmAdd;

impl FuseGemmAdd {
    pub fn new() -> Self {
        Self
    }

    fn try_fuse(
        &self,
        ctx: &mut GraphContext,
        gemm_name: &str,
        add_name: &str,
    ) -> OnnxResult<bool> {
        let gemm = ctx
            .get_node(gemm_name)
            .ok_or_else(|| TransformError::InvalidNode(gemm_name.to_string()))?
            .clone();
        let add = ctx
            .get_node(add_name)
            .ok_or_else(|| TransformError::InvalidNode(add_name.to_string()))?
            .clone();

        if gemm.input.len() >= 3 && !gemm.input[2].is_empty() {
            return Ok(false);
        }

        if !can_fuse(ctx, gemm_name, add_name) {
            return Ok(false);
        }

        // Check input rank (must be 2 for Gemm)
        if let Some(input0) = ctx.get_value_info(&gemm.input[0]) {
            if let Some(crate::proto::onnx::type_proto::Value::TensorType(tt)) =
                &input0.r#type.as_ref().and_then(|t| t.value.as_ref())
            {
                if let Some(shape) = &tt.shape {
                    if shape.dim.len() != 2 {
                        return Ok(false);
                    }
                } else {
                    return Ok(false); // Shape unknown
                }
            } else {
                return Ok(false); // Not a tensor?
            }
        } else {
            return Ok(false); // Value info missing
        }

        let gemm_output = gemm.output.first().cloned().unwrap_or_default();
        let bias_name = if add.input.len() >= 2 {
            if add.input[0] == gemm_output {
                add.input.get(1).cloned()
            } else if add.input[1] == gemm_output {
                add.input.first().cloned()
            } else {
                None
            }
        } else {
            None
        };

        let bias_name = match bias_name {
            Some(name) if !name.is_empty() => name,
            _ => return Ok(false),
        };

        if ctx.get_initializer(&bias_name).is_none() {
            return Ok(false);
        }

        if let Some(entry) = ctx.get_entry_mut(gemm_name) {
            while entry.node.input.len() < 3 {
                entry.node.input.push(String::new());
            }
            entry.node.input[2] = bias_name;

            if let Some(add_output) = add.output.first() {
                entry.node.output[0] = add_output.clone();
                ctx.update_producer(add_output, gemm_name);
            }
        }

        ctx.mark_eliminated(add_name);
        Ok(true)
    }
}

impl OnnxTransformer for FuseGemmAdd {
    fn name(&self) -> &'static str {
        "FuseGemmAdd"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        loop {
            let patterns: Vec<(String, String)> = ctx
                .nodes()
                .filter(|n| n.op_type == "Add" && !ctx.is_eliminated(&n.name))
                .filter_map(|add| {
                    for input in &add.input {
                        if let Some(pname) = ctx.get_producer_name(input) {
                            if let Some(pnode) = ctx.get_node(pname) {
                                if pnode.op_type == "Gemm" && !ctx.is_eliminated(pname) {
                                    return Some((pname.clone(), add.name.clone()));
                                }
                            }
                        }
                    }
                    None
                })
                .collect();

            if patterns.is_empty() {
                break;
            }
            result.patterns_matched += patterns.len();

            let mut any = false;
            for (g, a) in patterns {
                if ctx.is_eliminated(&g) || ctx.is_eliminated(&a) {
                    continue;
                }
                if self.try_fuse(ctx, &g, &a).unwrap_or(false) {
                    result.record(&g);
                    result.record_elimination(&a);
                    any = true;
                }
            }
            if !any {
                break;
            }
        }
        Ok(result)
    }

    fn is_applicable(&self, ctx: &GraphContext) -> bool {
        ctx.nodes().any(|n| n.op_type == "Gemm")
    }
}

/// Fuse MatMul + Add into Gemm
#[derive(Debug, Default)]
pub struct FuseMatMulAdd;

impl FuseMatMulAdd {
    pub fn new() -> Self {
        Self
    }

    fn try_fuse(
        &self,
        ctx: &mut GraphContext,
        matmul_name: &str,
        add_name: &str,
    ) -> OnnxResult<bool> {
        let matmul = ctx
            .get_node(matmul_name)
            .ok_or_else(|| TransformError::InvalidNode(matmul_name.to_string()))?
            .clone();
        let add = ctx
            .get_node(add_name)
            .ok_or_else(|| TransformError::InvalidNode(add_name.to_string()))?
            .clone();

        if !can_fuse(ctx, matmul_name, add_name) {
            return Ok(false);
        }

        // Check input rank (must be 2 for Gemm)
        if let Some(input0) = ctx.get_value_info(&matmul.input[0]) {
            if let Some(crate::proto::onnx::type_proto::Value::TensorType(tt)) =
                &input0.r#type.as_ref().and_then(|t| t.value.as_ref())
            {
                if let Some(shape) = &tt.shape {
                    if shape.dim.len() != 2 {
                        return Ok(false);
                    }
                } else {
                    return Ok(false); // Shape unknown
                }
            } else {
                return Ok(false); // Not a tensor?
            }
        } else {
            return Ok(false); // Value info missing
        }

        if matmul.input.len() < 2 {
            return Ok(false);
        }

        let a_input = matmul.input[0].clone();
        let b_input = matmul.input[1].clone();
        let matmul_output = matmul.output.first().cloned().unwrap_or_default();

        let bias_name = if add.input.len() >= 2 {
            if add.input[0] == matmul_output {
                add.input.get(1).cloned()
            } else if add.input[1] == matmul_output {
                add.input.first().cloned()
            } else {
                None
            }
        } else {
            None
        };

        let bias_name = match bias_name {
            Some(name) if !name.is_empty() => name,
            _ => return Ok(false),
        };

        if ctx.get_initializer(&bias_name).is_none() {
            return Ok(false);
        }

        if let Some(entry) = ctx.get_entry_mut(matmul_name) {
            entry.node.op_type = "Gemm".to_string();
            entry.node.input = vec![a_input, b_input, bias_name];
            set_attr_f(&mut entry.node, "alpha", 1.0);
            set_attr_f(&mut entry.node, "beta", 1.0);
            set_attr_i(&mut entry.node, "transA", 0);
            set_attr_i(&mut entry.node, "transB", 0);

            if let Some(add_output) = add.output.first() {
                entry.node.output[0] = add_output.clone();
                ctx.update_producer(add_output, matmul_name);
            }
        }

        ctx.mark_eliminated(add_name);
        Ok(true)
    }
}

impl OnnxTransformer for FuseMatMulAdd {
    fn name(&self) -> &'static str {
        "FuseMatMulAdd"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        loop {
            let patterns: Vec<(String, String)> = ctx
                .nodes()
                .filter(|n| n.op_type == "Add" && !ctx.is_eliminated(&n.name))
                .filter_map(|add| {
                    for input in &add.input {
                        if let Some(pname) = ctx.get_producer_name(input) {
                            if let Some(pnode) = ctx.get_node(pname) {
                                if pnode.op_type == "MatMul" && !ctx.is_eliminated(pname) {
                                    return Some((pname.clone(), add.name.clone()));
                                }
                            }
                        }
                    }
                    None
                })
                .collect();

            if patterns.is_empty() {
                break;
            }
            result.patterns_matched += patterns.len();

            let mut any = false;
            for (m, a) in patterns {
                if ctx.is_eliminated(&m) || ctx.is_eliminated(&a) {
                    continue;
                }
                if self.try_fuse(ctx, &m, &a).unwrap_or(false) {
                    result.record(&m);
                    result.record_elimination(&a);
                    any = true;
                }
            }
            if !any {
                break;
            }
        }
        Ok(result)
    }

    fn is_applicable(&self, ctx: &GraphContext) -> bool {
        ctx.nodes().any(|n| n.op_type == "MatMul")
    }
}

/// Combined Gemm/MatMul fusion
#[derive(Debug, Default)]
pub struct FuseGemmAll;

impl FuseGemmAll {
    pub fn new() -> Self {
        Self
    }
}

impl OnnxTransformer for FuseGemmAll {
    fn name(&self) -> &'static str {
        "FuseGemmAll"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();
        result.merge(FuseGemmAdd::new().transform(ctx)?);
        result.merge(FuseMatMulAdd::new().transform(ctx)?);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::{GraphProto, TensorProto, ValueInfoProto};

    #[test]
    fn test_fuse_gemm_add() {
        let bias = TensorProto {
            name: "bias".to_string(),
            dims: vec![10],
            data_type: 1,
            float_data: vec![0.0; 10],
            ..Default::default()
        };
        let graph = GraphProto {
            node: vec![
                make_node("Gemm", &["X", "W"], &["gemm_out"], "gemm_0"),
                make_node("Add", &["gemm_out", "bias"], &["Y"], "add_0"),
            ],
            input: vec![ValueInfoProto {
                name: "X".to_string(),
                ..Default::default()
            }],
            output: vec![ValueInfoProto {
                name: "Y".to_string(),
                ..Default::default()
            }],
            initializer: vec![bias],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let result = FuseGemmAdd::new().transform(&mut ctx).unwrap();

        assert_eq!(result.transforms_applied, 1);
        let gemm = ctx.get_node("gemm_0").unwrap();
        assert_eq!(gemm.input.len(), 3);
        assert_eq!(gemm.input[2], "bias");
    }

    #[test]
    fn test_fuse_matmul_add() {
        let bias = TensorProto {
            name: "bias".to_string(),
            dims: vec![10],
            data_type: 1,
            float_data: vec![0.0; 10],
            ..Default::default()
        };
        let graph = GraphProto {
            node: vec![
                make_node("MatMul", &["X", "W"], &["matmul_out"], "matmul_0"),
                make_node("Add", &["matmul_out", "bias"], &["Y"], "add_0"),
            ],
            input: vec![ValueInfoProto {
                name: "X".to_string(),
                ..Default::default()
            }],
            output: vec![ValueInfoProto {
                name: "Y".to_string(),
                ..Default::default()
            }],
            initializer: vec![bias],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let result = FuseMatMulAdd::new().transform(&mut ctx).unwrap();

        assert_eq!(result.transforms_applied, 1);
        let gemm = ctx.get_node("matmul_0").unwrap();
        assert_eq!(gemm.op_type, "Gemm");
    }
    #[test]
    fn test_fuse_matmul_add_invalid_rank() {
        let bias = TensorProto {
            name: "bias".to_string(),
            dims: vec![10],
            data_type: 1,
            float_data: vec![0.0; 10],
            ..Default::default()
        };
        // X에 대한 3D 입력 형상
        let input_shape = crate::proto::TypeProto {
            value: Some(crate::proto::type_proto::Value::TensorType(
                crate::proto::type_proto::Tensor {
                    elem_type: 1,
                    shape: Some(crate::proto::TensorShapeProto {
                        dim: vec![
                            crate::proto::tensor_shape_proto::Dimension {
                                value: Some(
                                    crate::proto::tensor_shape_proto::dimension::Value::DimValue(1),
                                ),
                                denotation: String::new(),
                            },
                            crate::proto::tensor_shape_proto::Dimension {
                                value: Some(
                                    crate::proto::tensor_shape_proto::dimension::Value::DimValue(
                                        128,
                                    ),
                                ),
                                denotation: String::new(),
                            },
                            crate::proto::tensor_shape_proto::Dimension {
                                value: Some(
                                    crate::proto::tensor_shape_proto::dimension::Value::DimValue(
                                        64,
                                    ),
                                ),
                                denotation: String::new(),
                            },
                        ],
                    }),
                },
            )),
            ..Default::default()
        };

        let graph = GraphProto {
            node: vec![
                make_node("MatMul", &["X", "W"], &["matmul_out"], "matmul_0"),
                make_node("Add", &["matmul_out", "bias"], &["Y"], "add_0"),
            ],
            input: vec![ValueInfoProto {
                name: "X".to_string(),
                r#type: Some(input_shape),
                ..Default::default()
            }],
            output: vec![ValueInfoProto {
                name: "Y".to_string(),
                ..Default::default()
            }],
            initializer: vec![bias],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let result = FuseMatMulAdd::new().transform(&mut ctx).unwrap();

        // 입력 X가 3D이므로 융합되지 않아야 함
        assert_eq!(result.transforms_applied, 0);
        let matmul = ctx.get_node("matmul_0").unwrap();
        assert_eq!(matmul.op_type, "MatMul");
    }
}
