//! Reshape optimization transformers

#![allow(missing_docs)]

use crate::error::{OnnxResult, TransformError};
use crate::graph::GraphContext;
use crate::transform::can_fuse;

use super::common::{OnnxTransformer, TransformResult};

/// Merge consecutive Reshape operations
#[derive(Debug, Default)]
pub struct MergeReshape;

impl MergeReshape {
    pub fn new() -> Self {
        Self
    }

    fn try_merge(
        &self,
        ctx: &mut GraphContext,
        first_name: &str,
        second_name: &str,
    ) -> OnnxResult<bool> {
        let first = ctx
            .get_node(first_name)
            .ok_or_else(|| TransformError::InvalidNode(first_name.to_string()))?
            .clone();
        let _second = ctx
            .get_node(second_name)
            .ok_or_else(|| TransformError::InvalidNode(second_name.to_string()))?
            .clone();

        if !can_fuse(ctx, first_name, second_name) {
            return Ok(false);
        }

        let data_input = first.input.first().cloned().unwrap_or_default();
        if data_input.is_empty() {
            return Ok(false);
        }

        if let Some(entry) = ctx.get_entry_mut(second_name) {
            if !entry.node.input.is_empty() {
                entry.node.input[0] = data_input;
            }
        }

        ctx.mark_eliminated(first_name);
        Ok(true)
    }
}

impl OnnxTransformer for MergeReshape {
    fn name(&self) -> &'static str {
        "MergeReshape"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        loop {
            let patterns: Vec<(String, String)> = ctx
                .nodes()
                .filter(|n| n.op_type == "Reshape" && !ctx.is_eliminated(&n.name))
                .filter_map(|second| {
                    let input = second.input.first()?;
                    let pname = ctx.get_producer_name(input)?;
                    let pnode = ctx.get_node(pname)?;
                    if pnode.op_type == "Reshape" && !ctx.is_eliminated(pname) {
                        Some((pname.clone(), second.name.clone()))
                    } else {
                        None
                    }
                })
                .collect();

            if patterns.is_empty() {
                break;
            }
            result.patterns_matched += patterns.len();

            let mut any = false;
            for (f, s) in patterns {
                if ctx.is_eliminated(&f) || ctx.is_eliminated(&s) {
                    continue;
                }
                if self.try_merge(ctx, &f, &s).unwrap_or(false) {
                    result.record(&s);
                    result.record_elimination(&f);
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
        ctx.nodes().filter(|n| n.op_type == "Reshape").count() >= 2
    }
}

/// Merge consecutive Squeeze operations
#[derive(Debug, Default)]
pub struct MergeSqueeze;

impl MergeSqueeze {
    pub fn new() -> Self {
        Self
    }
}

impl OnnxTransformer for MergeSqueeze {
    fn name(&self) -> &'static str {
        "MergeSqueeze"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        loop {
            let patterns: Vec<(String, String)> = ctx
                .nodes()
                .filter(|n| n.op_type == "Squeeze" && !ctx.is_eliminated(&n.name))
                .filter_map(|second| {
                    let input = second.input.first()?;
                    let pname = ctx.get_producer_name(input)?;
                    let pnode = ctx.get_node(pname)?;
                    if pnode.op_type == "Squeeze" && !ctx.is_eliminated(pname) {
                        Some((pname.clone(), second.name.clone()))
                    } else {
                        None
                    }
                })
                .collect();

            if patterns.is_empty() {
                break;
            }
            result.patterns_matched += patterns.len();

            let mut any = false;
            for (f, s) in patterns {
                if ctx.is_eliminated(&f) || ctx.is_eliminated(&s) {
                    continue;
                }
                if !can_fuse(ctx, &f, &s) {
                    continue;
                }

                if let Some(first_node) = ctx.get_node(&f).cloned() {
                    let data_input = first_node.input.first().cloned().unwrap_or_default();
                    if let Some(entry) = ctx.get_entry_mut(&s) {
                        if !entry.node.input.is_empty() {
                            entry.node.input[0] = data_input;
                        }
                    }
                    ctx.mark_eliminated(&f);
                    result.record(&s);
                    result.record_elimination(&f);
                    any = true;
                }
            }
            if !any {
                break;
            }
        }
        Ok(result)
    }
}

/// Merge consecutive Unsqueeze operations
#[derive(Debug, Default)]
pub struct MergeUnsqueeze;

impl MergeUnsqueeze {
    pub fn new() -> Self {
        Self
    }
}

impl OnnxTransformer for MergeUnsqueeze {
    fn name(&self) -> &'static str {
        "MergeUnsqueeze"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        loop {
            let patterns: Vec<(String, String)> = ctx
                .nodes()
                .filter(|n| n.op_type == "Unsqueeze" && !ctx.is_eliminated(&n.name))
                .filter_map(|second| {
                    let input = second.input.first()?;
                    let pname = ctx.get_producer_name(input)?;
                    let pnode = ctx.get_node(pname)?;
                    if pnode.op_type == "Unsqueeze" && !ctx.is_eliminated(pname) {
                        Some((pname.clone(), second.name.clone()))
                    } else {
                        None
                    }
                })
                .collect();

            if patterns.is_empty() {
                break;
            }
            result.patterns_matched += patterns.len();

            let mut any = false;
            for (f, s) in patterns {
                if ctx.is_eliminated(&f) || ctx.is_eliminated(&s) {
                    continue;
                }
                if !can_fuse(ctx, &f, &s) {
                    continue;
                }

                if let Some(first_node) = ctx.get_node(&f).cloned() {
                    let data_input = first_node.input.first().cloned().unwrap_or_default();
                    if let Some(entry) = ctx.get_entry_mut(&s) {
                        if !entry.node.input.is_empty() {
                            entry.node.input[0] = data_input;
                        }
                    }
                    ctx.mark_eliminated(&f);
                    result.record(&s);
                    result.record_elimination(&f);
                    any = true;
                }
            }
            if !any {
                break;
            }
        }
        Ok(result)
    }
}

/// Eliminate Flatten followed by Reshape
#[derive(Debug, Default)]
pub struct SimplifyFlattenReshape;

impl SimplifyFlattenReshape {
    pub fn new() -> Self {
        Self
    }
}

impl OnnxTransformer for SimplifyFlattenReshape {
    fn name(&self) -> &'static str {
        "SimplifyFlattenReshape"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        let patterns: Vec<(String, String)> = ctx
            .nodes()
            .filter(|n| n.op_type == "Reshape" && !ctx.is_eliminated(&n.name))
            .filter_map(|reshape| {
                let input = reshape.input.first()?;
                let pname = ctx.get_producer_name(input)?;
                let pnode = ctx.get_node(pname)?;
                if pnode.op_type == "Flatten" && !ctx.is_eliminated(pname) {
                    Some((pname.clone(), reshape.name.clone()))
                } else {
                    None
                }
            })
            .collect();

        result.patterns_matched = patterns.len();

        for (flatten_name, reshape_name) in patterns {
            if ctx.is_eliminated(&flatten_name) || ctx.is_eliminated(&reshape_name) {
                continue;
            }
            if !can_fuse(ctx, &flatten_name, &reshape_name) {
                continue;
            }

            if let Some(flatten) = ctx.get_node(&flatten_name).cloned() {
                let data_input = flatten.input.first().cloned().unwrap_or_default();
                if let Some(entry) = ctx.get_entry_mut(&reshape_name) {
                    if !entry.node.input.is_empty() {
                        entry.node.input[0] = data_input;
                    }
                }
                ctx.mark_eliminated(&flatten_name);
                result.record(&reshape_name);
                result.record_elimination(&flatten_name);
            }
        }
        Ok(result)
    }
}

/// Combined reshape optimizer
#[derive(Debug, Default)]
pub struct OptimizeReshapeAll;

impl OptimizeReshapeAll {
    pub fn new() -> Self {
        Self
    }
}

impl OnnxTransformer for OptimizeReshapeAll {
    fn name(&self) -> &'static str {
        "OptimizeReshapeAll"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();
        result.merge(MergeReshape::new().transform(ctx)?);
        result.merge(MergeSqueeze::new().transform(ctx)?);
        result.merge(MergeUnsqueeze::new().transform(ctx)?);
        result.merge(SimplifyFlattenReshape::new().transform(ctx)?);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::{GraphProto, TensorProto, ValueInfoProto};

    #[test]
    fn test_merge_reshape() {
        let shape1 = TensorProto {
            name: "shape1".to_string(),
            dims: vec![2],
            data_type: 7,
            int64_data: vec![2, 8],
            ..Default::default()
        };
        let shape2 = TensorProto {
            name: "shape2".to_string(),
            dims: vec![2],
            data_type: 7,
            int64_data: vec![4, 4],
            ..Default::default()
        };

        let graph = GraphProto {
            node: vec![
                make_node("Reshape", &["X", "shape1"], &["reshape1_out"], "reshape_0"),
                make_node("Reshape", &["reshape1_out", "shape2"], &["Y"], "reshape_1"),
            ],
            input: vec![ValueInfoProto {
                name: "X".to_string(),
                ..Default::default()
            }],
            output: vec![ValueInfoProto {
                name: "Y".to_string(),
                ..Default::default()
            }],
            initializer: vec![shape1, shape2],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let result = MergeReshape::new().transform(&mut ctx).unwrap();

        assert_eq!(result.nodes_eliminated, 1);
        assert!(ctx.is_eliminated("reshape_0"));
        let reshape = ctx.get_node("reshape_1").unwrap();
        assert_eq!(reshape.input[0], "X");
    }
}
