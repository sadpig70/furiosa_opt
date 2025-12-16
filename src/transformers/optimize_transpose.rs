//! Transpose optimization transformers

#![allow(missing_docs)]

use crate::error::{OnnxResult, TransformError};
use crate::graph::GraphContext;
use crate::transform::can_fuse;

use super::common::{get_attr_ints, set_attr_ints, OnnxTransformer, TransformResult};

/// Merge consecutive Transpose operations
#[derive(Debug, Default)]
pub struct MergeTranspose;

impl MergeTranspose {
    pub fn new() -> Self {
        Self
    }

    fn compose_perms(perm1: &[i64], perm2: &[i64]) -> Vec<i64> {
        perm2.iter().map(|&i| perm1[i as usize]).collect()
    }

    fn is_identity_perm(perm: &[i64]) -> bool {
        perm.iter().enumerate().all(|(i, &p)| p == i as i64)
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
        let second = ctx
            .get_node(second_name)
            .ok_or_else(|| TransformError::InvalidNode(second_name.to_string()))?
            .clone();

        if !can_fuse(ctx, first_name, second_name) {
            return Ok(false);
        }

        let perm1 = get_attr_ints(&first, "perm").map(|p| p.to_vec());
        let perm2 = get_attr_ints(&second, "perm").map(|p| p.to_vec());

        match (perm1, perm2) {
            (Some(p1), Some(p2)) if p1.len() == p2.len() => {
                let combined = Self::compose_perms(&p1, &p2);

                if Self::is_identity_perm(&combined) {
                    // Both cancel out - bypass both
                    let data_input = first.input.first().cloned().unwrap_or_default();
                    let output = second.output.first().cloned().unwrap_or_default();

                    // Update consumers
                    let consumers: Vec<String> = ctx
                        .nodes()
                        .filter(|n| n.input.contains(&output))
                        .map(|n| n.name.clone())
                        .collect();

                    for consumer in consumers {
                        if let Some(entry) = ctx.get_entry_mut(&consumer) {
                            for input in &mut entry.node.input {
                                if *input == output {
                                    *input = data_input.clone();
                                }
                            }
                        }
                    }

                    ctx.mark_eliminated(first_name);
                    ctx.mark_eliminated(second_name);
                } else {
                    // Merge into second
                    let data_input = first.input.first().cloned().unwrap_or_default();

                    if let Some(entry) = ctx.get_entry_mut(second_name) {
                        if !entry.node.input.is_empty() {
                            entry.node.input[0] = data_input;
                        }
                        set_attr_ints(&mut entry.node, "perm", combined);
                    }

                    ctx.mark_eliminated(first_name);
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}

impl OnnxTransformer for MergeTranspose {
    fn name(&self) -> &'static str {
        "MergeTranspose"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        loop {
            let patterns: Vec<(String, String)> = ctx
                .nodes()
                .filter(|n| n.op_type == "Transpose" && !ctx.is_eliminated(&n.name))
                .filter_map(|second| {
                    let input = second.input.first()?;
                    let pname = ctx.get_producer_name(input)?;
                    let pnode = ctx.get_node(pname)?;
                    if pnode.op_type == "Transpose" && !ctx.is_eliminated(pname) {
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
        ctx.nodes().filter(|n| n.op_type == "Transpose").count() >= 2
    }
}

/// Cancel inverse Transpose pairs
#[derive(Debug, Default)]
pub struct CancelInverseTranspose;

impl CancelInverseTranspose {
    pub fn new() -> Self {
        Self
    }

    fn inverse_perm(perm: &[i64]) -> Vec<i64> {
        let mut inv = vec![0i64; perm.len()];
        for (i, &p) in perm.iter().enumerate() {
            inv[p as usize] = i as i64;
        }
        inv
    }

    fn are_inverse(perm1: &[i64], perm2: &[i64]) -> bool {
        if perm1.len() != perm2.len() {
            return false;
        }
        Self::inverse_perm(perm1) == perm2
    }
}

impl OnnxTransformer for CancelInverseTranspose {
    fn name(&self) -> &'static str {
        "CancelInverseTranspose"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        let patterns: Vec<(String, String)> = ctx
            .nodes()
            .filter(|n| n.op_type == "Transpose" && !ctx.is_eliminated(&n.name))
            .filter_map(|second| {
                let input = second.input.first()?;
                let pname = ctx.get_producer_name(input)?;
                let first = ctx.get_node(pname)?;
                if first.op_type != "Transpose" || ctx.is_eliminated(pname) {
                    return None;
                }

                let perm1 = get_attr_ints(first, "perm")?;
                let perm2 = get_attr_ints(second, "perm")?;

                if Self::are_inverse(perm1, perm2) {
                    Some((pname.clone(), second.name.clone()))
                } else {
                    None
                }
            })
            .collect();

        result.patterns_matched = patterns.len();

        for (first_name, second_name) in patterns {
            if ctx.is_eliminated(&first_name) || ctx.is_eliminated(&second_name) {
                continue;
            }
            if !can_fuse(ctx, &first_name, &second_name) {
                continue;
            }

            if let (Some(first), Some(second)) = (
                ctx.get_node(&first_name).cloned(),
                ctx.get_node(&second_name).cloned(),
            ) {
                let data_input = first.input.first().cloned().unwrap_or_default();
                let output = second.output.first().cloned().unwrap_or_default();

                let consumers: Vec<String> = ctx
                    .nodes()
                    .filter(|n| n.input.contains(&output))
                    .map(|n| n.name.clone())
                    .collect();

                for consumer in consumers {
                    if let Some(entry) = ctx.get_entry_mut(&consumer) {
                        for input in &mut entry.node.input {
                            if *input == output {
                                *input = data_input.clone();
                            }
                        }
                    }
                }

                ctx.mark_eliminated(&first_name);
                ctx.mark_eliminated(&second_name);
                result.record_elimination(&first_name);
                result.record_elimination(&second_name);
            }
        }
        Ok(result)
    }
}

/// Sink Transpose through element-wise operations
#[derive(Debug, Default)]
pub struct SinkTranspose;

impl SinkTranspose {
    pub fn new() -> Self {
        Self
    }
}

impl OnnxTransformer for SinkTranspose {
    fn name(&self) -> &'static str {
        "SinkTranspose"
    }

    fn transform(&self, _ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        // Simplified - just return empty result
        Ok(TransformResult::new())
    }
}

/// Combined Transpose optimizer
#[derive(Debug, Default)]
pub struct OptimizeTransposeAll;

impl OptimizeTransposeAll {
    pub fn new() -> Self {
        Self
    }
}

impl OnnxTransformer for OptimizeTransposeAll {
    fn name(&self) -> &'static str {
        "OptimizeTransposeAll"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();
        result.merge(CancelInverseTranspose::new().transform(ctx)?);
        result.merge(MergeTranspose::new().transform(ctx)?);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::extensions::make_node;
    use crate::proto::{AttributeProto, GraphProto, ValueInfoProto};

    fn make_transpose_node(
        name: &str,
        input: &str,
        output: &str,
        perm: &[i64],
    ) -> crate::proto::NodeProto {
        let mut node = make_node("Transpose", &[input], &[output], name);
        node.attribute.push(AttributeProto {
            name: "perm".to_string(),
            ints: perm.to_vec(),
            r#type: 7,
            ..Default::default()
        });
        node
    }

    #[test]
    fn test_compose_perms() {
        let perm1 = vec![1, 0, 2];
        let perm2 = vec![2, 0, 1];
        let combined = MergeTranspose::compose_perms(&perm1, &perm2);
        assert_eq!(combined, vec![2, 1, 0]);
    }

    #[test]
    fn test_inverse_perm() {
        let perm = vec![2, 0, 1];
        let inv = CancelInverseTranspose::inverse_perm(&perm);
        assert_eq!(inv, vec![1, 2, 0]);
    }

    #[test]
    fn test_merge_transpose_to_identity() {
        let graph = GraphProto {
            node: vec![
                make_transpose_node("trans_0", "X", "mid", &[0, 2, 1]),
                make_transpose_node("trans_1", "mid", "Y", &[0, 2, 1]),
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
        let result = MergeTranspose::new().transform(&mut ctx).unwrap();

        assert!(result.nodes_eliminated >= 1);
    }

    #[test]
    fn test_cancel_inverse_transpose() {
        let graph = GraphProto {
            node: vec![
                make_transpose_node("trans_0", "X", "mid", &[1, 2, 0]),
                make_transpose_node("trans_1", "mid", "Y", &[2, 0, 1]),
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
        let result = CancelInverseTranspose::new().transform(&mut ctx).unwrap();

        assert_eq!(result.nodes_eliminated, 2);
    }
}
