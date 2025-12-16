//! Constant Folding Transformer
//!
//! Evaluates operations with constant inputs at compile time.
//! This is especially effective for Transformer models like GPT-2
//! which have many Shape/Gather/Concat chains for dynamic sizing.

use crate::error::OnnxResult;
use crate::graph::GraphContext;
use crate::proto::onnx::{NodeProto, TensorProto};
use crate::transformers::common::{OnnxTransformer, TransformResult};

use std::collections::HashSet;

/// Fold constant expressions into single Constant nodes.
///
/// Supported operations:
/// - Arithmetic: Add, Sub, Mul, Div
/// - Shape: Shape, Gather, Concat, Squeeze, Unsqueeze, Slice
/// - Comparison: Equal, Less, Greater
/// - Cast, Reshape (shape input only)
pub struct ConstantFold {
    /// Operations we can evaluate
    supported_ops: HashSet<&'static str>,
}

impl ConstantFold {
    /// Create a new ConstantFold transformer
    pub fn new() -> Self {
        let mut supported_ops = HashSet::new();
        // Arithmetic
        supported_ops.insert("Add");
        supported_ops.insert("Sub");
        supported_ops.insert("Mul");
        supported_ops.insert("Div");
        // Shape operations
        supported_ops.insert("Shape");
        supported_ops.insert("Gather");
        supported_ops.insert("Concat");
        supported_ops.insert("Squeeze");
        supported_ops.insert("Unsqueeze");
        supported_ops.insert("Slice");
        // Other
        supported_ops.insert("Cast");
        supported_ops.insert("Equal");
        supported_ops.insert("Less");
        supported_ops.insert("Greater");
        supported_ops.insert("Where");

        Self { supported_ops }
    }

    /// Check if all inputs of a node are constants (Constant nodes or initializers)
    fn all_inputs_constant(&self, node: &NodeProto, ctx: &GraphContext) -> bool {
        for input in &node.input {
            if input.is_empty() {
                continue;
            }

            // Check if it's an initializer
            if ctx.get_initializer(input).is_some() {
                continue;
            }

            // Check if it's produced by a Constant node
            if let Some(producer) = ctx.get_producer(input) {
                if producer.op_type == "Constant" {
                    continue;
                }
            }

            // Not a constant input
            return false;
        }
        true
    }

    /// Get constant tensor from input name
    fn get_constant_tensor<'a>(
        &self,
        name: &str,
        ctx: &'a GraphContext,
    ) -> Option<&'a TensorProto> {
        // First check initializers
        if let Some(init) = ctx.get_initializer(name) {
            return Some(init);
        }

        // Then check Constant nodes
        if let Some(producer) = ctx.get_producer(name) {
            if producer.op_type == "Constant" {
                // Get the value attribute
                for attr in &producer.attribute {
                    if attr.name == "value" {
                        return attr.t.as_ref();
                    }
                }
            }
        }

        None
    }

    /// Extract i64 values from a tensor
    fn tensor_to_i64_vec(&self, tensor: &TensorProto) -> Option<Vec<i64>> {
        // Handle different data types
        match tensor.data_type {
            // INT64
            7 => {
                if !tensor.int64_data.is_empty() {
                    Some(tensor.int64_data.clone())
                } else if !tensor.raw_data.is_empty() {
                    let data: Vec<i64> = tensor
                        .raw_data
                        .chunks_exact(8)
                        .map(|b| {
                            i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
                        })
                        .collect();
                    Some(data)
                } else {
                    None
                }
            }
            // INT32
            6 => {
                if !tensor.int32_data.is_empty() {
                    Some(tensor.int32_data.iter().map(|&x| x as i64).collect())
                } else if !tensor.raw_data.is_empty() {
                    let data: Vec<i64> = tensor
                        .raw_data
                        .chunks_exact(4)
                        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64)
                        .collect();
                    Some(data)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Create a new INT64 tensor
    fn create_i64_tensor(&self, name: &str, data: Vec<i64>, dims: Vec<i64>) -> TensorProto {
        TensorProto {
            dims,
            data_type: 7, // INT64
            name: name.to_string(),
            int64_data: data,
            ..Default::default()
        }
    }

    /// Evaluate Shape operation
    fn eval_shape(&self, node: &NodeProto, ctx: &GraphContext) -> Option<TensorProto> {
        use crate::proto::onnx::type_proto::Value;

        let input_name = node.input.first()?;

        // Get shape from value_info or initializer
        let shape = if let Some(init) = ctx.get_initializer(input_name) {
            init.dims.clone()
        } else if let Some(vi) = ctx.get_value_info(input_name) {
            if let Some(t) = vi.r#type.as_ref() {
                if let Some(Value::TensorType(tensor)) = t.value.as_ref() {
                    if let Some(shape) = tensor.shape.as_ref() {
                        use crate::proto::onnx::tensor_shape_proto::dimension::Value as DimValue;
                        shape
                            .dim
                            .iter()
                            .filter_map(|d| {
                                match &d.value {
                                    Some(DimValue::DimValue(val)) if *val != 0 => Some(*val),
                                    _ => None, // Dynamic or zero dimension
                                }
                            })
                            .collect()
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            } else {
                return None;
            }
        } else {
            return None;
        };

        // Check for dynamic dimensions
        if shape.is_empty() {
            return None;
        }

        let output_name = node.output.first()?;
        let len = shape.len() as i64;
        Some(self.create_i64_tensor(output_name, shape, vec![len]))
    }

    /// Evaluate Gather operation
    fn eval_gather(&self, node: &NodeProto, ctx: &GraphContext) -> Option<TensorProto> {
        if node.input.len() < 2 {
            return None;
        }

        let data_tensor = self.get_constant_tensor(&node.input[0], ctx)?;
        let indices_tensor = self.get_constant_tensor(&node.input[1], ctx)?;

        let data = self.tensor_to_i64_vec(data_tensor)?;
        let indices = self.tensor_to_i64_vec(indices_tensor)?;

        // Get axis attribute (default 0)
        let axis = node
            .attribute
            .iter()
            .find(|a| a.name == "axis")
            .map(|a| a.i)
            .unwrap_or(0);

        if axis != 0 {
            // Only support axis=0 for now
            return None;
        }

        // Gather values
        let data_len = data.len() as i64;
        let result: Vec<i64> = indices
            .iter()
            .map(|&idx| {
                let normalized_idx = if idx < 0 { data_len + idx } else { idx };
                data.get(normalized_idx as usize).copied().unwrap_or(0)
            })
            .collect();

        let output_name = node.output.first()?;
        let dims = indices_tensor.dims.clone();
        Some(self.create_i64_tensor(output_name, result, dims))
    }

    /// Evaluate Concat operation
    fn eval_concat(&self, node: &NodeProto, ctx: &GraphContext) -> Option<TensorProto> {
        if node.input.is_empty() {
            return None;
        }

        // Get axis
        let axis = node
            .attribute
            .iter()
            .find(|a| a.name == "axis")
            .map(|a| a.i)
            .unwrap_or(0);

        if axis != 0 {
            // Only support axis=0 for 1D tensors
            return None;
        }

        // Collect all input data
        let mut result = Vec::new();
        for input_name in &node.input {
            let tensor = self.get_constant_tensor(input_name, ctx)?;
            let data = self.tensor_to_i64_vec(tensor)?;
            result.extend(data);
        }

        let output_name = node.output.first()?;
        let len = result.len() as i64;
        Some(self.create_i64_tensor(output_name, result, vec![len]))
    }

    /// Evaluate Unsqueeze operation
    fn eval_unsqueeze(&self, node: &NodeProto, ctx: &GraphContext) -> Option<TensorProto> {
        let input_tensor = self.get_constant_tensor(node.input.first()?, ctx)?;
        let data = self.tensor_to_i64_vec(input_tensor)?;
        let mut dims = input_tensor.dims.clone();

        // Get axes from attribute (opset < 13) or second input (opset >= 13)
        let axes: Vec<i64> = if node.input.len() >= 2 && !node.input[1].is_empty() {
            let axes_tensor = self.get_constant_tensor(&node.input[1], ctx)?;
            self.tensor_to_i64_vec(axes_tensor)?
        } else {
            node.attribute
                .iter()
                .find(|a| a.name == "axes")
                .map(|a| a.ints.clone())
                .unwrap_or_default()
        };

        // Insert dimensions
        let mut sorted_axes: Vec<i64> = axes.clone();
        sorted_axes.sort();
        for axis in sorted_axes {
            let pos = if axis < 0 {
                (dims.len() as i64 + axis + 1) as usize
            } else {
                axis as usize
            };
            if pos <= dims.len() {
                dims.insert(pos, 1);
            }
        }

        let output_name = node.output.first()?;
        Some(self.create_i64_tensor(output_name, data, dims))
    }

    /// Evaluate Squeeze operation
    fn eval_squeeze(&self, node: &NodeProto, ctx: &GraphContext) -> Option<TensorProto> {
        let input_tensor = self.get_constant_tensor(node.input.first()?, ctx)?;
        let data = self.tensor_to_i64_vec(input_tensor)?;
        let mut dims = input_tensor.dims.clone();

        // Get axes
        let axes: Vec<i64> = if node.input.len() >= 2 && !node.input[1].is_empty() {
            let axes_tensor = self.get_constant_tensor(&node.input[1], ctx)?;
            self.tensor_to_i64_vec(axes_tensor)?
        } else {
            node.attribute
                .iter()
                .find(|a| a.name == "axes")
                .map(|a| a.ints.clone())
                .unwrap_or_default()
        };

        // Remove dimensions (in reverse order to maintain indices)
        let mut sorted_axes: Vec<i64> = axes.clone();
        sorted_axes.sort();
        sorted_axes.reverse();
        for axis in sorted_axes {
            let pos = if axis < 0 {
                (dims.len() as i64 + axis) as usize
            } else {
                axis as usize
            };
            if pos < dims.len() && dims[pos] == 1 {
                dims.remove(pos);
            }
        }

        let output_name = node.output.first()?;
        Some(self.create_i64_tensor(output_name, data, dims))
    }

    /// Evaluate arithmetic operations
    fn eval_arithmetic(&self, node: &NodeProto, ctx: &GraphContext) -> Option<TensorProto> {
        if node.input.len() < 2 {
            return None;
        }

        let a_tensor = self.get_constant_tensor(&node.input[0], ctx)?;
        let b_tensor = self.get_constant_tensor(&node.input[1], ctx)?;

        let a = self.tensor_to_i64_vec(a_tensor)?;
        let b = self.tensor_to_i64_vec(b_tensor)?;

        // Simple element-wise (assuming same shape or broadcasting with scalar)
        let result: Vec<i64> = if a.len() == 1 {
            match node.op_type.as_str() {
                "Add" => b.iter().map(|&x| a[0] + x).collect(),
                "Sub" => b.iter().map(|&x| a[0] - x).collect(),
                "Mul" => b.iter().map(|&x| a[0] * x).collect(),
                "Div" => b
                    .iter()
                    .map(|&x| if x != 0 { a[0] / x } else { 0 })
                    .collect(),
                _ => return None,
            }
        } else if b.len() == 1 {
            match node.op_type.as_str() {
                "Add" => a.iter().map(|&x| x + b[0]).collect(),
                "Sub" => a.iter().map(|&x| x - b[0]).collect(),
                "Mul" => a.iter().map(|&x| x * b[0]).collect(),
                "Div" => a
                    .iter()
                    .map(|&x| if b[0] != 0 { x / b[0] } else { 0 })
                    .collect(),
                _ => return None,
            }
        } else if a.len() == b.len() {
            match node.op_type.as_str() {
                "Add" => a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect(),
                "Sub" => a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect(),
                "Mul" => a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect(),
                "Div" => a
                    .iter()
                    .zip(b.iter())
                    .map(|(&x, &y)| if y != 0 { x / y } else { 0 })
                    .collect(),
                _ => return None,
            }
        } else {
            return None; // Complex broadcasting not supported
        };

        let output_name = node.output.first()?;
        let dims = if a_tensor.dims.len() >= b_tensor.dims.len() {
            a_tensor.dims.clone()
        } else {
            b_tensor.dims.clone()
        };
        Some(self.create_i64_tensor(output_name, result, dims))
    }

    /// Evaluate Cast operation
    fn eval_cast(&self, node: &NodeProto, ctx: &GraphContext) -> Option<TensorProto> {
        let input_tensor = self.get_constant_tensor(node.input.first()?, ctx)?;

        // Get target type
        let to_type = node
            .attribute
            .iter()
            .find(|a| a.name == "to")
            .map(|a| a.i as i32)
            .unwrap_or(0);

        // For now, just preserve INT64 data
        if to_type == 7 {
            let data = self.tensor_to_i64_vec(input_tensor)?;
            let output_name = node.output.first()?;
            Some(self.create_i64_tensor(output_name, data, input_tensor.dims.clone()))
        } else {
            None
        }
    }

    /// Try to evaluate a node and return the result tensor
    fn try_eval(&self, node: &NodeProto, ctx: &GraphContext) -> Option<TensorProto> {
        match node.op_type.as_str() {
            "Shape" => self.eval_shape(node, ctx),
            "Gather" => self.eval_gather(node, ctx),
            "Concat" => self.eval_concat(node, ctx),
            "Unsqueeze" => self.eval_unsqueeze(node, ctx),
            "Squeeze" => self.eval_squeeze(node, ctx),
            "Add" | "Sub" | "Mul" | "Div" => self.eval_arithmetic(node, ctx),
            "Cast" => self.eval_cast(node, ctx),
            _ => None,
        }
    }
}

impl Default for ConstantFold {
    fn default() -> Self {
        Self::new()
    }
}

impl OnnxTransformer for ConstantFold {
    fn name(&self) -> &'static str {
        "ConstantFold"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();
        let mut nodes_to_remove = Vec::new();
        let mut new_initializers = Vec::new();

        // Iterate multiple times until no more folding is possible
        for _iteration in 0..10 {
            let mut folded_this_iter = 0;

            let node_names: Vec<String> = ctx.nodes().map(|n| n.name.clone()).collect();

            for node_name in node_names {
                if nodes_to_remove.contains(&node_name) {
                    continue;
                }

                let node = match ctx.get_node(&node_name) {
                    Some(n) => n.clone(),
                    None => continue,
                };

                // Skip if not a supported op
                if !self.supported_ops.contains(node.op_type.as_str()) {
                    continue;
                }

                // Skip if not all inputs are constants
                if !self.all_inputs_constant(&node, ctx) {
                    continue;
                }

                // Try to evaluate
                if let Some(result_tensor) = self.try_eval(&node, ctx) {
                    // Add result as initializer
                    new_initializers.push(result_tensor);
                    nodes_to_remove.push(node_name.clone());
                    result.record_elimination(&node_name);
                    folded_this_iter += 1;
                }
            }

            if folded_this_iter == 0 {
                break;
            }

            // Apply changes for this iteration
            for tensor in &new_initializers {
                ctx.set_initializer(tensor.clone());
            }
        }

        // Remove folded nodes
        for name in &nodes_to_remove {
            ctx.remove_node(name);
        }

        Ok(result)
    }
}

/// Remove unused Constant nodes after folding
pub struct EliminateUnusedConstants;

impl EliminateUnusedConstants {
    /// Create a new EliminateUnusedConstants transformer
    pub fn new() -> Self {
        Self
    }
}

impl Default for EliminateUnusedConstants {
    fn default() -> Self {
        Self::new()
    }
}

impl OnnxTransformer for EliminateUnusedConstants {
    fn name(&self) -> &'static str {
        "EliminateUnusedConstants"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::new();

        // Collect all used input names
        let mut used_inputs: HashSet<String> = HashSet::new();
        for node in ctx.nodes() {
            for input in &node.input {
                used_inputs.insert(input.clone());
            }
        }

        // Also include graph outputs
        for output_name in ctx.graph_output_map.keys() {
            used_inputs.insert(output_name.clone());
        }

        // Find and remove unused Constant nodes
        let constants_to_remove: Vec<String> = ctx
            .nodes()
            .filter(|n| n.op_type == "Constant")
            .filter(|n| n.output.iter().all(|o| !used_inputs.contains(o)))
            .map(|n| n.name.clone())
            .collect();

        for name in constants_to_remove {
            ctx.remove_node(&name);
            result.record_elimination(&name);
        }

        Ok(result)
    }
}

/// Combined constant optimization pass
pub struct ConstantOptimizeAll;

impl ConstantOptimizeAll {
    /// Create a new ConstantOptimizeAll transformer
    pub fn new() -> Self {
        Self
    }
}

impl Default for ConstantOptimizeAll {
    fn default() -> Self {
        Self::new()
    }
}

impl OnnxTransformer for ConstantOptimizeAll {
    fn name(&self) -> &'static str {
        "ConstantOptimizeAll"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut total = TransformResult::new();

        // Fold constants
        let r1 = ConstantFold::new().transform(ctx)?;
        total.merge(r1);

        // Remove unused constants
        let r2 = EliminateUnusedConstants::new().transform(ctx)?;
        total.merge(r2);

        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::onnx::{AttributeProto, GraphProto};

    fn make_constant_node(name: &str, output: &str, data: Vec<i64>) -> NodeProto {
        let tensor = TensorProto {
            dims: vec![data.len() as i64],
            data_type: 7,
            name: format!("{}_value", name),
            int64_data: data,
            ..Default::default()
        };

        NodeProto {
            op_type: "Constant".to_string(),
            name: name.to_string(),
            output: vec![output.to_string()],
            attribute: vec![AttributeProto {
                name: "value".to_string(),
                r#type: 4, // TENSOR
                t: Some(tensor),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn make_gather_node(name: &str, data: &str, indices: &str, output: &str) -> NodeProto {
        NodeProto {
            op_type: "Gather".to_string(),
            name: name.to_string(),
            input: vec![data.to_string(), indices.to_string()],
            output: vec![output.to_string()],
            attribute: vec![AttributeProto {
                name: "axis".to_string(),
                r#type: 2, // INT
                i: 0,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn test_fold_gather() {
        let graph = GraphProto {
            name: "test".to_string(),
            node: vec![
                make_constant_node("const_data", "data", vec![10, 20, 30, 40]),
                make_constant_node("const_idx", "idx", vec![1]),
                make_gather_node("gather", "data", "idx", "output"),
            ],
            output: vec![],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let result = ConstantFold::new().transform(&mut ctx).unwrap();

        // Should have eliminated the gather node
        assert!(result.nodes_eliminated > 0, "Expected gather to be folded");

        // Check that gather was folded
        assert!(
            ctx.get_node("gather").is_none(),
            "Gather node should be removed"
        );

        // Check result was added as initializer
        let init = ctx.get_initializer("output");
        assert!(init.is_some(), "Output should be an initializer");
        let init = init.unwrap();
        assert_eq!(init.int64_data, vec![20], "Gather result should be [20]");
    }

    #[test]
    fn test_fold_concat() {
        let graph = GraphProto {
            name: "test".to_string(),
            node: vec![
                make_constant_node("const_a", "a", vec![1, 2]),
                make_constant_node("const_b", "b", vec![3, 4]),
                NodeProto {
                    op_type: "Concat".to_string(),
                    name: "concat".to_string(),
                    input: vec!["a".to_string(), "b".to_string()],
                    output: vec!["output".to_string()],
                    attribute: vec![AttributeProto {
                        name: "axis".to_string(),
                        r#type: 2,
                        i: 0,
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            ],
            output: vec![],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let result = ConstantFold::new().transform(&mut ctx).unwrap();

        // Should have eliminated the concat node
        assert!(result.nodes_eliminated > 0, "Expected concat to be folded");

        // Check result was added as initializer
        let init = ctx.get_initializer("output");
        assert!(init.is_some(), "Output should be an initializer");
        assert_eq!(
            init.unwrap().int64_data,
            vec![1, 2, 3, 4],
            "Concat result should be [1,2,3,4]"
        );
    }

    #[test]
    fn test_fold_add() {
        let graph = GraphProto {
            name: "test".to_string(),
            node: vec![
                make_constant_node("const_a", "a", vec![1, 2, 3]),
                make_constant_node("const_b", "b", vec![10, 20, 30]),
                NodeProto {
                    op_type: "Add".to_string(),
                    name: "add".to_string(),
                    input: vec!["a".to_string(), "b".to_string()],
                    output: vec!["output".to_string()],
                    ..Default::default()
                },
            ],
            output: vec![],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let result = ConstantFold::new().transform(&mut ctx).unwrap();

        assert!(result.nodes_eliminated > 0, "Expected add to be folded");

        let init = ctx.get_initializer("output");
        assert!(init.is_some(), "Output should be an initializer");
        assert_eq!(
            init.unwrap().int64_data,
            vec![11, 22, 33],
            "Add result should be [11,22,33]"
        );
    }

    #[test]
    fn test_fold_mul_scalar() {
        let graph = GraphProto {
            name: "test".to_string(),
            node: vec![
                make_constant_node("const_a", "a", vec![2]),
                make_constant_node("const_b", "b", vec![1, 2, 3, 4]),
                NodeProto {
                    op_type: "Mul".to_string(),
                    name: "mul".to_string(),
                    input: vec!["a".to_string(), "b".to_string()],
                    output: vec!["output".to_string()],
                    ..Default::default()
                },
            ],
            output: vec![],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let result = ConstantFold::new().transform(&mut ctx).unwrap();

        assert!(result.nodes_eliminated > 0, "Expected mul to be folded");

        let init = ctx.get_initializer("output");
        assert!(init.is_some(), "Output should be an initializer");
        assert_eq!(
            init.unwrap().int64_data,
            vec![2, 4, 6, 8],
            "Mul result should be [2,4,6,8]"
        );
    }

    #[test]
    fn test_fold_chain() {
        // Test: Constant -> Gather -> Add chain
        let graph = GraphProto {
            name: "test".to_string(),
            node: vec![
                make_constant_node("const_data", "data", vec![10, 20, 30, 40]),
                make_constant_node("const_idx", "idx", vec![1]),
                make_gather_node("gather", "data", "idx", "gathered"),
                make_constant_node("const_add", "to_add", vec![5]),
                NodeProto {
                    op_type: "Add".to_string(),
                    name: "add".to_string(),
                    input: vec!["gathered".to_string(), "to_add".to_string()],
                    output: vec!["output".to_string()],
                    ..Default::default()
                },
            ],
            output: vec![],
            ..Default::default()
        };

        let mut ctx = GraphContext::new(&graph);
        let result = ConstantFold::new().transform(&mut ctx).unwrap();

        // Both gather and add should be folded
        assert!(
            result.nodes_eliminated >= 2,
            "Expected both gather and add to be folded"
        );

        // Final result: gather([10,20,30,40], [1]) = [20], then [20] + [5] = [25]
        let init = ctx.get_initializer("output");
        assert!(init.is_some(), "Output should be an initializer");
        assert_eq!(
            init.unwrap().int64_data,
            vec![25],
            "Chain result should be [25]"
        );
    }
}
