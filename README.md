# Furiosa Optimizer (Rust)
![infographic](https://github.com/user-attachments/assets/f2ddd358-4da4-4e6b-9a42-2a6955b59da5)

ONNX model optimizer for Furiosa NPU - Rust implementation.

## Overview

This crate provides graph-level optimizations for ONNX models, ported from the Python [furiosa-optimizer](https://github.com/furiosa-ai/furiosa-sdk) implementation.

## Features

- **ONNX I/O**: Load and save ONNX models with validation
- **Pattern Matching**: Identify common subgraph patterns for optimization
- **Node Fusion**: Fuse BatchNorm into Conv, etc.
- **Node Elimination**: Remove Identity, Dropout, and no-op nodes
- **Graph Cleanup**: Remove unused nodes, initializers, and value_info
- **Zero-copy Design**: Efficient memory usage with minimal allocations

## Quick Start

```rust
use furiosa_optimizer::prelude::*;

fn main() -> OnnxResult<()> {
    // One-line optimization
    let stats = optimize_file("input.onnx", "output.onnx", OptimizeOptions::default())?;
    
    println!("Reduced {} nodes ({:.1}%)", 
             stats.nodes_reduced, 
             stats.node_reduction_percent());
    Ok(())
}
```

## CLI Usage

```bash
# Optimize a model
cargo run --example optimize_model -- input.onnx output.onnx

# Show model info
cargo run --example optimize_model -- model.onnx --info

# Validate a model
cargo run --example optimize_model -- model.onnx --validate

# Optimize without fusion
cargo run --example optimize_model -- input.onnx output.onnx --no-fuse
```

## Supported Transformations

| Transformer | Description | Status |
|-------------|-------------|--------|
| **Fusion** | | |
| FuseConvBN | Fuse Conv + BatchNormalization | ✅ Done |
| FuseGemmAdd | Fuse Gemm + Add (bias) | ✅ Done |
| FuseMatMulAdd | Fuse MatMul + Add → Gemm | ✅ Done |
| FuseLayerNorm | Fuse LayerNorm pattern (opset 17+) | ✅ Done |
| **Elimination** | | |
| EliminateIdentity | Remove Identity nodes | ✅ Done |
| EliminateDropout | Remove Dropout nodes (inference) | ✅ Done |
| EliminateNopCast | Remove same-type Cast | ✅ Done |
| EliminateNopTranspose | Remove identity Transpose | ✅ Done |
| **Shape Optimization** | | |
| MergeReshape | Merge consecutive Reshapes | ✅ Done |
| MergeTranspose | Merge consecutive Transposes | ✅ Done |
| CancelInverseTranspose | Cancel inverse Transpose pairs | ✅ Done |
| **Constant Folding** | | |
| ConstantFold | Fold constant expressions | ✅ Done |
| EliminateUnusedConstants | Remove unused constants | ✅ Done |
| **Inference** | | |
| InferSqueezeAxes | Infer missing Squeeze axes | ✅ Done |
| InferUnsqueezeAxes | Infer missing Unsqueeze axes | ✅ Done |

### LayerNorm Fusion (Standalone)

For Transformer models, use `FuseLayerNorm` separately from the default pipeline:

```rust
use furiosa_optimizer::transformers::{FuseLayerNorm, OnnxTransformer};

let result = FuseLayerNorm::new().transform(&mut ctx)?;
println!("Fused {} LayerNorm patterns", result.transforms_applied);
```

This detects the pattern: `ReduceMean → Sub → Pow → ReduceMean → Add → Sqrt → Div → Mul → Add`
and replaces it with a single `LayerNormalization` node (requires opset 17+).

### Opset Upgrader

To use `LayerNormalization` (opset 17+), upgrade older models:

```rust
use furiosa_optimizer::opset::{upgrade_model, get_opset_version};

let model = load_model("model_opset9.onnx")?;
println!("Original opset: {}", get_opset_version(&model));

// Upgrade to opset 17
let upgraded = upgrade_model(&model, 17)?;
println!("Upgraded opset: {}", get_opset_version(&upgraded));

// Now LayerNorm fusion can be applied
let mut ctx = GraphContext::new(upgraded.graph.as_ref().unwrap());
FuseLayerNorm::new().transform(&mut ctx)?;
```

Supported upgrades:
- Opset 9-12 → 13+: Squeeze/Unsqueeze axes to input
- Opset 9-12 → 13+: Split sizes to input
- Any → 17+: Enables LayerNormalization

## Advanced Usage

### Custom Optimization Pipeline

```rust
use furiosa_optimizer::prelude::*;
use furiosa_optimizer::transformers::{FuseConvBN, EliminateIdentity, OnnxTransformer};

let model = load_model("input.onnx")?;
let graph = model.graph.as_ref().unwrap();
let mut ctx = GraphContext::new(graph);

// Apply specific transformers
EliminateIdentity::new().transform(&mut ctx)?;
FuseConvBN::new().transform(&mut ctx)?;

// Build optimized model
let optimized = build_optimized_model(&ctx, &model);
save_model(&optimized, "output.onnx")?;
```

### Model Validation

```rust
use furiosa_optimizer::io::{validate_model, ValidationOptions};

let model = load_model("model.onnx")?;
let result = validate_model(&model);

if !result.is_valid {
    for error in &result.errors {
        eprintln!("Error: {}", error);
    }
}
```

## Project Structure

```
src/
├── lib.rs          # Crate entry point
├── error/          # Error types
├── proto/          # ONNX protobuf types
├── traits/         # Core traits
├── tensor/         # Tensor utilities (dtype, shape, convert)
├── graph/          # Graph context and manipulation
├── pattern/        # Pattern matching engine
├── transform/      # Transformation operations
├── builder/        # Model builder and cleanup
├── transformers/   # Concrete transformer implementations
└── io/             # ONNX file I/O and validation
```

## Requirements

- Rust 1.70+
- Protocol Buffers compiler (for development)

## Installation

```toml
[dependencies]
furiosa-optimizer = "0.1"
```

## Development

```bash
# Build
cargo build

# Test
cargo test

# Lint
cargo clippy

# Benchmark
cargo bench
```

## Python Bindings

### Installation

```bash
# From wheel
pip install furiosa_optimizer-*.whl

# From source (requires Rust and maturin)
pip install maturin
maturin develop --features python
```

### Python Usage

```python
import furiosa_optimizer as opt

# Simple optimization
result = opt.optimize_model("model.onnx", "optimized.onnx")
print(f"Reduced {result.nodes_removed} nodes ({result.reduction_percent:.1f}%)")

# With custom configuration
config = opt.OptimizationConfig(
    fuse_conv_bn=True,
    fuse_gemm_add=True,
    fuse_matmul_add=True,
    iterations=5
)
result = opt.optimize_model("model.onnx", "out.onnx", config)

# Model analysis
info = opt.analyze_model("model.onnx")
print(f"Nodes: {info.node_count}")
print(f"Top ops: {info.top_ops(5)}")

# Validation
is_valid = opt.validate("model.onnx")
```

### Configuration Presets

```python
# Default (balanced)
config = opt.OptimizationConfig()

# Minimal (fast, essential only)
config = opt.OptimizationConfig.minimal()

# Aggressive (maximum optimization)
config = opt.OptimizationConfig.aggressive()
```

## Statistics

| Module | Lines of Code | Tests |
|--------|--------------|-------|
| Total | ~12,100 | 165 |

## License

Apache-2.0

## Author

Jung Wook Yang <sadpig70@gmail.com>
