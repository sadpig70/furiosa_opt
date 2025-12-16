# Furiosa Optimizer

[![PyPI version](https://badge.fury.io/py/furiosa-optimizer.svg)](https://badge.fury.io/py/furiosa-optimizer)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

High-performance ONNX model optimizer implemented in Rust with Python bindings.

## Features

- **Conv+BatchNorm Fusion**: Up to 34% node reduction for CNN models
- **MatMul+Add → Gemm**: 6%+ reduction for Transformer models
- **LayerNorm Fusion**: 17% reduction with opset upgrade
- **Identity/Dropout Elimination**: Clean up inference graphs
- **10-100x faster** than Python implementations

## Installation

```bash
pip install furiosa-optimizer
```

## Quick Start

```python
import furiosa_optimizer

# Simple optimization
result = furiosa_optimizer.optimize_model("model.onnx", "optimized.onnx")
print(f"Reduced {result.nodes_removed} nodes ({result.reduction_percent:.1f}%)")

# With configuration
config = furiosa_optimizer.OptimizationConfig(
    fuse_conv_bn=True,
    fuse_gemm_add=True,
    eliminate_identity=True,
)
result = furiosa_optimizer.optimize_model("model.onnx", "out.onnx", config)

# Model analysis
info = furiosa_optimizer.analyze_model("model.onnx")
print(f"Nodes: {info.node_count}, Opset: {info.opset_version}")
```

## Benchmarks

| Model | Original | Optimized | Reduction |
|-------|----------|-----------|-----------|
| ResNet-18 | 69 | 49 | 29.0% |
| MobileNetV2 | 155 | 102 | 34.2% |
| RoBERTa | 1180 | 1107 | 6.2% |
| GPT-2 | 2534 | 2534 | 0.0%* |

*GPT-2 requires LayerNorm fusion with opset upgrade for optimization.

## API Reference

### Functions

- `optimize_model(input, output, config=None)` - Optimize ONNX model
- `analyze_model(path)` - Get model information
- `validate(path)` - Validate ONNX model
- `version()` - Get library version

### Classes

- `OptimizationConfig` - Configuration for optimization passes
- `OptimizationResult` - Result containing statistics
- `ModelInfo` - Model metadata

## Requirements

- Python 3.8+
- ONNX models (opset 9-17)

## License

Apache 2.0

## Links

- [GitHub Repository](https://github.com/prtx/furiosa-optimizer-rs)
- [Documentation](https://github.com/prtx/furiosa-optimizer-rs#readme)
