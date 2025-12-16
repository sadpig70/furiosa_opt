# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-12-03

### Added

- Initial release of furiosa-optimizer-rs
- Core ONNX parsing and serialization
- Graph context for efficient traversal and mutation
- Pattern matching engine for subgraph detection

### Transformers

- **FuseConvBN**: Fuse Conv + BatchNormalization (29-34% reduction for CNNs)
- **FuseGemmAll**: MatMul + Add → Gemm, Gemm + activation fusion (6% for Transformers)
- **FuseLayerNorm**: 9-node pattern → LayerNormalization (17% for Transformers, opset 17+)
- **EliminateIdentity**: Remove Identity nodes
- **EliminateDropout**: Remove Dropout nodes for inference
- **EliminateCast**: Remove redundant Cast nodes
- **OptimizeReshape**: Merge consecutive Reshape nodes
- **OptimizeTranspose**: Merge consecutive Transpose nodes
- **ConstantFold**: Fold constant expressions (Shape, Gather, Concat, arithmetic)
- **InferShapes**: Shape inference pass

### Utilities

- **OpsetUpgrader**: Upgrade models from opset 9-12 to 13-17
- Model validation with detailed error reporting
- Python bindings via PyO3

### Benchmarks

| Model | Nodes | Optimized | Reduction |
|-------|-------|-----------|-----------|
| ResNet-18 | 69 | 49 | 29.0% |
| MobileNetV2 | 155 | 102 | 34.2% |
| RoBERTa | 1180 | 1107 | 6.2% |
| SqueezeNet | 66 | 65 | 1.5% |

### Python API

```python
import furiosa_optimizer

result = furiosa_optimizer.optimize_model("in.onnx", "out.onnx")
print(f"Reduced {result.reduction_percent:.1f}%")
```

[Unreleased]: https://github.com/prtx/furiosa-optimizer-rs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/prtx/furiosa-optimizer-rs/releases/tag/v0.1.0
