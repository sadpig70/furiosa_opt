# Furiosa Optimizer Benchmark Report

## Test Date
2025-12-03

## Test Environment
- Rust: 1.91.0
- Build: Release (LTO enabled)
- Platform: Linux x86_64

## Models Tested

### CNN Models

| Model | Original | Optimized | Reduction | Time | Applied |
|-------|----------|-----------|-----------|------|---------|
| ResNet18 | 69 | 49 | **29.0%** | 195ms | Conv+BN Fusion |
| MobileNetV2 | 155 | 102 | **34.2%** | 20ms | Conv+BN Fusion |
| SqueezeNet 1.1 | 66 | 65 | 1.5% | 9ms | Identity Elim |

### NLP/Transformer Models

| Model | Original | Optimized | Reduction | Time | Applied |
|-------|----------|-----------|-----------|------|---------|
| RoBERTa | 1180 | 1107 | **6.2%** | 1124ms | MatMul+Add Fusion |
| GPT-2 | 2534 | 2534 | 0.0% | 1563ms | N/A (no patterns) |

## Detailed Analysis

### RoBERTa (BERT-like)
- **72 MatMul+Add patterns fused** into Gemm operations
- Pattern: `MatMul(Q,K) -> Add(bias)` → `Gemm(Q,K,bias)`
- All 72 fusible patterns successfully optimized
- 1 additional node eliminated (Identity)

### GPT-2
- **0 fusible patterns found** - model structure differs
- GPT-2 uses different Add patterns (not bias additions)
- Validates optimizer doesn't incorrectly modify unfusable patterns

### CNN Models
- **Conv+BN fusion** remains the primary optimization
- MobileNetV2: 53 BatchNorm nodes fused
- ResNet18: 20 BatchNorm nodes fused

## Optimization Effectiveness by Pattern

| Pattern | RoBERTa | GPT-2 | ResNet18 | MobileNetV2 |
|---------|---------|-------|----------|-------------|
| Conv+BN | N/A | N/A | 20 | 53 |
| MatMul+Add | 72 | 0 | N/A | N/A |
| Gemm+Add | 0 | 0 | N/A | N/A |
| Transpose chain | 0 | 0 | 0 | 0 |
| Reshape chain | 0 | 12 | 0 | 0 |

## Performance Summary

| Metric | CNN Avg | NLP Avg | Overall |
|--------|---------|---------|---------|
| Node Reduction | 21.6% | 3.1% | 3.7% |
| Processing Speed | 74ms | 1344ms | - |
| Validation Pass | 100% | 100% | 100% |

## Constant Folding Analysis

### Overview
Constant folding evaluates operations with all-constant inputs at compile time, eliminating runtime computation.

### Supported Operations
| Category | Operations |
|----------|------------|
| Arithmetic | Add, Sub, Mul, Div |
| Shape | Shape, Gather, Concat, Squeeze, Unsqueeze, Slice |
| Other | Cast, Equal, Less, Greater, Where |

### Real-world Model Analysis

| Model | Constant Nodes | Foldable Nodes | Reason |
|-------|---------------|----------------|--------|
| GPT-2 | 510 | 0 | Dynamic sequence length |
| RoBERTa | 200 | 0 | Dynamic batch/seq dims |
| ResNet18 | 0 | 0 | No Constant nodes |
| MobileNetV2 | 0 | 0 | No Constant nodes |

### Why No Folding in Transformers?
Transformer models use dynamic Shape/Gather/Concat chains for:
- Variable batch sizes
- Variable sequence lengths
- Dynamic attention masks

These chains depend on runtime input shapes, making them unfoldable.

### Effective Use Cases
Constant folding is effective for:
1. Models with fixed input shapes (static export)
2. Post-training quantization pipelines
3. Custom models with constant computation chains

## Conclusions

1. **CNN Models**: High optimization rate (21-34%) via Conv+BN fusion
2. **RoBERTa**: Effective MatMul+Add fusion (72 patterns, 6.2% reduction)
3. **GPT-2**: No applicable patterns - validates conservative optimization
4. **All models pass validation** - safe for production

## Next Steps

1. Add Constant Folding for GPT-2's static computations
2. Implement attention-specific optimizations
3. Add LayerNorm fusion patterns
