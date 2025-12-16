#!/usr/bin/env python3
"""
Furiosa Optimizer - Usage Examples

Install:
    pip install furiosa_optimizer-*.whl

Or build from source:
    maturin develop --features python
"""

import furiosa_optimizer as opt


def example_basic():
    """Basic optimization example."""
    print("\n=== Basic Optimization ===")
    
    result = opt.optimize_model(
        "model.onnx",
        "optimized.onnx"
    )
    
    print(f"Original nodes: {result.original_nodes}")
    print(f"Optimized nodes: {result.optimized_nodes}")
    print(f"Reduction: {result.reduction_percent:.1f}%")
    print(f"Valid: {result.is_valid}")


def example_custom_config():
    """Custom configuration example."""
    print("\n=== Custom Configuration ===")
    
    # Create config with specific optimizations
    config = opt.OptimizationConfig(
        fuse_conv_bn=True,
        fuse_gemm_add=True,
        fuse_matmul_add=True,
        eliminate_identity=True,
        eliminate_dropout=True,
        infer_axes=True,
        optimize_reshape=True,
        optimize_transpose=True,
        iterations=5
    )
    
    print(f"Config: {config}")
    
    result = opt.optimize_model("model.onnx", "optimized.onnx", config)
    print(f"Result: {result}")


def example_minimal_config():
    """Minimal optimization for fast processing."""
    print("\n=== Minimal Configuration ===")
    
    config = opt.OptimizationConfig.minimal()
    print(f"Minimal config: {config}")


def example_aggressive_config():
    """Aggressive optimization for maximum reduction."""
    print("\n=== Aggressive Configuration ===")
    
    config = opt.OptimizationConfig.aggressive()
    print(f"Aggressive config: {config}")


def example_analyze():
    """Model analysis example."""
    print("\n=== Model Analysis ===")
    
    info = opt.analyze_model("model.onnx")
    
    print(f"IR Version: {info.ir_version}")
    print(f"Producer: {info.producer}")
    print(f"Nodes: {info.node_count}")
    print(f"Initializers: {info.initializer_count}")
    print(f"Inputs: {info.input_count}")
    print(f"Outputs: {info.output_count}")
    print(f"Valid: {info.is_valid}")
    
    print("\nTop operations:")
    for op, count in info.top_ops(10):
        print(f"  {op}: {count}")


def example_validate():
    """Model validation example."""
    print("\n=== Model Validation ===")
    
    is_valid = opt.validate("model.onnx")
    print(f"Model is valid: {is_valid}")


def example_batch_optimize():
    """Batch optimization example."""
    print("\n=== Batch Optimization ===")
    
    import os
    import time
    
    models_dir = "models"
    output_dir = "optimized"
    os.makedirs(output_dir, exist_ok=True)
    
    config = opt.OptimizationConfig.aggressive()
    
    total_original = 0
    total_optimized = 0
    
    for filename in os.listdir(models_dir):
        if not filename.endswith(".onnx"):
            continue
            
        input_path = os.path.join(models_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        start = time.time()
        result = opt.optimize_model(input_path, output_path, config)
        elapsed = (time.time() - start) * 1000
        
        total_original += result.original_nodes
        total_optimized += result.optimized_nodes
        
        print(f"{filename}: {result.original_nodes} → {result.optimized_nodes} "
              f"({result.reduction_percent:.1f}%) [{elapsed:.0f}ms]")
    
    total_reduction = (1 - total_optimized / total_original) * 100 if total_original > 0 else 0
    print(f"\nTotal: {total_original} → {total_optimized} ({total_reduction:.1f}%)")


def example_result_to_dict():
    """Export result to dictionary."""
    print("\n=== Result to Dictionary ===")
    
    result = opt.optimize_model("model.onnx", "optimized.onnx")
    
    # Convert to dict for JSON serialization
    import json
    data = result.to_dict()
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    print("=" * 60)
    print("Furiosa Optimizer Python Examples")
    print("=" * 60)
    print(f"Version: {opt.version()}")
    
    # Run examples with a test model
    import sys
    if len(sys.argv) < 2:
        print("\nUsage: python examples.py <model.onnx>")
        print("\nAvailable functions:")
        print("  - optimize_model(input, output, config?)")
        print("  - analyze_model(path)")
        print("  - validate(path)")
        print("  - version()")
        print("\nConfiguration presets:")
        print("  - OptimizationConfig() - default")
        print("  - OptimizationConfig.minimal()")
        print("  - OptimizationConfig.aggressive()")
    else:
        # Quick demo with provided model
        model_path = sys.argv[1]
        
        print(f"\nAnalyzing: {model_path}")
        info = opt.analyze_model(model_path)
        print(f"  Nodes: {info.node_count}")
        print(f"  Valid: {info.is_valid}")
        print(f"  Top ops: {info.top_ops(5)}")
        
        print(f"\nOptimizing: {model_path}")
        result = opt.optimize_model(model_path, "/tmp/optimized.onnx")
        print(f"  {result.original_nodes} → {result.optimized_nodes}")
        print(f"  Reduction: {result.reduction_percent:.1f}%")
        print(f"  Time: {result.optimize_time_ms:.0f}ms")
