"""
Furiosa Optimizer - High-performance ONNX model optimizer

This package provides fast ONNX model optimization implemented in Rust.

Example usage:
    >>> import furiosa_optimizer
    >>>
    >>> # Simple optimization
    >>> result = furiosa_optimizer.optimize_model("model.onnx", "optimized.onnx")
    >>> print(f"Reduced {result.nodes_removed} nodes ({result.reduction_percent:.1f}%)")
    >>>
    >>> # With custom configuration
    >>> config = furiosa_optimizer.OptimizationConfig(
    ...     fuse_conv_bn=True,
    ...     fuse_gemm_add=True,
    ...     eliminate_identity=True,
    ... )
    >>> result = furiosa_optimizer.optimize_model("model.onnx", "out.onnx", config)
    >>>
    >>> # Analysis only
    >>> info = furiosa_optimizer.analyze_model("model.onnx")
    >>> print(f"Model has {info.node_count} nodes")
"""

from .furiosa_optimizer import (
    # Configuration
    OptimizationConfig,
    
    # Results
    OptimizationResult,
    ModelInfo,
    
    # Functions
    optimize_model,
    analyze_model,
    validate,
    version,
    
    # Module metadata
    __version__,
    __author__,
)

__all__ = [
    "OptimizationConfig",
    "OptimizationResult",
    "ModelInfo",
    "optimize_model",
    "analyze_model",
    "validate",
    "version",
    "__version__",
    "__author__",
]
