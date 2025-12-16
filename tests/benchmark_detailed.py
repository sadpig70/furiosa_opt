#!/usr/bin/env python3
"""
Detailed benchmark for furiosa-optimizer

Tests:
1. Optimization correctness (numerical accuracy)
2. Inference performance before/after optimization
3. Model size comparison
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, List
import json

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent.parent))
import furiosa_optimizer


def benchmark_model(model_path: str, num_runs: int = 50) -> Dict:
    """Run comprehensive benchmark on a single model."""
    model_name = Path(model_path).stem
    results = {
        "model": model_name,
        "original_path": model_path,
    }
    
    # Get original model info
    info = furiosa_optimizer.analyze_model(model_path)
    results["original_nodes"] = info.node_count
    results["original_opset"] = info.opset_version
    results["original_size_kb"] = os.path.getsize(model_path) / 1024
    
    # Optimize
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        optimized_path = f.name
    
    opt_result = furiosa_optimizer.optimize_model(model_path, optimized_path)
    results["optimized_nodes"] = opt_result.final_nodes
    results["node_reduction"] = opt_result.reduction_percent
    results["optimize_time_ms"] = opt_result.optimize_time_ms
    results["optimized_size_kb"] = os.path.getsize(optimized_path) / 1024
    results["size_reduction"] = (1 - results["optimized_size_kb"] / results["original_size_kb"]) * 100
    
    # Inference benchmark
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    
    try:
        orig_session = ort.InferenceSession(model_path, sess_options)
        opt_session = ort.InferenceSession(optimized_path, sess_options)
        
        # Generate inputs
        inputs = {}
        for inp in orig_session.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            if inp.type == 'tensor(float)':
                inputs[inp.name] = np.random.randn(*shape).astype(np.float32)
            elif inp.type in ('tensor(int64)', 'tensor(int32)'):
                dtype = np.int64 if 'int64' in inp.type else np.int32
                inputs[inp.name] = np.random.randint(0, 100, shape).astype(dtype)
            else:
                inputs[inp.name] = np.random.randn(*shape).astype(np.float32)
        
        # Warmup
        for _ in range(5):
            orig_session.run(None, inputs)
            opt_session.run(None, inputs)
        
        # Benchmark original
        orig_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            orig_out = orig_session.run(None, inputs)
            orig_times.append((time.perf_counter() - start) * 1000)
        
        # Benchmark optimized
        opt_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            opt_out = opt_session.run(None, inputs)
            opt_times.append((time.perf_counter() - start) * 1000)
        
        # Accuracy check
        max_diff = 0.0
        for o1, o2 in zip(orig_out, opt_out):
            max_diff = max(max_diff, float(np.max(np.abs(o1 - o2))))
        
        results["original_latency_ms"] = np.mean(orig_times)
        results["original_latency_std"] = np.std(orig_times)
        results["optimized_latency_ms"] = np.mean(opt_times)
        results["optimized_latency_std"] = np.std(opt_times)
        results["speedup"] = results["original_latency_ms"] / results["optimized_latency_ms"]
        results["max_diff"] = max_diff
        results["accuracy_pass"] = max_diff < 1e-4
        
    except Exception as e:
        results["error"] = str(e)
    
    # Cleanup
    os.unlink(optimized_path)
    
    return results


def main():
    models_dir = Path(__file__).parent.parent / "benches" / "models"
    
    # CNN models only (have static shapes)
    cnn_models = ["resnet18.onnx", "mobilenetv2.onnx", "squeezenet1.1.onnx"]
    
    print("=" * 70)
    print("Furiosa Optimizer - Detailed Benchmark Report")
    print("=" * 70)
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"Optimizer: {furiosa_optimizer.version()}")
    print()
    
    all_results = []
    for model_name in cnn_models:
        model_path = models_dir / model_name
        if not model_path.exists():
            print(f"Skipping {model_name} (not found)")
            continue
        
        print(f"Benchmarking {model_name}...")
        results = benchmark_model(str(model_path))
        all_results.append(results)
        
        print(f"  Nodes: {results['original_nodes']} → {results['optimized_nodes']} ({results['node_reduction']:.1f}% reduction)")
        print(f"  Size: {results['original_size_kb']:.1f}KB → {results['optimized_size_kb']:.1f}KB ({results['size_reduction']:.1f}% reduction)")
        if 'original_latency_ms' in results:
            print(f"  Latency: {results['original_latency_ms']:.2f}ms → {results['optimized_latency_ms']:.2f}ms ({results['speedup']:.2f}x)")
            print(f"  Max diff: {results['max_diff']:.2e} {'✅' if results['accuracy_pass'] else '❌'}")
        print()
    
    # Summary table
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Model':<15} {'Nodes':<12} {'Size(KB)':<15} {'Latency(ms)':<15} {'Accuracy':<10}")
    print("-" * 70)
    
    for r in all_results:
        if 'error' not in r:
            nodes = f"{r['original_nodes']}→{r['optimized_nodes']}"
            size = f"{r['original_size_kb']:.0f}→{r['optimized_size_kb']:.0f}"
            latency = f"{r['original_latency_ms']:.1f}→{r['optimized_latency_ms']:.1f}"
            acc = "✅" if r['accuracy_pass'] else "❌"
            print(f"{r['model']:<15} {nodes:<12} {size:<15} {latency:<15} {acc:<10}")
    
    # Save results
    output_path = models_dir.parent / "results" / "benchmark_detailed.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
