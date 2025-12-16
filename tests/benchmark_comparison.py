import torch
import onnx
import time
import sys
import os
import numpy as np
from pathlib import Path
import tempfile
from transformers import RobertaConfig, RobertaModel

# Add paths
rust_lib_path = str(Path(__file__).parent.parent)
python_lib_path = str(Path(__file__).parent.parent.parent / "python" / "furiosa-optimizer")

sys.path.insert(0, rust_lib_path)
sys.path.insert(0, python_lib_path)

# Try importing both
try:
    import furiosa_optimizer as rust_opt
    print("✅ Loaded Rust optimizer")
except ImportError:
    print("❌ Failed to load Rust optimizer")
    rust_opt = None

try:
    from furiosa.optimizer import optimize_model as python_optimize_model
    print("✅ Loaded Python optimizer")
except ImportError as e:
    print(f"❌ Failed to load Python optimizer: {e}")
    # Fallback for structure check
    python_optimize_model = None

def create_roberta_onnx(path):
    print("Creating RoBERTa model...")
    config = RobertaConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=512,
        attn_implementation="eager"
    )
    model = RobertaModel(config)
    model.eval()

    dummy_input = torch.randint(0, 1000, (1, 128))
    
    # Disable SDPA to allow export to opset 13
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
         torch.onnx.export(
            model,
            dummy_input,
            path,
            input_names=['input_ids'],
            output_names=['last_hidden_state', 'pooler_output'],
            opset_version=13,
            do_constant_folding=False
        )

def run_benchmark(name, optimize_func, orig_path, opt_path, iterations=10, is_python=False):
    if optimize_func is None:
        print(f"Skipping {name} (not available)")
        return None

    print(f"\nBenchmarking {name}...")
    
    # Warmup
    try:
        if is_python:
            model = onnx.load(orig_path)
            model.ir_version = 7 
            print(f"DEBUG: Model IR version set to {model.ir_version}")
            optimized_model = optimize_func(model)
            onnx.save(optimized_model, opt_path)
        else:
            optimize_func(orig_path, opt_path)
    except Exception as e:
        print(f"{name} warmup failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    times = []
    for i in range(iterations):
        start = time.perf_counter()
        if is_python:
            # Include loading/saving time? Usually optimizer benchmark includes I/O if the Rust one does.
            # Rust one: optimize_model(input_path, output_path) -> includes I/O.
            # So for Python we should also include I/O to be fair.
            model = onnx.load(orig_path)
            model.ir_version = 8
            optimized_model = optimize_func(model)
            onnx.save(optimized_model, opt_path)
        else:
            optimize_func(orig_path, opt_path)
        end = time.perf_counter()
        times.append(end - start)
        print(f"  Iter {i+1}: {times[-1]:.4f}s")

    return {
        "avg": np.mean(times),
        "min": np.min(times),
        "max": np.max(times),
        "std": np.std(times)
    }

def main():
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        orig_path = f.name
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        opt_path_rust = f.name
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        opt_path_py = f.name

    try:
        create_roberta_onnx(orig_path)
        
        results_rust = run_benchmark("Rust Optimizer", rust_opt.optimize_model if rust_opt else None, orig_path, opt_path_rust, is_python=False)
        
        results_py = run_benchmark("Python Optimizer", python_optimize_model, orig_path, opt_path_py, is_python=True)

        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        
        if results_rust:
            print(f"Rust   : Avg {results_rust['avg']:.4f}s (Std {results_rust['std']:.4f}s)")
        
        if results_py:
            print(f"Python : Avg {results_py['avg']:.4f}s (Std {results_py['std']:.4f}s)")
            
        if results_rust and results_py:
            speedup = results_py['avg'] / results_rust['avg']
            print(f"\nSpeedup: {speedup:.2f}x")

    finally:
        if os.path.exists(orig_path): os.unlink(orig_path)
        if os.path.exists(opt_path_rust): os.unlink(opt_path_rust)
        if os.path.exists(opt_path_py): os.unlink(opt_path_py)

if __name__ == "__main__":
    main()
