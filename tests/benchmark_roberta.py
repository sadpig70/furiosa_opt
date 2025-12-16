import torch
import onnx
import time
import sys
import os
import numpy as np
from pathlib import Path
import tempfile
from transformers import RobertaConfig, RobertaModel

# Add furiosa_optimizer to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import furiosa_optimizer

def create_roberta_onnx(path):
    print("Creating RoBERTa model...")
    config = RobertaConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=512
    )
    model = RobertaModel(config)
    model.eval()

    dummy_input = torch.randint(0, 1000, (1, 128)) # Batch 1, Seq 128
    
    torch.onnx.export(
        model,
        dummy_input,
        path,
        input_names=['input_ids'],
        output_names=['last_hidden_state', 'pooler_output'],
        opset_version=14,
        do_constant_folding=False
    )

def benchmark_roberta_optimization():
    print("="*60)
    print("RoBERTa Optimization Benchmark")
    print("="*60)

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        orig_path = f.name
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        opt_path = f.name

    try:
        # 1. Create Model
        create_roberta_onnx(orig_path)
        
        # 2. Warmup
        print("Warming up...")
        furiosa_optimizer.optimize_model(orig_path, opt_path)
        
        # 3. Benchmark
        iterations = 20
        print(f"Running {iterations} iterations...")
        
        times = []
        for i in range(iterations):
            start_time = time.perf_counter()
            furiosa_optimizer.optimize_model(orig_path, opt_path)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            print(f"Iteration {i+1}: {times[-1]:.4f}s")
            
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print("-" * 60)
        print(f"Average Time: {avg_time:.4f}s")
        print(f"Std Dev:      {std_time:.4f}s")
        print(f"Min Time:     {min_time:.4f}s")
        print(f"Max Time:     {max_time:.4f}s")
        print("-" * 60)
        
        return True

    finally:
        if os.path.exists(orig_path):
            os.unlink(orig_path)
        if os.path.exists(opt_path):
            os.unlink(opt_path)

if __name__ == "__main__":
    benchmark_roberta_optimization()
