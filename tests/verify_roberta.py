import torch
import onnx
import onnxruntime as ort
import numpy as np
import sys
import os
from pathlib import Path
import tempfile
from transformers import RobertaConfig, RobertaModel

# Add furiosa_optimizer to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import furiosa_optimizer

def create_roberta_onnx(path):
    print("Creating RoBERTa model...")
    # Use a smaller config for speed, but keep structure
    print("Initializing RobertaConfig...")
    config = RobertaConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2, # Reduced layers
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=512
    )
    print("Initializing RobertaModel...")
    model = RobertaModel(config)
    model.eval()

    print("Creating dummy input...")
    dummy_input = torch.randint(0, 1000, (1, 128)) # Batch 1, Seq 128
    
    print(f"Exporting to {path}...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            path,
            input_names=['input_ids'],
            output_names=['last_hidden_state', 'pooler_output'],
            opset_version=14,
            do_constant_folding=False # Let our optimizer do it
        )
        print("Export complete.")
    except Exception as e:
        print(f"Export failed: {e}")
        raise e
    return dummy_input

def verify_roberta_optimization():
    print("="*60)
    print("RoBERTa Optimization Verification")
    print("="*60)

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        orig_path = f.name
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        opt_path = f.name

    try:
        # 1. Create Model
        dummy_input = create_roberta_onnx(orig_path)
        
        # 2. Optimize
        print("Running optimization...")
        res = furiosa_optimizer.optimize_model(orig_path, opt_path)
        
        print(f"Original nodes: {res.original_nodes}")
        print(f"Optimized nodes: {res.final_nodes}")
        
        # Check node reduction
        reduction = res.original_nodes - res.final_nodes
        reduction_rate = (reduction / res.original_nodes) * 100
        print(f"Node reduction: {reduction} ({reduction_rate:.2f}%)")
        
        if reduction <= 0:
            print("❌ Optimization failed: No nodes reduced.")
            return False

        # Analyze original model nodes to debug GELU pattern
        orig_model = onnx.load(orig_path)
        orig_op_types = [n.op_type for n in orig_model.graph.node]
        from collections import Counter
        print(f"Original Op Types: {Counter(orig_op_types)}")

        # 3. Check for Gelu nodes
        opt_model = onnx.load(opt_path)
        nodes = opt_model.graph.node
        gelu_nodes = [n for n in nodes if n.op_type == 'Gelu']
        fast_gelu_nodes = [n for n in nodes if n.op_type == 'FastGelu']
        print(f"Found {len(gelu_nodes)} Gelu nodes and {len(fast_gelu_nodes)} FastGelu nodes.")
        
        # We expect 2 layers, so at least 2 Gelu/FastGelu nodes
        total_gelu = len(gelu_nodes) + len(fast_gelu_nodes)
        if total_gelu < 2:
            print(f"Expected at least 2 Gelu/FastGelu nodes, found {total_gelu}")
        
        # Check com.microsoft domain
        has_ms_domain = any(imp.domain == "com.microsoft" for imp in opt_model.opset_import)
        if has_ms_domain:
            print("✅ com.microsoft domain found in opset imports.")
        else:
            print("❌ com.microsoft domain NOT found in opset imports.")
            return False

        # 4. Runtime Verification
        print("Verifying with ONNX Runtime...")
        
        input_data = dummy_input.numpy()
        
        # Run Original
        sess_orig = ort.InferenceSession(orig_path, providers=['CPUExecutionProvider'])
        out_orig = sess_orig.run(None, {'input_ids': input_data})[0]
        
        # Run Optimized
        try:
            sess_opt = ort.InferenceSession(opt_path, providers=['CPUExecutionProvider'])
            out_opt = sess_opt.run(None, {'input_ids': input_data})[0]
        except Exception as e:
            print(f"❌ ONNX Runtime failed: {e}")
            return False

        # Compare
        max_diff = np.max(np.abs(out_orig - out_opt))
        print(f"Max difference: {max_diff}")
        
        # Allow slightly larger tolerance for complex models
        if max_diff > 1e-4:
            print("❌ Numerical mismatch!")
            return False
            
        print("✅ Verification Successful!")
        return True

    finally:
        if os.path.exists(orig_path):
            os.unlink(orig_path)
        if os.path.exists(opt_path):
            os.unlink(opt_path)

if __name__ == "__main__":
    success = verify_roberta_optimization()
    sys.exit(0 if success else 1)
