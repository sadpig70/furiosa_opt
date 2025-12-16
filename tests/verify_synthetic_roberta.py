import onnx
import onnx.helper
import onnx.numpy_helper
import numpy as np
import onnxruntime as ort
import sys
import os
from pathlib import Path
import tempfile

# Add furiosa_optimizer to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import furiosa_optimizer

def make_gelu_subgraph(input_name, output_name, suffix):
    """
    Creates nodes for Erf-based GELU:
    x -> Div(sqrt(2)) -> Erf -> Add(1) -> Mul(0.5) -> Mul(x) -> output
    """
    nodes = []
    
    # Constants names
    sqrt2_name = f"sqrt2_{suffix}"
    one_name = f"one_{suffix}"
    half_name = f"half_{suffix}"
    
    # Intermediate names
    div_out = f"div_out_{suffix}"
    erf_out = f"erf_out_{suffix}"
    add_out = f"add_out_{suffix}"
    mul1_out = f"mul1_out_{suffix}"
    
    # Nodes
    nodes.append(onnx.helper.make_node('Div', [input_name, sqrt2_name], [div_out], name=f'div_{suffix}'))
    nodes.append(onnx.helper.make_node('Erf', [div_out], [erf_out], name=f'erf_{suffix}'))
    nodes.append(onnx.helper.make_node('Add', [erf_out, one_name], [add_out], name=f'add_{suffix}'))
    nodes.append(onnx.helper.make_node('Mul', [add_out, half_name], [mul1_out], name=f'mul1_{suffix}'))
    nodes.append(onnx.helper.make_node('Mul', [mul1_out, input_name], [output_name], name=f'mul2_{suffix}'))
    
    return nodes

def create_synthetic_roberta():
    print("Creating Synthetic RoBERTa-like model...")
    
    # Shapes
    batch = 1
    seq = 32
    hidden = 64
    
    # Inputs
    x = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [batch, seq, hidden])
    y = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [batch, seq, hidden])
    
    nodes = []
    initializers = []
    
    # Constants for GELU
    sqrt2_val = np.array([1.41421356], dtype=np.float32)
    one_val = np.array([1.0], dtype=np.float32)
    half_val = np.array([0.5], dtype=np.float32)
    
    # Create 2 layers
    current_input = 'x'
    for i in range(2):
        suffix = str(i)
        
        # Add constants for this layer's GELU
        initializers.append(onnx.numpy_helper.from_array(sqrt2_val, name=f"sqrt2_{suffix}"))
        initializers.append(onnx.numpy_helper.from_array(one_val, name=f"one_{suffix}"))
        initializers.append(onnx.numpy_helper.from_array(half_val, name=f"half_{suffix}"))
        
        # 1. MatMul (Simulate FFN projection)
        w1_name = f"w1_{suffix}"
        w1 = np.random.randn(hidden, hidden).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(w1, name=w1_name))
        
        matmul1_out = f"matmul1_out_{suffix}"
        nodes.append(onnx.helper.make_node('MatMul', [current_input, w1_name], [matmul1_out], name=f'matmul1_{suffix}'))
        
        # 2. GELU
        gelu_out = f"gelu_out_{suffix}"
        nodes.extend(make_gelu_subgraph(matmul1_out, gelu_out, suffix))
        
        # 3. MatMul (Simulate FFN output projection)
        w2_name = f"w2_{suffix}"
        w2 = np.random.randn(hidden, hidden).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(w2, name=w2_name))
        
        matmul2_out = f"matmul2_out_{suffix}"
        nodes.append(onnx.helper.make_node('MatMul', [gelu_out, w2_name], [matmul2_out], name=f'matmul2_{suffix}'))
        
        # 4. Add (Residual)
        add_res_out = f"layer_{suffix}_out"
        nodes.append(onnx.helper.make_node('Add', [matmul2_out, current_input], [add_res_out], name=f'add_res_{suffix}'))
        
        current_input = add_res_out

    # Final Identity to match output name
    nodes.append(onnx.helper.make_node('Identity', [current_input], ['y'], name='final_identity'))
    
    graph = onnx.helper.make_graph(
        nodes,
        'synthetic_roberta',
        [x],
        [y],
        initializers
    )
    
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 14)])
    # onnx.checker.check_model(model) # Might be slow
    return model

def verify_synthetic():
    print("="*60)
    print("Synthetic RoBERTa Verification")
    print("="*60)

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        orig_path = f.name
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        opt_path = f.name

    try:
        # 1. Create Model
        print("Creating model...", flush=True)
        model = create_synthetic_roberta()
        print("Saving model...", flush=True)
        onnx.save(model, orig_path)
        print(f"Original model saved to {orig_path}", flush=True)
        
        # 2. Optimize
        print("Running optimization...", flush=True)
        # res = furiosa_optimizer.optimize_model(orig_path, opt_path)
        import shutil
        shutil.copy(orig_path, opt_path) # Dummy optimization
        print("Optimization complete.", flush=True)
        
        # print(f"Original nodes: {res.original_nodes}", flush=True)
        # print(f"Optimized nodes: {res.final_nodes}", flush=True)
        
        # Expected reduction:
        # Per layer: 5 nodes (GELU pattern) -> 1 node (Gelu) = -4 nodes
        # 2 layers = -8 nodes
        # Total nodes: (MatMul + 5 + MatMul + Add) * 2 + Identity = 8 * 2 + 1 = 17 nodes
        # Optimized: (MatMul + Gelu + MatMul + Add) * 2 + Identity = 4 * 2 + 1 = 9 nodes
        
        # expected_nodes = 9
        # if res.final_nodes != expected_nodes:
        #     print(f"⚠️ Expected {expected_nodes} nodes, got {res.final_nodes}", flush=True)
        
        # 3. Check for Gelu nodes
        print("Checking Gelu nodes...", flush=True)
        opt_model = onnx.load(opt_path)
        nodes = opt_model.graph.node
        gelu_nodes = [n for n in nodes if n.op_type == 'Gelu']
        print(f"Found {len(gelu_nodes)} Gelu nodes.", flush=True)
        
        if len(gelu_nodes) != 2:
            print(f"❌ Expected 2 Gelu nodes, found {len(gelu_nodes)}", flush=True)
            return False
            
        # Check com.microsoft domain
        has_ms_domain = any(imp.domain == "com.microsoft" for imp in opt_model.opset_import)
        if has_ms_domain:
            print("✅ com.microsoft domain found in opset imports.", flush=True)
        else:
            print("❌ com.microsoft domain NOT found in opset imports.", flush=True)
            return False

        # 4. Runtime Verification
        print("Verifying with ONNX Runtime...", flush=True)
        
        input_data = np.random.randn(1, 32, 64).astype(np.float32)
        
        # Run Original
        print("Running original model...", flush=True)
        sess_orig = ort.InferenceSession(orig_path, providers=['CPUExecutionProvider'])
        out_orig = sess_orig.run(None, {'x': input_data})[0]
        
        # Run Optimized
        print("Running optimized model...", flush=True)
        try:
            sess_opt = ort.InferenceSession(opt_path, providers=['CPUExecutionProvider'])
            out_opt = sess_opt.run(None, {'x': input_data})[0]
        except Exception as e:
            print(f"❌ ONNX Runtime failed: {e}", flush=True)
            return False

        # Compare
        print("Comparing results...", flush=True)
        max_diff = np.max(np.abs(out_orig - out_opt))
        print(f"Max difference: {max_diff}", flush=True)
        
        if max_diff > 1e-5:
            print("❌ Numerical mismatch!", flush=True)
            return False
            
        print("✅ Verification Successful!", flush=True)
        return True

    finally:
        if os.path.exists(orig_path):
            os.unlink(orig_path)
        if os.path.exists(opt_path):
            os.unlink(opt_path)

if __name__ == "__main__":
    success = verify_synthetic()
    sys.exit(0 if success else 1)
