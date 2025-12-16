import onnx
import onnx.helper
import onnx.numpy_helper
import numpy as np
import onnxruntime as ort
import sys
import os
import tempfile
from pathlib import Path

# Try to import furiosa_optimizer, if fails, add potential paths
try:
    import furiosa_optimizer
except ImportError:
    # Assuming we are in rust/tests
    # Try adding python/furiosa-optimizer to path
    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root / "python" / "furiosa-optimizer"))
    import furiosa_optimizer

def make_gelu_graph(input_name, output_name):
    nodes = []
    nodes.append(onnx.helper.make_node('Div', [input_name, 'sqrt2'], ['div_out'], name='div'))
    nodes.append(onnx.helper.make_node('Erf', ['div_out'], ['erf_out'], name='erf'))
    nodes.append(onnx.helper.make_node('Add', ['erf_out', 'one'], ['add_out'], name='add'))
    nodes.append(onnx.helper.make_node('Mul', ['add_out', 'half'], ['mul1_out'], name='mul1'))
    nodes.append(onnx.helper.make_node('Mul', ['mul1_out', input_name], [output_name], name='mul2'))
    
    input_vi = onnx.helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, [1, 4])
    output_vi = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, [1, 4])
    
    graph = onnx.helper.make_graph(
        nodes,
        'gelu_graph',
        [input_vi],
        [output_vi],
        [
            onnx.helper.make_tensor('sqrt2', onnx.TensorProto.FLOAT, [], [1.41421356]),
            onnx.helper.make_tensor('one', onnx.TensorProto.FLOAT, [], [1.0]),
            onnx.helper.make_tensor('half', onnx.TensorProto.FLOAT, [], [0.5]),
        ]
    )
    return graph

def test_gelu_fusion_runtime():
    print("="*60)
    print("GELU Fusion Runtime Verification")
    print("="*60)
    
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        orig_path = f.name
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        opt_path = f.name
        
    try:
        graph = make_gelu_graph('x', 'y')
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 14)])
        model.ir_version = 8
        onnx.save(model, orig_path)
        print(f"Original model saved to {orig_path}")
        
        print("Running optimization...")
        res = furiosa_optimizer.optimize_model(orig_path, opt_path)
        
        print(f"Original nodes: {res.original_nodes}")
        print(f"Optimized nodes: {res.final_nodes}")
        
        opt_model = onnx.load(opt_path)
        
        # Debug: Print all nodes
        print("Optimized Graph Nodes:")
        for n in opt_model.graph.node:
            print(f" - {n.op_type} (name={n.name}, domain={n.domain})")

        gelu_nodes = [n for n in opt_model.graph.node if n.op_type == 'Gelu']
        print(f"Found {len(gelu_nodes)} Gelu nodes.")
        
        if len(gelu_nodes) != 1:
            print(f"Expected 1 Gelu node, found {len(gelu_nodes)}")
            return False

        # Check com.microsoft domain
        has_ms_domain = any(imp.domain == "com.microsoft" for imp in opt_model.opset_import)
        if has_ms_domain:
            print("com.microsoft domain found in opset imports.")
        else:
            print("com.microsoft domain NOT found in opset imports.")

        print("Verifying with ONNX Runtime...")
        input_data = np.array([[1.0, 2.0, -1.0, -2.0]], dtype=np.float32)
        
        sess_orig = ort.InferenceSession(orig_path, providers=['CPUExecutionProvider'])
        out_orig = sess_orig.run(None, {'x': input_data})[0]
        
        sess_opt = ort.InferenceSession(opt_path, providers=['CPUExecutionProvider'])
        out_opt = sess_opt.run(None, {'x': input_data})[0]
        
        max_diff = np.max(np.abs(out_orig - out_opt))
        print(f"Max difference: {max_diff}")
        
        if max_diff > 1e-3:
            print("Numerical mismatch!")
            return False
            
        print("Verification Successful!")
        return True
        
    finally:
        if os.path.exists(orig_path):
            os.unlink(orig_path)
        if os.path.exists(opt_path):
            os.unlink(opt_path)

def make_gelu_subgraph(input_name, output_name, suffix):
    nodes = []
    sqrt2_name = f"sqrt2_{suffix}"
    one_name = f"one_{suffix}"
    half_name = f"half_{suffix}"
    div_out = f"div_out_{suffix}"
    erf_out = f"erf_out_{suffix}"
    add_out = f"add_out_{suffix}"
    mul1_out = f"mul1_out_{suffix}"
    
    nodes.append(onnx.helper.make_node('Div', [input_name, sqrt2_name], [div_out], name=f'div_{suffix}'))
    nodes.append(onnx.helper.make_node('Erf', [div_out], [erf_out], name=f'erf_{suffix}'))
    nodes.append(onnx.helper.make_node('Add', [erf_out, one_name], [add_out], name=f'add_{suffix}'))
    nodes.append(onnx.helper.make_node('Mul', [add_out, half_name], [mul1_out], name=f'mul1_{suffix}'))
    nodes.append(onnx.helper.make_node('Mul', [mul1_out, input_name], [output_name], name=f'mul2_{suffix}'))
    return nodes

def create_synthetic_roberta():
    print("Creating Synthetic RoBERTa-like model...")
    batch, seq, hidden = 1, 32, 64
    x = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [batch, seq, hidden])
    y = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [batch, seq, hidden])
    
    nodes = []
    initializers = []
    sqrt2_val = np.array([1.41421356], dtype=np.float32)
    one_val = np.array([1.0], dtype=np.float32)
    half_val = np.array([0.5], dtype=np.float32)
    
    current_input = 'x'
    for i in range(2):
        suffix = str(i)
        initializers.append(onnx.numpy_helper.from_array(sqrt2_val, name=f"sqrt2_{suffix}"))
        initializers.append(onnx.numpy_helper.from_array(one_val, name=f"one_{suffix}"))
        initializers.append(onnx.numpy_helper.from_array(half_val, name=f"half_{suffix}"))
        
        w1_name = f"w1_{suffix}"
        w1 = np.random.randn(hidden, hidden).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(w1, name=w1_name))
        matmul1_out = f"matmul1_out_{suffix}"
        nodes.append(onnx.helper.make_node('MatMul', [current_input, w1_name], [matmul1_out], name=f'matmul1_{suffix}'))
        
        gelu_out = f"gelu_out_{suffix}"
        nodes.extend(make_gelu_subgraph(matmul1_out, gelu_out, suffix))
        
        w2_name = f"w2_{suffix}"
        w2 = np.random.randn(hidden, hidden).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(w2, name=w2_name))
        matmul2_out = f"matmul2_out_{suffix}"
        nodes.append(onnx.helper.make_node('MatMul', [gelu_out, w2_name], [matmul2_out], name=f'matmul2_{suffix}'))
        
        add_res_out = f"layer_{suffix}_out"
        nodes.append(onnx.helper.make_node('Add', [matmul2_out, current_input], [add_res_out], name=f'add_res_{suffix}'))
        
        current_input = add_res_out

    nodes.append(onnx.helper.make_node('Identity', [current_input], ['y'], name='final_identity'))
    
    graph = onnx.helper.make_graph(nodes, 'synthetic_roberta', [x], [y], initializers)
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 14)])
    model.ir_version = 8 # Fix IR version
    return model

def test_synthetic_roberta():
    print("\n" + "="*60)
    print("Synthetic RoBERTa Verification")
    print("="*60)

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        orig_path = f.name
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        opt_path = f.name

    try:
        model = create_synthetic_roberta()
        onnx.save(model, orig_path)
        print(f"Original model saved to {orig_path}")
        
        print("Running optimization...")
        res = furiosa_optimizer.optimize_model(orig_path, opt_path)
        
        print(f"Original nodes: {res.original_nodes}")
        print(f"Optimized nodes: {res.final_nodes}")
        
        opt_model = onnx.load(opt_path)
        
        # Debug: Print all nodes
        print("Optimized Graph Nodes:")
        for n in opt_model.graph.node:
            print(f" - {n.op_type} (name={n.name}, domain={n.domain})")
            
        gelu_nodes = [n for n in opt_model.graph.node if n.op_type == 'Gelu']
        print(f"Found {len(gelu_nodes)} Gelu nodes.")
        
        if len(gelu_nodes) != 2:
            print(f"Expected 2 Gelu nodes, found {len(gelu_nodes)}")
            return False
            
        print("Opset imports:")
        for imp in opt_model.opset_import:
            print(f"  Domain: '{imp.domain}', Version: {imp.version}")
            
        has_ms_domain = any(imp.domain == "com.microsoft" for imp in opt_model.opset_import)
        if has_ms_domain:
            print("com.microsoft domain found in opset imports.")
        else:
            print("com.microsoft domain NOT found in opset imports.")

        print("Verifying with ONNX Runtime...")
        input_data = np.random.randn(1, 32, 64).astype(np.float32)
        
        sess_orig = ort.InferenceSession(orig_path, providers=['CPUExecutionProvider'])
        out_orig = sess_orig.run(None, {'x': input_data})[0]
        
        try:
            sess_opt = ort.InferenceSession(opt_path, providers=['CPUExecutionProvider'])
            out_opt = sess_opt.run(None, {'x': input_data})[0]
        except Exception as e:
            print(f"ONNX Runtime failed: {e}")
            return False

        max_diff = np.max(np.abs(out_orig - out_opt))
        print(f"Max difference: {max_diff}")
        
        if max_diff > 1e-3:
            print("Numerical mismatch!")
            return False
            
        print("Verification Successful!")
        return True

    finally:
        if os.path.exists(orig_path):
            os.unlink(orig_path)
        if os.path.exists(opt_path):
            os.unlink(opt_path)

if __name__ == "__main__":
    success1 = test_gelu_fusion_runtime()
    success2 = test_synthetic_roberta()
    sys.exit(0 if success1 and success2 else 1)
