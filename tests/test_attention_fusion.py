
import onnx
import onnx.helper
import numpy as np
import furiosa_optimizer
from onnx import TensorProto

def make_attention_graph(input_name, output_name):
    nodes = []
    
    # Q, K, V Projections (Simplified: Input -> MatMul -> Reshape -> Transpose)
    # We skip Add(Bias) for simplicity in this test pattern
    
    # Q Path
    nodes.append(onnx.helper.make_node('MatMul', [input_name, 'W_Q'], ['q_mm'], name='q_mm'))
    nodes.append(onnx.helper.make_node('Reshape', ['q_mm', 'shape_5d'], ['q_rs'], name='q_rs'))
    nodes.append(onnx.helper.make_node('Transpose', ['q_rs'], ['q_tp'], name='q_tp', perm=[0, 2, 1, 3]))
    
    # K Path
    nodes.append(onnx.helper.make_node('MatMul', [input_name, 'W_K'], ['k_mm'], name='k_mm'))
    nodes.append(onnx.helper.make_node('Reshape', ['k_mm', 'shape_5d'], ['k_rs'], name='k_rs'))
    nodes.append(onnx.helper.make_node('Transpose', ['k_rs'], ['k_tp'], name='k_tp', perm=[0, 2, 1, 3]))
    
    # V Path
    nodes.append(onnx.helper.make_node('MatMul', [input_name, 'W_V'], ['v_mm'], name='v_mm'))
    nodes.append(onnx.helper.make_node('Reshape', ['v_mm', 'shape_5d'], ['v_rs'], name='v_rs'))
    nodes.append(onnx.helper.make_node('Transpose', ['v_rs'], ['v_tp'], name='v_tp', perm=[0, 2, 1, 3]))
    
    # Core Attention
    # Q * K^T
    nodes.append(onnx.helper.make_node('MatMul', ['q_tp', 'k_tp'], ['qk'], name='qk'))
    # Scale (Div)
    nodes.append(onnx.helper.make_node('Div', ['qk', 'scale'], ['scaled_qk'], name='div_scale'))
    # Softmax
    nodes.append(onnx.helper.make_node('Softmax', ['scaled_qk'], ['probs'], name='softmax'))
    # Probs * V
    nodes.append(onnx.helper.make_node('MatMul', ['probs', 'v_tp'], ['context'], name='context'))
    
    # Output Projection
    nodes.append(onnx.helper.make_node('Transpose', ['context'], ['out_tp'], name='out_tp', perm=[0, 2, 1, 3]))
    nodes.append(onnx.helper.make_node('Reshape', ['out_tp', 'shape_flat'], ['out_rs'], name='out_rs'))
    nodes.append(onnx.helper.make_node('MatMul', ['out_rs', 'W_O'], [output_name], name='out_mm'))
    
    # Initializers
    initializers = []
    initializers.append(onnx.helper.make_tensor('W_Q', TensorProto.FLOAT, [768, 768], np.random.randn(768, 768).astype(np.float32).flatten()))
    initializers.append(onnx.helper.make_tensor('W_K', TensorProto.FLOAT, [768, 768], np.random.randn(768, 768).astype(np.float32).flatten()))
    initializers.append(onnx.helper.make_tensor('W_V', TensorProto.FLOAT, [768, 768], np.random.randn(768, 768).astype(np.float32).flatten()))
    initializers.append(onnx.helper.make_tensor('W_O', TensorProto.FLOAT, [768, 768], np.random.randn(768, 768).astype(np.float32).flatten()))
    initializers.append(onnx.helper.make_tensor('scale', TensorProto.FLOAT, [], [8.0]))
    initializers.append(onnx.helper.make_tensor('shape_5d', TensorProto.INT64, [4], [1, 12, 64, 64])) # Batch, Heads, Seq, HeadDim
    initializers.append(onnx.helper.make_tensor('shape_flat', TensorProto.INT64, [2], [1, 768]))
    
    graph = onnx.helper.make_graph(
        nodes,
        'AttentionGraph',
        [onnx.helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 768])],
        [onnx.helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 768])],
        initializers
    )
    
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    return model

def test_attention_fusion():
    print("Creating Attention Graph...")
    model = make_attention_graph('x', 'y')
    input_path = 'attention_test.onnx'
    output_path = 'optimized_attention.onnx'
    onnx.save(model, input_path)
    
    print("Optimizing...")
    # optimize_model takes paths, not objects
    furiosa_optimizer.optimize_model(input_path, output_path)
    
    # Load optimized model to check results
    opt_model = onnx.load(output_path)
    
    print(f"Original nodes: {len(model.graph.node)}")
    print(f"Optimized nodes: {len(opt_model.graph.node)}")
    
    # We expect significant reduction. 
    # Original: 3(QKV) * 3(MatMul,Reshape,Transpose) + 4(Core) + 3(Out) = 16 nodes
    # Fused: Should remove most of them. 
    # Current implementation only marks them as eliminated, so we expect node count to drop.
    
    if len(opt_model.graph.node) < len(model.graph.node):
        print("Fusion Successful! Nodes reduced.")
        return True
    else:
        print("Fusion Failed! Node count unchanged.")
        return False

if __name__ == '__main__':
    try:
        if test_attention_fusion():
            print("Test Passed!")
            exit(0)
        else:
            print("Test Failed!")
            exit(1)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
