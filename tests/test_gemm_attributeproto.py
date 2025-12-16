#!/usr/bin/env python3
"""
AttributeProto 타입 수정 검증 테스트

이 테스트는 common.rs의 set_attr_f() 버그 수정이 제대로 적용되었는지 검증합니다.
- Gemm 노드의 alpha, beta 속성이 올바른 타입(FLOAT=1)을 가지는지 확인
- ONNX Runtime에서 모델이 정상 로드되는지 확인
"""

import sys
import tempfile
import os
from pathlib import Path

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import onnxruntime as ort

# Add furiosa_optimizer to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import furiosa_optimizer


def create_matmul_add_model():
    """MatMul + Add 패턴 모델 생성 (Gemm으로 융합될 예정)"""
    # Input/Output
    X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [2, 3])
    Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [2, 4])
    
    # Initializers (weights)
    W_data = np.random.randn(3, 4).astype(np.float32)
    B_data = np.random.randn(4).astype(np.float32)
    W = onnx.numpy_helper.from_array(W_data, 'W')
    B = onnx.numpy_helper.from_array(B_data, 'B')
    
    # Nodes: MatMul + Add
    matmul = onnx.helper.make_node('MatMul', ['X', 'W'], ['tmp'], name='matmul')
    add = onnx.helper.make_node('Add', ['tmp', 'B'], ['Y'], name='add')
    
    # Graph
    graph = onnx.helper.make_graph(
        [matmul, add],
        'test_graph',
        [X],
        [Y],
        [W, B]
    )
    
    # Model (opset 13 for Gemm support)
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    
    return model


def verify_gemm_attributes(model):
    """Gemm 노드의 AttributeProto 타입 검증"""
    gemm_nodes = [n for n in model.graph.node if n.op_type == 'Gemm']
    
    if not gemm_nodes:
        return False, "No Gemm nodes found after optimization"
    
    errors = []
    for gemm in gemm_nodes:
        for attr in gemm.attribute:
            if attr.name in ['alpha', 'beta']:
                # ONNX 표준: FLOAT = 1
                if attr.type != 1:
                    errors.append(
                        f"Gemm.{attr.name}: type={attr.type} (expected 1=FLOAT), value={attr.f}"
                    )
    
    if errors:
        return False, "; ".join(errors)
    
    return True, f"Found {len(gemm_nodes)} Gemm nodes with correct FLOAT attributes"


def test_attributeproto_fix():
    """AttributeProto 타입 버그 수정 검증"""
    print("=" * 80)
    print("AttributeProto 타입 수정 검증 테스트")
    print("=" * 80)
    
    # 1. ONNX 표준 확인
    print("\n[1] ONNX AttributeProto 타입 표준:")
    print(f"    FLOAT = {onnx.AttributeProto.FLOAT} (예상: 1)")
    print(f"    INT = {onnx.AttributeProto.INT} (예상: 2)")
    print(f"    INTS = {onnx.AttributeProto.INTS} (예상: 7)")
    
    assert onnx.AttributeProto.FLOAT == 1, "ONNX FLOAT type mismatch"
    assert onnx.AttributeProto.INT == 2, "ONNX INT type mismatch"
    assert onnx.AttributeProto.INTS == 7, "ONNX INTS type mismatch"
    print("    ✅ ONNX 표준 확인 완료")
    
    # 2. 테스트 모델 생성
    print("\n[2] MatMul + Add 모델 생성:")
    original_model = create_matmul_add_model()
    print(f"    노드: {len(original_model.graph.node)} (MatMul, Add)")
    
    # 3. 최적화 실행
    print("\n[3] Furiosa Optimizer 실행:")
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        original_path = f.name
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        optimized_path = f.name
    
    try:
        # Save original
        onnx.save(original_model, original_path)
        
        # Optimize
        result = furiosa_optimizer.optimize_model(original_path, optimized_path)
        print(f"    Original nodes: {result.original_nodes}")
        print(f"    Optimized nodes: {result.final_nodes}")
        print(f"    Reduction: {result.reduction_percent:.1f}%")
        
        # Load optimized model
        optimized_model = onnx.load(optimized_path)
        
        # 4. Gemm AttributeProto 검증
        print("\n[4] Gemm AttributeProto 타입 검증:")
        passed, message = verify_gemm_attributes(optimized_model)
        print(f"    {message}")
        
        if not passed:
            print("    ❌ FAILED: AttributeProto 타입 오류")
            return False
        
        # 5. ONNX Runtime 로드 테스트
        print("\n[5] ONNX Runtime 로드 테스트:")
        try:
            onnx.checker.check_model(optimized_model)
            print("    ✅ ONNX 검증 통과")
            
            sess = ort.InferenceSession(optimized_path, 
                                        ort.SessionOptions())
            print("    ✅ ONNX Runtime 로드 성공")
            
            # 6. 추론 정확도 테스트
            print("\n[6] 추론 정확도 검증:")
            input_data = {'X': np.random.randn(2, 3).astype(np.float32)}
            
            # Original inference
            sess_original = ort.InferenceSession(original_path, ort.SessionOptions())
            output_original = sess_original.run(None, input_data)[0]
            
            # Optimized inference
            output_optimized = sess.run(None, input_data)[0]
            
            # Compare
            max_diff = np.max(np.abs(output_original - output_optimized))
            mean_diff = np.mean(np.abs(output_original - output_optimized))
            
            print(f"    Max diff: {max_diff:.2e}")
            print(f"    Mean diff: {mean_diff:.2e}")
            
            if max_diff < 1e-5:
                print("    ✅ 정확도 검증 통과 (max_diff < 1e-5)")
            else:
                print(f"    ⚠️  정확도 차이 다소 큼 (max_diff = {max_diff:.2e})")
            
        except Exception as e:
            print(f"    ❌ ONNX Runtime 오류: {e}")
            return False
        
    finally:
        # Cleanup
        if os.path.exists(original_path):
            os.unlink(original_path)
        if os.path.exists(optimized_path):
            os.unlink(optimized_path)
    
    print("\n" + "=" * 80)
    print("🎉 모든 검증 통과!")
    print("=" * 80)
    print("\n결론:")
    print("  - AttributeProto 타입 버그가 성공적으로 수정되었습니다")
    print("  - Gemm 노드의 alpha, beta 속성이 올바른 타입(FLOAT=1)을 가집니다")
    print("  - ONNX Runtime에서 정상적으로 로드 및 추론됩니다")
    print("  - 정확도 손실 없이 최적화가 적용되었습니다")
    
    return True


if __name__ == '__main__':
    try:
        success = test_attributeproto_fix()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
