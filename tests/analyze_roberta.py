import torch
from transformers import RobertaConfig, RobertaModel
import onnx
from collections import Counter
import sys

def analyze():
    print("Creating config...")
    # Minimal config for speed
    config = RobertaConfig(
        vocab_size=100, 
        hidden_size=64, 
        num_hidden_layers=1, 
        num_attention_heads=2, 
        intermediate_size=128,
        max_position_embeddings=128
    )
    model = RobertaModel(config)
    model.eval()
    dummy_input = torch.randint(0, 100, (1, 32))
    
    print("Exporting...")
    torch.onnx.export(
        model, 
        dummy_input, 
        "roberta_debug.onnx", 
        opset_version=14,
        input_names=['input_ids'],
        output_names=['last_hidden_state', 'pooler_output']
    )
    
    print("Loading...")
    model = onnx.load("roberta_debug.onnx")
    ops = [n.op_type for n in model.graph.node]
    print(f"Op Types: {Counter(ops)}")
    
    # Check for Erf
    if "Erf" in ops:
        print("Erf found! This matches FuseGeluErf pattern.")
    else:
        print("Erf NOT found! FuseGeluErf will NOT work.")
        # Check for other potential activations
        for op in ["Tanh", "Sigmoid", "Relu", "Gelu"]:
            if op in ops:
                print(f"Found alternative activation: {op}")

if __name__ == "__main__":
    analyze()
