import onnx
import numpy as np
import sys

def get_const_value(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            return onnx.numpy_helper.to_array(init)
    return None

def debug_gelu_pattern():
    print("Loading roberta_debug.onnx...")
    try:
        model = onnx.load("roberta_debug.onnx")
    except FileNotFoundError:
        print("roberta_debug.onnx not found. Please run analyze_roberta.py first.")
        return

    print("Searching for Erf nodes...")
    erf_nodes = [n for n in model.graph.node if n.op_type == "Erf"]
    
    if not erf_nodes:
        print("No Erf nodes found.")
        return

    print(f"Found {len(erf_nodes)} Erf nodes. Inspecting the first one...")
    erf = erf_nodes[0]
    print(f"Erf Node: {erf.name}, Input: {erf.input}, Output: {erf.output}")

    # Trace back to Div
    div_node = None
    for n in model.graph.node:
        if erf.input[0] in n.output and n.op_type == "Div":
            div_node = n
            break
    
    if div_node:
        print(f"  Preceding Div Node: {div_node.name}, Inputs: {div_node.input}")
        # Check divisor
        if len(div_node.input) > 1:
            val = get_const_value(model, div_node.input[1])
            print(f"    Divisor (Input 1) value: {val}")
            val0 = get_const_value(model, div_node.input[0])
            print(f"    Divisor (Input 0) value: {val0}")
    else:
        print("  No preceding Div node found (or it's not a direct parent).")

    # Trace forward to Add
    add_node = None
    for n in model.graph.node:
        if erf.output[0] in n.input and n.op_type == "Add":
            add_node = n
            break
            
    if add_node:
        print(f"  Following Add Node: {add_node.name}, Inputs: {add_node.input}")
        # Check added value
        for inp in add_node.input:
            if inp != erf.output[0]:
                val = get_const_value(model, inp)
                print(f"    Added value ({inp}): {val}")
    else:
        print("  No following Add node found.")

    # Trace forward to Mul (0.5)
    mul1_node = None
    if add_node:
        for n in model.graph.node:
            if add_node.output[0] in n.input and n.op_type == "Mul":
                mul1_node = n
                break
    
    if mul1_node:
        print(f"  Following Mul1 Node: {mul1_node.name}, Inputs: {mul1_node.input}")
        for inp in mul1_node.input:
            if inp != add_node.output[0]:
                val = get_const_value(model, inp)
                print(f"    Multiplied value ({inp}): {val}")
    else:
        print("  No following Mul1 node found.")

    # Trace forward to Mul (x)
    mul2_node = None
    if mul1_node:
        for n in model.graph.node:
            if mul1_node.output[0] in n.input and n.op_type == "Mul":
                mul2_node = n
                break
    
    if mul2_node:
        print(f"  Following Mul2 Node: {mul2_node.name}, Inputs: {mul2_node.input}")
        print(f"    Inputs: {mul2_node.input}")
        if div_node:
             print(f"    Original Input should be: {div_node.input[0]}")
    else:
        print("  No following Mul2 node found.")

if __name__ == "__main__":
    debug_gelu_pattern()
