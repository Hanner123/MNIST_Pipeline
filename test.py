import tensorrt as trt
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
import time
import json
import onnx_tool
import torch
import psutil
import gc
from pyJoules.energy_meter import measure_energy
from pyJoules.device.rapl_device import RaplPackageDomain
from pyJoules.device.nvidia_device import NvidiaGPUDomain
import pynvml
import onnx
from onnx import numpy_helper

onnx_model_path = "mnist_model.onnx"
#profile = onnx_tool.model_profile(onnx_model_path, None, None)
onnx_model = onnx.load(onnx_model_path)
print("\nModel Inputs:")
for input_tensor in onnx_model.graph.input:
    print(input_tensor)

print("\nModel Outputs:")
for output_tensor in onnx_model.graph.output:
    print(output_tensor)

# Inspektiere die Layer (Operatoren, Weight-Tensoren etc.)
all_parameters = 0  # Summe der Parameter
print("\nModel Nodes (Operations):")
for node in onnx_model.graph.node:
    print(f"- {node.name}: {node.op_type}")

# Inspektiere Gewichtungen und Speicherplatz
for initializer in onnx_model.graph.initializer:
    weight = numpy_helper.to_array(initializer)
    all_parameters += weight.size
    print(f"Name: {initializer.name}, Shape: {weight.shape}, DataType: {weight.dtype}")
print(f"\nTotal Parameters: {all_parameters}")

# Speicherplatz schätzen (typischerweise float32 = 4 Bytes)
total_memory = all_parameters * 4 / (1024 ** 2)  # Speicherplatz in MB
print(f"Total memory usage (approx.): {total_memory:.2f} MB")
