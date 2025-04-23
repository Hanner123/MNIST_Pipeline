import json
import torch
import yaml
from train import MnistModel  # Modell importieren - sonst Fehler: Can't get attribute 'MnistModel' on <module '__main__' from '/home/student/git/Simple_NN_Tests/export_model.py'>

def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params



params = load_params()
batch_size = params["train"]["batch_size"]
for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    example_inputs = torch.randn(batch_size, 1, 28, 28).float() # eventuell andere Batch sizes nötig/ häufiger exportieren


    # vergleich: inteferenz mit 16, 32, 64, 128 Batch size auf 128 Batch Size Model (Latenz) #
    # vergleich: inteferenz mit 16, 32, 64, 128 Batch size auf 16, 32, 64, 128 Batch Size Model 
    # es gibt einen Unterschied!!!

    model = torch.load("full_model.pth", map_location="cpu", weights_only=False)

    onnx_program = torch.onnx.export(model, example_inputs, dynamo=True)
    # onnx_program.optimize()
    name = "mnist_model_" + str(batch_size) + ".onnx"
    onnx_program.save(name)
    print("onnx saved")
    # view with https://netron.app/