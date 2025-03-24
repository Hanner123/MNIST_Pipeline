import tensorrt as trt
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time

def to_device(data,device):
    if isinstance(data, (list,tuple)): #The isinstance() function returns True if the specified object is of the specified type, otherwise False.
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)

class DeviceDataLoader():
    def __init__(self,dl,device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b,self.device)
    
    def __len__(self):
        return len(self.dl)

def accuracy(labels, outputs):
    correct_predictions = 0
    total_predictions = 0
    i = 0
    for label in labels:
        _, predicted = torch.max(torch.tensor(outputs[i]), dim=0)
        total_predictions = total_predictions + 1
        if predicted == label:
            correct_predictions = correct_predictions + 1
        i = i+1
    # print("accuracy", float(correct_predictions)/float(total_predictions))
    # print("Correct Predictions:", correct_predictions)
    # print("Total Predictions:", total_predictions)
    return correct_predictions, total_predictions


def measure_latency(context, test_loader, device_input, device_output, stream_ptr, torch_stream, batch_size=1):
    """
    Funktion zur Bestimmung der Inferenzlatenz.
    """
    total_time = 0
    total_time_synchronize = 0
    total_time_datatransfer = 0  # Gesamte Laufzeit aller gemessenen Batches
    iterations = 0  # Anzahl gemessener Batches
    k_max = int(10000/batch_size)
    for images, labels in test_loader:
        images = images.float()

 
        start_time_datatransfer = time.time()  # Startzeit messen
        device_input.copy_(images)  # Eingabe auf GPU übertragen

        start_time_synchronize = time.time()  # Startzeit messen
        torch_stream.synchronize()  

        start_time_inteference = time.time()  # Startzeit messen

        with torch.cuda.stream(torch_stream):
            context.execute_async_v3(stream_ptr)  # TensorRT-Inferenz durchführen
        torch_stream.synchronize()  # GPU-Synchronisation nach Inferenz

        end_time = time.time()

        output = device_output.cpu().numpy()
        end_time_datatransfer = time.time() 

        latency = end_time - start_time_inteference  # Latenz für diesen Batch
        latency_synchronize = end_time - start_time_synchronize  # Latenz für diesen Batch
        latency_datatransfer = end_time_datatransfer - start_time_datatransfer  # Latenz für diesen Batch

        total_time += latency
        total_time_synchronize += latency_synchronize
        total_time_datatransfer += latency_datatransfer


        iterations += 1
        if iterations == k_max: break

        # labels auswerten - zeit messen, bar plots

    average_latency = (total_time / iterations) * 1000  # In Millisekunden
    average_latency_synchronize = (total_time_synchronize / iterations) * 1000  # In Millisekunden
    average_latency_datatransfer = (total_time_datatransfer / iterations) * 1000  # In Millisekunden


    return average_latency, average_latency_synchronize, average_latency_datatransfer

def test():
    start_time = time.time()
    time.sleep(0.1)  # Pause für eine Sekunde
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Verstrichene Zeit: {elapsed_time} Sekunden")


batch_size = 1
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

success = parser.parse_from_file("mnist_model.onnx")    # parser defines the network
for idx in range(parser.num_errors):
    print(parser.get_error(idx))
if not success:
    print("no success")
    pass # Error handling code here

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) # 1 MiB
serialized_engine = builder.build_serialized_network(network, config) # Error Code 9: API Usage Error (Target GPU SM 61 is not supported by this TensorRT release.)
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(serialized_engine)

with open("mnist_model.onnx", "rb") as f:
    model_data = f.read()

#test data
input_name = "xb"
output_name = "linear_1"
input_shape = (batch_size, 1, 28, 28)  
output_shape = (batch_size, 10)
input_size = trt.volume(input_shape) * trt.float32.itemsize  # Größe in Bytes
output_size = trt.volume(output_shape) * trt.float32.itemsize  # Größe in Bytes

test_data = torch.load("test_data.pt", map_location="cpu",weights_only=False) 
test_loader = DeviceDataLoader(DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True), "cpu")  # test_loader in die cpu laden, wegen numpy im evaluate_model


#inteference
context = engine.create_execution_context()

# input/output buffers
device_input = torch.empty(input_shape, dtype=torch.float32, device='cuda')  # Eingabe auf der GPU
device_output = torch.empty(output_shape, dtype=torch.float32, device='cuda')  # Ausgabe auf der GPU

torch_stream = torch.cuda.Stream()
stream_ptr = torch_stream.cuda_stream

context.set_tensor_address(input_name, device_input.data_ptr()) 
context.set_tensor_address(output_name, device_output.data_ptr())  # AusgabeTensor verknüpfen

# inteference

total_predictions = 0
correct_predictions = 0
iterations = 0
iterations_max = int(10000/batch_size)
for images, labels in test_loader:
    images = images.float()
    # image = images[0].squeeze()
    # plt.imsave("output_image.png", image.numpy(), cmap='gray' if image.ndimension() == 2 else None)
    device_input.copy_(images)
    #inteferenz
    with torch.cuda.stream(torch_stream):
        context.execute_async_v3(stream_ptr)  # Richtiger Aufruf der Methode
        torch_stream.synchronize()  # Warten, bis die GPU die Inferenz abgeschlossen hat
    output = device_output.cpu().numpy()
    correct, total = accuracy(labels, output)
    total_predictions = total + total_predictions
    correct_predictions = correct + correct_predictions
    iterations = iterations+1
    if iterations == iterations_max:
        break 

print("Correct Predictions: ",correct_predictions)
print("Total Predictions: ",total_predictions)
print("Accuracy: ", float(correct_predictions)/float(total_predictions))

#print(f"Inferenz-Ergebnis (Shape {host_output.shape}):\n{output}")

np.savetxt("tensorrt_inteference.txt", output)

for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1048, 2096]:
    #Latency:
    print("For Batch Size: ", batch_size)
    start_time = time.time()
    latency_ms, latency_synchronize, latency_datatransfer = measure_latency(
        context=context,
        test_loader=test_loader,
        device_input=device_input,
        device_output=device_output,
        stream_ptr=stream_ptr,
        torch_stream=torch_stream,
        batch_size=batch_size
    )
    end_time = time.time()
    print(f"Gemessene durchschnittliche Latenz für Inteferenz: {latency_ms:.4f} ms")
    print(f"Gemessene durchschnittliche Latenz mit Synchronisation: {latency_synchronize:.4f} ms")
    print(f"Gemessene durchschnittliche Latenz mit Datentransfer: {latency_datatransfer:.4f} ms")
    print(f"Gesamtzeit: {end_time-start_time:.4f} s")
    num_batches = int(10000/batch_size)
    print(f"Throughput: {num_batches/(end_time-start_time):.4f} Batches/Sekunde")
    print(f"Throughput: {(num_batches*batch_size)/(end_time-start_time):.4f} Bilder/Sekunde")
