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
    return correct_predictions, total_predictions

def save_json(log, filepath):
    with open(filepath, "w") as f:
        json.dump(log, f, indent=4)

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

def print_latency(latency_ms, latency_synchronize, latency_datatransfer, end_time, start_time, num_batches, throughput_batches, throughput_images):
    print("For Batch Size: ", batch_size)
    print(f"Gemessene durchschnittliche Latenz für Inteferenz : {latency_ms:.4f} ms")
    print(f"Gemessene durchschnittliche Latenz mit Synchronisation : {latency_synchronize:.4f} ms")
    print(f"Gemessene durchschnittliche Latenz mit Datentransfer : {latency_datatransfer:.4f} ms")
    print(f"Gesamtzeit: {end_time-start_time:.4f} s")
    print("num_batches", num_batches)
    print(f"Throughput: {throughput_batches:.4f} Batches/Sekunde")
    print(f"Throughput: {throughput_images:.4f} Bilder/Sekunde")

def build_tensorrt_engine(onnx_model_path):
    """
    Erstellt und gibt die TensorRT-Engine und den Kontext zurück.
    :param onnx_model_path: Pfad zur ONNX-Modell-Datei.
    :param logger: TensorRT-Logger.
    :return: TensorRT-Engine und Execution Context.
    """

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    success = parser.parse_from_file(onnx_model_path)
    if not success:
        for idx in range(parser.num_errors):
            print(parser.get_error(idx))
        raise RuntimeError(f"Fehler beim Parsen von {onnx_model_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)  # 1 MiB

    serialized_engine = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()

    return engine, context


def create_test_dataloader(data_path, batch_size, device):
    """
    Erstellt den DataLoader für die Testdaten.
    :param data_path: Pfad zur Testdaten-Datei.
    :param batch_size: Die Batchgröße.
    :param device: Zielgerät (z. B. 'cpu' oder 'cuda').
    :return: DataLoader-Objekt für die Testdaten.
    """
    test_data = torch.load(data_path, map_location=device, weights_only=False) 
    test_loader = DeviceDataLoader(DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True), device)  # test_loader in die cpu laden, wegen numpy im evaluate_model
    return test_loader

def run_inference(
        context, test_loader, device_input, device_output, stream_ptr, torch_stream, batch_size):
    """pynvml-Stream-Pointer.
    :param torch_stream: PyTorch CUDA-Stream.
    :param max_iterations: Maximalanzahl der Iterationen.
    :return: (Anzahl der korrekten Vorhersagen, Gesamtanzahl der Vorhersagen).
    """
    iterations_max=int(10000/batch_size)
    total_predictions = 0
    correct_predictions = 0

    for idx, (images, labels) in enumerate(test_loader):
        if idx == iterations_max: 
            break  # Begrenzung der Iterationen

        images = images.float()
        device_input.copy_(images)
        
        with torch.cuda.stream(torch_stream):
            context.execute_async_v3(stream_ptr)
            torch_stream.synchronize()
        
        output = device_output.cpu().numpy()
        correct, total = accuracy(labels, output)
        total_predictions += total
        correct_predictions += correct

    np.savetxt("tensorrt_inteference.txt", output)

    return correct_predictions, total_predictions

def calculate_latency_and_throughput(context, test_loader, device_input, device_output, stream_ptr, torch_stream, batch_sizes):
    """
    Berechnet die durchschnittliche Latenz und den Durchsatz (Bilder und Batches pro Sekunde) für verschiedene Batchgrößen.
    :param context: TensorRT-Execution-Context.
    :param test_loader: DataLoader mit Testdaten.
    :param device_input: Eingabebuffer auf der GPU.
    :param device_output: Ausgabebuffer auf der GPU.
    :param stream_ptr: CUDA-Stream-Pointer.
    :param torch_stream: PyTorch CUDA-Stream.
    :param batch_sizes: Liste der Batchgrößen.
    :return: (Throughput-Log, Latenz-Log).
    """
    throughput_log = []
    latency_log = []

    for batch_size in batch_sizes:
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
        num_batches = int(10000/batch_size)
        throughput_batches = num_batches/(end_time-start_time) 
        throughput_images = (num_batches*batch_size)/(end_time-start_time)
        log_latency_inteference = {"batch_size": batch_size, "type":"inteference", "value": latency_ms}
        log_latency_synchronize = {"batch_size": batch_size, "type":"synchronize", "value": (latency_synchronize-latency_ms)}
        log_latency_datatransfer = {"batch_size": batch_size, "type":"datatransfer", "value": (latency_datatransfer-latency_synchronize)}
        throughput = {"batch_size": batch_size, "throughput_images_per_s": throughput_images, "throughput_batches_per_s": throughput_batches}
        throughput_log.append(throughput)
        latency_log.extend([log_latency_inteference, log_latency_synchronize, log_latency_datatransfer])
        print_latency(latency_ms, latency_synchronize, latency_datatransfer, end_time, start_time, num_batches, throughput_batches, throughput_images)

    return throughput_log, latency_log

def test_data():
    input_name = "xb"
    output_name = "linear_1"
    input_shape = (batch_size, 1, 28, 28)  
    output_shape = (batch_size, 10)
    device_input = torch.empty(input_shape, dtype=torch.float32, device='cuda')  # Eingabe auf der GPU
    device_output = torch.empty(output_shape, dtype=torch.float32, device='cuda')  # Ausgabe auf der GPU
    torch_stream = torch.cuda.Stream()
    stream_ptr = torch_stream.cuda_stream
    context.set_tensor_address(input_name, device_input.data_ptr()) 
    context.set_tensor_address(output_name, device_output.data_ptr())  # AusgabeTensor verknüpfen
    return device_input, device_output, stream_ptr, torch_stream

def measure_memory(batch_sizes, context, test_loader, device_input, device_output, stream_ptr, torch_stream):

    # nvidia-smi -l 1
    memory_log = []

    for batch_size in batch_sizes:
        gc.collect()
        torch.cuda.empty_cache()  # GPU-Speicher freigeben
        time.sleep(1)  # Stabilitätspause

        torch.cuda.synchronize()  # GPU-Synchronisation
        start_vram = torch.cuda.memory_allocated(device='cuda') #nur torch?

        start_sys_vram = torch.cuda.max_memory_allocated(device='cuda')  # Maximaler GPU-Speicher
        start_ram = psutil.virtual_memory().used / (1024 ** 2)  # RAM vorher in MB

        # Führe Inferenz durch
        for images, _ in test_loader:
            images = images.float().to('cuda')  # Daten auf die GPU verschieben
            if images.size(0) != batch_size:
                continue
            device_input.copy_(images)  # In den GPU-Input-Speicher kopieren

            with torch.cuda.stream(torch_stream):
                context.execute_async_v3(stream_ptr)
            torch.cuda.synchronize()  # Synchronisiere GPU, um abgeschlossene Operationen sicherzustellen
            break  # Nur ein Batch ablaufen


        # Speicherverbrauch nach Inferenz
        end_vram = torch.cuda.memory_allocated(device='cuda')
        end_sys_vram = torch.cuda.max_memory_allocated(device='cuda')
        end_ram = psutil.virtual_memory().used / (1024 ** 2)
        # Differenz berechnen
        vram_usage = (end_vram - start_vram) / (1024 ** 2)  # in MB
        sys_vram_usage = (end_sys_vram - start_sys_vram) / (1024 ** 2)  # Maximaler GPU-Speicher
        ram_usage = end_ram - start_ram  # in MB

        memory_log.append({"batch_size": batch_size, "vram_usage_mb": vram_usage, "sys_vram_usage_mb": sys_vram_usage, "ram_usage_mb": ram_usage})
        print(f"Batch Size: {batch_size}, VRAM Usage: {vram_usage:.2f} MB, Peak VRAM Usage: {sys_vram_usage:.2f} MB, RAM Usage: {ram_usage:.2f} MB")
    print("5", images.device)

    return memory_log



# Energiemessung
def run_inference_with_energy_measurements():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    total_energy_joules = 0
    interval_seconds = 0.1

    # Beispielhafte Batch-Inferenz
    num_batches = 100
    for batch_idx in range(num_batches):
        start_time = time.time()
        
        # Simuliere Inferenz
        time.sleep(0.05)  # (Hier kommt der TensorRT-Inferenzcode hin)
        
        # Energie in diesem Intervall berechnen
        power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watt
        elapsed_time = time.time() - start_time
        total_energy_joules += power_draw * elapsed_time

    total_energy_mwh = total_energy_joules / 3600 * 1000  # Joule in Milliwattstunden umrechnen
    print(f"Total Energy Consumption: {total_energy_mwh:.4f} mWh")

if __name__ == "__main__":
    onnx_model_path="mnist_model.onnx"
    data_path = "test_data.pt"
    batch_size = 1

    engine, context = build_tensorrt_engine(onnx_model_path)

    test_loader = create_test_dataloader(data_path, batch_size, "cpu")

    device_input, device_output, stream_ptr, torch_stream = test_data()

    # correct_predictions, total_predictions = run_inference(context, test_loader, device_input, device_output, stream_ptr, torch_stream, batch_size)
    # print(f"Accuracy: {correct_predictions / total_predictions:.2%}")

    batch_sizes = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 606, 700, 750, 800, 850, 900, 950, 1024, 1280, 1460, 1536, 1600, 1792, 2048, 2560, 3072, 3584, 4096]
    throughput_log, latency_log = calculate_latency_and_throughput(context, test_loader, device_input, device_output, stream_ptr, torch_stream, batch_sizes)

    profile = onnx_tool.model_profile(onnx_model_path, None, None)

    save_json(throughput_log, "throughput_results.json")
    save_json(throughput_log, "throughput_results_2.json")
    save_json(latency_log, "latency_results.json")

    batch_sizes = [1, 64, 128]  # Beispiel-Batchgrößen
    memory_log = measure_memory(batch_sizes, context, test_loader, device_input, device_output, stream_ptr, torch_stream) #negative werte -> schwankungen, geringe auslastung
    # run_inference_with_energy_measurements() #probleme mit berechtigung: mit sudo ausführen -> tensorrt fehlt
    # sudo /home/student/git/Simple_NN_Tests/venv/bin/python measure.py
    #`nvmlDeviceGetTotalEnergyConsumption` ist nicht unterstützt: Die Funktion `nvmlDeviceGetTotalEnergyConsumption()` wurde in neueren NVIDIA-Treibern und GPUs hinzugefügt. Deine aktuelle Kombination aus GPU (GTX 1050 Ti) und NVIDIA-Treiber unterstützt diese Funktion möglicherweise nicht.
    #nvidia-smi --query-gpu=power.draw --format=csv power.draw [W] [N/A]
    # Die Ausgabe `power.draw [W] [N/A]` bedeutet, dass deine GPU (GTX 1050 Ti) oder der verwendete NVIDIA-Treiber keine Leistungswerte in Watt (`power.draw`) bereitstellen kann. Dies ist leider bei einigen Consumer-GPUs wie der GTX-1050-Ti ein bekanntes Problem. 



# Funktion schreiben: Speicher mit parametern des Modells berechnen, mit nvidia-smi überprüfen




