import numpy as np
import time
import torch

# Tensor auf der GPU erstellen (100 MiB)
import sys

arguments = sys.argv  # Liste der Kommandozeilenargumente
print("Erhaltene Argumente:", arguments)
i = int(sys.argv[1])
size_in_bytes = i * (2 ** 20)  # 100 MiB
num_values = size_in_bytes // 4  # Anzahl für float32 (4 Bytes)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
data = torch.rand(num_values, device=device, dtype=torch.float32)
# Rechenoperationen auf der GPU
result = torch.sin(data) + torch.log(data) - torch.sqrt(data)
sum_result = torch.sum(result)  # Summe der Ergebnisse
print(f"Summe der Berechnungen: {sum_result}")

print("Warte 3 Sekunden...", i)
time.sleep(3)


