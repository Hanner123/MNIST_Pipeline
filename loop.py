import numpy as np
import time
import torch

# Tensor auf der GPU erstellen (100 MiB)

    


import subprocess

# Starte ein Programm im Hintergrund#
list = list(range(40,200+1,30))
for i in list:
    subprocess.run(["python3", "measure_for_mem.py", str(i)])


for idx, speicher in enumerate(list):
    print("von Python reservierter speicher: 72 MiB") #immer 72...
    print("Batch Größe: ", list[idx])
    print("-" * 30)  # Trennlinie für Lesbarkeit

print("Das Skript läuft im Hintergrund...")


# das gleiche nochmal mit measure machen (latency für verschiedene Batch sizes)