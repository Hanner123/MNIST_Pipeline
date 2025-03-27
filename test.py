import pynvml

pynvml.nvmlInit()

# Erhalte Gerätegriff der ersten verfügbaren GPU
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Überprüfe, ob `nvmlDeviceGetTotalEnergyConsumption` verfügbar ist
try:
    energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    print(f"Total Energy Consumption: {energy} mWh")
except AttributeError:
    print("The method nvmlDeviceGetTotalEnergyConsumption is not supported on this system/GPU.")
except pynvml.NVMLError as e:
    print(f"Error querying energy consumption: {e}")