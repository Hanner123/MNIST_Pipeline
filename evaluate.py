import json
import matplotlib.pyplot as plt
import torch
from train import MnistModel  # Modell importieren
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import confusion_matrix
import csv


#function that can move data and model to a chosen device. (wichtig für GPU-Nutzung, da standartmäßig oft die CPU verwendet wird - es gibt probleme, wenn manche berechnungen auf der GPU und manche auf der CPU laufen)
# RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
def to_device(data,device):
    if isinstance(data, (list,tuple)): #The isinstance() function returns True if the specified object is of the specified type, otherwise False.
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)

# um dataloaders auf gpu zu verschieben 
# Es stellt sicher, dass jede Batch von Daten, die im DataLoader verarbeitet wird, auf das angegebene Gerät verschoben wird, bevor sie ins Modell übergeben wird.
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
    

def evaluate_model(model, test_loader):
    model.eval()  # In den Evaluierungsmodus setzen
    correct_predictions = 0
    total_predictions = 0
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()  # Beispiel für eine Verlustfunktion

    all_labels = []
    all_preds = []

    with torch.no_grad():  # Keine Gradienten berechnen während der Evaluation
        for inputs, labels in test_loader:
            # wieder zurück auf die GPU schieben
            inputs, labels = inputs.to("cuda:0"), labels.to("cuda:0")
            
            # Forward Pass
            outputs = model(inputs)
            
            # Verlust berechnen
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Vorhersagen berechnen
            _, predicted = torch.max(outputs, 1)
            # print("pred: ", predicted[0:5])
            # print("lable: ",labels[0:5])

            # Sammle die Vorhersagen und Labels für Confusion Matrix
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
    print("Correct: ", correct_predictions)
    print("Total: ", total_predictions)
    accuracy = correct_predictions / total_predictions
    average_loss = total_loss / len(test_loader)
    print(f"Test Loss: {average_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Berechnung der Confusion Matrix, das ist eine 10x10 Matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Speichern der Confusion Matrix als CSV
    with open('confusion_matrix.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Schreibe die Kopfzeile (Index der Klassen)
        writer.writerow([f"True/Predicted"] + [str(i) for i in range(cm.shape[0])])
        
        # Schreibe jede Zeile der Confusion Matrix
        for i in range(cm.shape[0]):
            writer.writerow([str(i)] + cm[i].tolist())
            
    print("Confusion Matrix:")
    print(cm)

    metrics = {}
    # Berechnung von Recall und Specificity
    for x in range(10):
        print("  ")
        print("Metrics for Number: ", x)
        TP = cm[x, x]  # True Positive
        #print("TP:", TP)

        # True Negative
        TN = 0
        for i in range(10):
            TN = TN + cm[i, i]
        TN = TN - cm[x, x]
        #print("TN:", TN)

        # False Positive
        FP = 0
        for i in range(10):
            FP = FP+ cm[i, x]
        FP = FP - cm[x, x]
        #print("FP:", FP)

        # False Negative
        FN = 0
        for i in range(10):
            FN = FN + cm[x, i]
        FN = FN - cm[x, x]
        #print("FN:", FN)

        # Recall (Sensitivität)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        # Specificity
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        print(f"Recall: {recall:.6f}")
        print(f"Specificity: {specificity:.6f}")

        metrics[x] = {
        'Recall': round(recall, 6),
        'Specificity': round(specificity, 6)
        }

    with open('metrics.json', 'w') as json_file:
        json.dump(metrics, json_file, indent=4)

    return average_loss, accuracy



test_data = torch.load("test_data.pt", map_location="cpu",weights_only=False)                                               # test daten in cpu laden
model = torch.load("full_model.pth", map_location="cuda:0", weights_only=False)                                             # Modell in gpu laden

test_loader = DeviceDataLoader(DataLoader(test_data, batch_size=128, shuffle=True, num_workers=0, pin_memory=True), "cpu")  # test_loader in die cpu laden, wegen numpy im evaluate_model

evaluate_model(model, test_loader)
