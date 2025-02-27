import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from torchmetrics.classification import Accuracy
import json
import yaml

def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def accuracy(outputs,labels):
    _,preds = torch.max(outputs,dim = 1) ## _ here max prob will come and we don't require it now
    return torch.tensor(torch.sum(preds == labels).item()/len(preds))

# NN Klasse
class MnistModel(nn.Module):
    def __init__(self,input_size,hidden_size,out_size):
        super().__init__()
        #Modell hat drei Schichten: Input, Hidden und Output
        self.linear1 = nn.Linear(input_size,hidden_size) #parameter 1 (input -> hidden)
        self.linear2 = nn.Linear(hidden_size,out_size) #parameter 2 (hidden -> output)

    def forward(self,xb):
        # forward bedeutet, dass ein Schritt der Klassifizierung mit den vorhandenen Gewichten ausgeführt wird.
        xb = xb.view(xb.size(0),-1) ## same as .reshape()
        out = self.linear1(xb)  # erste lineare transformation (out=xb⋅W1​+b1) xb: eingabetensor, W1: Gewichte der ersten schicht, b_1: Bias-Vektor
        out = F.relu(out)       # Aktivierungsfunktion ReLu
        out = self.linear2(out) # zweite lineare transformation
        return out
    
    def training_step(self,batch):
        # berechnung des losses für einen Schritt (für Backpropagation)
        images,labels = batch #extract the images and labels from the batch
        out = self(images) #self(images) ruft die forward methode auf! Das hier sind also die Prediction-Wahrscheinlichkeiten für einen step
        loss = F.cross_entropy(out,labels) #jetzt wird geschaut, wie groß der Fehler ist, zwischen prediction und tatsächlichen labels
        return loss
    
    def validation_step(self,batch):
        # genauso wie training, aber berechnet zusätzlich accuracy und printet die werte aus
        images,labels = batch
        out = self(images) #höherer Wert -> höhere Wahrscheinlichkeit für die Klasse, argmax -> wahrscheinlichste Klasse
        loss = F.cross_entropy(out,labels)
        acc = accuracy(out,labels)
        return {'val_loss': loss, 'val_acc': acc}
    
    def validation_epoch_end(self,outputs):
        # validierungsfunktion für das ende der Epoche (Dict)
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_acc = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {'val_loss': epoch_loss.item(),'val_acc': epoch_acc.item()}
    
    def epoch_end(self,epoch,result): # print funktion für das dictionary
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch+1, result['val_loss'], result['val_acc']))




#function that can move data and model to a chosen device. (wichtig für GPU-Nutzung, da standartmäßig oft die CPU verwendet wird - es gibt probleme, wenn manche berechnungen auf der GPU und manche auf der CPU laufen)
# RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
def to_device(data,device):
    if isinstance(data, (list,tuple)): #The isinstance() function returns True if the specified object is of the specified type, otherwise False.
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)

# um dataloaders auf gpu zu verschieben 
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



def evaluate(model,val_loader):
    # aktuelles Modell betrachten (mit Gewichten), predicten (mit validierungsdatensatz) und Metriken berechnen
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs,lr,model,train_loder,val_loader,opt_func = torch.optim.SGD):
    # führt das training aus, mit backpropagation
    history = []
    optimizer = opt_func(model.parameters(),lr)
    # führt die epochen durch
    for epoch in range(epochs):
        for batch in train_loder:
            loss = model.training_step(batch) # predicts (forward-step) and determines loss
            loss.backward() # berechnet die Gradienten des Verlusts bezüglich der Modellparameter (Gewichte und Biases), bei den Modellparametern gespeichert, Kettenregel Defferentation, Richtung und Gröe für Änderung der Gewichte
            optimizer.step() # used to update the parameters
            optimizer.zero_grad() # Clears the gradients of optimizer
        result = evaluate(model,val_loader) # evaluierung am ende der epoche
        model.epoch_end(epoch,result) # print
        history.append(result) # history: alle evaluierungen der Epoche
    return history



def train():
    # Lade Parameter aus der YAML-Datei
    params = load_params()
    print(params)
    batch_size = params["train"]["batch_size"]
    learning_rate = params["train"]["lr"]
    # Prüfen, ob eine GPU verfügbar ist, andernfalls CPU verwenden
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Verwende Gerät: {device}")

    # download dataset
    dataset = MNIST(root = 'data/',train=True,transform=ToTensor())
    dataset_test = MNIST(root = 'data/',train=False,transform=ToTensor())
    torch.save(dataset_test, "test_data.pt")

    # split in validation and train data
    val_size = 10000
    train_size = len(dataset) - val_size # 50000
    train_ds,val_ds = random_split(dataset,[train_size,val_size])
    print(len(train_ds), len(val_ds))

    # batch size for every iteration in each spoch
    #dataloaders: to load dataset in batches and to shuffle the data -> no overfitting, parallel processes
    train_loder = DataLoader(train_ds,batch_size,shuffle=True,num_workers=4,pin_memory=True)
    val_loader = DataLoader(val_ds,batch_size,shuffle=True,num_workers=4,pin_memory=True)



    input_size = 784 #28x28 Pixel
    out_size = 10 #anzahl klassen
    hidden_size = 32 #anzahl neuronen in versteckter schicht

    # model wird initialsiert
    model = MnistModel(input_size,hidden_size,out_size)

    train_loder = DeviceDataLoader(train_loder,device)
    val_loader = DeviceDataLoader(val_loader,device)

    to_device(model, device)

    print(evaluate(model,val_loader)) # Modell wird vor dem training evaluiert, um es vergleichen zu können (es sollte ca 10% genauigkeit ergeben)

    history = fit(10,learning_rate,model,train_loder,val_loader) # Modell trainieren und für epochen prints ausgeben

    with open("training_data2.json", "w") as f:
        json.dump(history, f, indent = 4)

    torch.save(model, "full_model.pth")



if __name__ == "__main__":
    train()  # Nur wenn dieses Skript direkt ausgeführt wird
