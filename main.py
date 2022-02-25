#2243392 Pınar Dilbaz
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import numpy

def main():
    # Fix the randomness
    seed = 1234
    torch.manual_seed(seed)

    # For datasets
    train_transform = T. Compose ([T. ToTensor () , T. Normalize ( mean =(0.5 ,) , std=(0.5 ,) ) ])
    val_transform = test_transform = T. Compose ([T. ToTensor () ,T. Grayscale () ,T. Normalize ( mean =(0.5 ,) , std=(0.5 ,) )])

    train_set = CIFAR10(root="Dataset", train=True, transform=train_transform, download=True)
    train_set_length = int(0.8 * len(train_set))
    val_set_length = len(train_set) - train_set_length

    train_set, val_set = torch.utils.data.random_split(train_set, [train_set_length, val_set_length])

    test_set = CIFAR10(root="Dataset", train=False, transform=train_transform, download=True)

    batch_size = 32
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #num_epochs = 50
    num_epochs = 3
    layer_numbers = [1, 2, 3]
    layer_sizes = [150, 200, 350]
    activation_functions = ['relu', 'hardswish', 'tanh']
    learning_rates = [1E-2, 1E-3, 1E-4, 1E-5, 1E-6, 1E-7]

    for learning_rate in learning_rates:

        for layer_number in layer_numbers:

            for layer_size in layer_sizes:

                if (layer_number == 1):
                    print(f'\nLayer Number: {layer_number} Activation Function: none  Size: {layer_size} Learning Rate: {learning_rate}')
                    model = MyModel(layer_number, layer_size, 'none').to(device)
                    loss_function = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
                    train(model, train_loader, optimizer, val_loader, device, loss_function, num_epochs, test_loader)
                    
                elif (layer_number == 2): 
                    for activation_function in activation_functions:
                        print(f'\nLayer Number: {layer_number} Activation Function: {activation_function}  Size: {layer_size} Learning Rate: {learning_rate}\n')
                        model = MyModel(layer_number, layer_size, activation_function).to(device)
                        loss_function = nn.CrossEntropyLoss()
                        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
                        train(model, train_loader, optimizer, val_loader, device, loss_function, num_epochs, test_loader)

                elif (layer_number == 3):
                    for activation_function in activation_functions: 
                        print(f'\nLayer Number: {layer_number} Activation Function: {activation_function}  Size: {layer_size} Learning Rate: {learning_rate}\n')
                        model = MyModel(layer_number, layer_size, activation_function).to(device)
                        loss_function = nn.CrossEntropyLoss()
                        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
                        train(model, train_loader, optimizer, val_loader, device, loss_function, num_epochs, test_loader)  
    


    #for try to the best result with 50 epoch and find the train and validation loss graph
    #The Best Result
    # model = MyModel(3, 150, 'hardswish').to(device)
    # loss_function = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # train(model, train_loader, optimizer, val_loader, device, loss_function, num_epochs, test_loader)



#here is my Model
class MyModel(nn.Module):
    def __init__(self, layerNum, layerSize, activationFunction,):
        super(MyModel, self).__init__()
        #for layer-1
        if layerNum   == 1:
            self.layer1 = nn.Linear(in_features=32*32*3, out_features=10)
            self.layerNum = 1
        #for layer-2
        elif layerNum == 2:
            self.layer1 = nn.Linear(in_features=32*32*3, out_features=layerSize)
            self.layer2 = nn.Linear(in_features=layerSize,out_features=10)
            self.layerNum = 2
            self.activationFunction = activationFunction
        #for layer-3
        elif layerNum == 3:
            self.layer1 = nn.Linear(in_features=32*32*3, out_features=layerSize)
            self.layer2 = nn.Linear(in_features=layerSize, out_features=layerSize)
            self.layer3 = nn.Linear(in_features=layerSize, out_features=10)
            self.layerNum = 3
            self.activationFunction = activationFunction

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.hardswish(self.layer1(x))
        if self.layerNum > 1: #for layer-2
            #if the activation function is relu
            if (self.activationFunction == 'relu'):
                x = self.layer2(F.relu(x))
            #if the activation function is hardswish
            elif (self.activationFunction   == 'hardswish'):
                x = self.layer2(F.hardswish(x))
            #if the activation function is tanh
            elif (self.activationFunction == 'tanh'):
                x = self.layer2(F.tanh(x))

        elif self.layerNum > 2: #for layer-3

            if self.activationFunction == 'relu':
                x = self.layer3(F.relu(x))

            elif self.activationFunction   == 'hardswish':
                x = self.layer3(F.hardswish(x))

            elif self.activationFunction == 'tanh':
                x = self.layer3(F.tanh(x))

        return x
    


#here is my train function   
def train(model, train_loader, optimizer, val_loader, device, loss_function, num_epochs, test_loader):
    
    for epoch in tqdm(range(num_epochs)):
        #training
        model.train()
        accum_train_loss = 0
        for i, (imgs, labels) in enumerate(train_loader, start=1):
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            loss = loss_function(output, labels)

            # accumlate the loss
            accum_train_loss += loss.item()
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        # Validation
        model.eval()
        accum_val_loss = 0
        with torch.no_grad():
            correct_val = total_val = 0
            for j, (imgs, labels) in enumerate(val_loader, start=1):
                imgs, labels = imgs.to(device), labels.to(device)
                output = model(imgs)
                accum_val_loss += loss_function(output, labels).item()
                
                _, predicted_labels = torch.max(output, 1)
                correct_val += (predicted_labels == labels).sum()
                total_val += labels.size(0)

        # print statistics of the epoch
        print(f' Epoch = {epoch} | Train Loss = {accum_train_loss / i:.4f}\tValidation Loss = {accum_val_loss / j:.4f}')
        print(f'Validation Accuracy = { 100 * correct_val/total_val :.3f}%')
    
    
    model.eval()
    with torch.no_grad():
        correct = total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            
            _, predicted_labels = torch.max(output, 1)
            correct += (predicted_labels == labels).sum()
            total += labels.size(0)

    print(f'Test Accuracy = { 100 * correct/total :.3f}%')
    

if __name__ == '__main__':
    main()