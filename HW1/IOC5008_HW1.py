#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms # Transformations we can perform on our dataset
import torchvision
import os
import pandas as pd
from skimage import io
from torch.utils.data import Dataset, DataLoader # Gives easier dataset managment and creates mini batches
from pathlib import Path
from PIL import Image
import numpy as np


# In[4]:


#Create data and label list
file = open("./train.txt")
data_list = []
label_list = []
count = 0
while True:
    try:
        data = file.readline().split(' ')
        # get imgae list & groundtruth list
        data_list.append(data[0])
        label_list.append(int(data[1]))
    except:
        break
data_list.pop(-1)


# In[5]:


#Seperate the training data into 9:1 (training:validation)
train_list = data_list[:int(len(data_list)*0.9)]
test_list = data_list[int(len(data_list)*0.9):]
train_label_list = label_list[:int(len(label_list)*0.9)]
test_label_list = label_list[int(len(label_list)*0.9):]


# In[6]:


#Create custom dataset
class CarDataset(Dataset):
    def __init__(self, data_list, data_label_list, root_dir, transform=None):
        self.data_list = data_list
        self.data_label_list = data_label_list
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        
        self.data_list[index]
        img_name = self.data_list[index]
        img_path = self.root_dir + img_name
        img = Image.open(img_path)
        img =img.convert("RGB")
        img = self.transform(img)
    
        return img, self.data_label_list[index]


# In[7]:


#Transform out images to the same size
transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([300,300]),
        transforms.ToTensor()])
Training_data = CarDataset (train_list, train_label_list, './training_data/training_data/', transform)
train_loader = DataLoader(dataset=Training_data, batch_size=4, shuffle=True)

Testing_data = CarDataset (test_list, test_label_list, './training_data/training_data/', transform)
test_loader = DataLoader(dataset=Testing_data, batch_size=4, shuffle=True)

device = torch.device('cuda')

# Model
model = torchvision.models.resnet101(pretrained=True)
model.fc = nn.Linear(2048, 196) #Turn the Resnet18 from 512 to 196 #Turn the Resnet101 from 2048 to 196
model.to(device)


# In[8]:


# Hyperparameters
in_channel = 3
num_classes = 196
learning_rate = 1e-4
batch_size = 4
#batch_size = 32
num_epochs = 60


# In[10]:


# Check accuracy on training to see how good our model is
acc = 0.0
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            predictions = scores.argmax(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f} %') 
    
    return float(num_correct)/float(num_samples)


# In[ ]:


# Loss and optimizer
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    losses = []
    val_losses = []
    model.train()
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        
        # gradient descent or adam step
        optimizer.step()
    print("======================")    
    print(f'Cost at epoch {epoch} is {sum(losses)/len(losses)}')
    print("Checking accuracy on Training Set")
        
    accuarcy = check_accuracy(train_loader, model)
    print("----------------------")


        
    for batch_idx, (data, targets) in enumerate(test_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        val_losses.append(loss.item())
        
    print(f'Cost at epoch {epoch} is {sum(val_losses)/len(val_losses)}')       
    print("Checking accuracy on Testing Set")
    accuracy = check_accuracy(test_loader, model)
    #Save the model
    if accuracy > acc:
        torch.save({'state_dict': model.state_dict()}, './cnn2.pkl')
        acc = accuracy

