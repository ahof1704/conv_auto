#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 10:32:22 2019

@author: antonio
Just playing around with some autoenconders

"""
# matplotlib inline

import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
#import torch.sigmoid as sig


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        #enconder layers
        #Conv layer (depth from 1-> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(1,16,3, padding=1)
        # Conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16,4,3, padding=1)
        # Pooling layer to reduce dimension 
        self.pool = nn.MaxPool2d(2,2)
        
        ## Decoder Layer ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4,16,2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16,1,2, stride=2)
    
    def forward(self, x):
        ## Enconde ##
        # Add hidden layers with relu activation function and maxpooling
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # Add a second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x) #compressed representation

        ##Decode ##
        #Add transpose conv layers, with relu actication function
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = torch.sigmoid(self.t_conv2(x))

        return x

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# load the training and test datasets
train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                                  download=True, transform=transform)

# Create training and test dataloaders

num_workers = 0
# how many samples per batch to load
batch_size = 20

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

import matplotlib.pyplot as plt
#matplotlib inline
    
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

# get one image from the batch
img = np.squeeze(images[0])

fig = plt.figure(figsize = (5,5)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')

#initialize the NN
model = ConvAutoencoder()
print(model)

## Training the NN ##
#Specify Loss Function
criterion = nn.MSELoss()

#Specify optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Number of epoch for training
n_epochs = 30

for epoch in range (1, n_epochs+1):
    #Monitor training Loss
    train_loss = 0.0
    
    #Train the model
    for data in train_loader:
        # _ stands in for labels, here no need t flatten images
        images, _ = data
        #Clear the gradients of all optimized variables
        optimizer.zero_grad()
        #Forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(images)
        #Calculate the loss
        loss = criterion(outputs, images)
        #backward pass: compute graditent of the loss with respect to the model parameters
        loss.backward()
        # Perform single optimization step (parameter update)
        optimizer.step()
        #Update running training loss
        train_loss += loss.item()*images.size(0)
        
    #print average training stats
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch,train_loss))
    
## Checking results ##
#Batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()

#Get sample outputs
output = model(images)
#Prep images for display
images = images.numpy()

#Output is resized in a batch of images
output = output.view(batch_size,1,28,28)
#use deatch when it's an output that requires grad
output = output.detach().numpy()

#Plot the first ten input images and then reconstruct images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))

#Input images on the top row and reconstructed on the bottom
for images, row in zip([images, output], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)