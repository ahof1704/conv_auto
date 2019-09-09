#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 10:32:22 2019

@author: antonio
Just playing around with some autoenconders
example: https://github.com/yangzhangalmo/pytorch-examples/blob/master/ae_cnn.py
https://gist.github.com/okiriza/fe874412f540a6f7eb0111c4f6649afe

"""
# matplotlib inline

import torch
import numpy as np
import copy
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from   torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import save_image
from torchsummary import summary
#import torch.sigmoid as sig
import os

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')

## Choose on structure
# 'only_convs' -> two conv layers for encoding and two for decoding. No dropout or fully connected
structure_net = 'only_convs'

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 436, 436)
    return x

class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

if structure_net == 'only_convs'
    class ConvAutoencoder(nn.Module):
        def __init__(self, code_size):
            self.code_size = code_size
            super().__init__()
            self.enc_cnn_1 = nn.Conv2d(1, 16, 3, stride=3, padding=1)  # b, 16, 10, 10
            self.enc_cnn_2 = nn.Conv2d(16, 8, 3, stride=2, padding=2)  # b, 8, 3, 3
            # add regularuzation 
            # Add dropout after ReLU
            # add fc layer to 100dim here or another conv
            # check for getting enconding from conv auto enconder (how to check the latent space)
            # 109 * 8 * 8 = 6976
            self.enc_linear_1 = nn.Linear(in_features=2888, out_features=4000)   #Flattened image is fed into linear NN and reduced to half size
            self.enc_linear_2 = nn.Linear(in_features=4000, out_features=100)
            self.enc_linear_3 = nn.Linear(in_features=100, out_features=self.code_size)
            self.dec_linear_1 = nn.Linear(in_features=self.code_size, out_features=4000)
            self.dec_linear_3 = nn.Linear(in_features=4000, out_features=2888)
            self.dec_convT_1 = nn.ConvTranspose2d(8, 16, 3, stride=2, padding=2)  # b, 16, 5, 5
            self.dec_convT_2 = nn.ConvTranspose2d(16, 1, 3, stride=3, padding=1)  # b, 8, 15, 15
            
        def forward(self, images):
            code,indices1,indices2 = self.encoder(images)
            out = self.decoder(code,indices1,indices2)
            return out, code

        def encoder(self, images):
            print(images.shape)
            x = self.enc_cnn_1(images) # [436,436,1], K=3, P=1, S=3 -> W2 =(W−F+2P)/S+1 = (436-3+2*1)/3+1= 146
            print(x.shape)
            x, indices1 = F.max_pool2d(x,kernel_size=2, return_indices=True) #indices for unpooling, #146/2 = 73
            print(x.shape)
            x = F.leaky_relu(x)
            x = self.enc_cnn_2(x) #[73,73,16] -> W2 = (73-3+2*2)/2+1 = 38
            print(x.shape)
            x, indices2 = F.max_pool2d(x,kernel_size=2, return_indices=True) #38/2 = 19 -> [19,19,8]
            code = F.leaky_relu(x)
            # print(x.shape)
            # x = x.view([images.size(0), -1])
            # print(x.shape)
            # x = F.relu(self.enc_linear_1(x))
            # print(x.shape)
            # x = F.dropout(x,p=0.5)
            # print(x.shape)
            # x = F.relu(self.enc_linear_2(x))
            # print(x.shape)
            # x = F.dropout(x,p=0.5)
            # print(x.shape)
            # code = self.enc_linear_3(x)
            print(code.shape)
            return code, indices1, indices2

        def decoder(self, code, indices1, indices2):
            print(code.shape)
            # x = F.relu(self.dec_linear_1(code))
            # print(x.shape)
            # x = F.relu(self.dec_linear_2(x))
            # x = F.relu(self.dec_linear_3(x))
            # print(x.shape)
            # x = self.dec_linear_4(x)
            # x = x.view([code.shape[0], 8, 19, 19])
            # print(x.shape)
            x = F.max_unpool2d(code, indices2, 2)
            x = F.leaky_relu(self.dec_convT_1(x))
            print(x.shape)
            # x = F.relu(self.dec_convT_2(x))
            x = F.max_unpool2d(x, indices1, 2)
    #        print(x.shape)
            out = torch.sigmoid(self.dec_convT_2(x))
            print(out.shape)
    #        decoded = F.tanh(x)
            return out




# convert data to torch.FloatTensor
# transform = transforms.ToTensor()

# load the training and test datasets
# train_data = datasets.MNIST(root='data', train=True,
#                                    download=True, transform=transform)
# test_data = datasets.MNIST(root='data', train=False,
                                  # download=True, transform=transform)

# Create training and test dataloaders

# num_workers = 0
# how many samples per batch to load
# batch_size = 20

phase = 'train'
curr_path	 = os.getcwd()
dataset_dir      = os.path.join(curr_path, "data/All_samples_noise")
batch_size       = 128
validation_split = .1 # -- split training set into train/val sets
n_epochs           = 30
print('training params:\n curr_path: {}\n dataset dir: {}\n batch size: {}\n val split: {}\n epochs: {}'.format(curr_path, dataset_dir, batch_size, validation_split, n_epochs))

# -- transforms to use
normalize = transforms.Normalize(mean=[0.5],
                                     std=[0.5])

trans         = transforms.Compose([
    transforms.Resize(436),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    normalize
])

# -- create dataset
image_dataset = datasets.ImageFolder(dataset_dir, transform=trans)
class_names   = image_dataset.classes
num_classes   = len(class_names)
dataset_size  = len(image_dataset)
print('dataset has {} images'.format(dataset_size))
print('dataset has {} classes:'.format(num_classes))
print(class_names)

# -- split dataset
indices       = list(range(dataset_size))
split         = int(np.floor(validation_split*dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# -- create dataloaders
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

dataloaders   = {
    'train': torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, num_workers=4, sampler=train_sampler),
      'val': torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, num_workers=4, sampler=valid_sampler)
}

# prepare data loaders
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

import matplotlib.pyplot as plt
#matplotlib inline
    
# obtain one batch of training images
dataiter = iter(dataloaders['train'])
images, labels = dataiter.next()
print('Dim of raw sample image: {}\n'.format(images.shape))
images = images.numpy()

# get one image from the batch
img = np.squeeze(images[0])
print('Dim of sample image: {}\n'.format(img.shape))
fig = plt.figure(figsize = (5,5)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
plt.savefig('USV_example_v5.png')

#initialize the NN
#model = ConvAutoencoder()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
code_size = 100 #dimension of the latent space
model = ConvAutoencoder(code_size).to(device)
#print(model)
#summary(model, input_size=(batch_size, 1, 436, 436))

## Training the NN ##
#Specify Loss Function
criterion = nn.MSELoss()

#Specify optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Number of epoch for training
#n_epochs = 10 #Make it stop before overfitting
best_loss = 100.0

if phase == 'train':
    for epoch in range (1, n_epochs+1):
        #Monitor training Loss
        train_loss = 0.0
        
        #Train the model
        #for data in dataloaders['train']:
        for data in dataloaders['train']:
            # _ stands in for labels, here no need t flatten images
            images, _ = data
            images = Variable(images).to(device)
#            print('Dim of input image: {}\n'.format(images.shape))
            #Clear the gradients of all optimized variables
            out, code = model(Variable(images))
            optimizer.zero_grad()
            #Forward pass: compute predicted outputs by passing inputs to the model
#            print('Dim of output image: {}\n'.format(out.shape))
            #Calculate the loss
            loss = criterion(out, images)
            #backward pass: compute graditent of the loss with respect to the model parameters
            loss.backward()
            # Perform single optimization step (parameter update)
            optimizer.step()
            #Update running training loss
            train_loss += loss.item()*images.size(0)
            
        #print average training stats
        train_loss = train_loss/len(dataloaders['train'])
        print('Epoch: {}/{} \tTraining Loss: {:.6f}'.format(epoch,n_epochs,train_loss))

        if train_loss < best_loss:
            best_loss = train_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, 'best_net_test5.pth')

        if epoch % 5 == 0:
            pic = to_img(out.cpu().data)
            save_image(pic, './dc_img/image_{}.png'.format(epoch))

else:
    model.load_state_dict(torch.load('best_net_test3.pth'))

## Checking results ##
#Batch of test images
dataiter = iter(dataloaders['val'])
images, labels = dataiter.next()
images = Variable(images).cuda()

#Get sample outputs

output, _ = model(images)
#Prep images for display
images = images.cpu().numpy()

#Output is resized in a batch of images
output = output.view(batch_size,1,436,436)
#use deatch when it's an output that requires grad
output = output.detach().cpu().numpy()

#Plot the first ten input images and then reconstruct images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))

#Input images on the top row and reconstructed on the bottom
for images, row in zip([images, output], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.savefig('conv_auto_output_test5.png')
