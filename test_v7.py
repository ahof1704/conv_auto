#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 10:32:22 2019

@author: antonio
Just playing around with some autoenconders
example: https://github.com/yangzhangalmo/pytorch-examples/blob/master/ae_cnn.py
https://gist.github.com/okiriza/fe874412f540a6f7eb0111c4f6649afe
https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
"""
# matplotlib inline

import torch
import time
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
import os
from pytorchtools import EarlyStopping

import phate
import scprep
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale
phate_op = phate.PHATE()

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')

## Choose on structure
# 'only_convs_with_maxpool' -> two conv layers for encoding and two for decoding. No dropout or fully connected
# 'convs_withDense_withUnpool' -> with unpool layers 
# 'convs_simple' -> like the one I did first... doesn't have a latent space to plot. Maybe just turn it in an array?
# 'convs_simple_two_dense' -> Adding two fcs in the middle
# 'convs_simple_3layersConv' -> adding one more conv layer to bring it to one channel
# 'convs_simple_two_dense_v2' -> making the fc more narrow (500)
# 'convs_simple_two_dense_v3' -> adding a 3rd conv for encoder and making the fc more narrow (500)
structure_net = 'convs_simple_two_dense_v2'

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 436, 436)
    return x

class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

if structure_net == 'only_convs_with_maxpool':
    class ConvAutoencoder(nn.Module):
        def __init__(self, code_size):
            self.code_size = code_size
            super().__init__()
            self.enc_cnn_1 = nn.Conv2d(1, 16, 3, stride=3, padding=1)  # b, 16, 10, 10
            self.enc_cnn_2 = nn.Conv2d(16, 8, 3, stride=2, padding=2)  # b, 8, 3, 3
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
            # print(images.shape)
            x = self.enc_cnn_1(images) # [436,436,1], K=3, P=1, S=3 -> W2 =(W−F+2P)/S+1 = (436-3+2*1)/3+1= 146
            # print(x.shape)
            x, indices1 = F.max_pool2d(x,kernel_size=2, return_indices=True) #indices for unpooling, #146/2 = 73
            # print(x.shape)
            x = F.leaky_relu(x)
            x = self.enc_cnn_2(x) #[73,73,16] -> W2 = (73-3+2*2)/2+1 = 38
            # print(x.shape)
            x, indices2 = F.max_pool2d(x,kernel_size=2, return_indices=True) #38/2 = 19 -> [19,19,8]
            x = F.leaky_relu(x)
            # print(x.shape)
            code = x.view([images.size(0), -1])
            # print(code.shape)
            # x = F.relu(self.enc_linear_1(x))
            # print(x.shape)
            # x = F.dropout(x,p=0.5)
            # print(x.shape)
            # x = F.relu(self.enc_linear_2(x))
            # print(x.shape)
            # x = F.dropout(x,p=0.5)
            # print(x.shape)
            # code = self.enc_linear_3(x)
            # print(code.shape)
            return code, indices1, indices2

        def decoder(self, code, indices1, indices2):
            # print(code.shape)
            # x = F.relu(self.dec_linear_1(code))
            # print(x.shape)
            # x = F.relu(self.dec_linear_2(x))
            # x = F.relu(self.dec_linear_3(x))
            # print(x.shape)
            # x = self.dec_linear_4(x)
            x = code.view([code.shape[0], 8, 19, 19])
            # print(x.shape)
            x = F.max_unpool2d(x, indices2, 2)
            x = F.leaky_relu(self.dec_convT_1(x))
            # print(x.shape)
            # x = F.relu(self.dec_convT_2(x))
            x = F.max_unpool2d(x, indices1, 2)
    #        print(x.shape)
            out = torch.sigmoid(self.dec_convT_2(x))
            # print(out.shape)
    #        decoded = F.tanh(x)
            return out

elif structure_net == 'convs_withDense_withUnpool':
    class ConvAutoencoder(nn.Module):
        def __init__(self, code_size):
            self.code_size = code_size
            super().__init__()
            self.enc_cnn_1 = nn.Conv2d(1, 16, 3, stride=3, padding=1)  # b, 16, 10, 10
            self.enc_cnn_2 = nn.Conv2d(16, 8, 3, stride=2, padding=2)  # b, 8, 3, 3
            self.enc_linear_1 = nn.Linear(in_features=2888, out_features=4000)   #Flattened image is fed into linear NN and reduced to half size
            self.enc_linear_2 = nn.Linear(in_features=4000, out_features=self.code_size)
            # self.enc_linear_3 = nn.Linear(in_features=100, out_features=self.code_size)
            self.dec_linear_1 = nn.Linear(in_features=self.code_size, out_features=4000)
            self.dec_linear_2 = nn.Linear(in_features=4000, out_features=2888)
            self.dec_convT_1 = nn.ConvTranspose2d(8, 16, 3, stride=2, padding=2)  # b, 16, 5, 5
            self.dec_convT_2 = nn.ConvTranspose2d(16, 1, 3, stride=3, padding=1)  # b, 8, 15, 15
            
        def forward(self, images):
            code,indices1,indices2 = self.encoder(images)
            out = self.decoder(code,indices1,indices2)
            return out, code

        def encoder(self, images):
            #print(images.shape)
            x = self.enc_cnn_1(images) # [436,436,1], K=3, P=1, S=3 -> W2 =(W−F+2P)/S+1 = (436-3+2*1)/3+1= 146
            #print(x.shape)
            x, indices1 = F.max_pool2d(x,kernel_size=2, return_indices=True) #indices for unpooling, #146/2 = 73
            #print(x.shape)
            x = F.leaky_relu(x)
            x = self.enc_cnn_2(x) #[73,73,16] -> W2 = (73-3+2*2)/2+1 = 38
            #print(x.shape)
            x, indices2 = F.max_pool2d(x,kernel_size=2, return_indices=True) #38/2 = 19 -> [19,19,8]
            x = F.leaky_relu(x)
            #print(x.shape)
            x = x.view([images.size(0), -1])
            #print(x.shape)
            x = F.leaky_relu(self.enc_linear_1(x))
            #print(x.shape)
            # x = F.dropout(x,p=0.5)
            # print(x.shape)
            code = self.enc_linear_2(x)
            # print(x.shape)
            # x = F.dropout(x,p=0.5)
            # print(x.shape)
            # code = self.enc_linear_3(x)
            #print(code.shape)
            return code, indices1, indices2

        def decoder(self, code, indices1, indices2):
            #print(code.shape)
            x = F.leaky_relu(self.dec_linear_1(code))
            #print(x.shape)
            x = F.leaky_relu(self.dec_linear_2(x))
            # x = F.relu(self.dec_linear_3(x))
            #print(x.shape)
            # x = self.dec_linear_4(x)
            x = x.view([code.shape[0], 8, 19, 19])
            #print('Dim of x: {}'.format(x.shape))
            #print('Dim indices2: {}'.format(indices2.shape))
            x = F.max_unpool2d(x, indices2, 2)
            x = F.leaky_relu(self.dec_convT_1(x)) #W2 =(W-1)*S-2P+K = (19-1)*2-2*2+3
            #print('Dim of x: {}'.format(x.shape))
            x = F.max_unpool2d(x, indices1, 2)
            #print('Dim indices1: {}'.format(indices1.shape))
            out = F.torch.sigmoid(self.dec_convT_2(x))
                        
            # print(x.shape)
            # out = torch.sigmoid(self.dec_convT_2(x))
            #print(out.shape)
    #        decoded = F.tanh(x)
            return out

elif structure_net == 'convs_simple':
    class ConvAutoencoder(nn.Module):
        def __init__(self, code_size):
            self.code_size = code_size
            super().__init__()
            self.enc_cnn_1 = nn.Conv2d(1, 16, 3, stride=3, padding=1)  # b, 16, 10, 10
            self.enc_cnn_2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)  # b, 8, 3, 3
            # self.enc_linear_1 = nn.Linear(in_features=2888, out_features=4000)   #Flattened image is fed into linear NN and reduced to half size
            # self.enc_linear_2 = nn.Linear(in_features=4000, out_features=self.code_size)
            # self.enc_linear_3 = nn.Linear(in_features=100, out_features=self.code_size)
            # self.dec_linear_1 = nn.Linear(in_features=self.code_size, out_features=4000)
            # self.dec_linear_2 = nn.Linear(in_features=4000, out_features=2888)
            self.dec_convT_1 = nn.ConvTranspose2d(8, 16, 3, stride=2)  # b, 16, 5, 5
            self.dec_convT_2 = nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1)  # b, 8, 15, 15
            self.dec_convT_3 = nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1)  # b, 8, 15, 15
            
        def forward(self, images):
            code = self.encoder(images)
            out = self.decoder(code)
            return out, code

        def encoder(self, images):
            # print('Enconder:')
            # print(images.shape)
            x = F.leaky_relu(self.enc_cnn_1(images)) # [436,436,1], K=3, P=1, S=3 -> W2 =(W−F+2P)/S+1 = (436-3+2*1)/3+1= 146
            # print(x.shape)
            x = F.max_pool2d(x, kernel_size=2, stride=2) #indices for unpooling, #146/2 = 73
            # print(x.shape)
            # x = F.leaky_relu(x)
            x = F.leaky_relu( self.enc_cnn_2(x)) #[73,73,16] -> W2 = (73-3+2*2)/2+1 = 38
            # print(x.shape)
            x = F.max_pool2d(x, kernel_size=2, stride=1) #38/2 = 19 -> [19,19,8]
            # print(x.shape)
            code = x.view([images.size(0), -1]) #export flatten image
            # x = F.leaky_relu(x)
            #print(x.shape)
            # x = x.view([images.size(0), -1])
            #print(x.shape)
            # x = F.leaky_relu(self.enc_linear_1(x))
            #print(x.shape)
            # x = F.dropout(x,p=0.5)
            # print(x.shape)
            # code = self.enc_linear_2(x)
            # print(x.shape)
            # x = F.dropout(x,p=0.5)
            # print(x.shape)
            # code = self.enc_linear_3(x)
            # print(code.shape)
            return code

        def decoder(self, code):
            # print('\nDecoder:')
            # print(code.shape)
            # x = F.leaky_relu(self.dec_linear_1(code))
            #print(x.shape)
            # x = F.leaky_relu(self.dec_linear_2(x))
            # x = F.relu(self.dec_linear_3(x))
            #print(x.shape)
            # x = self.dec_linear_4(x)
            x = code.view([code.shape[0], 8, 36, 36]) #turning into matrix again
            #print('Dim of x: {}'.format(x.shape))
            #print('Dim indices2: {}'.format(indices2.shape))
            # x = F.max_unpool2d(x, indices2, 2)
            x = F.leaky_relu(self.dec_convT_1(x)) #W2 =(W-1)*S-2P+K = (19-1)*2-2*2+3
            # print(x.shape)
            # x = F.max_unpool2d(x, indices1, 2)
            #print('Dim indices1: {}'.format(indices1.shape))
            x = F.leaky_relu(self.dec_convT_2(x))           
            # print(x.shape)
            out = torch.sigmoid(self.dec_convT_3(x))
            # print(out.shape)
    #        decoded = F.tanh(x)
            return out

elif structure_net == 'convs_simple_3layersConv':
    class ConvAutoencoder(nn.Module):
        def __init__(self, code_size):
            self.code_size = code_size
            super().__init__()
            self.enc_cnn_1 = nn.Conv2d(1, 16, 3, stride=3, padding=1)  
            self.enc_cnn_2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)  
            self.enc_cnn_3 = nn.Conv2d(8, 1, 4, stride=2, padding=1)  
            # self.enc_linear_1 = nn.Linear(in_features=2888, out_features=4000)   #Flattened image is fed into linear NN and reduced to half size
            # self.enc_linear_2 = nn.Linear(in_features=4000, out_features=self.code_size)
            # self.enc_linear_3 = nn.Linear(in_features=100, out_features=self.code_size)
            # self.dec_linear_1 = nn.Linear(in_features=self.code_size, out_features=4000)
            # self.dec_linear_2 = nn.Linear(in_features=4000, out_features=2888)
            self.dec_convT_1 = nn.ConvTranspose2d(1, 8, 4, stride=2, padding=1)  
            self.dec_convT_2 = nn.ConvTranspose2d(8, 16, 5, stride=2, padding=1) 
            self.dec_convT_3 = nn.ConvTranspose2d(16, 1, 5, stride=2, padding=1) 
            self.dec_convT_4 = nn.ConvTranspose2d(1, 1, 2, stride=3, padding=2)  
            
        def forward(self, images):
            code = self.encoder(images)
            out = self.decoder(code)
            return out, code

        def encoder(self, images):
            # print(images.shape)
            x = F.leaky_relu(self.enc_cnn_1(images)) # [436,436,1], K=3, P=1, S=3 -> W2 =(W−F+2P)/S+1 = (436-3+2*1)/3+1= 146
            # print(x.shape)
            x = F.max_pool2d(x, kernel_size=2, stride=2) #indices for unpooling, #146/2 = 73
            # print(x.shape)
            # x = F.leaky_relu(x)
            x = F.leaky_relu( self.enc_cnn_2(x)) #[73,73,16] -> W2 = (73-3+2*1)/2+1 = 37
            # print(x.shape)
            x = F.max_pool2d(x, kernel_size=2, stride=1) #38/2 = 19 -> [19,19,8]
            # print(x.shape)
            x = F.leaky_relu( self.enc_cnn_3(x)) #[36,36,8] -> W2 = (36-4+2*1)/2+1 = 38
            # x = F.leaky_relu(x)
            # print(x.shape)
            code = x.view([images.size(0), -1])
            # print(x.shape)
            # x = F.leaky_relu(self.enc_linear_1(x))
            #print(x.shape)
            # x = F.dropout(x,p=0.5)
            # print(x.shape)
            # code = self.enc_linear_2(x)
            # print(x.shape)
            # x = F.dropout(x,p=0.5)
            # print(x.shape)
            # code = self.enc_linear_3(x)
            # print(code.shape)
            return code

        def decoder(self, code):
            # print(code.shape)
            # x = F.leaky_relu(self.dec_linear_1(code))
            #print(x.shape)
            # x = F.leaky_relu(self.dec_linear_2(x))
            # x = F.relu(self.dec_linear_3(x))
            #print(x.shape)
            # x = self.dec_linear_4(x)
            x = code.view([code.shape[0], 1, 18, 18])
            #print('Dim of x: {}'.format(x.shape))
            #print('Dim indices2: {}'.format(indices2.shape))
            # x = F.max_unpool2d(x, indices2, 2)
            x = F.leaky_relu(self.dec_convT_1(x)) #W2 =(W-1)*S-2P+K = (18-1)*2-2*1+4 = 36
            # print(x.shape)
            # x = F.max_unpool2d(x, indices1, 2)
            #print('Dim indices1: {}'.format(indices1.shape))
            x = F.leaky_relu(self.dec_convT_2(x))  # (36-1)*2-2*1+5 = 73
            # print(x.shape)
            x = F.leaky_relu(self.dec_convT_3(x)) # (73-1)*2-2*1+5 = 147
            # print(x.shape)
            out = torch.sigmoid(self.dec_convT_4(x)) # (147-1)*3-2*2+2 = 436
    #        decoded = F.tanh(x)
            return out

elif structure_net == 'convs_simple_two_dense':
    class ConvAutoencoder(nn.Module):
        def __init__(self, code_size):
            self.code_size = code_size
            super().__init__()
            self.enc_cnn_1 = nn.Conv2d(1, 16, 3, stride=3, padding=1)  # b, 16, 10, 10
            self.enc_cnn_2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)  # b, 8, 3, 3
            self.enc_cnn_3 = nn.Conv2d(8, 1, 4, stride=2, padding=1) 
            self.enc_linear_1 = nn.Linear(in_features=10368, out_features=4000)   #Flattened image is fed into linear NN and reduced to half size
            self.enc_linear_2 = nn.Linear(in_features=4000, out_features=self.code_size)
            # self.enc_linear_3 = nn.Linear(in_features=100, out_features=self.code_size)
            self.dec_linear_1 = nn.Linear(in_features=self.code_size, out_features=4000)
            self.dec_linear_2 = nn.Linear(in_features=4000, out_features=10368)
            self.dec_convT_1 = nn.ConvTranspose2d(8, 16, 3, stride=2)  # b, 16, 5, 5
            self.dec_convT_2 = nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1)  # b, 8, 15, 15
            self.dec_convT_3 = nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1)  # b, 8, 15, 15
            
        def forward(self, images):
            code = self.encoder(images)
            out = self.decoder(code)
            return out, code

        def encoder(self, images):
            # print('Enconding:')
            # print(images.shape)
            x = F.leaky_relu(self.enc_cnn_1(images)) # [436,436,1], K=3, P=1, S=3 -> W2 =(W−F+2P)/S+1 = (436-3+2*1)/3+1= 146
            # print(x.shape)
            x = F.max_pool2d(x, kernel_size=2, stride=2) #indices for unpooling, #146/2 = 73
            # print(x.shape)
            x = F.leaky_relu( self.enc_cnn_2(x)) #[73,73,16] -> W2 = (73-3+2*1)/2+1 = 38
            # print(x.shape)
            x = F.max_pool2d(x, kernel_size=2, stride=1) #38/2 = 19 -> [19,19,8]
            # print(x.shape)
            code = F.leaky_relu( self.enc_cnn_3(x)) #[36,36,8] -> W2 = (36-4+2*1)/2+1 = 38
            # x = F.leaky_relu(x)
            # print(x.shape)
            x = x.view([images.size(0), -1])
            # print(x.shape)
            x = F.leaky_relu(self.enc_linear_1(x))
            # print(x.shape)
            # x = F.dropout(x,p=0.5)
            # print(x.shape)
            code = self.enc_linear_2(x)
            # print(x.shape)
            # x = F.dropout(x,p=0.5)
            # print(x.shape)
            # code = self.enc_linear_3(x)
            # print(code.shape)
            return code

        def decoder(self, code):
            # print('Deconding:')
            # print(code.shape)
            x = F.leaky_relu(self.dec_linear_1(code))
            # print(x.shape)
            x = F.leaky_relu(self.dec_linear_2(x))
            # x = F.relu(self.dec_linear_3(x))
            # print(x.shape)
            # x = self.dec_linear_4(x)
            x = x.view([code.shape[0], 8, 36, 36])
            # print(x.shape)
            #print('Dim indices2: {}'.format(indices2.shape))
            # x = F.max_unpool2d(x, indices2, 2)
            x = F.leaky_relu(self.dec_convT_1(x)) #W2 =(W-1)*S-2P+K = (19-1)*2-2*2+3
            # print(x.shape)
            # x = F.max_unpool2d(x, indices1, 2)
            #print('Dim indices1: {}'.format(indices1.shape))
            x = F.leaky_relu(self.dec_convT_2(x))           
            # print(x.shape)
            out = torch.sigmoid(self.dec_convT_3(x))
            # print(out.shape)
    #        decoded = F.tanh(x)
            return out

elif structure_net == 'convs_simple_two_dense_v2':
    class ConvAutoencoder(nn.Module):
        def __init__(self, code_size):
            self.code_size = code_size
            super().__init__()
            self.enc_cnn_1 = nn.Conv2d(1, 16, 3, stride=3, padding=1)  # b, 16, 10, 10
            self.enc_cnn_2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)  # b, 8, 3, 3
            self.enc_linear_1 = nn.Linear(in_features=10368, out_features=500)   #Flattened image is fed into linear NN and reduced to half size
            self.enc_linear_2 = nn.Linear(in_features=500, out_features=self.code_size)
            # self.enc_linear_3 = nn.Linear(in_features=100, out_features=self.code_size)
            self.dec_linear_1 = nn.Linear(in_features=self.code_size, out_features=500)
            self.dec_linear_2 = nn.Linear(in_features=500, out_features=10368)
            self.dec_convT_1 = nn.ConvTranspose2d(8, 16, 3, stride=2)  # b, 16, 5, 5
            self.dec_convT_2 = nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1)  # b, 8, 15, 15
            self.dec_convT_3 = nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1)  # b, 8, 15, 15
            
        def forward(self, images):
            code = self.encoder(images)
            out = self.decoder(code)
            return out, code

        def encoder(self, images):
            # print('Enconding:')
            # print(images.shape)
            x = F.leaky_relu(self.enc_cnn_1(images)) # [436,436,1], K=3, P=1, S=3 -> W2 =(W−F+2P)/S+1 = (436-3+2*1)/3+1= 146
            # print(x.shape)
            x = F.max_pool2d(x, kernel_size=2, stride=2) #indices for unpooling, #146/2 = 73
            # print(x.shape)
            x = F.leaky_relu( self.enc_cnn_2(x)) #[73,73,16] -> W2 = (73-3+2*1)/2+1 = 38
            # print(x.shape)
            x = F.max_pool2d(x, kernel_size=2, stride=1) #38/2 = 19 -> [19,19,8]
            # x = F.leaky_relu(x)
            # print(x.shape)
            x = x.view([images.size(0), -1])
            # print(x.shape)
            x = F.leaky_relu(self.enc_linear_1(x))
            # print(x.shape)
            # x = F.dropout(x,p=0.5)
            # print(x.shape)
            code = self.enc_linear_2(x)
            # print(x.shape)
            # x = F.dropout(x,p=0.5)
            # print(x.shape)
            # code = self.enc_linear_3(x)
            # print(code.shape)
            return code

        def decoder(self, code):
            # print('Deconding:')
            # print(code.shape)
            x = F.leaky_relu(self.dec_linear_1(code))
            # print(x.shape)
            x = F.leaky_relu(self.dec_linear_2(x))
            # x = F.relu(self.dec_linear_3(x))
            # print(x.shape)
            # x = self.dec_linear_4(x)
            x = x.view([code.shape[0], 8, 36, 36])
            # print(x.shape)
            #print('Dim indices2: {}'.format(indices2.shape))
            # x = F.max_unpool2d(x, indices2, 2)
            x = F.leaky_relu(self.dec_convT_1(x)) #W2 =(W-1)*S-2P+K = (19-1)*2-2*2+3
            # print(x.shape)
            # x = F.max_unpool2d(x, indices1, 2)
            #print('Dim indices1: {}'.format(indices1.shape))
            x = F.leaky_relu(self.dec_convT_2(x))           
            # print(x.shape)
            out = torch.sigmoid(self.dec_convT_3(x))
            # print(out.shape)
    #        decoded = F.tanh(x)
            return out

elif structure_net == 'convs_simple_two_dense_v3':
    class ConvAutoencoder(nn.Module):
        def __init__(self, code_size):
            self.code_size = code_size
            super().__init__()
            self.enc_cnn_1 = nn.Conv2d(1, 16, 3, stride=3, padding=1)  # b, 16, 10, 10
            self.enc_cnn_2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)  # b, 8, 3, 3
            self.enc_cnn_3 = nn.Conv2d(8, 1, 4, stride=2, padding=1) 
            self.enc_linear_1 = nn.Linear(in_features=324, out_features=500)   #Flattened image is fed into linear NN and reduced to half size
            self.enc_linear_2 = nn.Linear(in_features=500, out_features=self.code_size)
            # self.enc_linear_3 = nn.Linear(in_features=100, out_features=self.code_size)
            self.dec_linear_1 = nn.Linear(in_features=self.code_size, out_features=500)
            self.dec_linear_2 = nn.Linear(in_features=500, out_features=324)
            self.dec_convT_4 = nn.ConvTranspose2d(1, 8, 3, stride=2)  # b, 16, 5, 5
            self.dec_convT_1 = nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1)  # b, 16, 5, 5
            self.dec_convT_2 = nn.ConvTranspose2d(16, 8, 2, stride=2, padding=0)  # b, 8, 15, 15
            self.dec_convT_3 = nn.ConvTranspose2d(8, 1, 3, stride=3, padding=1)  # b, 8, 15, 15
            
        def forward(self, images):
            code = self.encoder(images)
            out = self.decoder(code)
            return out, code

        def encoder(self, images):
            # print('Enconding:')
            # print(images.shape)
            x = F.leaky_relu(self.enc_cnn_1(images)) # [436,436,1], K=3, P=1, S=3 -> W2 =(W−F+2P)/S+1 = (436-3+2*1)/3+1= 146
            # print(x.shape)
            x = F.max_pool2d(x, kernel_size=2, stride=2) #indices for unpooling, #146/2 = 73
            # print(x.shape)
            x = F.leaky_relu( self.enc_cnn_2(x)) #[73,73,16] -> W2 = (73-3+2*1)/2+1 = 37
            # print(x.shape)
            x = F.max_pool2d(x, kernel_size=2, stride=1) # 36
            # print(x.shape)
            x = F.leaky_relu( self.enc_cnn_3(x)) #[36,36,8] -> W2 = (36-4+2*1)/2+1 = 18
            # x = F.leaky_relu(x)
            # print(x.shape)
            x = x.view([images.size(0), -1])
            # print(x.shape)
            x = F.leaky_relu(self.enc_linear_1(x))
            # print(x.shape)
            # x = F.dropout(x,p=0.5)
            # print(x.shape)
            code = self.enc_linear_2(x)
            # print(x.shape)
            # x = F.dropout(x,p=0.5)
            # print(x.shape)
            # code = self.enc_linear_3(x)
            # print(code.shape)
            return code

        def decoder(self, code):
            # print('Deconding:')
            # print(code.shape)
            x = F.leaky_relu(self.dec_linear_1(code))
            # print(x.shape)
            x = F.leaky_relu(self.dec_linear_2(x))
            # x = F.relu(self.dec_linear_3(x))
            # print(x.shape)
            # x = self.dec_linear_4(x)
            x = x.view([code.shape[0], 1, 18, 18])
            # print(x.shape)
            #print('Dim indices2: {}'.format(indices2.shape))
            # x = F.max_unpool2d(x, indices2, 2)
            x = F.leaky_relu(self.dec_convT_4(x)) #W2 =(W-1)*S-2P+K = (18-1)*2-2*0+3 = 37
            # print(x.shape)
            x = F.leaky_relu(self.dec_convT_1(x)) #W2 =(W-1)*S-2P+K = (37-1)*2-2*1+3 = 73
            # print(x.shape)
            # x = F.max_unpool2d(x, indices1, 2)
            #print('Dim indices1: {}'.format(indices1.shape))
            x = F.leaky_relu(self.dec_convT_2(x))  #W2 =(W-1)*S-2P+K = (73-1)*2-2*0+2 = 146
            # print(x.shape)
            out = torch.sigmoid(self.dec_convT_3(x)) #W2 =(W-1)*S-2P+K = (146-1)*3-2*1+3 = 436
            # print(out.shape)
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

phase = 'val'
curr_path	 = os.getcwd()
dataset_dir      = os.path.join(curr_path, "data/All_samples_noise")
testset_dir      = os.path.join(curr_path,"data/testset")
batch_size       = 128
validation_split = .1 # -- split training set into train/val sets
n_epochs           = 50
code_size = 100 #dimension of the latent space
patience = 10
print('training params:\n phase: {}\n curr_path: {}\n dataset dir: {}\n batch size: {}\n val split: {}\n epochs: {}\n structure: {}\n code_size: {}\n patience: {}\n'.format(phase, curr_path, dataset_dir, batch_size, validation_split, n_epochs, structure_net, code_size, patience))

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

test_dataset = datasets.ImageFolder(testset_dir, transform=trans)
class_names   = test_dataset.classes
num_classes_test   = len(class_names)
dataset_size_test  = len(test_dataset)
print('\nFor test dataset:')
print('dataset has {} images'.format(dataset_size_test))
print('dataset has {} classes:'.format(num_classes_test))
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
    'val': torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, num_workers=4, sampler=valid_sampler),
    'test': torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, num_workers=4, shuffle=False),
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
# fig = plt.figure(figsize = (5,5)) 
# ax = fig.add_subplot(111)
# ax.imshow(img, cmap='gray')
# plt.savefig('USV_example_v5.png')

#initialize the NN
#model = ConvAutoencoder()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = ConvAutoencoder(code_size).to(device)

#print(model)
summary(model, input_size=(1, 436, 436))


## Training the NN ##
#Specify Loss Function
criterion = nn.MSELoss()

#Specify optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Number of epoch for training
#n_epochs = 10 #Make it stop before overfitting
best_loss = 100.0

def train_model(model, batch_size, patience, n_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=False)
    # early_stop=0
    # epoch=1

    for epoch in range (1, n_epochs+1):
        # print('EarlyStopping: {}\n'.format(early_stop))
        #Monitor training Loss
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0

            #Train the model
            #for data in dataloaders['train']:
            for data in dataloaders[phase]:
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

                if phase == 'train':
                    loss = criterion(out, images)
                    #backward pass: compute graditent of the loss with respect to the model parameters
                    loss.backward()
                    # Perform single optimization step (parameter update)
                    optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                if phase == 'train':
                    train_losses.append(loss.item()*images.size(0))
                    epoch_loss = running_loss / (len(dataloaders[phase].dataset)*0.9)
                else:
                    epoch_loss = running_loss / (len(dataloaders[phase].dataset)*0.1)
                    valid_losses.append(loss.item()*images.size(0))
        
            # print training/validation statistics 
            # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(n_epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                 f'train_loss: {train_loss:.5f} ' +
                 f'valid_loss: {valid_loss:.5f}')

        print(print_msg)
        # print('validation_loss: {}\n model: {}\n structure_net: {}'.format(valid_loss,model,structure_net))
        early_stopping(valid_loss, model, structure_net)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        if early_stopping.early_stop:
            print("Early stopping")
            break


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(torch.load('checkpoint' + structure_net + '.pt'))
    return  model, avg_train_losses, avg_valid_losses
        # epoch=epoch+1
# else:
#     model.load_state_dict(torch.load('best_net_test' + structure_net + '.pth'))

## Checking results ##
#Batch of test images
dataiter = iter(dataloaders['val'])
images, labels = dataiter.next()
images = Variable(images).cuda()

if phase == 'train':
    model, train_loss, valid_loss = train_model(model, batch_size, patience, n_epochs)
    best_model_wts = copy.deepcopy(model.state_dict())
    torch.save(best_model_wts, 'best_model' + structure_net + '.pth')
else:
    model.load_state_dict(torch.load('best_model' + structure_net + '.pth'))
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

fig.suptitle('convAE_valset_' + structure_net + '_' +str(n_epochs) + 'epochs', fontsize=16)
plt.savefig('convAE_valset_' + structure_net + '_' +str(n_epochs) + 'epochs.png')

## Now for testset
#Batch of test images
dataiter = iter(dataloaders['test'])
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

fig.suptitle('convAE_testset_' + structure_net + '_' +str(n_epochs) + 'epochs', fontsize=16)
plt.savefig('convAE_testset_' + structure_net + '_' +str(n_epochs) + 'epochs.png')

# visualize the loss as the network trained
if phase == 'train':
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    # plt.ylim(0, 0.5) # consistent scale
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.title('Loss_' + structure_net + '_' + str(n_epochs) + 'epochs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('loss_plot' + structure_net + '_' + str(n_epochs) + 'epochs.png', bbox_inches='tight')

# # NOW A REAL TEST#
# # -- create dataset
# test_dataset = datasets.ImageFolder(testset_dir, transform=trans)
# class_names   = test_dataset.classes
# num_classes   = len(test_dataset)
# dataset_size  = len(test_dataset)
# print('Starting unseen dataset \n dataset has {} images'.format(dataset_size))

# #Load labels
# # curr_path    = os.getcwd()
# # excel_path      =  '/home/antonio/Documents/conv_auto/data/correct_classifier/1303_Agrp-Trpv1_1st/1303_Agrp-Trpv1_1st_GT.xlsx'
# # tabela = pd.read_excel(excel_path)
# # labels_all = tabela['GT']

# dataloaders   = {
#     'train': torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, num_workers=4),
#     'test': torch.utils.data.DataLoader(test_dataset,  batch_size=dataset_size, num_workers=4, shuffle=False),
# }

# # model.load_state_dict(torch.load('best_model' + structure_net + '.pth'))

# dataiter = iter(dataloaders['test'])
# images, labels = dataiter.next()
# labels = labels.numpy()
# print('labels: {}'.format(labels))
# images = Variable(images).cuda()
# output, code = model(images)

# #Prep images for display
# images = images.cpu().numpy()

# # torch.save(code.cpu(), '/home/antonio/Documents/conv_auto/code.pt')
# # np.save('/home/antonio/Documents/conv_auto/code_np' + structure_net, code.cpu().detach().numpy())

# phate_operator = phate.PHATE(n_components=3, k=5, a=20, t=150)
# # data_phate = phate_operator.fit_transform(code.cpu().detach().numpy())
# # print('data_phate shape: {}\n'.format(data_phate.shape))

# # print(labels_all)
# # phate.plot.scatter2d(data_phate, c=labels_all, cmap="Spectral",filename="test_sqrt_Sept11.png", title="Test1", ticks=False, label_prefix="PHATE")
# # phate.plot.rotate_scatter3d(data_phate, c=labels, filename="embedding_" + structure_net + ".gif", title=structure_net)


#Get PHATE plot for training data#
print('\nEmbedding training data:')
image_dataset = datasets.ImageFolder(dataset_dir, transform=trans)
class_names   = image_dataset.classes
num_classes   = len(class_names)
dataset_size  = len(image_dataset)
print('dataset has {} images'.format(dataset_size))
print('dataset has {} classes:'.format(num_classes))
print(class_names)

dataloaders   = {
    'train': torch.utils.data.DataLoader(image_dataset, batch_size=2000, num_workers=4),
}

dataiter = iter(dataloaders['train'])
images, labels = dataiter.next()
labels = labels.numpy()
# print('labels: {}'.format(labels))
images = Variable(images).cuda()
output, code = model(images)

#Prep images for display
# images = images.cpu().numpy()
phate_operator = phate.PHATE(n_components=3, k=5, a=20, t=150)
training_phate = phate_operator.fit_transform(code.cpu().detach().numpy())
print('training_phate shape: {}\n'.format(training_phate.shape))

# print(labels_all)
# phate.plot.scatter2d(data_phate, c=labels_all, cmap="Spectral",filename="test_sqrt_Sept11.png", title="Test1", ticks=False, label_prefix="PHATE")
phate.plot.rotate_scatter3d(training_phate, filename="embedding_trainingData" + structure_net + ".gif", title=structure_net)