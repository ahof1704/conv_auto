#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 10:32:22 2019

@author: antonio
Just playing around with some autoenconders
example: https://github.com/yangzhangalmo/pytorch-examples/blob/master/ae_cnn.py
https://gist.github.com/okiriza/fe874412f540a6f7eb0111c4f6649afe
https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
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
from   tqdm              import tqdm
from   torch.optim       import lr_scheduler as lrs

import phate
import scprep
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
phate_op = phate.PHATE()

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

import nvidia_smi
import gc
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
# 'convs_simple_two_dense_v2_batchNorm' -> like convs_simple_two_dense_v2 but with batch normalization
# 'convs_simple_two_dense_batchNorm' -> normalization also in the middle layers
# 'convs_two_dense_batchNorm_dropOut0.5' -> adding dropout
structure_net = 'convs_simple_two_dense_batchNorm'

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 436, 436)
    return x

def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert(isinstance(size, torch.Size))
    return " × ".join(map(str, size))

def dump_tensors(gpu_only=True):
    """Prints a list of the Tensors being tracked by the garbage collector."""
    import gc
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print("%s:%s%s %s" % (type(obj).__name__, 
                                          " GPU" if obj.is_cuda else "",
                                          " pinned" if obj.is_pinned else "",
                                          pretty_size(obj.size())))
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    print("%s → %s:%s%s%s%s %s" % (type(obj).__name__, 
                                                   type(obj.data).__name__, 
                                                   " GPU" if obj.is_cuda else "",
                                                   " pinned" if obj.data.is_pinned else "",
                                                   " grad" if obj.requires_grad else "", 
                                                   " volatile" if obj.volatile else "",
                                                   pretty_size(obj.data.size())))
                    total_size += obj.data.numel()
        except Exception as e:
            pass        
    print("Total size:", total_size)

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

# elif structure_net == 'convs_withDense_withUnpool':
#     class ConvAutoencoder(nn.Module):
#         def __init__(self, code_size):
#             self.code_size = code_size
#             super().__init__()
#             self.enc_cnn_1 = nn.Conv2d(1, 16, 3, stride=3, padding=1)  # b, 16, 10, 10
#             self.enc_cnn_2 = nn.Conv2d(16, 8, 3, stride=2, padding=2)  # b, 8, 3, 3
#             self.enc_linear_1 = nn.Linear(in_features=2888, out_features=4000)   #Flattened image is fed into linear NN and reduced to half size
#             self.enc_linear_2 = nn.Linear(in_features=4000, out_features=self.code_size)
#             # self.enc_linear_3 = nn.Linear(in_features=100, out_features=self.code_size)
#             self.dec_linear_1 = nn.Linear(in_features=self.code_size, out_features=4000)
#             self.dec_linear_2 = nn.Linear(in_features=4000, out_features=2888)
#             self.dec_convT_1 = nn.ConvTranspose2d(8, 16, 3, stride=2, padding=2)  # b, 16, 5, 5
#             self.dec_convT_2 = nn.ConvTranspose2d(16, 1, 3, stride=3, padding=1)  # b, 8, 15, 15
            
#         def forward(self, images):
#             code,indices1,indices2 = self.encoder(images)
#             out = self.decoder(code,indices1,indices2)
#             return out, code

#         def encoder(self, images):
#             #print(images.shape)
#             x = self.enc_cnn_1(images) # [436,436,1], K=3, P=1, S=3 -> W2 =(W−F+2P)/S+1 = (436-3+2*1)/3+1= 146
#             #print(x.shape)
#             x, indices1 = F.max_pool2d(x,kernel_size=2, return_indices=True) #indices for unpooling, #146/2 = 73
#             #print(x.shape)
#             x = F.leaky_relu(x)
#             x = self.enc_cnn_2(x) #[73,73,16] -> W2 = (73-3+2*2)/2+1 = 38
#             #print(x.shape)
#             x, indices2 = F.max_pool2d(x,kernel_size=2, return_indices=True) #38/2 = 19 -> [19,19,8]
#             x = F.leaky_relu(x)
#             #print(x.shape)
#             x = x.view([images.size(0), -1])
#             #print(x.shape)
#             x = F.leaky_relu(self.enc_linear_1(x))
#             #print(x.shape)
#             # x = F.dropout(x,p=0.5)
#             # print(x.shape)
#             code = self.enc_linear_2(x)
#             # print(x.shape)
#             # x = F.dropout(x,p=0.5)
#             # print(x.shape)
#             # code = self.enc_linear_3(x)
#             #print(code.shape)
#             return code, indices1, indices2

#         def decoder(self, code, indices1, indices2):
#             #print(code.shape)
#             x = F.leaky_relu(self.dec_linear_1(code))
#             #print(x.shape)
#             x = F.leaky_relu(self.dec_linear_2(x))
#             # x = F.relu(self.dec_linear_3(x))
#             #print(x.shape)
#             # x = self.dec_linear_4(x)
#             x = x.view([code.shape[0], 8, 19, 19])
#             #print('Dim of x: {}'.format(x.shape))
#             #print('Dim indices2: {}'.format(indices2.shape))
#             x = F.max_unpool2d(x, indices2, 2)
#             x = F.leaky_relu(self.dec_convT_1(x)) #W2 =(W-1)*S-2P+K = (19-1)*2-2*2+3
#             #print('Dim of x: {}'.format(x.shape))
#             x = F.max_unpool2d(x, indices1, 2)
#             #print('Dim indices1: {}'.format(indices1.shape))
#             out = F.torch.sigmoid(self.dec_convT_2(x))
                        
#             # print(x.shape)
#             # out = torch.sigmoid(self.dec_convT_2(x))
#             #print(out.shape)
#     #        decoded = F.tanh(x)
#             return out

# elif structure_net == 'convs_simple':
#     class ConvAutoencoder(nn.Module):
#         def __init__(self, code_size):
#             self.code_size = code_size
#             super().__init__()
#             self.enc_cnn_1 = nn.Conv2d(1, 16, 3, stride=3, padding=1)  # b, 16, 10, 10
#             self.enc_cnn_2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)  # b, 8, 3, 3
#             # self.enc_linear_1 = nn.Linear(in_features=2888, out_features=4000)   #Flattened image is fed into linear NN and reduced to half size
#             # self.enc_linear_2 = nn.Linear(in_features=4000, out_features=self.code_size)
#             # self.enc_linear_3 = nn.Linear(in_features=100, out_features=self.code_size)
#             # self.dec_linear_1 = nn.Linear(in_features=self.code_size, out_features=4000)
#             # self.dec_linear_2 = nn.Linear(in_features=4000, out_features=2888)
#             self.dec_convT_1 = nn.ConvTranspose2d(8, 16, 3, stride=2)  # b, 16, 5, 5
#             self.dec_convT_2 = nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1)  # b, 8, 15, 15
#             self.dec_convT_3 = nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1)  # b, 8, 15, 15
            
#         def forward(self, images):
#             code = self.encoder(images)
#             out = self.decoder(code)
#             return out, code

#         def encoder(self, images):
#             # print('Enconder:')
#             # print(images.shape)
#             x = F.leaky_relu(self.enc_cnn_1(images)) # [436,436,1], K=3, P=1, S=3 -> W2 =(W−F+2P)/S+1 = (436-3+2*1)/3+1= 146
#             # print(x.shape)
#             x = F.max_pool2d(x, kernel_size=2, stride=2) #indices for unpooling, #146/2 = 73
#             # print(x.shape)
#             # x = F.leaky_relu(x)
#             x = F.leaky_relu( self.enc_cnn_2(x)) #[73,73,16] -> W2 = (73-3+2*2)/2+1 = 38
#             # print(x.shape)
#             x = F.max_pool2d(x, kernel_size=2, stride=1) #38/2 = 19 -> [19,19,8]
#             # print(x.shape)
#             code = x.view([images.size(0), -1]) #export flatten image
#             # x = F.leaky_relu(x)
#             #print(x.shape)
#             # x = x.view([images.size(0), -1])
#             #print(x.shape)
#             # x = F.leaky_relu(self.enc_linear_1(x))
#             #print(x.shape)
#             # x = F.dropout(x,p=0.5)
#             # print(x.shape)
#             # code = self.enc_linear_2(x)
#             # print(x.shape)
#             # x = F.dropout(x,p=0.5)
#             # print(x.shape)
#             # code = self.enc_linear_3(x)
#             # print(code.shape)
#             return code

#         def decoder(self, code):
#             # print('\nDecoder:')
#             # print(code.shape)
#             # x = F.leaky_relu(self.dec_linear_1(code))
#             #print(x.shape)
#             # x = F.leaky_relu(self.dec_linear_2(x))
#             # x = F.relu(self.dec_linear_3(x))
#             #print(x.shape)
#             # x = self.dec_linear_4(x)
#             x = code.view([code.shape[0], 8, 36, 36]) #turning into matrix again
#             #print('Dim of x: {}'.format(x.shape))
#             #print('Dim indices2: {}'.format(indices2.shape))
#             # x = F.max_unpool2d(x, indices2, 2)
#             x = F.leaky_relu(self.dec_convT_1(x)) #W2 =(W-1)*S-2P+K = (19-1)*2-2*2+3
#             # print(x.shape)
#             # x = F.max_unpool2d(x, indices1, 2)
#             #print('Dim indices1: {}'.format(indices1.shape))
#             x = F.leaky_relu(self.dec_convT_2(x))           
#             # print(x.shape)
#             out = torch.sigmoid(self.dec_convT_3(x))
#             # print(out.shape)
#     #        decoded = F.tanh(x)
#             return out

# elif structure_net == 'convs_simple_3layersConv':
#     class ConvAutoencoder(nn.Module):
#         def __init__(self, code_size):
#             self.code_size = code_size
#             super().__init__()
#             self.enc_cnn_1 = nn.Conv2d(1, 16, 3, stride=3, padding=1)  
#             self.enc_cnn_2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)  
#             self.enc_cnn_3 = nn.Conv2d(8, 1, 4, stride=2, padding=1)  
#             # self.enc_linear_1 = nn.Linear(in_features=2888, out_features=4000)   #Flattened image is fed into linear NN and reduced to half size
#             # self.enc_linear_2 = nn.Linear(in_features=4000, out_features=self.code_size)
#             # self.enc_linear_3 = nn.Linear(in_features=100, out_features=self.code_size)
#             # self.dec_linear_1 = nn.Linear(in_features=self.code_size, out_features=4000)
#             # self.dec_linear_2 = nn.Linear(in_features=4000, out_features=2888)
#             self.dec_convT_1 = nn.ConvTranspose2d(1, 8, 4, stride=2, padding=1)  
#             self.dec_convT_2 = nn.ConvTranspose2d(8, 16, 5, stride=2, padding=1) 
#             self.dec_convT_3 = nn.ConvTranspose2d(16, 1, 5, stride=2, padding=1) 
#             self.dec_convT_4 = nn.ConvTranspose2d(1, 1, 2, stride=3, padding=2)  
            
#         def forward(self, images):
#             code = self.encoder(images)
#             out = self.decoder(code)
#             return out, code

#         def encoder(self, images):
#             # print(images.shape)
#             x = F.leaky_relu(self.enc_cnn_1(images)) # [436,436,1], K=3, P=1, S=3 -> W2 =(W−F+2P)/S+1 = (436-3+2*1)/3+1= 146
#             # print(x.shape)
#             x = F.max_pool2d(x, kernel_size=2, stride=2) #indices for unpooling, #146/2 = 73
#             # print(x.shape)
#             # x = F.leaky_relu(x)
#             x = F.leaky_relu( self.enc_cnn_2(x)) #[73,73,16] -> W2 = (73-3+2*1)/2+1 = 37
#             # print(x.shape)
#             x = F.max_pool2d(x, kernel_size=2, stride=1) #38/2 = 19 -> [19,19,8]
#             # print(x.shape)
#             x = F.leaky_relu( self.enc_cnn_3(x)) #[36,36,8] -> W2 = (36-4+2*1)/2+1 = 38
#             # x = F.leaky_relu(x)
#             # print(x.shape)
#             code = x.view([images.size(0), -1])
#             # print(x.shape)
#             # x = F.leaky_relu(self.enc_linear_1(x))
#             #print(x.shape)
#             # x = F.dropout(x,p=0.5)
#             # print(x.shape)
#             # code = self.enc_linear_2(x)
#             # print(x.shape)
#             # x = F.dropout(x,p=0.5)
#             # print(x.shape)
#             # code = self.enc_linear_3(x)
#             # print(code.shape)
#             return code

#         def decoder(self, code):
#             # print(code.shape)
#             # x = F.leaky_relu(self.dec_linear_1(code))
#             #print(x.shape)
#             # x = F.leaky_relu(self.dec_linear_2(x))
#             # x = F.relu(self.dec_linear_3(x))
#             #print(x.shape)
#             # x = self.dec_linear_4(x)
#             x = code.view([code.shape[0], 1, 18, 18])
#             #print('Dim of x: {}'.format(x.shape))
#             #print('Dim indices2: {}'.format(indices2.shape))
#             # x = F.max_unpool2d(x, indices2, 2)
#             x = F.leaky_relu(self.dec_convT_1(x)) #W2 =(W-1)*S-2P+K = (18-1)*2-2*1+4 = 36
#             # print(x.shape)
#             # x = F.max_unpool2d(x, indices1, 2)
#             #print('Dim indices1: {}'.format(indices1.shape))
#             x = F.leaky_relu(self.dec_convT_2(x))  # (36-1)*2-2*1+5 = 73
#             # print(x.shape)
#             x = F.leaky_relu(self.dec_convT_3(x)) # (73-1)*2-2*1+5 = 147
#             # print(x.shape)
#             out = torch.sigmoid(self.dec_convT_4(x)) # (147-1)*3-2*2+2 = 436
#     #        decoded = F.tanh(x)
#             return out

# elif structure_net == 'convs_simple_two_dense':
#     class ConvAutoencoder(nn.Module):
#         def __init__(self, code_size):
#             self.code_size = code_size
#             super().__init__()
#             self.enc_cnn_1 = nn.Conv2d(1, 16, 3, stride=3, padding=1)  # b, 16, 10, 10
#             self.enc_cnn_2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)  # b, 8, 3, 3
#             self.enc_cnn_3 = nn.Conv2d(8, 1, 4, stride=2, padding=1) 
#             self.enc_linear_1 = nn.Linear(in_features=10368, out_features=4000)   #Flattened image is fed into linear NN and reduced to half size
#             self.enc_linear_2 = nn.Linear(in_features=4000, out_features=self.code_size)
#             # self.enc_linear_3 = nn.Linear(in_features=100, out_features=self.code_size)
#             self.dec_linear_1 = nn.Linear(in_features=self.code_size, out_features=4000)
#             self.dec_linear_2 = nn.Linear(in_features=4000, out_features=10368)
#             self.dec_convT_1 = nn.ConvTranspose2d(8, 16, 3, stride=2)  # b, 16, 5, 5
#             self.dec_convT_2 = nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1)  # b, 8, 15, 15
#             self.dec_convT_3 = nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1)  # b, 8, 15, 15
            
#         def forward(self, images):
#             code = self.encoder(images)
#             out = self.decoder(code)
#             return out, code

#         def encoder(self, images):
#             # print('Enconding:')
#             # print(images.shape)
#             x = F.leaky_relu(self.enc_cnn_1(images)) # [436,436,1], K=3, P=1, S=3 -> W2 =(W−F+2P)/S+1 = (436-3+2*1)/3+1= 146
#             # print(x.shape)
#             x = F.max_pool2d(x, kernel_size=2, stride=2) #indices for unpooling, #146/2 = 73
#             # print(x.shape)
#             x = F.leaky_relu( self.enc_cnn_2(x)) #[73,73,16] -> W2 = (73-3+2*1)/2+1 = 38
#             # print(x.shape)
#             x = F.max_pool2d(x, kernel_size=2, stride=1) #38/2 = 19 -> [19,19,8]
#             # print(x.shape)
#             code = F.leaky_relu( self.enc_cnn_3(x)) #[36,36,8] -> W2 = (36-4+2*1)/2+1 = 38
#             # x = F.leaky_relu(x)
#             # print(x.shape)
#             x = x.view([images.size(0), -1])
#             # print(x.shape)
#             x = F.leaky_relu(self.enc_linear_1(x))
#             # print(x.shape)
#             # x = F.dropout(x,p=0.5)
#             # print(x.shape)
#             code = self.enc_linear_2(x)
#             # print(x.shape)
#             # x = F.dropout(x,p=0.5)
#             # print(x.shape)
#             # code = self.enc_linear_3(x)
#             # print(code.shape)
#             return code

#         def decoder(self, code):
#             # print('Deconding:')
#             # print(code.shape)
#             x = F.leaky_relu(self.dec_linear_1(code))
#             # print(x.shape)
#             x = F.leaky_relu(self.dec_linear_2(x))
#             # x = F.relu(self.dec_linear_3(x))
#             # print(x.shape)
#             # x = self.dec_linear_4(x)
#             x = x.view([code.shape[0], 8, 36, 36])
#             # print(x.shape)
#             #print('Dim indices2: {}'.format(indices2.shape))
#             # x = F.max_unpool2d(x, indices2, 2)
#             x = F.leaky_relu(self.dec_convT_1(x)) #W2 =(W-1)*S-2P+K = (19-1)*2-2*2+3
#             # print(x.shape)
#             # x = F.max_unpool2d(x, indices1, 2)
#             #print('Dim indices1: {}'.format(indices1.shape))
#             x = F.leaky_relu(self.dec_convT_2(x))           
#             # print(x.shape)
#             out = torch.sigmoid(self.dec_convT_3(x))
#             # print(out.shape)
#     #        decoded = F.tanh(x)
#             return out

# elif structure_net == 'convs_simple_two_dense_v2':
#     class ConvAutoencoder(nn.Module):
#         def __init__(self, code_size):
#             self.code_size = code_size
#             super().__init__()
#             self.enc_cnn_1 = nn.Conv2d(1, 16, 3, stride=3, padding=1)  # b, 16, 10, 10
#             self.enc_cnn_2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)  # b, 8, 3, 3
#             self.enc_linear_1 = nn.Linear(in_features=10368, out_features=500)   #Flattened image is fed into linear NN and reduced to half size
#             self.enc_linear_2 = nn.Linear(in_features=500, out_features=self.code_size)
#             # self.enc_linear_3 = nn.Linear(in_features=100, out_features=self.code_size)
#             self.dec_linear_1 = nn.Linear(in_features=self.code_size, out_features=500)
#             self.dec_linear_2 = nn.Linear(in_features=500, out_features=10368)
#             self.dec_convT_1 = nn.ConvTranspose2d(8, 16, 3, stride=2)  # b, 16, 5, 5
#             self.dec_convT_2 = nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1)  # b, 8, 15, 15
#             self.dec_convT_3 = nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1)  # b, 8, 15, 15
            
#         def forward(self, images):
#             code = self.encoder(images)
#             out = self.decoder(code)
#             return out, code

#         def encoder(self, images):
#             # print('Enconding:')
#             # print(images.shape)
#             x = F.leaky_relu(self.enc_cnn_1(images)) # [436,436,1], K=3, P=1, S=3 -> W2 =(W−F+2P)/S+1 = (436-3+2*1)/3+1= 146
#             # print(x.shape)
#             x = F.max_pool2d(x, kernel_size=2, stride=2) #indices for unpooling, #146/2 = 73
#             # print(x.shape)
#             x = F.leaky_relu( self.enc_cnn_2(x)) #[73,73,16] -> W2 = (73-3+2*1)/2+1 = 38
#             # print(x.shape)
#             x = F.max_pool2d(x, kernel_size=2, stride=1) #38/2 = 19 -> [19,19,8]
#             # x = F.leaky_relu(x)
#             # print(x.shape)
#             x = x.view([images.size(0), -1])
#             # print(x.shape)
#             x = F.leaky_relu(self.enc_linear_1(x))
#             # print(x.shape)
#             # x = F.dropout(x,p=0.5)
#             # print(x.shape)
#             code = self.enc_linear_2(x)
#             # print(x.shape)
#             # x = F.dropout(x,p=0.5)
#             # print(x.shape)
#             # code = self.enc_linear_3(x)
#             # print(code.shape)
#             return code

#         def decoder(self, code):
#             # print('Deconding:')
#             # print(code.shape)
#             x = F.leaky_relu(self.dec_linear_1(code))
#             # print(x.shape)
#             x = F.leaky_relu(self.dec_linear_2(x))
#             # x = F.relu(self.dec_linear_3(x))
#             # print(x.shape)
#             # x = self.dec_linear_4(x)
#             x = x.view([code.shape[0], 8, 36, 36])
#             # print(x.shape)
#             #print('Dim indices2: {}'.format(indices2.shape))
#             # x = F.max_unpool2d(x, indices2, 2)
#             x = F.leaky_relu(self.dec_convT_1(x)) #W2 =(W-1)*S-2P+K = (19-1)*2-2*2+3
#             # print(x.shape)
#             # x = F.max_unpool2d(x, indices1, 2)
#             #print('Dim indices1: {}'.format(indices1.shape))
#             x = F.leaky_relu(self.dec_convT_2(x))           
#             # print(x.shape)
#             out = torch.sigmoid(self.dec_convT_3(x))
#             # print(out.shape)
#     #        decoded = F.tanh(x)
#             return out

# elif structure_net == 'convs_simple_two_dense_v3':
#     class ConvAutoencoder(nn.Module):
#         def __init__(self, code_size):
#             self.code_size = code_size
#             super().__init__()
#             self.enc_cnn_1 = nn.Conv2d(1, 16, 3, stride=3, padding=1)  # b, 16, 10, 10
#             self.enc_cnn_2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)  # b, 8, 3, 3
#             self.enc_cnn_3 = nn.Conv2d(8, 1, 4, stride=2, padding=1) 
#             self.enc_linear_1 = nn.Linear(in_features=324, out_features=500)   #Flattened image is fed into linear NN and reduced to half size
#             self.enc_linear_2 = nn.Linear(in_features=500, out_features=self.code_size)
#             # self.enc_linear_3 = nn.Linear(in_features=100, out_features=self.code_size)
#             self.dec_linear_1 = nn.Linear(in_features=self.code_size, out_features=500)
#             self.dec_linear_2 = nn.Linear(in_features=500, out_features=324)
#             self.dec_convT_4 = nn.ConvTranspose2d(1, 8, 3, stride=2)  # b, 16, 5, 5
#             self.dec_convT_1 = nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1)  # b, 16, 5, 5
#             self.dec_convT_2 = nn.ConvTranspose2d(16, 8, 2, stride=2, padding=0)  # b, 8, 15, 15
#             self.dec_convT_3 = nn.ConvTranspose2d(8, 1, 3, stride=3, padding=1)  # b, 8, 15, 15
            
#         def forward(self, images):
#             code = self.encoder(images)
#             out = self.decoder(code)
#             return out, code

#         def encoder(self, images):
#             # print('Enconding:')
#             # print(images.shape)
#             x = F.leaky_relu(self.enc_cnn_1(images)) # [436,436,1], K=3, P=1, S=3 -> W2 =(W−F+2P)/S+1 = (436-3+2*1)/3+1= 146
#             # print(x.shape)
#             x = F.max_pool2d(x, kernel_size=2, stride=2) #indices for unpooling, #146/2 = 73
#             # print(x.shape)
#             x = F.leaky_relu( self.enc_cnn_2(x)) #[73,73,16] -> W2 = (73-3+2*1)/2+1 = 37
#             # print(x.shape)
#             x = F.max_pool2d(x, kernel_size=2, stride=1) # 36
#             # print(x.shape)
#             x = F.leaky_relu( self.enc_cnn_3(x)) #[36,36,8] -> W2 = (36-4+2*1)/2+1 = 18
#             # x = F.leaky_relu(x)
#             # print(x.shape)
#             x = x.view([images.size(0), -1])
#             # print(x.shape)
#             x = F.leaky_relu(self.enc_linear_1(x))
#             # print(x.shape)
#             # x = F.dropout(x,p=0.5)
#             # print(x.shape)
#             code = self.enc_linear_2(x)
#             # print(x.shape)
#             # x = F.dropout(x,p=0.5)
#             # print(x.shape)
#             # code = self.enc_linear_3(x)
#             # print(code.shape)
#             return code

#         def decoder(self, code):
#             # print('Deconding:')
#             # print(code.shape)
#             x = F.leaky_relu(self.dec_linear_1(code))
#             # print(x.shape)
#             x = F.leaky_relu(self.dec_linear_2(x))
#             # x = F.relu(self.dec_linear_3(x))
#             # print(x.shape)
#             # x = self.dec_linear_4(x)
#             x = x.view([code.shape[0], 1, 18, 18])
#             # print(x.shape)
#             #print('Dim indices2: {}'.format(indices2.shape))
#             # x = F.max_unpool2d(x, indices2, 2)
#             x = F.leaky_relu(self.dec_convT_4(x)) #W2 =(W-1)*S-2P+K = (18-1)*2-2*0+3 = 37
#             # print(x.shape)
#             x = F.leaky_relu(self.dec_convT_1(x)) #W2 =(W-1)*S-2P+K = (37-1)*2-2*1+3 = 73
#             # print(x.shape)
#             # x = F.max_unpool2d(x, indices1, 2)
#             #print('Dim indices1: {}'.format(indices1.shape))
#             x = F.leaky_relu(self.dec_convT_2(x))  #W2 =(W-1)*S-2P+K = (73-1)*2-2*0+2 = 146
#             # print(x.shape)
#             out = torch.sigmoid(self.dec_convT_3(x)) #W2 =(W-1)*S-2P+K = (146-1)*3-2*1+3 = 436
#             # print(out.shape)
#     #        decoded = F.tanh(x)
#             return out

# elif structure_net == 'convs_simple_two_dense_v2_batchNorm':
#     class ConvAutoencoder(nn.Module):
#         def __init__(self, code_size):
#             self.code_size = code_size
#             super().__init__()
#             self.enc_cnn_1 = nn.Conv2d(1, 16, 3, stride=3, padding=1)  # b, 16, 10, 10
#             self.batch_enc_1 =  nn.BatchNorm2d(16)
#             self.enc_cnn_2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)  # b, 8, 3, 3
#             self.batch_enc_2 = nn.BatchNorm2d(8)
#             self.enc_linear_1 = nn.Linear(in_features=10368, out_features=500)   #Flattened image is fed into linear NN and reduced to half size
#             self.enc_linear_2 = nn.Linear(in_features=500, out_features=self.code_size)
#             # self.enc_linear_3 = nn.Linear(in_features=100, out_features=self.code_size)
#             self.dec_linear_1 = nn.Linear(in_features=self.code_size, out_features=500)
#             self.dec_linear_2 = nn.Linear(in_features=500, out_features=10368)
#             self.dec_convT_1 = nn.ConvTranspose2d(8, 16, 3, stride=2)  # b, 16, 5, 5
#             self.batch_dec_1 =  nn.BatchNorm2d(16)
#             self.dec_convT_2 = nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1)  # b, 8, 15, 15
#             self.batch_dec_2 =  nn.BatchNorm2d(8)
#             self.dec_convT_3 = nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1)  # b, 8, 15, 15
#             # self.batch_dec_3 =  nn.BatchNorm2d(1) #not really doing anything here I think
            
#         def forward(self, images):
#             code = self.encoder(images)
#             out = self.decoder(code)
#             return out, code

#         def encoder(self, images):
#             # print('Enconding:')
#             # print(images.shape)
#             x = F.leaky_relu(self.enc_cnn_1(images)) # [436,436,1], K=3, P=1, S=3 -> W2 =(W−F+2P)/S+1 = (436-3+2*1)/3+1= 146
#             # print(x.shape)
#             x = self.batch_enc_1(x)
#             x = F.max_pool2d(x, kernel_size=2, stride=2) #indices for unpooling, #146/2 = 73
#             # print(x.shape)
#             x = F.leaky_relu( self.enc_cnn_2(x)) #[73,73,16] -> W2 = (73-3+2*1)/2+1 = 38
#             # print(x.shape)
#             x = self.batch_enc_2(x)
#             x = F.max_pool2d(x, kernel_size=2, stride=1) #38/2 = 19 -> [19,19,8]
#             # x = F.leaky_relu(x)
#             # print(x.shape)
#             x = x.view([images.size(0), -1])
#             # print(x.shape)
#             x = F.leaky_relu(self.enc_linear_1(x))
#             # print(x.shape)
#             # x = F.dropout(x,p=0.5)
#             # print(x.shape)
#             code = self.enc_linear_2(x)
#             # print(x.shape)
#             # x = F.dropout(x,p=0.5)
#             # print(x.shape)
#             # code = self.enc_linear_3(x)
#             # print(code.shape)
#             return code

#         def decoder(self, code):
#             # print('Deconding:')
#             # print(code.shape)
#             x = F.leaky_relu(self.dec_linear_1(code))
#             # print(x.shape)
#             x = F.leaky_relu(self.dec_linear_2(x))
#             # x = F.relu(self.dec_linear_3(x))
#             # print(x.shape)
#             # x = self.dec_linear_4(x)
#             x = x.view([code.shape[0], 8, 36, 36])
#             # print(x.shape)
#             #print('Dim indices2: {}'.format(indices2.shape))
#             # x = F.max_unpool2d(x, indices2, 2)
#             x = F.leaky_relu(self.dec_convT_1(x)) #W2 =(W-1)*S-2P+K = (19-1)*2-2*2+3
#             # print(x.shape)
#             x = self.batch_dec_1(x)
#             # x = F.max_unpool2d(x, indices1, 2)
#             #print('Dim indices1: {}'.format(indices1.shape))
#             x = F.leaky_relu(self.dec_convT_2(x))           
#             # print(x.shape)
#             x = self.batch_dec_2(x)
#             out = torch.sigmoid(self.dec_convT_3(x))
#             # print(out.shape)
#     #        decoded = F.tanh(x)
#             return out

elif structure_net == 'convs_simple_two_dense_batchNorm':
    class ConvAutoencoder(nn.Module):
        def __init__(self, code_size):
            self.code_size = code_size
            super().__init__()
            self.enc_cnn_1 = nn.Conv2d(1, 16, 3, stride=3, padding=1)  # b, 16, 10, 10
            self.batch_enc_1 =  nn.BatchNorm2d(16)
            self.enc_cnn_2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)  # b, 8, 3, 3
            self.batch_enc_2 = nn.BatchNorm2d(8)
            self.enc_linear_1 = nn.Linear(in_features=10368, out_features=500)   #Flattened image is fed into linear NN and reduced to half size
            self.batch_enc_den_1 = nn.BatchNorm1d(500)
            self.enc_linear_2 = nn.Linear(in_features=500, out_features=self.code_size)
            # self.enc_linear_3 = nn.Linear(in_features=100, out_features=self.code_size)
            self.dec_linear_1 = nn.Linear(in_features=self.code_size, out_features=500)
            self.batch_dec_den_1 = nn.BatchNorm1d(500)
            self.dec_linear_2 = nn.Linear(in_features=500, out_features=10368)
            self.batch_dec_den_2 = nn.BatchNorm1d(10368)
            self.dec_convT_1 = nn.ConvTranspose2d(8, 16, 3, stride=2)  # b, 16, 5, 5
            self.batch_dec_1 =  nn.BatchNorm2d(16)
            self.dec_convT_2 = nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1)  # b, 8, 15, 15
            self.batch_dec_2 =  nn.BatchNorm2d(8)
            self.dec_convT_3 = nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1)  # b, 8, 15, 15
            # self.batch_dec_3 =  nn.BatchNorm2d(1) #not really doing anything here I think
            
        def forward(self, images):
            code = self.encoder(images)
            out = self.decoder(code)
            return out, code

        def encoder(self, images):
            # print('Enconding:')
            # print(images.shape)
            x = F.leaky_relu(self.enc_cnn_1(images)) # [436,436,1], K=3, P=1, S=3 -> W2 =(W−F+2P)/S+1 = (436-3+2*1)/3+1= 146
            # print(x.shape)
            x = self.batch_enc_1(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2) #indices for unpooling, #146/2 = 73
            # print(x.shape)
            x = F.leaky_relu( self.enc_cnn_2(x)) #[73,73,16] -> W2 = (73-3+2*1)/2+1 = 38
            # print(x.shape)
            x = self.batch_enc_2(x)
            x = F.max_pool2d(x, kernel_size=2, stride=1) #38/2 = 19 -> [19,19,8]
            # x = F.leaky_relu(x)
            # print(x.shape)
            x = x.view([images.size(0), -1])
            # print(x.shape)
            x = F.leaky_relu(self.enc_linear_1(x))
            x = self.batch_enc_den_1(x)
            # print(x.shape)
            # x = F.dropout(x,p=0.5)
            # print(x.shape)
            x = self.enc_linear_2(x)
            code = torch.tanh(x)
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
            x = self.batch_dec_den_1(x)
            # print(x.shape)
            x = F.leaky_relu(self.dec_linear_2(x))
            x = self.batch_dec_den_2(x)
            # x = F.relu(self.dec_linear_3(x))
            # print(x.shape)
            # x = self.dec_linear_4(x)
            x = x.view([code.shape[0], 8, 36, 36])
            # print(x.shape)
            #print('Dim indices2: {}'.format(indices2.shape))
            # x = F.max_unpool2d(x, indices2, 2)
            x = F.leaky_relu(self.dec_convT_1(x)) #W2 =(W-1)*S-2P+K = (19-1)*2-2*2+3
            # print(x.shape)
            x = self.batch_dec_1(x)
            # x = F.max_unpool2d(x, indices1, 2)
            #print('Dim indices1: {}'.format(indices1.shape))
            x = F.leaky_relu(self.dec_convT_2(x))           
            # print(x.shape)
            x = self.batch_dec_2(x)
            out = torch.tanh(self.dec_convT_3(x))
            # print(out.shape)
    #        decoded = F.tanh(x)
            return out

# elif structure_net == 'convs_two_dense_batchNorm_dropOut0.0':
#     class ConvAutoencoder(nn.Module):
#         def __init__(self, code_size):
#             self.code_size = code_size
#             super().__init__()
#             self.enc_cnn_1 = nn.Conv2d(1, 16, 3, stride=3, padding=1)  # b, 16, 10, 10
#             self.batch_enc_1 =  nn.BatchNorm2d(16)
#             self.enc_cnn_2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)  # b, 8, 3, 3
#             self.batch_enc_2 = nn.BatchNorm2d(8)
#             self.enc_linear_1 = nn.Linear(in_features=10368, out_features=500)   #Flattened image is fed into linear NN and reduced to half size
#             self.batch_enc_den_1 = nn.BatchNorm1d(500)
#             self.enc_linear_2 = nn.Linear(in_features=500, out_features=self.code_size)
#             # self.enc_linear_3 = nn.Linear(in_features=100, out_features=self.code_size)
#             self.dec_linear_1 = nn.Linear(in_features=self.code_size, out_features=500)
#             self.batch_dec_den_1 = nn.BatchNorm1d(500)
#             self.dec_linear_2 = nn.Linear(in_features=500, out_features=10368)
#             self.batch_dec_den_2 = nn.BatchNorm1d(10368)
#             self.dec_convT_1 = nn.ConvTranspose2d(8, 16, 3, stride=2)  # b, 16, 5, 5
#             self.batch_dec_1 =  nn.BatchNorm2d(16)
#             self.dec_convT_2 = nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1)  # b, 8, 15, 15
#             self.batch_dec_2 =  nn.BatchNorm2d(8)
#             self.dec_convT_3 = nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1)  # b, 8, 15, 15
#             # self.batch_dec_3 =  nn.BatchNorm2d(1) #not really doing anything here I think
            
#         def forward(self, images):
#             code = self.encoder(images)
#             out = self.decoder(code)
#             return out, code

#         def encoder(self, images):
#             # print('Enconding:')
#             # print(images.shape)
#             x = F.leaky_relu(self.enc_cnn_1(images)) # [436,436,1], K=3, P=1, S=3 -> W2 =(W−F+2P)/S+1 = (436-3+2*1)/3+1= 146
#             # x = F.dropout(x,p=0.0)
#             # print(x.shape)
#             x = self.batch_enc_1(x)
#             x = F.max_pool2d(x, kernel_size=2, stride=2) #indices for unpooling, #146/2 = 73
#             x = F.dropout(x,p=0.0)
#             # print(x.shape)
#             x = F.leaky_relu( self.enc_cnn_2(x)) #[73,73,16] -> W2 = (73-3+2*1)/2+1 = 38
#             # x = F.dropout(x,p=0.0)
#             # print(x.shape)
#             x = self.batch_enc_2(x)
#             x = F.max_pool2d(x, kernel_size=2, stride=1) #38/2 = 19 -> [19,19,8]
#             x = F.dropout(x,p=0.0)
#             # x = F.leaky_relu(x)
#             # print(x.shape)
#             x = x.view([images.size(0), -1])
#             # print(x.shape)
#             x = F.leaky_relu(self.enc_linear_1(x))
#             # x = F.dropout(x,p=0.0)
#             x = self.batch_enc_den_1(x)
#             # print(x.shape)
#             # print(x.shape)
#             code = self.enc_linear_2(x)
#             # x = F.dropout(x,p=0.0)
#             # print(x.shape)
#             # x = F.dropout(x,p=0.0)
#             # print(x.shape)
#             # code = self.enc_linear_3(x)
#             # print(code.shape)
#             return code

#         def decoder(self, code):
#             # print('Deconding:')
#             # print(code.shape)
#             x = F.leaky_relu(self.dec_linear_1(code))
#             x = F.dropout(x,p=0.0)
#             x = self.batch_dec_den_1(x)
#             # print(x.shape)
#             x = F.leaky_relu(self.dec_linear_2(x))
#             x = F.dropout(x,p=0.0)
#             x = self.batch_dec_den_2(x)
#             # x = F.relu(self.dec_linear_3(x))
#             # print(x.shape)
#             # x = self.dec_linear_4(x)
#             x = x.view([code.shape[0], 8, 36, 36])
#             # print(x.shape)
#             #print('Dim indices2: {}'.format(indices2.shape))
#             # x = F.max_unpool2d(x, indices2, 2)
#             x = F.leaky_relu(self.dec_convT_1(x)) #W2 =(W-1)*S-2P+K = (19-1)*2-2*2+3
#             x = F.dropout(x,p=0.0)
#             # print(x.shape)
#             x = self.batch_dec_1(x)
#             # x = F.max_unpool2d(x, indices1, 2)
#             #print('Dim indices1: {}'.format(indices1.shape))
#             x = F.leaky_relu(self.dec_convT_2(x))           
#             x = F.dropout(x,p=0.0)
#             # print(x.shape)
#             x = self.batch_dec_2(x)
#             out = torch.tanh(self.dec_convT_3(x))
#             # print(out.shape)
#     #        decoded = F.tanh(x)
#             return out

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
dataset_dir      = os.path.join(curr_path, "data/18-november-augment")
testset_dir      = os.path.join(curr_path,"data/testset")
batch_size       = 128
validation_split = 0.2 # -- split training set into train/val sets
n_epochs           = 200
code_size_range = [100, 50, 12] #dimension of the latent space
patience = 3
weight_decay_range = [1e-3, 1e-4, 1e-5]
lr_range = [0.01, 0.001, 0.0001]
momentum_range = [0.8, 0.9, 0.99]

normalize = transforms.Normalize(mean=[0.5],std=[0.5])

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

# Check if normalization worked
train_loader = torch.utils.data.DataLoader(image_dataset, batch_size=10000, num_workers=4)
dataiter = iter(train_loader)
images, labels = dataiter.next()
print('Dim of raw sample image: {}\n'.format(images.shape))
print('max {}, min {}'.format(torch.max(images),torch.min(images)))
del train_loader


for code_size in code_size_range:
    for momentum in momentum_range:
        for lr in lr_range:
            for weight_decay in weight_decay_range:
                torch.cuda.empty_cache() 
                string_name = structure_net + '_' + str(n_epochs) + 'epochs_codeSize' + str(code_size) + '_lr' + str(lr) + '_weightDecay' + str(weight_decay) + '_split' + str(validation_split) + '_moment' + str(momentum) + '_adam'
                print('\n\n ------NEW RUN------ \n training params:\n phase: {}\n curr_path: {}\n dataset dir: {}\n weight_decay: {}\n lr: {}\n batch size: {}\n val split: {}\n epochs: {}\n structure: {}\n code_size: {}\n patience: {}\n string_name: {}\n'.format(phase, curr_path, dataset_dir, weight_decay, lr, batch_size, validation_split, n_epochs, structure_net, code_size, patience, string_name))

                # nvidia_smi.nvmlInit()
                # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0) 
                # res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                # print(f'gpu_1: {res.gpu}%, gpu-mem: {res.memory}%')

                
                # prepare data loaders
                # train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
                # test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
                    
                # obtain one batch of training images
                dataiter = iter(dataloaders['train'])
                images, labels = dataiter.next()
                print('Dim of raw sample image: {}\n'.format(images.shape))
                print('max {}, min {}'.format(torch.max(images),torch.min(images)))
                # images = images.numpy()

                # get one image from the batch
                # img = np.squeeze(images[0])
                # print('Dim of sample image: {}\n'.format(img.shape))
                # fig = plt.figure(figsize = (5,5)) 
                # ax = fig.add_subplot(111)
                # ax.imshow(img, cmap='gray')
                # plt.savefig('USV_example_v5.png')

                #initialize the NN
                #model = ConvAutoencoder()
                # dump_tensors()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
                model = ConvAutoencoder(code_size).to(device)
                # del ConvAutoencoder
                #print(model)
                # summary(model, input_size=(1, 436, 436))
                # dump_tensors()

                ## Training the NN ##
                #Specify Loss Function
                # criterion = nn.BCELoss()
                criterion = nn.MSELoss()

                #Specify optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay,amsgrad=False)
                # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

                #Number of epoch for training
                #n_epochs = 10 #Make it stop before overfitting
                best_loss = 100.0

                def train_model(model, batch_size, scheduler, dataloaders, patience, n_epochs):
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

                        model.train()
                        for itr, (images, labels) in enumerate(dataloaders['train']):
                            images = images.to(device)
                            #Clear the gradients of all optimized variables
                            optimizer.zero_grad()
                            out, code = model(Variable(images))
                            # out, code = model(images)
                            loss = criterion(out, images)
                            #backward pass: compute graditent of the loss with respect to the model parameters
                            loss.backward()
                            # Perform single optimization step (parameter update)
                            optimizer.step()
                            train_losses.append(loss.item()*images.size(0))
                            del images, labels, loss
                        model.eval()
                        # with torch.no_grad():
                        for data in dataloaders['val']:
                            images, _ = data
                            images = Variable(images).to(device)
                            out, code = model(Variable(images))
                            loss = criterion(out, images)
                            valid_losses.append(loss.item()*images.size(0))
                            del images, data, loss

                        train_loss = np.average(train_losses) #/ (len(dataloaders['train'].dataset))
                        valid_loss = np.average(valid_losses) #/ (len(dataloaders['val'].dataset))
                        avg_train_losses.append(train_loss)
                        avg_valid_losses.append(valid_loss)
                        scheduler.step(valid_loss)

                            # t.set_postfix(loss='{:05.3f}'.format(avg_valid_losses))
                            # t.update()
                            
                        epoch_len = len(str(n_epochs))
                        
                        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                                 f'train_loss: {train_loss:.5f} ' +
                                 f'valid_loss: {valid_loss:.5f}')

                        print(print_msg)
                        # print('validation_loss: {}\n model: {}\n structure_net: {}'.format(valid_loss,model,structure_net))
                        early_stopping(valid_loss, model, string_name)
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
                    model.load_state_dict(torch.load('checkpoint_' + string_name + '.pt'))
                    return  model, avg_train_losses, avg_valid_losses
                        # epoch=epoch+1
                # else:
                #     model.load_state_dict(torch.load('best_net_test' + structure_net + '.pth'))

                #Choosing mode
                if phase == 'train':
                    scheduler = lrs.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=5)
                    model, train_loss, valid_loss = train_model(model, batch_size, scheduler, dataloaders, patience, n_epochs)
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, 'best_model' + string_name + '.pth')
                else:
                    model.load_state_dict(torch.load('best_model' + string_name + '.pth'))

                ## Checking results ##
                #Batch of test images
                dataiter = iter(dataloaders['val'])
                images, labels = dataiter.next()
                images = Variable(images).cuda()
                # print('Axis 1: max {}, min {}'.format(images[...,1].max(), images[...,1].min())

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

                fig.suptitle('convAE_valset_' + string_name , fontsize=16)
                plt.savefig('convAE_valset_' + string_name + '.png')

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

                fig.suptitle('convAE_testset_' + string_name , fontsize=16)
                plt.savefig('convAE_testset_' + string_name + '.png')

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
                    plt.title('Loss_' + string_name)
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.show()
                    fig.savefig('loss_plot' + string_name + '.png', bbox_inches='tight')

                # NOW A REAL TEST#
                # -- create dataset
                # test_dataset = datasets.ImageFolder(testset_dir, transform=trans)
                # class_names   = test_dataset.classes
                # num_classes   = len(test_dataset)
                # dataset_size  = len(test_dataset)
                # print('Starting unseen dataset \n dataset has {} images'.format(dataset_size))

                #Load labels
                curr_path    = os.getcwd()
                excel_path      =  '/home/antonio/Documents/conv_auto/data/correct_classifier/1303_Agrp-Trpv1_1st/1303_Agrp-Trpv1_1st_GT.xlsx'
                tabela = pd.read_excel(excel_path)
                labels_all = tabela['GT']

                # dataloaders   = {
                #     # 'train': torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, num_workers=4),
                #     'test': torch.utils.data.DataLoader(test_dataset,  batch_size=dataset_size, num_workers=4, shuffle=False),
                # }

                # model.load_state_dict(torch.load('best_model' + structure_net + '.pth'))

                dataiter = iter(dataloaders['test'])
                images, labels = dataiter.next()
                labels = labels.numpy()
                print('labels: {}'.format(labels))
                images = Variable(images).cuda()
                
                # nvidia_smi.nvmlInit()
                # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0) 
                # res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                # print(f'gpu_1: {res.gpu}%, gpu-mem: {res.memory}%')  

                del output, train_loss
                torch.cuda.empty_cache()
                output, code = model(images)
                
                # nvidia_smi.nvmlInit()
                # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0) 
                # res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                # print(f'gpu_1: {res.gpu}%, gpu-mem: {res.memory}%')  

                del model, images, output
                torch.cuda.empty_cache()

                # nvidia_smi.nvmlInit()
                # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0) 
                # res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                # print(f'gpu_before_phate: {res.gpu}%, gpu-mem: {res.memory}%')

                # #Prep images for display
                # # images = images.cpu().numpy()

                # torch.save(code.cpu(), '/home/antonio/Documents/conv_auto/code.pt')
                # np.save('/home/antonio/Documents/conv_auto/code_np' + structure_net, code.cpu().detach().numpy())
                gamma=0
                phate_operator = phate.PHATE(n_components=3, k=5, a=20, t=150, gamma=gamma)
                data_phate = phate_operator.fit_transform(code.cpu().detach().numpy())
                # print('data_phate shape: {}\n'.format(data_phate.shape))

                # # print(labels_all)
                fig1 = scprep.plot.scatter2d(data_phate, c=labels, filename= string_name + ".png", title=string_name, ticks=False, label_prefix="PHATE")
                fig2 = scprep.plot.rotate_scatter3d(data_phate, c=labels, filename=string_name + ".gif", title=string_name)

                # nvidia_smi.nvmlInit()
                # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0) 
                # res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                # print(f'gpu_post_phate: {res.gpu}%, gpu-mem: {res.memory}%')

                # # fig2.clf()
                plt.clf()
                plt.close()
                gc.collect()

                del fig1, fig2

                del data_phate, phate_operator
                torch.cuda.empty_cache()
                # nvidia_smi.nvmlInit()
                # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0) 
                # res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                # print(f'gpu_post_cleaning_phate_vars: {res.gpu}%, gpu-mem: {res.memory}%')

                # dump_tensors()

                # #Get PHATE plot for training data#
                # print('\nEmbedding training data:')
                # image_dataset = datasets.ImageFolder(dataset_dir, transform=trans)
                # class_names   = image_dataset.classes
                # num_classes   = len(class_names)
                # dataset_size  = len(image_dataset)
                # print('dataset has {} images'.format(dataset_size))
                # print('dataset has {} classes:'.format(num_classes))
                # print(class_names)

                # dataloaders   = {
                #     'train': torch.utils.data.DataLoader(image_dataset, batch_size=2000, num_workers=4),
                # }

                # dataiter = iter(dataloaders['train'])
                # images, labels = dataiter.next()
                # labels = labels.numpy()
                # # print('labels: {}'.format(labels))
                # images = Variable(images).cuda()
                # output, code = model(images)

                # #Prep images for display
                # # images = images.cpu().numpy()
                # phate_operator = phate.PHATE(n_components=3, k=5, a=20, t=150)
                # training_phate = phate_operator.fit_transform(code.cpu().detach().numpy())
                # print('training_phate shape: {}\n'.format(training_phate.shape))

                # # print(labels_all)
                # phate.plot.scatter2d(training_phate, cmap="Spectral",filename="embedding_trainingData" + structure_net + ".png", title=structure_net, ticks=False, label_prefix="PHATE")
                # phate.plot.rotate_scatter3d(training_phate, filename="embedding_trainingData" + structure_net + ".gif", title=structure_net)