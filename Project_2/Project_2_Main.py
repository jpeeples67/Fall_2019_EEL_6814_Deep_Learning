# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 18:31:56 2019
Main code for project
@author: jpeeples
"""

import os
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch
import pdb
from torchvision import datasets,transforms
from train_model import train
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


folder = 'Results/'
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, folder)

#Window size, number of epochs, batch size, learning rate and loss function to use
N = 100
epochs = 100
batch_size = 64
eta = .01
Noise = False
device = 'cpu'
loss_fxn = 'MSE' #MSE or MEE
bandwidth = .1

#Load data to compute bandwidth using Silverman's rule

#Create Stacked autoencoder
class SAE(nn.Module):
    def __init__(self,bottleneck=100):
        super(SAE,self).__init__()
        
        self.encoder = nn.Sequential(nn.Linear(784,500),nn.ReLU(),
                                     nn.Linear(500,200),nn.ReLU(),
                                     nn.Linear(200,bottleneck),nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(bottleneck,200),nn.ReLU(),
                                     nn.Linear(200,500),nn.ReLU(),
                                     nn.Linear(500,784),nn.ReLU())
    def forward(self,x):
        #Encoder
        x = self.encoder(x)
        #Decoder
        x = self.decoder(x)
        return x

#Load data and create dataloader
data_transforms = transforms.ToTensor()
train_dataset = datasets.FashionMNIST('../data', train=True, download=True,transform = data_transforms)
test_dataset = datasets.FashionMNIST('../data', train=False, download=True, transform = data_transforms)

#Flatten data and normalize
scaler = StandardScaler()
train_data = scaler.fit_transform(torch.flatten(train_dataset.data,start_dim=1).numpy())
test_data = scaler.transform(torch.flatten(test_dataset.data,start_dim=1).numpy())
train_targets = train_dataset.targets
test_targets = test_dataset.targets
MG_dataloader = torch.utils.data.DataLoader(dataset=MG_series,batch_size=batch_size,
                                            shuffle=False,num_workers=0)

#Define FIR filter of order N, loss function, and optimizer
model = SAE(bottleneck=N)
optimizer = optim.Adam(model.parameters(),lr=eta)

#Train model and get predictions
(model, best_model_wts, train_error_history,best_epoch,
            actual_series,predicted_series) = train(model,MG_dataloader,loss_fxn,
                                          optimizer,device,num_epochs = epochs,
                                          bw = bandwidth)

#Generate figures and save results
results_folder = os.path.join(final_directory,loss_fxn)
if(Noise):
    results_folder = results_folder + '/Noise/'
else:
    results_folder = results_folder + '/No_Noise/'
 
#Create directory 
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
#Learning curve
fig1 = plt.figure()
plt.plot(train_error_history)
plt.suptitle('Learning Curve for {} Loss'.format(loss_fxn))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training Loss'], loc='upper right')
plt.show()
fig1.savefig((results_folder + 'Learning Curve.png'), dpi=fig1.dpi)
#plt.close()

#Plot Actual vs Prediction
fig2 = plt.figure()
plt.plot(actual_series)
plt.plot(predicted_series)
plt.suptitle('Actual vs Predicted Mackey-Glass nonlinear time series {} Loss'.format(loss_fxn))
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend(['Actual','Predicted'], loc='upper right')
plt.show()
fig2.savefig((results_folder + 'Series Data.png'), dpi=fig2.dpi)
#plt.close()
