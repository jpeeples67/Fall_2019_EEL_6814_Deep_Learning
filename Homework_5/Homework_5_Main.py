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
from MGdataloader import MG_data
from torchvision import transforms
from train_model import train
import matplotlib.pyplot as plt


folder = 'Results/'
data_filename = 'MGdata.txt'
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, folder)

#Window size, number of epochs, batch size, learning rate and loss function to use
N = 20
epochs = 100
batch_size = 64
eta = .01
Noise = False
device = 'cpu'
loss_fxn = 'MSE' #MSE or MEE
bandwidth = .1

#Load data to compute bandwidth using Silverman's rule

#Create FIR filter
class FIR_filter(nn.Module):
    def __init__(self,model_order=20):
        super(FIR_filter,self).__init__()
        self.fc1 = nn.Linear(model_order,1,bias=False)
        
    def forward(self,x):
        x = self.fc1(x)
        return x

#Load data and create dataloader
MG_series = MG_data(data_filename,window=N,Noise=Noise)
MG_dataloader = torch.utils.data.DataLoader(dataset=MG_series,batch_size=batch_size,
                                            shuffle=False,num_workers=0)

#Define FIR filter of order N, loss function, and optimizer
model = FIR_filter(model_order=N)
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
