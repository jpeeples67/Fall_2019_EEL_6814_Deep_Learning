# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 16:24:50 2019

@author: Joshua Peeples
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from HMWK_2_functions import train_model,view_results

#Load data
patient_1_input = pd.read_csv('Sleepdata1 Input.asc',sep='\t')
patient_1_desired = pd.read_csv('Sleepdata1 Desired.asc',sep='\t')
patient_2_input = pd.read_csv('Sleepdata2 Input.asc',sep='\t')
patient_2_desired = pd.read_csv('Sleepdata2 Desired.asc',sep='\t')
class_number = patient_1_desired.columns.shape[0]
class_names = patient_1_desired.columns
num_feats = patient_1_input.shape[1]
#Convert dataframes to tensors
patient_1_input = torch.FloatTensor(patient_1_input.values)
patient_1_desired = torch.FloatTensor(patient_1_desired.values)
patient_2_input = torch.FloatTensor(patient_2_input.values)
patient_2_desired = torch.FloatTensor(patient_2_desired.values)
#For cross entropy loss, need class label
patient_1_labels = patient_1_desired.argmax(dim=1)
patient_2_labels = patient_2_desired.argmax(dim=1)
#Loss to use, either 'MSE' or 'CE'
criterion = 'CE'
epochs = 10000 #10000
progress = 1000 #Print every progess # of epochs
lr_rate = 1e-3 #-e6
alpha = .9 #momentum
num_units = 50

#Build model
class SleepModel(nn.Module):
    def __init__(self, num_classes=class_number):
        super(SleepModel, self).__init__()
        self.hidden_layers = nn.Sequential(nn.Linear(num_feats,num_units))
#        self.hidden_layers = nn.Sequential(nn.Linear(num_feats,num_units),
#                                       nn.Sigmoid(),
#                                     nn.Linear(num_units,int(num_units/2)))
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_layers[-1].out_features, num_classes))
    def forward(self, x):
        x = torch.sigmoid(self.hidden_layers(x))
        x = self.classifier(x)
        return x

# P
#Train on first patient, test on second
model = SleepModel()
#Train model
best_model_wts,Error_track = train_model(model,epochs,criterion,lr_rate,
                                         alpha,patient_1_input,patient_1_labels,
                                         progress)

#Take best model and compute confusion matrix for training/test data
view_results(Error_track,model,best_model_wts,patient_1_input,patient_1_labels,
                 'Patient 1',patient_2_input,patient_2_labels,'Patient 2',
                 class_names)
model_2 = SleepModel()
#Train model
best_model_wts_2,Error_track = train_model(model_2,epochs,criterion,lr_rate,
                                         alpha,patient_2_input,patient_2_labels,
                                         progress)

#Take best model and compute confusion matrix for training/test data
view_results(Error_track,model_2,best_model_wts_2,patient_2_input,patient_2_labels,
                 'Patient 2',patient_1_input,patient_1_labels,'Patient 1',
                 class_names)
    
    
    
    