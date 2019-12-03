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
from Functions.train_model import train_SAE,train_classifier
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from Functions.Prepare_data import dataloaders_dict
from Functions.Get_Results import SAE_results,Classifier_results
from Functions.Networks import SAE,MLP,CNN


folder = 'Results/'
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, folder)

#Window size, number of epochs, batch size, learning rate and loss function to use
bottleneck_sizes = [100,50,25,12,2]
epochs = 100
eta = .001
num_classes = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Define SAE with bottleneck of size M, loss function, and optimizer
for M in bottleneck_sizes:
    SAE_model = SAE(bottleneck=M)
    SAE_optimizer = optim.Adam(SAE_model.parameters(),lr=eta)
    
    
    #Train model and get predictions
    (SAE_model, best_SAE_model_wts, train_error_history,val_error_history, best_loss,
                time_elapsed) = train_SAE(SAE_model,dataloaders_dict,
                                              SAE_optimizer,device=device,num_epochs = epochs)
    
    #Generate figures and save results
    SAE_results(final_directory,SAE_model,best_SAE_model_wts,train_error_history,
                val_error_history,time_elapsed,best_loss,M)
    
    #Train MLP that uses features from SAE to classify test set
    MLP_classifier = MLP(SAE_model.encoder,num_classes=num_classes)
    
    #Define optimizer for MLP
    MLP_optimizer = optim.Adam(MLP_classifier.parameters(),lr=eta)
    
    #Train MLP and classify test set
    (MLP_classifier, best_MLP_model_wts, train_error_history,val_error_history,best_acc,
                time_elapsed,GT,predictions) = train_classifier(MLP_classifier,
                                           dataloaders_dict,MLP_optimizer,num_epochs=epochs)
    #Get results for MLP classifier
    Classifier_results(final_directory,MLP_classifier,best_MLP_model_wts,
                       train_error_history,val_error_history,time_elapsed,best_acc,
                       GT,predictions,bottleneck_size=M,MLP=True)
    
    #Delete models to save memory
    del SAE_model,MLP_classifier
    torch.cuda.empty_cache()

#Train CNN and classify test set
CNN_classifier = CNN()
CNN_optimizer = optim.Adam(CNN_classifier.parameters(),lr=eta)

(CNN_classifier, best_CNN_model_wts, train_error_history,val_error_history,best_acc,
            time_elapsed,GT,predictions) = train_classifier(CNN_classifier,
                                       dataloaders_dict,CNN_optimizer,num_epochs=epochs)

#Get results for CNN classifier
Classifier_results(final_directory,CNN_classifier,best_CNN_model_wts,
                   train_error_history,val_error_history,time_elapsed,best_acc,
                   GT,predictions,MLP=False)






