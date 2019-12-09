# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 18:31:56 2019
Main code for project
@author: jpeeples
"""

import os
import torch.optim as optim
import torch
import pdb
from Functions.train_model import train_classifier
from Functions.Prepare_data import (encoded_dicts,bandwidths)
from Functions.Get_Results import Classifier_results_ITL
from Functions.Networks import MLP_ITL


folder = 'Results/ITL/'
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, folder)

#Window size, number of epochs, batch size, learning rate and loss function to use
bottleneck_sizes = [100,50,25,12,2]
epochs = 50
eta = .001
num_classes = 10
criterion = ['MCE','QMI']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Define Encoder with bottleneck of size M, loss function, and optimizer
#Initalize models
for M in bottleneck_sizes:
    for loss in criterion:
        for bandwidth in bandwidths:
            for encoding in encoded_dicts:
                
                #Set saving directory
                save_dir = os.path.join(final_directory,bandwidth,encoding)
                #Define 3 layer MLP with encoder and have output layer
                MLP_ITL_model = MLP_ITL(bottleneck=M,num_classes=num_classes)
                MLP_ITL_optimizer = optim.Adam(MLP_ITL_model.parameters(),lr=eta)
                
                #Train model and get predictions for encoded outputs without gaussian encoding
                (best_MLP_ITL, best_MLP_ITL_model_wts, train_error_history,val_error_history,best_acc,
                            time_elapsed,GT,predictions) = train_classifier(MLP_ITL_model,
                                                       encoded_dicts[encoding], MLP_ITL_optimizer,
                                                       criterion=loss,device=device,num_epochs=epochs,
                                                       bw=bandwidths[bandwidth],ITL=True)
                #Get results for MLP classifier
                Classifier_results_ITL(save_dir,best_MLP_ITL,best_MLP_ITL_model_wts,
                                   train_error_history,val_error_history,time_elapsed,best_acc,
                                   GT,predictions,bottleneck_size=M,QMI=(loss=='QMI'))
            
                #Delete models, optimizer, and save_dir
                del MLP_ITL_model,best_MLP_ITL,MLP_ITL_optimizer,save_dir
                torch.cuda.empty_cache()







