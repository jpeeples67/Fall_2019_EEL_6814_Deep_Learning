# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 18:31:56 2019
Main code for project
@author: jpeeples
"""

from Experimental_parameters import Networks
from Prepare_data import Reduced_datasets
from Visualizations import TSNE_Visual
from model_functions import train_model
import os
import torch.optim as optim
import torch.nn as nn
import pdb

visualize = 0
num_epochs = 100
fold_num = 3
batch_size = {'train': 1024, 'val': 2048}
folder = 'Results/'
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, folder)
dim_red_names = list(Reduced_datasets.keys())
Selected_Nets = [6]
#Run experiments to vary network architecture and dimensionality reduction
for name in dim_red_names:
    data_folder = final_directory + name
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    #Visualize dataset using TSNE (training)
    if (visualize):
        TSNE_Visual(Reduced_datasets[name]['train'].files,
                       Reduced_datasets[name]['train'].targets,data_folder,name)
    for network in Selected_Nets:
        #Train model using 3-fold CV and save visual results
        Network_folder = data_folder + '/' + 'Network ' + str(network) + '/'
        if not os.path.exists(Network_folder):
            os.makedirs(Network_folder)
        num_features = Reduced_datasets[name]['train'].files.shape[1]
        model = Networks[network+1](in_features=num_features)
        optimizer = optim.Adam(model.parameters(),lr=.001)
        criterion = nn.CrossEntropyLoss()
        #Compute number of parameters once and save to Network folder
        if name == dim_red_names[0]:
            #Print number of trainable parameters
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                # Write to text file
            with open((Network_folder + 'NumParams.txt'), "w") as output:
                output.write(str(num_params))
        train_model(model,Reduced_datasets[name],batch_size,criterion,optimizer,
                    num_epochs=num_epochs,k=fold_num,data_folder=Network_folder)