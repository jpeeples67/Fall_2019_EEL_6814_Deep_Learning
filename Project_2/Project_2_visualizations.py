# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 18:31:56 2019
Main code for project
@author: jpeeples
"""

import os
import torch.optim as optim
import torch
import torch.nn as nn
import pdb
from Functions.train_model import train_classifier
from Functions.Prepare_data import dataloaders_dict
from Functions.Get_Results import Classifier_results_ITL
from Functions.Networks import MLP_ITL
import matplotlib.pyplot as plt
from barbar import Bar
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.cm as colormap


folder = 'Results/SAE/'
ITL = 0
CNN = 0
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, folder)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Visualize fashion mnist dataset
train_data_example = dataloaders_dict['val'].dataset.data[dataloaders_dict['val'].sampler.indices[3000]]
plt.figure()
plt.imshow(train_data_example.numpy(),cmap ='gray')
##plt.imshow(train_data_example.numpy())
plt.axis('off')
class_names = np.array(['T-shirt/top','Trouser','Pullover','Dress',
                        'Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot'])
#Show reconstruction of training image
bottleneck_sizes = [100,50,25,12,2]

for M in bottleneck_sizes:
    #Look for folder with bottleneck model
    temp_folder = final_directory + 'Bottleneck_Size_' + str(M) + '/'

    #Load model
    temp_model = torch.load((temp_folder+'Model.pt'),map_location=device)
    temp_model.load_state_dict(torch.load(temp_folder+'Best_Weights.pt'))
    temp_model.eval()
    
    if not(ITL):
        #Show reconstructed image
        output = temp_model(train_data_example.unsqueeze(0).unsqueeze(1).float().to(device))
        output = torch.reshape(output,(28,28))
        fig1 = plt.figure()
        plt.imshow(output.detach().cpu().numpy(),cmap='gray')
        plt.axis('off')
        fig1.savefig(temp_folder+'Bottleneck_Reconstruction_'+str(M))
        plt.close()
    
    #T-SNE visual of test features
    inputs = (dataloaders_dict['test'].dataset.data)/255
    labels = dataloaders_dict['test'].dataset.targets.numpy()
    
    #Remove decoder portion
    if(ITL):
        temp_model.fc = nn.Sequential()
    else:
        temp_model.decoder = nn.Sequential()
    
    encoded_features = temp_model(inputs.unsqueeze(1).float().to(device))
    if (M==2):
        features_embedded = encoded_features.detach().cpu().numpy()
    else:
        features_embedded = TSNE(n_components=2,verbose=1).fit_transform(encoded_features.detach().cpu().numpy())

    fig2, ax2 = plt.subplots()
    colors = colormap.rainbow(np.linspace(0, 1, 10))
    for clothes in range (0, 10):
        x = features_embedded[np.where(labels==clothes),0]
        y = features_embedded[np.where(labels==clothes),1]
        
        plt.scatter(x, y, color = colors[clothes,:],label = class_names[clothes])
    ax2.legend(loc=1) 
    plt.title('TSNE Visualization of Test Data For Bottleneck of ' + str(M))
    plt.show()
    
    fig2.savefig((temp_folder + 'TSNE_testing.png'), dpi=fig2.dpi)
    plt.close() 
    
    #Print progress
    print('Finished bottleck_size of '+str(M))
    
if (CNN):
    #Look for folder with CNN model
    temp_folder = final_directory + 'CNN/'

    #Load model
    temp_model = torch.load((temp_folder+'Model.pt'),map_location=device)
    temp_model.load_state_dict(torch.load(temp_folder+'Best_Weights.pt'))
    temp_model.eval()
    
    #T-SNE visual of test features
    inputs = (dataloaders_dict['test'].dataset.data)/255
    labels = dataloaders_dict['test'].dataset.targets.numpy()
    
    #Remove decoder portion
    temp_model.fc1 = nn.Sequential()
    temp_model.fc2 = nn.Sequential()
    
    encoded_features = temp_model(inputs.unsqueeze(1).float().to(device))
    features_embedded = TSNE(n_components=3,verbose=1).fit_transform(encoded_features.detach().cpu().numpy())

    fig2, ax2 = plt.subplots()
    colors = colormap.rainbow(np.linspace(0, 1, 10))
    for clothes in range (0, 10):
        x = features_embedded[np.where(labels==clothes),0]
        y = features_embedded[np.where(labels==clothes),1]
        z = features_embedded[np.where(labels==clothes),2]
        
        plt.scatter(x, y, z, color = colors[clothes,:],label = class_names[clothes])
    ax2.legend(loc=1) 
    plt.title('TSNE Visualization of Test Data For CNN')
    plt.show()
    
    fig2.savefig((temp_folder + 'TSNE_testing.png'), dpi=fig2.dpi)
    plt.close() 
    



