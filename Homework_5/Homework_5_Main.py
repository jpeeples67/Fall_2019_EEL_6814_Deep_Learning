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
import pdb

num_epochs = 100
batch_size = {'train': 1024, 'val': 2048}
folder = 'Results/'
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, folder)

# Function to create training data and label 
# Source: https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq
#Load data
data = np.loadtxt('MGdata.txt')

#Define window size and compute desired value
N = 20
desired = data[N+1:-1:N+1]
