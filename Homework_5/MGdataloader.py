# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 16:12:38 2019
Data loader for Mackey-Glass nonlinear time series
@author: Joshua Peeples
"""

import os
from torch.utils.data import Dataset
import pdb
import torch
import numpy as np

class MG_data(Dataset):

    def __init__(self, directory, window = 20, Noise = False):
        
        #Labels
        self.targets = []
        #Load the data
        input_data = np.loadtxt(directory)
        
        #Compute dataset using windows
        # Source for windowing code: 
        # https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
        self.inout_seq = []
        L = len(input_data)
        for i in range(L-window):
            train_seq = input_data[i:i+window]
            train_label = input_data[i+window:i+window+1]
            #Add noise using Middleton model if desired
            if (Noise):
                train_label += (.95*np.random.normal(loc=0,scale=np.sqrt(.5)) +
                                .05*np.random.normal(loc=1,scale=np.sqrt(.5)))
            self.inout_seq.append({  # sequence and desired
                    "sequence": train_seq,
                    "label": train_label
                })
            self.targets.append(train_label)


    def __len__(self):
        return len(self.inout_seq)

    def __getitem__(self, index):

        datafiles = self.inout_seq[index]

        sequence = torch.tensor(datafiles["sequence"])

        label_file = datafiles["label"]
        label = torch.tensor(label_file)

        return sequence, label,index