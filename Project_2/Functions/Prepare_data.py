# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:13:58 2019

@author: jpeeples
"""
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import pdb
import numpy as np

#Create training, validation and test splits for fashion mnist data
batch_size = {'train': 1024, 'val': 2048, 'test': 2048}
split = .1
np.random.seed(0)
    
#Code to prepare train and test dataset
data_transforms = transforms.ToTensor()
train_dataset = datasets.FashionMNIST('../data', train=True, download=True,transform = data_transforms)
test_dataset = datasets.FashionMNIST('../data', train=False, download=True, transform = data_transforms)

#Divide training data into training and validation split
indices = np.arange(len(train_dataset))
y = train_dataset.targets.numpy()
    
#Use stratified split to balance training validation splits, set random state to be same for each encoding method
_,_,_,_,train_indices,val_indices = train_test_split(y,y,indices,stratify=y,test_size = split,random_state=42)

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

# Create training and validation dataloaders
dataloaders_dict = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size['train'],
                                           sampler=train_sampler, shuffle=False,num_workers=0),
                    'val': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size['val'],
                                           sampler=valid_sampler, shuffle=False,num_workers=0),
                    'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size['test'],
                                           shuffle=True,num_workers=0) }




