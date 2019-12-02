# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:13:58 2019

@author: jpeeples
"""
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import torch
import pdb
import numpy as np

#Number of images to process at a time
np.random.seed(0)
class Create_dataset(Dataset):
    def __init__(self,data_tuple,transform=None):
        self.files = data_tuple[0]
        self.targets = data_tuple[1]
        self.transform=transform
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        img = self.files[index]

        label = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label
#Code to prepare train and test dataset
data_transforms = transforms.ToTensor()
train_dataset = datasets.FashionMNIST('../data', train=True, download=True,transform = data_transforms)
test_dataset = datasets.FashionMNIST('../data', train=False, download=True, transform = data_transforms)

#Flatten data to perform dimensionality reduction
scaler = StandardScaler()
train_data = scaler.fit_transform(torch.flatten(train_dataset.data,start_dim=1).numpy())
test_data = scaler.transform(torch.flatten(test_dataset.data,start_dim=1).numpy())
train_targets = train_dataset.targets
test_targets = test_dataset.targets

#Create dataset object for dataloader in main code
SAE_train = Create_dataset((train_data,train_targets))
SAE_test = Create_dataset((test_data,test_targets))
FMNIST_train = Create_dataset((train_data,train_targets))
FMNIST_test = Create_dataset((test_data,test_targets))

#Create dataset for training and testing data for each dimensionality reduction method
SAE_dataset = {'train': SAE_train, 'test': SAE_test}  
FMNIST_dataset = {'train': FMNIST_train, 'test': FMNIST_test}  




