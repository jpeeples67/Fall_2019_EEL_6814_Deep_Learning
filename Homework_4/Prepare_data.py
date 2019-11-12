# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:13:58 2019

@author: jpeeples
"""
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import torch
from barbar import Bar
import pdb
import numpy as np

#Number of images to process at a time
numClasses = 10
var = .95
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
#Code to prepare train and test dataset using dimensionality reduction
data_transforms = transforms.ToTensor()
train_dataset = datasets.FashionMNIST('../data', train=True, download=True,transform = data_transforms)
test_dataset = datasets.FashionMNIST('../data', train=False, download=True, transform = data_transforms)

#Flatten data to perform dimensionality reduction
scaler = StandardScaler()
train_data = scaler.fit_transform(torch.flatten(train_dataset.data,start_dim=1).numpy())
test_data = scaler.transform(torch.flatten(test_dataset.data,start_dim=1).numpy())
train_targets = train_dataset.targets
test_targets = test_dataset.targets

#Run dimensionality reduction
print('Prepping data with dimensionality reduction')
print('PCA on training and test data')
pca = PCA(n_components=var)
pca_train_data = torch.from_numpy(pca.fit_transform(train_data))
pca_test_data = torch.from_numpy(pca.transform(test_data))
num_comps = pca_train_data.shape[1]

#Create dataset object for dataloader in main code
PCA_train = Create_dataset((pca_train_data,train_targets))
PCA_test = Create_dataset((pca_test_data,test_targets))


#Create dataset for training and testing data for each dimensionality reduction method
PCA_dataset = {'train': PCA_train, 'test': PCA_test}  
Reduced_datasets = {'PCA': PCA_dataset}




