# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:13:58 2019

@author: jpeeples
"""
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import pdb
import numpy as np

#Number of images to process at a time
var = .85
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
transforms = transforms.ToTensor()
full_train_dataset = datasets.FashionMNIST('../data', train=True, 
                                      download=True,transform = transforms)
full_test_dataset = datasets.FashionMNIST('../data', train=False, download=True,
                                     transform = transforms)
full_train_dataset_batch = full_train_dataset.data.unsqueeze(1)
full_test_dataset_batch = full_test_dataset.data.unsqueeze(1)
mid_train_dataset = F.interpolate(full_train_dataset_batch.to('cpu',dtype=float), size=[14,14])
mid_test_dataset = F.interpolate(full_test_dataset_batch.to('cpu',dtype=float), size=[14,14])
small_train_dataset = F.interpolate(full_train_dataset_batch.to('cpu',dtype=float), size=[7,7])
small_test_dataset = F.interpolate(full_test_dataset_batch.to('cpu',dtype=float), size=[7,7])

#Flatten data to perform dimensionality reduction
scaler_full = StandardScaler()
scaler_mid = StandardScaler()
scaler_small = StandardScaler()
#Normalize each scale of the data
full_train_data = scaler_full.fit_transform(torch.flatten(full_train_dataset.data,start_dim=1).numpy())
full_test_data = scaler_full.transform(torch.flatten(full_test_dataset.data,start_dim=1).numpy())
mid_train_data = scaler_mid.fit_transform(torch.flatten(mid_train_dataset.squeeze(1),start_dim=1).numpy())
mid_test_data = scaler_mid.transform(torch.flatten(mid_test_dataset.squeeze(1),start_dim=1).numpy())
small_train_data = scaler_small.fit_transform(torch.flatten(small_train_dataset.squeeze(1),start_dim=1).numpy())
small_test_data = scaler_small.transform(torch.flatten(small_test_dataset.squeeze(1),start_dim=1).numpy())
train_targets = full_train_dataset.targets
test_targets = full_test_dataset.targets

#Run dimensionality reduction
print('Prepping data with dimensionality reduction')
print('PCA on training and test data')
full_pca = PCA(n_components=var)
mid_pca = PCA(n_components=var)
small_pca = PCA(n_components=var)
full_pca_train_data = torch.from_numpy(full_pca.fit_transform(full_train_data))
full_pca_test_data = torch.from_numpy(full_pca.transform(full_test_data))
mid_pca_train_data = torch.from_numpy(mid_pca.fit_transform(mid_train_data))
mid_pca_test_data = torch.from_numpy(mid_pca.transform(mid_test_data))
small_pca_train_data = torch.from_numpy(small_pca.fit_transform(mid_train_data))
small_pca_test_data = torch.from_numpy(small_pca.transform(mid_test_data))

#Create dataset object for dataloader in main code, combine multiscale data
pca_train_data = torch.cat((full_pca_train_data,mid_pca_train_data,small_pca_train_data),dim=1)
pca_test_data = torch.cat((full_pca_test_data,mid_pca_test_data,small_pca_test_data),dim=1)
PCA_train = Create_dataset((pca_train_data,train_targets))
PCA_test = Create_dataset((pca_test_data,test_targets))


#Create dataset for training and testing data for each dimensionality reduction method
PCA_dataset = {'train': PCA_train, 'test': PCA_test}  
Reduced_datasets = {'PCA': PCA_dataset}




