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
print('ICA on training and test data')
ICA = FastICA(n_components=num_comps)
ICA_train_data = torch.from_numpy(ICA.fit_transform(train_data))
ICA_test_data = torch.from_numpy(ICA.transform(test_data))
print('LDA on training and test data')
LDA = LinearDiscriminantAnalysis(n_components=numClasses-1)
lda_train_data = torch.from_numpy(LDA.fit_transform(train_data,train_targets.numpy()))
lda_test_data = torch.from_numpy(LDA.transform(test_data))

#Create dataset object for dataloader in main code
PCA_train = Create_dataset((pca_train_data,train_targets))
PCA_test = Create_dataset((pca_test_data,test_targets))
ICA_train = Create_dataset((ICA_train_data,train_targets))
ICA_test = Create_dataset((ICA_test_data,test_targets))
LDA_train = Create_dataset((lda_train_data,train_targets))
LDA_test = Create_dataset((lda_test_data,test_targets))

#Create dataset for training and testing data for each dimensionality reduction method
PCA_dataset = {'train': PCA_train, 'test': PCA_test}  
ICA_dataset = {'train': ICA_train, 'test': ICA_test}  
LDA_dataset = {'train': LDA_train, 'test': LDA_test} 
Reduced_datasets = {'PCA': PCA_dataset, 'ICA': ICA_dataset, 'LDA': LDA_dataset}
#def PCA_dataset(data_loader,numComponents):
#    pca = PCA(n_components=numComponents)
#    PCA_set = []
#    labels = []
#    for idx, (x,t) in enumerate(Bar(data_loader)):
#        x = torch.flatten(x,start_dim=1).numpy()
#        pca.fit(x)
#        PCA_set.append(pca.transform(x))
#        labels.append(t.numpy())
#    #Change to array
#    PCA_set = np.asarray(PCA_set)
#    PCA_set = np.concatenate(PCA_set).astype(None)
#    labels = np.asarray(labels)
#    labels = np.concatenate(labels).astype(None)
#    return (torch.from_numpy(PCA_set),torch.from_numpy(labels))
#
#def ICA_dataset(data_loader,numComponents):
#    ICA = KernelPCA(n_components=numComponents,kernel='rbf')
#    ICA_set = []
#    labels = []
#    for idx, (x,t) in enumerate(Bar(data_loader)):
#        x = torch.flatten(x,start_dim=1).numpy()
#        ICA_set.append(ICA.fit_transform(x))
#        labels.append(t.numpy())
#    #Change to array
#    ICA_set = np.asarray(ICA_set)
#    ICA_set = np.concatenate(ICA_set).astype(None)
#    labels = np.asarray(labels)
#    labels = np.concatenate(labels).astype(None)
#    return (torch.from_numpy(ICA_set),torch.from_numpy(labels))
#
#def LDA_dataset(data_loader,numComponents):
#    LDA = LinearDiscriminantAnalysis()
#    LDA_set = []
#    labels = []
#    for idx, (x,t) in enumerate(Bar(data_loader)):
#        x = torch.flatten(x,start_dim=1).numpy()
#        LDA_set.append(LDA.fit(x,t).transform(x))
#        labels.append(t.numpy())
#    LDA_set = np.asarray(LDA_set)
#    LDA_set = np.concatenate(LDA_set).astype(None)
#    labels = np.asarray(labels)
#    labels = np.concatenate(labels).astype(None)
#    return (torch.from_numpy(LDA_set),torch.from_numpy(labels))

#train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size)
#test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size)

##Perform PCA on training and test data
#print('Prepping data with dimensionality reduction')
#print('PCA on training data')
#PCA_train = Create_dataset(PCA_dataset(train_loader,numComps))
#print('PCA on testing data')
#PCA_test = Create_dataset(PCA_dataset(test_loader,numComps))
##Perform PCA on training and test data
#print('Kernel PCA on training data')
#ICA_train = Create_dataset(ICA_dataset(train_loader,numComps))
#print('Kernel PCA on testing data')
#ICA_test = Create_dataset(ICA_dataset(test_loader,numComps))
##Perform PCA on training and test data
#print('LDA on training data')
#LDA_train = Create_dataset(LDA_dataset(train_loader,numClasses-1))
#print('Kernel PCA on testing data')
#LDA_test = Create_dataset(LDA_dataset(test_loader,numClasses-1))



