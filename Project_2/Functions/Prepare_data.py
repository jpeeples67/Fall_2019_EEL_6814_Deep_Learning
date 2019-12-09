# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:13:58 2019

@author: jpeeples
"""
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import pdb
import numpy as np
import scipy.stats as scs

#Create training, validation and test splits for fashion mnist data
batch_size = {'train': 512, 'val': 2048, 'test': 2048}
split = .1
np.random.seed(0)

class Create_dataset(Dataset):
    def __init__(self,data_tuple,Gauss=False,transform=None):
        self.data = data_tuple[0]
        #One hot encode targets
        num_classes = len(torch.unique(data_tuple[1]))
        self.targets = torch.eye(num_classes)[data_tuple[1]]
        #Use Gaussian distribution with location num_class D gaussian
        if Gauss:
            for i in range(0,len(data_tuple[1])):
                m =  torch.distributions.multivariate_normal.MultivariateNormal(self.targets[i],.005*torch.eye(num_classes))
                self.targets[i,:] = m.sample()
        self.transform=transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img = self.data[index].type(torch.FloatTensor)

        label = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label
    
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

#Compute bandwith using silverman's rule and Santos 
# N is the number of samples, 
N = train_dataset.data.shape[0]
d = train_dataset.data.shape[1]*train_dataset.data.shape[2]
L = len(torch.unique(train_dataset.targets))
#bw_silverman = np.std(train_dataset.data.numpy()/255)*(4*(1/N)*(1/((2*d)+1)))**(1/d+4)
bw_silverman = np.round(300*.9*min(np.std(train_dataset.data.numpy()/255),scs.iqr(train_dataset.data.numpy()/255)/1.34)*(N**(-1/5)),2)
bw_santos = np.round(30*25*np.sqrt(L/N),2)
bandwidths = {'Silverman': bw_silverman, 'Santos': bw_santos}

#One hot dataset and Gaussian encoded dataset
train_one_hot_dataset = Create_dataset((train_dataset.data,train_dataset.targets))
test_one_hot_dataset = Create_dataset((test_dataset.data,test_dataset.targets))
train_one_hot_Gauss_dataset = Create_dataset((train_dataset.data,train_dataset.targets),Gauss=True)
test_one_hot_Gauss_dataset = Create_dataset((test_dataset.data,test_dataset.targets),Gauss=True)

# Create training and validation dataloaders, one hot encoded and Gaussian sampling
dataloaders_dict = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size['train'],
                                           sampler=train_sampler, shuffle=False,num_workers=0),
                    'val': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size['val'],
                                           sampler=valid_sampler, shuffle=False,num_workers=0),
                    'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size['test'],
                                           shuffle=True,num_workers=0)}
                    
dataloaders_dict_one_hot = {'train': torch.utils.data.DataLoader(train_one_hot_dataset, batch_size=batch_size['train'],
                                           sampler=train_sampler, shuffle=False,num_workers=0),
                    'val': torch.utils.data.DataLoader(train_one_hot_dataset, batch_size=batch_size['val'],
                                           sampler=valid_sampler, shuffle=False,num_workers=0),
                    'test': torch.utils.data.DataLoader(test_one_hot_dataset, batch_size=batch_size['test'],
                                           shuffle=True,num_workers=0)}
                    
dataloaders_dict_one_hot_Gauss = {'train': torch.utils.data.DataLoader(train_one_hot_Gauss_dataset, batch_size=batch_size['train'],
                                           sampler=train_sampler, shuffle=False,num_workers=0),
                    'val': torch.utils.data.DataLoader(train_one_hot_Gauss_dataset, batch_size=batch_size['val'],
                                           sampler=valid_sampler, shuffle=False,num_workers=0),
                    'test': torch.utils.data.DataLoader(test_one_hot_Gauss_dataset, batch_size=batch_size['test'],
                                           shuffle=True,num_workers=0)}

encoded_dicts = {'One_Hot': dataloaders_dict_one_hot, 'One_Hot_Gaussian': dataloaders_dict_one_hot_Gauss}
#encoded_dicts = {'One_Hot_Gaussian': dataloaders_dict_one_hot_Gauss}