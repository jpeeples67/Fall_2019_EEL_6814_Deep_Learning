# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:36:13 2019

@author: jpeeples
"""
import torch.nn as nn
import torch
#Create Stacked autoencoder (SAE)
class SAE(nn.Module):
    def __init__(self,bottleneck=100):
        super(SAE,self).__init__()
        
        self.encoder = nn.Sequential(nn.Linear(784,500),nn.ReLU(),
                                     nn.Linear(500,200),nn.ReLU(),
                                     nn.Linear(200,bottleneck),nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(bottleneck,200),nn.ReLU(),
                                     nn.Linear(200,500),nn.ReLU(),
                                     nn.Linear(500,784),nn.Sigmoid())     
    def forward(self,x):
        #Flatten the input
        x = torch.flatten(x,start_dim=1)
        #Encoder
        x = self.encoder(x)
        #Decoder
        x = self.decoder(x)
        return x
#Create MLP classifier that uses encoder from SAE
class MLP(nn.Module):
    def __init__(self,encoder,num_classes=10):
        super(MLP,self).__init__()
        
        self.encoder = encoder
        #Fix the encoder portion of the MLP so the weights do not update
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(encoder[-2].out_features,num_classes) 
        
    def forward(self,x):
        #Flatten the input
        x = torch.flatten(x,start_dim=1)
        #Encoder
        x = self.encoder(x)
        #MLP
        x = self.fc(x)
        return x  
    
#Create CNN classifier for comparison, based on 2 conv + pooling net
#for mnist: 
#https://github.com/pytorch/examples/blob/master/mnist/main.py
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2,stride=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
#Create MLP classifier that uses encoder from SAE
class MLP_ITL(nn.Module):
    def __init__(self,bottleneck=100,num_classes=10):
        super(MLP_ITL,self).__init__()
        self.encoder = nn.Sequential(nn.Linear(784,500),nn.ReLU(),
                                     nn.Linear(500,200),nn.ReLU(),
                                     nn.Linear(200,bottleneck),nn.ReLU())
        self.fc = nn.Linear(self.encoder[-2].out_features,num_classes) 
        
    def forward(self,x):
        #Flatten the input
        x = torch.flatten(x,start_dim=1)
        #Encoder
        x = self.encoder(x)
        #MLP
        x = self.fc(x)
        return x  