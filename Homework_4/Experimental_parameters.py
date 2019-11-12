# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 18:40:47 2019

@author: jpeeples
"""

import torch.nn as nn

Units = {1: 10, 2: 25, 3: 50}
num_classes = 10
in_features = 39

#Create different networks
#Change number of units in input layer
class Net1(nn.Module):
    def __init__(self,in_features=in_features):
        super(Net1,self).__init__()
        
        self.features = nn.Sequential(nn.Linear(in_features,Units[1]),
                                      nn.ReLU())
        self.classifier = nn.Linear(self.features[-2].out_features,num_classes)
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x
        
class Net2(nn.Module):
    def __init__(self,in_features=in_features):
        super(Net2,self).__init__()
        
        self.features = nn.Sequential(nn.Linear(in_features,Units[2]),
                                      nn.ReLU())
        self.classifier = nn.Linear(self.features[-2].out_features,num_classes) 
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x 
       
class Net3(nn.Module):
    def __init__(self,in_features=in_features):
        super(Net3,self).__init__()
        
        self.features = nn.Sequential(nn.Linear(in_features,Units[3]),
                                      nn.ReLU())
        self.classifier = nn.Linear(self.features[-2].out_features,num_classes)
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x 
        
#Change number of units in input layer and single hidden layer
class Net4(nn.Module):
    def __init__(self,in_features=in_features):
        super(Net4,self).__init__()
        
        self.features = nn.Sequential(nn.Linear(in_features,Units[1]),
                                      nn.ReLU(),
                                      nn.Linear(Units[1],Units[1]),
                                      nn.ReLU())
        self.classifier = nn.Linear(self.features[-2].out_features,num_classes)
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x 
        
class Net5(nn.Module):
    def __init__(self,in_features=in_features):
        super(Net5,self).__init__()
        
        self.features = nn.Sequential(nn.Linear(in_features,Units[2]),
                                      nn.ReLU(),
                                      nn.Linear(Units[2],Units[2]),
                                      nn.ReLU())
        self.classifier = nn.Linear(self.features[-2].out_features,num_classes) 
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x 
                
class Net6(nn.Module):
    def __init__(self,in_features=in_features):
        super(Net6,self).__init__()
        
        self.features = nn.Sequential(nn.Linear(in_features,Units[3]),
                                      nn.ReLU(),
                                      nn.Linear(Units[3],Units[3]),
                                      nn.ReLU())
        self.classifier = nn.Linear(self.features[-2].out_features,num_classes)
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x 
                
#Change number of units in input layer and two hidden layers
class Net7(nn.Module):
    def __init__(self,in_features=in_features):
        super(Net7,self).__init__()
        
        self.features = nn.Sequential(nn.Linear(in_features,Units[1]),
                                      nn.ReLU(),
                                      nn.Linear(Units[1],Units[1]),
                                      nn.ReLU(),
                                      nn.Linear(Units[1],Units[1]),
                                      nn.ReLU())
        self.classifier = nn.Linear(self.features[-2].out_features,num_classes)
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x 
                
class Net8(nn.Module):
    def __init__(self,in_features=in_features):
        super(Net8,self).__init__()
        
        self.features = nn.Sequential(nn.Linear(in_features,Units[2]),
                                      nn.ReLU(),
                                      nn.Linear(Units[2],Units[2]),
                                      nn.ReLU(),
                                      nn.Linear(Units[2],Units[2]),
                                      nn.ReLU())
        self.classifier = nn.Linear(self.features[-2].out_features,num_classes) 
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x 
                
class Net9(nn.Module):
    def __init__(self,in_features=in_features):
        super(Net9,self).__init__()
        
        self.features = nn.Sequential(nn.Linear(in_features,Units[3]),
                                      nn.ReLU(),
                                      nn.Linear(Units[3],Units[3]),
                                      nn.ReLU(),
                                      nn.Linear(Units[3],Units[3]),
                                      nn.ReLU())
        self.classifier = nn.Linear(self.features[-2].out_features,num_classes)
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x 
 
Networks = {1: Net1, 2: Net2, 3: Net3, 4: Net4, 5: Net5, 6: Net6,
            7: Net7, 8: Net8, 9: Net9}                     
#Networks = {1: Net1}           
        
        
           