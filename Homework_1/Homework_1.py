# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:28:18 2019
Homework #1, Problem 2
@author: Joshua Peeples
"""
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import pdb
#Network parameters
lr_rate = 1e-1
num_units = 4
epochs = 2000 

#Create training data
training_data = np.array([[1,0],[0,1],[-1,0],[0,-1]
                          ,[0.5,0.5],[-0.5,0.5],[0.5,-0.5],[-0.5,-0.5]])
desired = np.array([[1,1,1,1,0,0,0,0]])
#Visualize training data
plt.figure
plt.scatter(training_data[:,0],training_data[:,1],c=desired[0])
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Training Data")
#plt.legend(labels=["Class 1","Class 2"])

#Generate initial weights and biases
# Add one for bias
pdb.set_trace()
training_data = np.concatenate((np.ones((training_data.shape[0],1)),training_data),axis=1)
num_feats = training_data.shape[1]
hidden_input_weights = np.random.rand(num_units,num_feats)
output_weights = np.random.rand(num_units,1)
Error_track = np.zeros([epochs])

#Activation function
def net(X):
    #Sigmoid
    Y = 1/(1+np.exp(-X))
    return Y
def dnet(X):
    Y = X*(1-X)
    return Y

for epoch in range(1,epochs+1):
    #Forward propogation
    #Pass through hidden layer
    hidden_layer_output = net(hidden_input_weights@(training_data.T))
    
    #Pass through output layer
    output = net((output_weights.T)@hidden_layer_output)
    
    #Compute loss
    Loss = .5*np.sum((desired-output)**2)
    Error_track[epoch-1] = Loss
    error = (desired-output)
    
    #Update weights
    #Output layer
#    pdb.set_trace()
    output_weights = output_weights+(lr_rate*(((error)*
                                (dnet(output)))@(hidden_layer_output.T))).T
    #Hidden layer
#    pdb.set_trace()
    local_error = (error.T)@(dnet(output))
    hidden_input_weights = hidden_input_weights+(lr_rate*(
        dnet(hidden_layer_output)@local_error@training_data))
    

plt.plot(Error_track)
    

    
    
    
    
    
    
    
    

