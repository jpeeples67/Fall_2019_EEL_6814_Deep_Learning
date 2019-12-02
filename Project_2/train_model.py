# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 16:51:12 2019

@author: Joshua Peeples
"""
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import time
import copy
import pdb
from barbar import Bar
from Loss_functions import MEE, MCE

def train(model, dataloader, criterion, optimizer, device='cpu',num_epochs=25,bw=.1):
    since = time.time()

    train_error_history = []
    MSE_loss = nn.MSELoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Train model
        model.double()
        model.train()  # Set model to training mode 
        running_loss = 0.0
        
        # Iterate over data.
        for inputs, labels, index in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            index = index.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)
            if criterion == 'MSE':
                loss = MSE_loss(outputs, labels)
            else: #MEE
                loss = MCE(outputs,labels,bw)
            # backward pass
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)

        #Record training epoch loss for learning curve
        epoch_loss = running_loss / len(dataloader.dataset)
        
        train_error_history.append(epoch_loss)

        print('Training Loss: {:.4f} '.format(epoch_loss))

        # deep copy the model
        if epoch_loss < best_loss:
            best_epoch = epoch
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
    print()

  
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best train loss: {:4f}'.format(best_loss))

    # load best model weights
    #Fit model on hold out test set
    model.load_state_dict(best_model_wts)
    
    #Get predicted time series for best model
    actual_series,predicted_series = predict(dataloader,model,device)
    #pdb.set_trace()
    return (model, best_model_wts, train_error_history,best_epoch,
            actual_series,predicted_series)

def predict(dataloader,model,device):
    #Initialize and accumalate ground truth and predictions
    GT = np.array(0)
    Predictions = np.array(0)

    model.eval()
        # Iterate over data.
    with torch.no_grad():
        #for idx, (inputs, labels,index) in Bar(enumerate(dataloader)):
        for inputs, labels, index in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            index = index.to(device)
    
            # forward
            outputs = model(inputs)
    
            #If validation, accumulate labels for confusion matrix
            GT = np.concatenate((GT,labels.detach().cpu().numpy()),axis=None)
            Predictions = np.concatenate((Predictions,outputs.detach().cpu().numpy()),axis=None)

    return GT[1:],Predictions[1:]