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
from Functions.Loss_functions import MEE, MCE

def train_SAE(model, dataloaders, optimizer, device='cpu',num_epochs=25):
    since = time.time()

    train_error_history = []
    val_error_history = []
    
    MSE_loss = nn.MSELoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    
    model = model.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode 
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0       
            # Iterate over data.
            for idx, (inputs, labels) in enumerate(Bar(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
    
                with torch.set_grad_enabled(phase=='train'):
                    
                    # forward pass
                    outputs = model(inputs)
                    loss = MSE_loss(outputs, torch.flatten(inputs,start_dim=1))
    
                    # backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            
            if phase == 'train':
                train_error_history.append(epoch_loss)

            print()
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_epoch = epoch+1
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                
            if phase == 'val':
                val_error_history.append(epoch_loss)
    print()

   
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation loss: {:4f} at Epoch {:.0f}'.format(best_loss,best_epoch))

    # load best model weights
    #Fit model on hold out test set
    model.load_state_dict(best_model_wts)
    
    return (model, best_model_wts, train_error_history,val_error_history,best_loss,time_elapsed)
    
def train_classifier(model, dataloaders, optimizer,criterion = 'CE', device='cpu',num_epochs=25,bw=.1):
    since = time.time()

    val_acc_history = []
    train_acc_history = []
    train_error_history = []
    val_error_history = []
    CE = nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    model = model.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode 
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
        
            # Iterate over data.
            for idx, (inputs, labels) in enumerate(Bar(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)

            # zero the parameter gradients
                optimizer.zero_grad()
    
                with torch.set_grad_enabled(phase=='train'):
                    
                    # forward pass
                    outputs = model(inputs)
                    if criterion == 'MSE': #Put MI for mutual information
                        loss = MEE(outputs, labels,bw)
                    elif criterion == 'MCE': #MCE
                        loss = MCE(outputs,labels,bw)
                    else:
                        loss = CE(outputs,labels)
                        
                    _, preds = torch.max(outputs, 1)
                    # backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            if phase == 'train':
                train_error_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)

            print()
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_epoch = epoch+1
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
            if phase == 'val':
                val_error_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
    print()

  
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation accuracy: {:4f} at Epoch {:.0f}'.format(best_acc,best_epoch))

    # load best model weights
    #Fit model on hold out test set
    model.load_state_dict(best_model_wts)
    
    #Get predictions for test set
    GT,predictions = predict(dataloaders['test'],model,device)
   
    return (model, best_model_wts, train_error_history,val_error_history,best_acc,
            time_elapsed,GT,predictions)

def predict(dataloader,model,device):
    #Initialize and accumalate ground truth and predictions
    GT = np.array(0)
    Predictions = np.array(0)
    running_corrects = 0
    model = model.to(device)
    model.eval()
        # Iterate over data.
    with torch.no_grad():
        #for idx, (inputs, labels,index) in Bar(enumerate(dataloader)):
        for idx, (inputs, labels) in enumerate(Bar(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs,1)
    
            #If validation, accumulate labels for confusion matrix
            GT = np.concatenate((GT,labels.detach().cpu().numpy()),axis=None)
            Predictions = np.concatenate((Predictions,preds.detach().cpu().numpy()),axis=None)
            running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / len(dataloader.dataset)
    print('Test Accuracy: {:4f}'.format(test_acc))           

    return GT[1:],Predictions[1:]


