# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 23:53:17 2019

@author: jpeeples
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler
from Visualizations import Val_Test_CM,Summary_Val_Test_CM,Learning_Curve
import torch
import time
import copy
from barbar import Bar
import os
import pdb

def train_model(model,dataset,batch_size,criterion,optimizer,num_epochs=100,k=3,data_folder=None):
    since = time.time()

    #Get dataset and divide training into folds, use same random seed so each model is trained and tested on the same data for fair comparision
    skf = StratifiedKFold(n_splits=k,shuffle=True,random_state=3)
    train_dataset = dataset['train']
    y = train_dataset.targets
    num_classes = len(np.unique(y))
    val_cm_stats = np.zeros((num_classes,num_classes,k))
    test_cm_stats = np.zeros((num_classes,num_classes,k))
    val_acc_stats = np.zeros(k)
    test_acc_stats = np.zeros(k)
    fold_count = 1
    for train_index, val_index in skf.split(train_dataset.files, y):
        Results_folder = data_folder + 'Fold_' + str(fold_count) + '/'
        if not os.path.exists(Results_folder):
            os.makedirs(Results_folder)
        temp_model = model
        train_sampler = SubsetRandomSampler(train_index)
        valid_sampler = SubsetRandomSampler(val_index)
        dataloaders_dict = {'train': torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size['train'],
                                                   sampler=train_sampler, shuffle=False,num_workers=0),
                            'val': torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size['val'],
                                                   sampler=valid_sampler, shuffle=False,num_workers=0),
                            'test': torch.utils.data.DataLoader(dataset['test'], batch_size=batch_size['val'],
                                                   shuffle=True,num_workers=0) }
        #Train model on fold and validate/test
        (model_ft, best_weights,val_acc,val_error,train_acc,train_error,
             best_epoch) = Cross_Val(temp_model,dataloaders_dict,num_epochs,optimizer,criterion)
        Val_GT, Val_Predictions = test_model(dataloaders_dict['val'],model_ft,device='cpu')
        Test_GT, Test_Predictions = test_model(dataloaders_dict['test'],model_ft,device='cpu')
        Learning_Curve(val_acc,val_error,train_error,train_acc,best_epoch,Results_folder)
        val_cm, test_cm, fold_val_acc, fold_test_acc = Val_Test_CM(Val_GT,Val_Predictions,Test_GT,Test_Predictions,Results_folder)
        val_cm_stats[:, :, fold_count - 1] = val_cm
        test_cm_stats[:, :, fold_count - 1] = test_cm
        val_acc_stats[fold_count-1] = fold_val_acc
        test_acc_stats[fold_count-1] = fold_test_acc
        fold_count += 1
    
    Summary_Val_Test_CM(val_cm_stats,test_cm_stats,val_acc_stats,test_acc_stats,data_folder)
    time_elapsed = time.time() - since
    print('Training, Validation and Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
def Cross_Val(model,dataloaders,num_epochs,optimizer,criterion,device='cpu'):
    val_acc_history = np.zeros(num_epochs)
    train_acc_history =  np.zeros(num_epochs)
    train_error_history =  np.zeros(num_epochs)
    val_error_history =  np.zeros(num_epochs)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
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
            for idx, (inputs,labels) in enumerate(Bar(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
    
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs.float())
                    loss = criterion(outputs, labels.long())
    
                    _, preds = torch.max(outputs, 1)
    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
    
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data.long())
    
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            if phase == 'train':
                train_error_history[epoch] = epoch_loss
                train_acc_history[epoch] = epoch_acc
            print()
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
            if phase == 'val':
                val_error_history[epoch] = epoch_loss
                val_acc_history[epoch] = epoch_acc
                
        print()
        
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # load best model weights
    #Fit model on hold out test set
    model.load_state_dict(best_model_wts)
    
    return (model, best_model_wts,val_acc_history, val_error_history, 
            train_acc_history, train_error_history,best_epoch)
    
def test_model(dataloader,model,device):
    #Initialize and accumalate ground truth, predictions, and image indices
    GT = np.array(0)
    Predictions = np.array(0)
    running_corrects = 0
    model.eval()
        # Iterate over data.
    with torch.no_grad():
        for idx, (inputs,labels) in enumerate(Bar(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            # forward
            outputs = model(inputs.float())
            _, preds = torch.max(outputs, 1)
    
            #If validation, accumulate labels for confusion matrix
            GT = np.concatenate((GT,labels.detach().cpu().numpy()),axis=None)
            Predictions = np.concatenate((Predictions,preds.detach().cpu().numpy()),axis=None)
            
            running_corrects += torch.sum(preds == labels.data.long())

    test_acc = running_corrects.double() / len(dataloader.dataset)
    print()
    print('Test Accuracy: {:4f}'.format(test_acc))
    
    return GT[1:],Predictions[1:]