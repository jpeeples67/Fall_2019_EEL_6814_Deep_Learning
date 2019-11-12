# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 18:06:02 2019

@author: Joshua Peeples
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import copy

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def train_model(model,epochs,criterion,lr_rate,alpha,training_data,labels,
                print_epochs):
    loss_fxn = {"MSE": nn.MSELoss(), "CE": nn.CrossEntropyLoss()}
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate, momentum=alpha)
    model.train()
    Error_track = np.zeros(epochs)
    error_check = np.inf
    for epoch in range(0,epochs):
        
        #Forward pass through model
        output = model(training_data)
        
        #Compute loss
        error = loss_fxn[criterion](output,labels)

        #Keep track of Error
        Error_track[epoch] = error.detach().numpy()
        
        #Update weights with backprop
        error.backward()
        optimizer.step()
        
        #Save best model
        if error < error_check:
            error_check = error
            best_model_wts = copy.deepcopy(model.state_dict())
        if epoch % print_epochs == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(
                epoch, error.item()))
    print()        
    return best_model_wts,Error_track

def view_results(Error_track,trained_model,best_model_wts,training_data,training_labels,
                 training_name,test_data,test_labels,test_name,class_names):
    #Plot learning curve
    plt.figure()
    plt.plot(Error_track)  
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    trained_model.load_state_dict(best_model_wts) 
    trained_model.eval()
    #Add softmax on output
    trained_model = nn.Sequential(trained_model,nn.Softmax(dim=None))
    #Pass through training/test data
    training_output = trained_model(training_data).argmax(dim=1)
    testing_output = trained_model(test_data).argmax(dim=1)
    #Create confusion matrix
    plot_confusion_matrix(training_labels.numpy(),training_output.numpy(),
                          class_names,title = training_name + ' Training Confusion Matrix')
    plot_confusion_matrix(test_labels.numpy(),testing_output.numpy(),
                          class_names,title = test_name + ' Testing Confusion Matrix')
    

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        #print("Normalized confusion matrix")
#    else:
#        #print('Confusion matrix, without normalization')
#
#    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax