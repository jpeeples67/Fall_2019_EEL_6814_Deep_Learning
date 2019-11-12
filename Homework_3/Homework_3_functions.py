# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:07:03 2019

@author: Joshua Peeples
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import copy

from sklearn.metrics import confusion_matrix

def train_model(model,epochs,criterion,lr_rate,alpha,training_data,labels,
                print_epochs):
    loss_fxn = {"MSE": nn.MSELoss(), "CE": nn.BCELoss()}
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

def view_results(Error_track,trained_model,best_model_wts,Training_name,
                 test_data,test_labels,test_name,class_names):
    #Plot learning curve
    plt.figure()
    plt.plot(Error_track)  
    plt.title(Training_name + ' Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    trained_model.load_state_dict(best_model_wts) 
    trained_model.eval()
    #Add softmax on output
    #trained_model = nn.Sequential(trained_model,nn.Sigmoid())
    #Pass through training/test data
    testing_output = trained_model(test_data)
    #Threshold output
    testing_output[testing_output>.5] = 1
    testing_output[testing_output<=.5] = 0
    plot_confusion_matrix(test_labels.detach().numpy().astype(int),
                          testing_output.detach().numpy().astype(int),
                          class_names,title = test_name + ' Confusion Matrix')
    

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
    #classes = classes[np.array([0,1])]

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
#Generate models
class TDNN_7(nn.Module):
    def __init__(self,sequence_len):
        super(TDNN_7, self).__init__()
        self.time_delay = nn.Conv1d(1,1,7,bias=True)
        self.fc_1 = nn.Linear(sequence_len-self.time_delay.kernel_size[0]+1,2)
        self.classifier = nn.Linear(self.fc_1.out_features,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.time_delay(x))
        x = torch.flatten(x,1)
        x = self.relu(self.fc_1(x))
        x = self.classifier(x)
        x = self.sigmoid(torch.flatten(x))
        return x
    
class TDNN_20(nn.Module):
    def __init__(self,sequence_len):
        super(TDNN_20, self).__init__()
        self.time_delay = nn.Conv1d(1,1,20,bias=True)
        self.fc_1 = nn.Linear(sequence_len-self.time_delay.kernel_size[0]+1,2)
        self.classifier = nn.Linear(self.fc_1.out_features,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.time_delay(x))
        x = self.relu(self.fc_1(x))
        x = self.classifier(x)
        x = self.sigmoid(torch.flatten(x))
        return x

class RNN(nn.Module):
    """
    The RNN model will be a RNN followed by a linear layer,
    i.e. a fully-connected layer
    """
    def __init__(self, sequence_len, num_outputs, input_size, hidden_size, num_layers):
        super().__init__()
        self.seq_len = sequence_len
        self.num_layers = num_layers
        self.input_size = input_size
        self.outputs = num_outputs
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first = True)
        self.linear = nn.Linear(self.seq_len*self.hidden_size, self.outputs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # assuming batch_first = True for RNN cells
        batch_size = x.size(0)
        hidden = self._init_hidden(batch_size)
        x = x.view(batch_size, self.seq_len, self.input_size)

        # apart from the output, rnn also gives us the hidden
        # cell, this gives us the opportunity to pass it to
        # the next cell if needed; we won't be needing it here
        # because the nn.RNN already computed all the time steps
        # for us. rnn_out will of size [batch_size, seq_len, hidden_size]
        rnn_out, _ = self.rnn(x, hidden)
        linear_out = self.sigmoid(torch.flatten(self.linear(torch.flatten(rnn_out,1))))
        return linear_out

    def _init_hidden(self, batch_size):
        """
        Initialize hidden cell states, assuming
        batch_first = True for RNN cells
        """
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)