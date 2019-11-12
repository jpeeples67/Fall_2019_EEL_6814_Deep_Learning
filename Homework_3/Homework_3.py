# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 20:56:32 2019

@author: Joshua Peeples
"""
import numpy as np
from Generate_data import Generate_training_data
import torch
from Homework_3_functions import train_model,view_results,TDNN_7,TDNN_20,RNN

#Loss to use, either 'MSE' or 'CE'
criterion = 'CE'
epochs = 100 #10000
progress = 10 #Print every progess # of epochs
lr_rate = 1e-3 #-e6
alpha = .9 #momentum

#Class names
class_names = ['Out of Grammar','In Grammar']
#Import test data
test_data = np.genfromtxt('hmw3test.csv',delimiter=',')

#Get labels and remove from test data
test_labels = test_data[:,-1]
test_data = test_data[:,:-1] 
sequence_len = test_data.shape[1]

# hyperparameters
seq_len = 80
input_size = 1   # one-hot size
num_layers = 1   # one-layer rnn
num_outputs = 1  # predicting 5 distinct character
hidden_size = 2  # output from the RNN

#Generate training set
training_data,training_labels = Generate_training_data()
#Pad data to be length 80
training_lengths = [len(sequence) for sequence in training_data]
# create an empty matrix with padding tokens
longest_sent = sequence_len
batch_size = len(training_data)
padded_training = np.ones((batch_size, longest_sent)) * 0# copy over the actual sequences
for i, x_len in enumerate(training_lengths):
  sequence = training_data[i]
  padded_training[i, 0:x_len] = sequence[:x_len]
        
#Change 0s to -1        
test_data[test_data==0] = -1 

#Convert to tensor values
padded_training = torch.FloatTensor(padded_training)
padded_training = padded_training.unsqueeze(1)
training_labels = torch.FloatTensor(training_labels)
#training_labels = training_labels.unsqueeze(1)
test_data = torch.FloatTensor(test_data)
test_data = test_data.unsqueeze(1)
test_labels = torch.FloatTensor(test_labels) 

##Train and get results for TDNN_7, TDNN_20 and RNN
#Train model
print("Starting TDNN_7")
model_1 = TDNN_7(sequence_len)
best_model_wts,Error_track = train_model(model_1,epochs,criterion,lr_rate,
                                         alpha,padded_training,training_labels,
                                         progress)
#Take best model and compute confusion matrix for test data
view_results(Error_track,model_1,best_model_wts,'TDNN with Window Size 7',
             test_data,test_labels,'TDNN with Window Size 7 Test Data',
                 class_names)

print("Starting TDNN_20")
#Train model
model_2 = TDNN_20(sequence_len)
best_model_wts_2,Error_track_2 = train_model(model_2,epochs,criterion,lr_rate,
                                         alpha,padded_training,training_labels,
                                         progress)
#Take best model and compute confusion matrix for test data
view_results(Error_track_2,model_2,best_model_wts_2,'TDNN with Window Size 20',
             test_data,test_labels,'TDNN with Window Size 20 Test Data',
                 class_names)

print("Starting TDNN_20")
#Train model
model_3 = RNN(seq_len, num_outputs, input_size, hidden_size, num_layers)
best_model_wts_3,Error_track_3 = train_model(model_3,epochs,criterion,lr_rate,
                                         alpha,padded_training,training_labels,
                                         progress)
#Take best model and compute confusion matrix for test data
view_results(Error_track_3,model_3,best_model_wts_3,'RNN',
             test_data,test_labels,'RNN Test Data',
                 class_names)     
        