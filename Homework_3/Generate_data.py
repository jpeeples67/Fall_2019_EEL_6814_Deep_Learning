# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 20:56:32 2019

@author: Joshua Peeples
"""
import numpy as np

def Generate_training_data():
    #Generate training set
    min_chars = 20
    max_chars = 60
    num_samples = 1000
    consecutive = 10
    var_length = np.arange(min_chars,max_chars+1)
    
    data_list = []
    labels = np.zeros(num_samples)
    sample_count = 0
    
    for sample in range(0,num_samples):
        #Generate length randomly
        length = var_length[np.random.randint(len(var_length))]
        #Generate samples for in grammar (label 1)
        if(sample_count<int(num_samples/2)):
            data = np.zeros(length)-1
            #Make sure it is in grammar, change some -1s to ones in range
            #of consecutive -1s
            for char in range(0,length,consecutive+1):
                temp_data = data[char:(char+consecutive+1)]
                #Randomly change non-consecutive values to be 1
                #If at the end of the window, change number of values
                if(len(temp_data)<(consecutive)):
                    #If only a single value is left, don't make any changes
                    if(len(temp_data)==1):
                        num_changes = 0
                    num_changes = int(np.random.randint(0,high=np.ceil(len(temp_data)/2)))
                else:
                    num_changes = np.random.randint(1,high=(consecutive/2)+1)
                for change in range(0,(2*num_changes),2):
                    temp_data[change] = 1
                data[char:(char+consecutive+1)] = temp_data
            #Save data and label in tuple
            data_list.append(data)
            labels[sample] = 1
        #Generate smaples for out of grammar (label -1)
        else:
            data = np.zeros(length)+1
            #Make sure it is in grammar, change some 1s to ones in range
            #of consecutive 1s
            for char in range(0,length,consecutive+1):
                temp_data = data[char:(char+consecutive+1)]
                #Randomly change non-consecutive values to be 1
                #If at the end of the window, change number of values
                if(len(temp_data)<(consecutive)):
                    #If only a single value is left, don't make any changes
                    if(len(temp_data)==1):
                        num_changes = 0
                    num_changes = int(np.random.randint(0,high=np.ceil(len(temp_data)/2)))
                else:
                    num_changes = np.random.randint(1,high=(consecutive/2)+1)
                for change in range(0,(2*num_changes),2):
                    temp_data[change] = -1
                data[char:(char+consecutive+1)] = temp_data
                    #Save data and label in tuple
            data_list.append(data)
            labels[sample] = 0
        #Increment sample count    
        sample_count+=1
    return data_list,labels
        
        
        
        
        
        