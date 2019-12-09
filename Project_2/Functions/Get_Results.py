# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:44:54 2019

@author: jpeeples
"""
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from Functions.Confusion_mats import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


def SAE_results(directory,model,best_model_wts,train_error_history,
                val_error_history,time_elapsed,best_loss,bottleneck_size):
    
    #Create directory to save results
    save_dir = os.path.join(directory,'Bottleneck_Size_'+str(bottleneck_size))
    
    #Create directory 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    #Generate learning curve and save
    fig1 = plt.figure()
    plt.plot(train_error_history)
    plt.plot(val_error_history)
    plt.suptitle('Learning Curve for Bottleneck Size of {}'.format(bottleneck_size))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training Loss','Validation Loss'], loc='upper right')
    plt.show()
    fig1.savefig((save_dir + '/Learning Curve.png'), dpi=fig1.dpi)
    plt.close()
    
    #Save model and weights
    torch.save(model, (save_dir + '/Model.pt'))
    torch.save(best_model_wts,(save_dir+'/Best_Weights.pt'))
    
    #Count the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    #Write elapsed time, best validation loss, and number of parameters to text file
    with open((save_dir + '/Computation_time_seconds.txt'), "w") as output:
        output.write(str(np.round(time_elapsed,2)) + ' seconds')
        
    with open((save_dir + '/Computation_time.txt'), "w") as output:
        output.write(str(np.round(time_elapsed // 60,2)) + 'm and ' + str(np.round(time_elapsed % 60,2)) + 's')
        
    with open((save_dir + '/Best_Validaton_Loss.txt'), "w") as output:
        output.write(np.format_float_scientific(best_loss,precision=2))
        
    with open((save_dir + '/Number_of_Parameters.txt'), "w") as output:
        output.write(str(num_params))
        
def Classifier_results(directory,model,best_model_wts,train_error_history,
                val_error_history,time_elapsed,best_acc,GT,predictions,
                bottleneck_size=None,MLP=False):
    #Set class names
    class_names = np.array(['T-shirt/top','Trouser','Pullover','Dress',
                        'Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot'])
    
    #Create directory to save results, change filename if MLP or CNN
    if(MLP):
        save_dir = os.path.join(directory,'MLP/Bottleneck_Size_'+str(bottleneck_size))
        cm_title = 'Test Confusion Matrix for MLP classifier with Bottleneck Size of {}'.format(bottleneck_size)
    else:
        save_dir = os.path.join(directory,'CNN')
        cm_title = 'Test Confusion Matrix for CNN classifier'
    #Create directory 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    #Generate learning curve and save
    fig1 = plt.figure()
    plt.plot(train_error_history)
    plt.plot(val_error_history)
    if(MLP):
        plt.suptitle('Learning Curve for MLP classifier with Bottleneck Size of {}'.format(bottleneck_size))
    else:
        plt.suptitle('Learning Curve for CNN classifier')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training Loss','Validation Loss'], loc='upper right')
    plt.show()
    fig1.savefig((save_dir + '/Learning Curve.png'), dpi=fig1.dpi)
    plt.close()
    
    #Create confusion matrix and compute accuracy
    test_cm = confusion_matrix(GT,predictions)
    test_acc = 100 * sum(np.diagonal(test_cm)) / sum(sum(test_cm))
    
    #visualize
    np.set_printoptions(precision=2)
    fig2 = plt.figure()
    plot_confusion_matrix(test_cm, classes=class_names, title= cm_title)
    fig2.savefig((save_dir + '/Test Confusion Matrix.png'), dpi=fig2.dpi)
    plt.close()
    
    #Save model and weights
    torch.save(model, (save_dir + '/Model.pt'))
    torch.save(best_model_wts,(save_dir+'/Best_Weights.pt'))
    
    #Count the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    #Write elapsed time, best validation/test accuracy, and number of parameters to text file
    with open((save_dir + '/Computation_time_seconds.txt'), "w") as output:
        output.write(str(np.round(time_elapsed,2)) + ' seconds')
        
    with open((save_dir + '/Computation_time.txt'), "w") as output:
        output.write(str(np.round(time_elapsed // 60,2)) + 'm and ' + str(np.round(time_elapsed % 60,2)) + 's')
        
    with open((save_dir + '/Best_Validaton_Acc.txt'), "w") as output:
        output.write(str(np.round(best_acc.cpu(),2)) + '%')
        
    with open((save_dir + '/Best_Test_Acc.txt'), "w") as output:
        output.write(str(np.round(test_acc,2)) + '%')
        
    with open((save_dir + '/Number_of_Parameters.txt'), "w") as output:
        output.write(str(num_params))
        
def Classifier_results_ITL(directory,model,best_model_wts,train_error_history,
                val_error_history,time_elapsed,best_acc,GT,predictions,
                bottleneck_size,QMI=False):
    #Set class names
    class_names = np.array(['T-shirt/top','Trouser','Pullover','Dress',
                        'Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot'])
    
    #Create directory to save results, change filename if QMI or XEnt
    if(QMI):
        save_dir = os.path.join(directory,'QMI_MLP/Bottleneck_Size_'+str(bottleneck_size))
        cm_title = 'Test Confusion Matrix for QMI MLP classifier with Bottleneck Size of {}'.format(bottleneck_size)
    else:
        save_dir = os.path.join(directory,'XEnt_MLP/Bottleneck_Size_'+str(bottleneck_size))
        cm_title = 'Test Confusion Matrix for XEnt MLP classifier with Bottleneck Size of {}'.format(bottleneck_size)
    #Create directory 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    #Generate learning curve and save
    fig1 = plt.figure()
    plt.plot(train_error_history)
    plt.plot(val_error_history)
    if(QMI):
        plt.suptitle('Learning Curve for QMI MLP classifier with Bottleneck Size of {}'.format(bottleneck_size))
    else:
        plt.suptitle('Learning Curve XEnt MLP classifier with Bottleneck Size of {}'.format(bottleneck_size))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training Loss','Validation Loss'], loc='upper right')
    plt.show()
    fig1.savefig((save_dir + '/Learning Curve.png'), dpi=fig1.dpi)
    plt.close()
    
    #Create confusion matrix and compute accuracy
    test_cm = confusion_matrix(GT,predictions)
    test_acc = 100 * sum(np.diagonal(test_cm)) / sum(sum(test_cm))
    
    #visualize
    np.set_printoptions(precision=2)
    fig2 = plt.figure()
    plot_confusion_matrix(test_cm, classes=class_names, title= cm_title)
    fig2.savefig((save_dir + '/Test Confusion Matrix.png'), dpi=fig2.dpi)
    plt.close()
    
    #Save model and weights
    torch.save(model, (save_dir + '/Model.pt'))
    torch.save(best_model_wts,(save_dir+'/Best_Weights.pt'))
    
    #Count the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    #Write elapsed time, best validation/test accuracy, and number of parameters to text file
    with open((save_dir + '/Computation_time_seconds.txt'), "w") as output:
        output.write(str(np.round(time_elapsed,2)) + ' seconds')
        
    with open((save_dir + '/Computation_time.txt'), "w") as output:
        output.write(str(np.round(time_elapsed // 60,2)) + 'm and ' + str(np.round(time_elapsed % 60,2)) + 's')
        
    with open((save_dir + '/Best_Validaton_Acc.txt'), "w") as output:
        output.write(str(np.round(best_acc.cpu(),2)) + '%')
        
    with open((save_dir + '/Best_Test_Acc.txt'), "w") as output:
        output.write(str(np.round(test_acc,2)) + '%')
        
    with open((save_dir + '/Number_of_Parameters.txt'), "w") as output:
        output.write(str(num_params))