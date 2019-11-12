# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 21:58:55 2019

@author: jpeeples
"""
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
from Confusion_mats import plot_confusion_matrix,plot_avg_confusion_matrix
from sklearn.metrics import confusion_matrix
import numpy as np

class_names = np.array(['T-shirt/top','Trouser','Pullover','Dress',
                        'Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot'])
def TSNE_Visual(data,labels,filename,name):
    
    features_embedded = TSNE(n_components=2,verbose=1).fit_transform(data)
    GT_train = labels
    fig6, ax6 = plt.subplots()
    colors = colormap.rainbow(np.linspace(0, 1, 10))
    for clothes in range (0, 10):
        x = features_embedded[np.where(GT_train==clothes),0]
        y = features_embedded[np.where(GT_train==clothes),1]
        
        plt.scatter(x, y, color = colors[clothes,:],label = class_names[clothes])
    ax6.legend(loc=1) 
    plt.title('TSNE Visualization of Training Data Features for ' + name)
    plt.show()
    
    fig6.savefig((filename + '/TSNE_training.png'), dpi=fig6.dpi)
    plt.close()   
    
def Val_Test_CM(Val_GT,Val_Predictions,Test_GT,Test_Predictions,folder):
    
    #Create the validation and test cm
    val_cm = confusion_matrix(Val_GT,Val_Predictions)
    test_cm = confusion_matrix(Test_GT,Test_Predictions)
    
    val_acc = 100 * sum(np.diagonal(val_cm)) / sum(sum(val_cm))
    test_acc = 100 * sum(np.diagonal(test_cm)) / sum(sum(test_cm))
    
    with open((folder + 'Validation Accuracy.txt'), "w") as output:
        output.write(str(val_acc))
        
    with open((folder + 'Test Accuracy.txt'), "w") as output:
        output.write(str(test_acc))    

    #Visualize Confusion Matrices
    np.set_printoptions(precision=2)
    fig4 = plt.figure()
    plot_confusion_matrix(val_cm, classes=class_names, title='Validation Confusion Matrix')
    fig4.savefig((folder + 'Validation Confusion Matrix.png'), dpi=fig4.dpi)
    plt.close()
    
    np.set_printoptions(precision=2)
    fig5 = plt.figure()
    plot_confusion_matrix(test_cm, classes=class_names, title='Test Confusion Matrix')
    fig5.savefig((folder + 'Test Confusion Matrix.png'), dpi=fig5.dpi)
    plt.close()
    
    return val_cm,test_cm,val_acc,test_acc

def Summary_Val_Test_CM(Val_cm,Test_cm,val_acc,test_acc,folder):
        
    np.set_printoptions(precision=2)
    fig6 = plt.figure(figsize=(11, 11))
    plot_avg_confusion_matrix(Val_cm, classes=class_names, title='Average Validation Confusion Matrix')
    fig6.savefig((folder + 'Average Validation Confusion Matrix.png'), dpi=fig6.dpi)
    plt.close()
    
    np.set_printoptions(precision=2)
    fig7 = plt.figure(figsize=(11,11))
    plot_avg_confusion_matrix(Test_cm, classes=class_names, title='Average Test Confusion Matrix')
    fig7.savefig((folder + 'Average Test Confusion Matrix.png'), dpi=fig7.dpi)
    plt.close()
    
    with open((folder + 'Validation Overall_Accuracy.txt'), "w") as output:
        output.write('Average accuracy: ' + str(np.mean(val_acc)) + ' Std: ' + str(np.std(val_acc)))

    with open((folder + 'Test Overall_Accuracy.txt'), "w") as output:
        output.write('Average accuracy: ' + str(np.mean(val_acc)) + ' Std: ' + str(np.std(val_acc)))
    
def Learning_Curve(Validation_Accuracy_track,Validation_Error_track,
                   Training_Error_track,Training_Accuracy_track,Best_epoch,folder):
    
    # visualize results
    fig2 = plt.figure()
    plt.plot(Training_Error_track)
    plt.plot(Validation_Error_track)
    # Mark best epoch and validation error
    plt.plot([Best_epoch], Validation_Error_track[Best_epoch], marker='x', markersize=3, color='red')
    plt.suptitle('Learning Curve for {} Epochs'.format(len(Training_Error_track)))
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.show()
    fig2.savefig((folder + 'Learning Curve.png'), dpi=fig2.dpi)
    plt.close()
    
        # visualize results
    fig3 = plt.figure()
    plt.plot(Training_Accuracy_track)
    plt.plot(Validation_Accuracy_track)
    # Mark best epoch and validation error
    plt.plot([Best_epoch], Validation_Accuracy_track[Best_epoch], marker='x', markersize=3, color='red')
    plt.suptitle('Accuracy for {} Epochs'.format(len(Training_Error_track)))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.show()
    fig3.savefig((folder + 'Accuracy Curve.png'), dpi=fig2.dpi)
    plt.close()
 
    