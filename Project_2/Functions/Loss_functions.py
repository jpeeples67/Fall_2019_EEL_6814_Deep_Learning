# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 18:33:07 2019

@author: Joshua Peeples
"""

import torch
import numpy as np
import torch.nn.functional as nn
#Compute minimum error entropy with kernel bandwidth = bw
def MEE(outputs,targets,bw):
    
    #Compute error
    error = targets - outputs
    n = targets.size(0)
    
    #Compute pairwise distances of error
    #Source for pairwise distance calculation:
    # https://github.com/jiyanggao/Video-Person-ReID/blob/master/losses.py#L47-L89
    diff = torch.pow(error,2).sum(dim=1,keepdim=True).expand(n,n)
    diff = diff + diff.t()
    diff.addmm_(1,-2,error,error.t())
    diff = diff.clamp(min=1e-12)

    #Compute normalizing constant
    const = 1/(np.sqrt(2*np.pi)*bw)
    loss = torch.sum(const*torch.exp(-((diff))/(2*(bw**2))))
            
    #Normalize
    loss = -torch.log(loss/(n**2)) 
    
    return loss

#Compute maximum cross entropy with kernel bandwidth = bw
def MCE(outputs,targets,bw):
    
    #Get number of samples
    n = targets.size(0)
    
    #Compute error
    error = targets - outputs
    n = targets.size(0)
    
    #Compute pairwise distances of error
    #Source for pairwise distance calculation:
    # https://github.com/jiyanggao/Video-Person-ReID/blob/master/losses.py#L47-L89
    diff = torch.pow(error,2).sum(dim=1,keepdim=True).expand(n,n)
    diff = diff + diff.t()
    diff.addmm_(1,-2,error,error.t())
    diff = diff.clamp(min=1e-12).sqrt()

    #Compute normalizing constant
    const = 1/(np.sqrt(2*np.pi)*bw)
    loss = torch.sum(const*torch.exp(-((diff)**2)/(2*(bw**2))))
    
    #Normalize
    loss = -torch.log(loss/(n**2)) 
    
    return loss

#Compute maximum cross entropy with kernel bandwidth = bw
def QMI(outputs,targets,bw):
    
    #Get number of samples
    n = targets.size(0)
    
    #Compute error
    error = targets - outputs
    n = targets.size(0)
    
    #Compute pairwise distances of error
    #Source for pairwise distance calculation:
    # https://github.com/jiyanggao/Video-Person-ReID/blob/master/losses.py#L47-L89
    cross_info = torch.pow(error,2).sum(dim=1,keepdim=True).expand(n,n)
    cross_info = cross_info + cross_info.t()
    cross_info.addmm_(1,-2,error,error.t())
    cross_info = cross_info.clamp(min=1e-12).sqrt()

    #Compute pairwise distances of output
    #Source for pairwise distance calculation:
    # https://github.com/jiyanggao/Video-Person-ReID/blob/master/losses.py#L47-L89
    output_info = torch.pow(outputs,2).sum(dim=1,keepdim=True).expand(n,n)
    output_info = output_info + output_info.t()
    output_info.addmm_(1,-2,outputs,outputs.t())
    output_info = output_info.clamp(min=1e-12).sqrt() 
    
    #Compute pairwise distances of desired
    #Source for pairwise distance calculation:
    # https://github.com/jiyanggao/Video-Person-ReID/blob/master/losses.py#L47-L89
    desired_info = torch.pow(targets,2).sum(dim=1,keepdim=True).expand(n,n)
    desired_info = desired_info + desired_info.t()
    desired_info.addmm_(1,-2,targets,targets.t())
    desired_info = desired_info.clamp(min=1e-12).sqrt() 
    
    #Compute normalizing constant
    const = 1/(np.sqrt(2*np.pi)*bw)
    numerator = torch.sum(const*torch.exp(-((cross_info)**2)/(2*(bw**2))))/n
    denominator = ((torch.sum(const*torch.exp(-((desired_info)**2)/(2*(bw**2))))/n)*
                   (torch.sum(const*torch.exp(-((output_info)**2)/(2*(bw**2))))/n))
    
    #Normalize, alpha = 2
    loss = -(1/2)*torch.log((1/n)*torch.sum(numerator/denominator)) 
    
    return loss
    
