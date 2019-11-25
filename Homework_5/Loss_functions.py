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
    diff = diff.clamp(min=1e-12).sqrt()

    #Compute normalizing constant
    const = 1/(np.sqrt(2*np.pi)*bw)
    loss = torch.sum(const*torch.exp(-((diff)**2)/(2*(bw**2))))
            
    #Normalize
    loss = torch.sum(loss)/(len(error)**2) 
    
    return loss
    
