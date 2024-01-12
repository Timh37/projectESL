#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:22:09 2023

@author: timhermans
"""

import numpy as np

def multivariate_normal_gpd_samples_from_covmat(scale,shape,cov,n,seed=None):
    if seed:
        np.random.seed(seed)
    gp_samples = np.random.multivariate_normal([shape,scale], cov,size=n)
    shape_samples = gp_samples[:,0]
    scale_samples = gp_samples[:,1]
    scale_samples[scale_samples<0] = 0 #no negative scales
    
    return scale_samples, shape_samples
   