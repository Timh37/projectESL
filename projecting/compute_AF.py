#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 13:13:08 2023

@author: timhermans
"""
import numpy as np


def compute_AF(f,z_hist,slr,refFreq):
    i_ref = np.argmin(np.abs(f-refFreq)) #find index of f closest to refFreq
    #f = np.repeat(f[:,None],len(slr),axis=1)
    if (z_hist.ndim == 1):
        
        if np.isscalar(slr)==False:
            z_hist = np.repeat(z_hist[:,None],len(slr),axis=1)
    
    z_fut = z_hist+slr
        
    refZ_hist = z_hist[i_ref] #historical reference height samples corresponding to refFreq
    
    iRefZ_hist_in_z_fut = np.nanargmin(np.abs(z_fut - refZ_hist),axis=0) #find future frequency corresponding to those heights
    #note that the highest frequency possible is the lower bound of the GPD (or the distribution used below the GPD) 
    
    AF = f[iRefZ_hist_in_z_fut]/refFreq #divide future frequency correspondong to refZ_hist by historical reference frequency

    return z_hist,z_fut,AF