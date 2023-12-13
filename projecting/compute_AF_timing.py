#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 13:13:08 2023

@author: timhermans
"""
import numpy as np

def compute_AF_timing(f,z_hist,slr,refFreq,AF):
    assert AF>1
    
    i_ref = np.argmin(np.abs(f-refFreq)) #find index of f closest to refFreq
    i_ref_AF = np.argmin(np.abs(f-refFreq*AF)) #find index of f closest to AF * refFreq

    if (z_hist.ndim == 1):
        
        if np.isscalar(slr)==False:
            z_hist = np.repeat(z_hist[:,None],len(slr),axis=1)

    req_slr = z_hist[i_ref] - z_hist[i_ref_AF] #sea-level rise required to go from refFreq to AF*refFreq

    #find first year in which SLR > required SLR
    slr_minus_required = slr.values - np.repeat(req_slr[:,np.newaxis],len(slr.years),axis=1)
    slr_minus_required[slr_minus_required<0] = 999
    imin = np.nanargmin(slr_minus_required,axis=-1)

    timing = slr.years.values[imin]
    
    try:
        timing[timing==slr.years.values[-1]] = np.nan #end of timeseries or later -> cannot evaluate timing, so set to nan
    except:
        pass
    
    return timing