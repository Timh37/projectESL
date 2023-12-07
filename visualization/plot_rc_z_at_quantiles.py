#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:40:17 2023

@author: timhermans
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

def plot_rc_z_at_quantiles(z_hist,quantiles)
plt.figure()
for i in [0,2,4]:
    plt.plot(z_hist_qnt_vals[i,:]+(gpd_params['loc']-gpd_params['mhhw']).values,f,color='red',label='historical')
    plt.plot(z_fut_qnt_vals[i,:]+(gpd_params['loc']-gpd_params['mhhw']).values,f,color='blue',label='future')
   
    #plt.plot(np.nanquantile(f_hist,[.05,.5,.95][i],axis=1),z,color='orange',linestyle='dashed')
    #plt.plot(np.nanquantile(f_fut,[.05,.5,.95][i],axis=1),z,color='cyan',linestyle='dashed')
    
    #plt.plot(f_fut[:,-1],z,color='green',linestyle='dashed')
   
plt.yscale('log')
plt.xlim([0,6])
plt.ylim([1e-6,365.25/2])
plt.xticks(np.arange(0,6.5,.5))
plt.grid()