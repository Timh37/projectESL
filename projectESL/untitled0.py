#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:32:57 2024

@author: timhermans
"""
import xarray as xr
import numpy as np


test = xr.open_dataset('/Users/timhermans/Documents/Data/CODEC/CODEC_amax_ERA5_1979_2017_coor_mask_GUM_RPS.nc')
location = test.GUM[0,0]
scale = test.GUM[0,1]
z = test.RPS[0,2].values

f=np.array([.1,.2])

def gum_Z_from_F(scale,loc,f): 
    try: #convert array input to float if using only 1 scale
        scale=scale.item()
    except:
        pass
    
    z = 0*f #initialize z
    z = loc - np.log( -np.log(1-f) ) * scale #rearranged from cdf formula exceedeance prob = 1-np.exp(-np.exp((location-z)/scale))

    z = np.where(z<loc,np.nan,z) #where z below location parameter, replace z by np.nan
    return z


def gum_Z_from_F_mhhw(scale,loc,f,mhhw): 
    try: #convert array input to float if using only 1 scale
        scale=scale.item()
    except:
        pass
    
    z = 0*f #initialize z
    z = loc - np.log( -np.log(1-f) ) * scale #see Eq 3.4 in Coles; rearranged from cdf formula exceedeance prob = 1-np.exp(-np.exp((location-z)/scale))

    avg_exceed = 1-np.exp(-1) #exceedance probability of location parameter
    
    #z = np.where(z<loc,np.nan,z) #where z below location parameter, replace z by np.nan
    
    #below location parameter, use a gumbel distribution between the location parameter and the mhhw (F=avg_exceed for z->0 with z=z0-loc (z0->loc)), use Gumbel distribution
    z[f>avg_exceed] = np.log(f[f>avg_exceed]/avg_exceed) * (mhhw-loc)/np.log((365.25/2)/avg_exceed) #lower bound of gumbel is z=z0-loc=(mhhw-loc), i.e. (loc-mhhw) below loc if mhhw follows from long-term mean of 2-day maximum

    #below lower bound of Gumbel?
    z = np.where(f>(365.25/2),np.nan,z) #where F>mhhwFreq, replace z by np.nan

    
    return z