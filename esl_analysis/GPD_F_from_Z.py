#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" 
Created on Tue Nov 9th 15:58:34 2021

Functions to uses GPD parameters to estimate the return heights Z corresponding 
to the queried return frequencies F. 

@author: Tim Hermans
tim(dot)hermans@nioz(dot)nl
"""
import numpy as np

def F_from_Z(scale,shape,loc,avg_exceed,z):
    #This function obtains frequencies for corresponding test heights. It uses a General Pareto Distribution for heights that exceed the threshold.
    #A Gumbel distribution is assumed between the MHHW and the threshold.
    
    #input:
    #scale: scale factor GPD
    #shape: shape factor GPD
    #loc: threshold historical
    #avg_exceed: frequency that threshold is exceeded
    #z: test height minus (historical or future) location parameter
    #mhhw: Mean Higher High Water
    #mhhwFreq: frequency of MHHW
    
    #output:
    #Return frequencies
    try: #convert array input to float if using only 1 shape & scale
        scale=scale.item()
        shape=shape.item()
    except:
        pass
    
    z[z<0] = np.nan
    f = 0*z #initialize f

    if np.isscalar(scale) and np.isscalar(shape):
        if shape==0:
            f = avg_exceed * np.exp(-z/scale)
        else:
            f = avg_exceed * np.power((1+(shape*z/scale)),(-1/shape)) #get frequencies from pareto distribution for testz>loc
    else:
        if np.any(shape)==0:
            f[shape==0] = avg_exceed * np.exp(-z[shape==0]/scale[shape==0])
            f[shape!=0] = avg_exceed * np.power((1+(shape[shape!=0]*z[shape!=0]/scale[shape!=0])),(-1/shape[shape!=0]))
        else:
            f = avg_exceed * np.power((1+(shape*z/scale)),(-1/shape)) #get frequencies from pareto distribution for testz>loc
    
    
    #log_freq[below_loc] = np.log(avg_exceed) + ( np.log(365.25/2)-np.log(avg_exceed) )*(z0[below_loc]/(mhhw-loc)) #from Gumbel (see Buchanan et al. 2016 & LocalizeSL) for testz<loc (logN linear in z, assume MHHW once every 2 days)
    #freq[below_loc] = np.exp( log_freq[below_loc] )
    
    f = np.where(z<0,np.nan,f)#mask out where z is negative (heights below loc)
    f = np.where((shape*z/scale)<-1,np.nan,f) #mask out where z is above upper bound of the GPD
    
    #f[np.less(f, 1e-6, where=~np.isnan(f))] = np.nan #throw away frequencies lower than 1e-6 (following SROCC scripts)
    
    return f

def F_from_Z_mhhw(scale,shape,loc,avg_exceed,z,mhhw):
    #This function obtains frequencies for corresponding test heights. It uses a General Pareto Distribution for heights that exceed the threshold.
    #A Gumbel distribution is assumed between the MHHW and the threshold.
    
    #input:
    #scale: scale factor GPD
    #shape: shape factor GPD
    #loc: threshold historical
    #avg_exceed: frequency that threshold is exceeded
    #z: test height minus (historical or future) location parameter
    #mhhw: Mean Higher High Water
    #mhhwFreq: frequency of MHHW
    
    #output:
    #Return frequencies
    try: #convert array input to float if using only 1 shape & scale
        scale=scale.item()
        shape=shape.item()
    except:
        pass

    f = 0*z #initialize f

    if np.isscalar(scale) and np.isscalar(shape):
        if shape==0:
            f = avg_exceed * np.exp(-z/scale)
        else:
            f = avg_exceed * np.power((1+(shape*z/scale)),(-1/shape)) #get frequencies from pareto distribution for testz>loc
    else:
        if np.any(shape)==0:
            f[shape==0] = avg_exceed * np.exp(-z[shape==0]/scale[shape==0])
            f[shape!=0] = avg_exceed * np.power((1+(shape[shape!=0]*z[shape!=0]/scale[shape!=0])),(-1/shape[shape!=0]))
        else:
            f = avg_exceed * np.power((1+(shape*z/scale)),(-1/shape)) #get frequencies from pareto distribution for testz>loc
    
    f = np.where(z<0,np.nan,f)#mask out where z is negative (heights below loc)
    f = np.where((shape*z/scale)<-1,np.nan,f) #mask out where z is above upper bound of the GPD
    
    #f[np.less(f, 1e-6, where=~np.isnan(f))] = np.nan #throw away frequencies lower than 1e-6 (following SROCC & Frederikse et al. 2020)
    
    below_loc = ((z<0) & (z>=mhhw-loc))
    
    f[below_loc] = np.exp(np.log(avg_exceed) + ( np.log(365.25/2)-np.log(avg_exceed) )*(z[below_loc]/(mhhw-loc))) #from Gumbel (see Buchanan et al. 2016 & LocalizeSL) for testz<loc (logN linear in z, assume MHHW once every 2 days)
  
    return f