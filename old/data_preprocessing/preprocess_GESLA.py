#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" .py
Created on Mon Oct 28 13:54:34 2019, adapted 20-04-22.
@author: Tim Hermans
tim(dot)hermans@nioz(dot)nl
"""
import numpy as np
import pandas as pd
from copy import deepcopy

def detrend_gesla_dfs(dfs):
    detrended_dfs = deepcopy(dfs)
    for k,df in detrended_dfs.items():
        x =  df.index.values.astype(np.int64) // 10 ** 9 #convert to seconds timestamp
        y =  df['sea_level'].values.astype('float64')
        lrcoefs = np.polyfit(x,y,1)
        trend = np.polyval(lrcoefs,x)
        
        df['sea_level'] = df['sea_level'] - trend + lrcoefs[-1]
    return detrended_dfs

def deseasonalize_gesla_dfs(dfs):
    deseasonalized_dfs = deepcopy(dfs)
    for k,df in deseasonalized_dfs.items():
        df['sea_level'] = df['sea_level'] - df.groupby(df.index.month).transform('mean')['sea_level'].astype('float64') + np.mean(df.groupby(df.index.month).mean()['sea_level'])
    return deseasonalized_dfs

def subtract_amean_from_gesla_dfs(dfs):
    dfs_no_amean = deepcopy(dfs)
    for k,df in dfs_no_amean.items():
        df['sea_level'] = df['sea_level'] - df.groupby(df.index.year).transform('mean')['sea_level'].astype('float64')
    return dfs_no_amean

    
def drop_shorter_gesla_neighbors(dfs,min_dist=3):
    filtered_dfs = deepcopy(dfs)
    
    for k,df in dfs.items():
        #all tgs in filtered_dfs
        lons    = np.array([k.attrs['longitude'] for k in filtered_dfs.values()])
        lats    = np.array([k.attrs['latitude'] for k in filtered_dfs.values()])
        lengths = np.array([len(k) for k in filtered_dfs.values()])
        
        #current tg
        lon = df.attrs['longitude']
        lat = df.attrs['latitude']
        length = len(df)
        
        #compute distances current tg to all tgs in filtered_dfs
        distances = 6371*2*np.arcsin( np.sqrt(
                np.sin( (np.pi/180) * 0.5*(lats-lat) )**2 +
                np.cos((np.pi/180)*lat)*np.cos((np.pi/180)*lats)*np.sin((np.pi/180)*0.5*(lons-lon))**2) ) #distances from current site to all included sites
        
        if np.sum(distances < min_dist)>1: #if other tgs than current tg within 3 km:
            if length != np.max(lengths[distances<min_dist]): #if current record shorther than nearby records
                filtered_dfs.pop(k) #remove current tg from filtered_dfs
                
    return filtered_dfs