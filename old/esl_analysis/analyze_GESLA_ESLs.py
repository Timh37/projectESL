#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" .py
Created on Mon Oct 28 13:54:34 2019, adapted 20-04-22.
@author: Tim Hermans
tim(dot)hermans@nioz(dot)nl
"""
import numpy as np
import pandas as pd
from gplike import gplike
from scipy.stats import genpareto

def pot_extremes_from_gesla_dfs(dfs,threshold_pct,declus_method=None,declus_window=None):
    #declus window in days
    
    assert ((threshold_pct>0) & (threshold_pct<100))
    extremes_dfs = {}
    
    for k,df in dfs.items():
        threshold = np.percentile(df['sea_level'],threshold_pct)
        extremes = df[df['sea_level']>=threshold]

        
        if declus_method == 'iterative_descending':
            #sorts extremes descendingly, and iteratively add extremes to declustered extremes if they are further than declus_window away from already included extremes
            extremes_sorted = extremes.sort_values(by=['sea_level'], ascending=False, kind='mergesort')
            
            decl_extremes = extremes_sorted.iloc[[0]] #initialize using the first (highest) extreme
            
            for i in range(1,len(extremes_sorted)): #for each subsequent extreme
                next_extreme = extremes_sorted.iloc[[i]]
                timediffs = decl_extremes.index - next_extreme.index[0] #calculate time differences
            
                # if none of the included extremes are less than 'window_len' days away from current extreme, add current extreme to declustered extremes
                if all( abs(timediffs / np.timedelta64(24*3600, 's')) >= declus_window): 
                    decl_extremes = pd.concat((decl_extremes,next_extreme)) #add next extreme
            
            extremes_dfs[k] = decl_extremes.sort_index()

        elif declus_method == 'rolling_max':
            #only include extremes that are the maximum of a rolling window of declus_window around that extreme
            extremes_dfs[k] = extremes['sea_level'].where(extremes['sea_level']==extremes['sea_level'].rolling(window=str(declus_window)+'D',center=True,min_periods=1).max()).dropna().to_frame()
            
        elif declus_method == None:
            extremes_dfs[k] = extremes
        
        if 'H' in df.attrs['resample_freq']:
            rps = pd.DataFrame(data=[(1 + len(df)/365.25/24)/ i for i in range(1,len(extremes_dfs[k])+1)],
                               index=extremes_dfs[k].sort_values(by=['sea_level'], ascending=False, kind='mergesort').index,
                               columns=['rp_empirical']) #determine return periods of declustered extremes empirically
            extremes_dfs[k] = extremes_dfs[k].join(rps.sort_index())
            
            extremes_dfs[k].attrs['avg_extr_pyear'] = 24 * 365.25 * len(extremes_dfs[k])/len(df)
        elif 'D' in df.attrs['resample_freq']:
            rps = pd.DataFrame(data=[(1 + len(df)/365.25)/ i for i in range(1,len(extremes_dfs[k])+1)],
                               index=extremes_dfs[k].sort_values(by=['sea_level'], ascending=False, kind='mergesort').index,
                               columns=['rp_empirical'])
            extremes_dfs[k] = extremes_dfs[k].join(rps.sort_index())
            
            extremes_dfs[k].attrs['avg_extr_pyear'] = 365.25 * len(extremes_dfs[k])/len(df)
        
        #add metadata
        extremes_dfs[k].attrs['mhhw'] = np.nanmean(df.groupby(pd.Grouper(freq='2D')).max()) #2d because some places don't see high tide every day
        extremes_dfs[k].attrs['threshold_pct'] = threshold_pct
        extremes_dfs[k].attrs['threshold'] = threshold
        extremes_dfs[k].attrs['declus_method'] = declus_method
        extremes_dfs[k].attrs['declus_window'] = declus_window
        
    return extremes_dfs

def fit_gpd_to_gesla_extremes(dfs):
    for k,df in dfs.items():
        loc = df.attrs['threshold'] #location parameter = threshold
        
        #provide initial guess of parameters using method of moments (code based on MATLAB script gpfit.m)
        xbar = np.mean(df['sea_level'].values-loc) #mean of extremes
        s2 = np.var(df['sea_level'].values-loc) #variance of extremes
        k0 = -.5 * ((xbar**2)/s2 - 1) #initial guesses
        sigma0 = .5 * xbar * ( (xbar**2) / s2 + 1)
        xmax = max(df['sea_level'].values-loc)
        
        if (k0 < 0 and xmax >= -sigma0/k0 ):# if method of moments invalid (code based on MATLAB script gpfit.m), #assume exponential distribution
            k0 = 0
            sigma0 = xbar
    
        gp_params = genpareto.fit(df['sea_level'].values-loc,loc=0,scale=sigma0) #fit gpd based on ESLs and initial guess, note that optimization vals differ slightly from using gpfit.m
        gp_nlogl, gp_cov = gplike(gp_params[0], gp_params[2],df['sea_level'].values-loc) #calculate covariance matrix of estimated parameters
    
        if gp_nlogl == np.Inf:
            #GPD parameters not supported, confidence intervals and standard errors cannot be computed reliably
            print(k+' - not included, estimated GPD parameters not supported.')
            continue
        
        d = {'loc':[loc], 'scale': [gp_params[-1]], 'shape': [gp_params[0]], 'cov': [gp_cov], 'avg_extr_pyear': df.attrs['avg_extr_pyear'], 'mhhw': df.attrs['mhhw'], 'key': [k]}
        gpd_df = pd.DataFrame(data=d).set_index('key')

        if k == list(dfs.keys())[0]:
            gpd_dfs = gpd_df    
        else:
            gpd_dfs = pd.concat((gpd_dfs,gpd_df))
            
    return gpd_dfs
