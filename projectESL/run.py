#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:51:03 2023

Execution of projectESL according to user configuration.

@author: timhermans
"""
import os
import xarray as xr
import numpy as np
import pandas as pd
from utils import mindist
import warnings
from I_O import load_config, store_esl_params, open_slr_projections, get_refFreqs
from esl_analysis import ESL_stats_from_raw_GESLA, multivariate_normal_gpd_samples_from_covmat, get_return_curve_gpd
from projecting import compute_AF, compute_AF_timing

cfg = load_config('/Users/timhermans/Documents/GitHub/projectESL/config.yml')
#sites = xr.open_dataset(os.path.join(home_path,'input',cfg['general']['sites_filename']))
sites = xr.open_mfdataset((os.path.join(cfg['general']['project_path'],'input',cfg['general']['sites_filename'])),concat_dim='locations',combine='nested')#.load()


out_qnts = np.array(cfg['output']['output_quantiles'].split(',')).astype('float')

f= 10**np.linspace(-6,2,num=2001) #input frequencies to compute return heights
f=np.append(f,np.arange(101,183))


#### 1. Get ESL information at sites
esl_statistics = {} #initialize dictionary

if cfg['esl_analysis']['input_type'] == 'raw': #derive GPD parameters from raw data
    if cfg['esl_analysis']['input_source'].lower() in ['gesla2','gesla3']: #if using raw data from GESLA
        esl_statistics = ESL_stats_from_raw_GESLA(sites,cfg,0.1)
    else:
        raise Exception('If input_type == "raw", "input_source" must be in ["gesla2","gesla3"].')

    if cfg['output']['store_distParams']:
        store_esl_params(cfg,sites,esl_statistics)
         
#read in pre-defined GPD parameters
#elif cfg['esl'][input_type'] == 'gpd_params'
#to be implemented
   
elif cfg['esl_analysis']['input_type'] == 'return_curves': #read in pre-defined return curves 
    if cfg['esl_analysis']['input_source'].lower() =='coast-rp':
        coast_rp_coords = pd.read_pickle(os.path.join(cfg['esl_analysis']['input_dir'],'pxyn_coastal_points.xyn'))
        min_idx = [mindist(x,y,coast_rp_coords['lat'].values,coast_rp_coords['lon'].values, 0.1) for x,y in zip(sites.lat.values, sites.lon.values)]
        
        for i in np.arange(len(sites.locations)):
            this_id = str(int(coast_rp_coords.iloc[min_idx[0][0]].name))
            rc = pd.read_pickle(os.path.join(cfg['esl_analysis']['input_dir'],'rp_full_empirical_station_'+this_id+'.pkl'))
            rc = rc['rp'][np.isfinite(rc['rp'])]
            
            esl_statistics[sites.locations.values[i]] = {}
            esl_statistics[sites.locations.values[i]]['z_hist'] = rc.index.values
            esl_statistics[sites.locations.values[i]]['f_hist'] = 1/rc.values 
    else:
        raise Exception('If input_type == "return_curves", "input_source" must be in ["coast-rp"].')   
        #others to be implemented
    
else:
    raise Exception('Input type not recognized.')
#### 
   
 
#### 2. Compute return curves & do projections
#a. open SLR projections & get reference frequencies for AF projections
if cfg['output']['output_AFs'] + cfg['output']['output_AF_timing'] > 0 : 
    slr = open_slr_projections(cfg)
    refFreqs = get_refFreqs(cfg,sites,esl_statistics)
  
    
    if cfg['output']['output_AFs']:
        target_years_ = np.array(str(cfg['projecting']['projection_settings']['target_years']).split(',')).astype('int')  #check if target_years in slr projections
        target_years = np.intersect1d(target_years_,slr.years)
        
        if len(target_years)!=len(target_years_):
            warnings.warn('SLR projections do not contain all target years, continuing with a smaller set of target years.')
    
#b. loop over sites to do computations (make this parallel later?):
if cfg['output']['output_AFs'] + cfg['output']['output_AF_timing'] + cfg['output']['store_RCs'] > 0:
    i=0    
    sites_output = []
 
    for site_id,stats in esl_statistics.items(): #for each site
        site_rcs = None
        site_AFs = None
        site_timing = None
        
        #compute/get return curves
        if cfg['esl_analysis']['input_type'] == 'raw': #generate return curves
            scale = stats['scale'].iloc[0]
            shape = stats['shape'].iloc[0]
            
            scale_samples,shape_samples = multivariate_normal_gpd_samples_from_covmat(scale,shape,stats['cov'].iloc[0],cfg['general']['num_mc'])
            z_hist = get_return_curve(f,scale_samples,shape_samples,stats['loc'].iloc[0],stats['avg_extr_pyear'].iloc[0],'mhhw',stats['mhhw'].iloc[0])
            f_ = f
        
        #to-do implement input type gpd params
        
        elif cfg['esl_analysis']['input_type'] == 'return_curves': #load in return curves
            z_hist = stats['z_hist']
            f_ = stats['f_hist']
        
        
        if cfg['output']['store_RCs']: #store return cures
            if len(np.shape(z_hist))==2:
                site_rcs = xr.Dataset(data_vars=dict(z_hist=(["qnt","f"],np.quantile(z_hist,q=out_qnts,axis=-1))),
                        coords=dict(qnt=(["qnt"], out_qnts),f=f_,site=site_id,),)
            else:
                site_rcs = xr.Dataset(data_vars=dict(z_hist=(["f"],z_hist)),
                        coords=dict(f=f_,site=site_id,),)
  
        if cfg['output']['output_AFs'] + cfg['output']['output_AF_timing'] > 0:
            refFreq = refFreqs[i] #get reference frequency for this site
            
            if ~np.isfinite(refFreq):
                print(cfg['projecting']['projection_settings']['refFreqs']+' protection standard unavailable for site: '+site_id+', moving on to next site.')
            else:
                
                #compute AFs:  
                if cfg['output']['output_AFs']: 
                    output = []
                    for yr in target_years:
                        z_fut,AF = compute_AF(f_,z_hist,slr.sel(locations=int(site_id)).sel(years=yr).sea_level_change.values,refFreq)
            
                        output_ds = xr.Dataset(data_vars=dict(z_fut=(["qnt",'f'], np.quantile(z_fut,q=out_qnts,axis=-1)),AF=(["qnt"],np.quantile(AF,out_qnts))),
                                coords=dict(f=(["f"], f),qnt=(["qnt"], out_qnts),year=yr,site=site_id,),)
                        output.append(output_ds)
                        
                    site_AFs = xr.concat(output,dim='year')
                    site_AFs['refFreq'] = ([],refFreq)
             
                #compute AF timing
                if cfg['output']['output_AF_timing']: #if projecting timing of AFs
                    annual_slr = slr.sel(locations=int(site_id)).interp(years=np.arange(slr.years[0],slr.years[-1]+1),method='quadratic') #interpolate SLR to annual timesteps
                    
                    if 'target_AFs' in cfg['projecting']['projection_settings']:
                        output = []
                        
                        target_AFs = np.array(str(cfg['projecting']['projection_settings']['target_AFs']).split(',')).astype('float')
                        
                        for target_AF in target_AFs:
                            timing = compute_AF_timing(f_,z_hist,annual_slr.sea_level_change,refFreq,AF=target_AF)
                            
                            output_ds = xr.Dataset(data_vars=dict(af_timing=(["qnt"],np.quantile(timing,q=out_qnts,axis=-1).astype('int'))),
                                    coords=dict(qnt=(["qnt"], out_qnts),target_AF=target_AF,site=site_id,),)
                            output.append(output_ds)
                        af_timing = xr.concat(output,dim='target_AF')
                        
                    if 'target_freqs' in cfg['projecting']['projection_settings']:
                        output = []
                        
                        target_freqs = np.array(str(cfg['projecting']['projection_settings']['target_freqs']).split(',')).astype('float')
                    
                        for target_freq in target_freqs:
                            timing = compute_AF_timing(f_,z_hist,annual_slr.sea_level_change,refFreq,AF=target_freq/refFreq)
                            
                            output_ds = xr.Dataset(data_vars=dict(freq_timing=(["qnt"],np.quantile(timing,q=out_qnts,axis=-1).astype('int'))),
                                    coords=dict(qnt=(["qnt"], out_qnts),target_f=target_freq,site=site_id,),)
                    
                            output.append(output_ds)
                        freq_timing = xr.concat(output,dim='target_f')
                    
                    site_timing = xr.merge((af_timing,freq_timing))
                    site_timing['refFreq'] = ([],refFreq)
                
            site_output = [k for k in [site_rcs,site_AFs,site_timing] if k!=None]
            if site_output!=[]:
                sites_output.append(xr.merge(site_output))
            i+=1
                
        output_ds = xr.concat(sites_output,dim='site')
        output_ds = output_ds.assign_coords({'lat':sites.lat.sel(locations=output_ds.site),'lon':sites.lon.sel(locations=output_ds.site)})
        output_ds.attrs['config'] = str(cfg)
        output_ds.to_netcdf(os.path.join(cfg['output']['output_dir'],'projectESL_output.nc'),mode='w')
            
