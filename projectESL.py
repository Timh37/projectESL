#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:51:03 2023

Execution of projectESL according to user configuration.

@author: timhermans
"""
import sys
sys.path.insert(0, 'data_preprocessing')
sys.path.insert(1, 'esl_analysis')
sys.path.insert(2, 'projecting')

import os
import yaml
import xarray as xr
import numpy as np
import pandas as pd
from utils import mindist
import warnings
import pickle

with open('config.yml', 'r') as f: #open configuration file
    cfg = yaml.safe_load(f)

out_qnts = np.array(cfg['output']['output_quantiles'].split(',')).astype('float')

f= 10**np.linspace(-6,2,num=2001) #input frequencies to compute return heights
f=np.append(f,np.arange(101,183))

#### 1. Get & manipulate SLR projections
slr = xr.open_mfdataset((cfg['projecting']['slr_filename']),concat_dim='locations',combine='nested').load()

if 'locations' not in slr.dims:
    slr = slr.expand_dims('locations')

if 'samples' in slr.dims:
    if cfg['global-options']['num_mc']>len(slr.samples):
        raise Exception('Insufficient SLR samples for desired number of Monte Carlo samples.')
    else:
        idx = np.sort(np.random.choice(len(slr.samples), size=cfg['global-options']['num_mc'], replace=False))
    slr = slr.isel(samples = idx)

try:
    slr = slr.rename({'latitude':'lat','longitude':'lon'})
except:
    pass
####

#### 2. Get ESL information near queried locations
esl_statistics = {} #initialize dictionary

if cfg['esl_analysis']['input_type'] == 'raw': #derive GPD parameters from raw data
    
    if cfg['esl_analysis']['input_source'].lower() in ['gesla2','gesla3']: #if using raw data from GESLA
        from ESL_stats_from_raw_GESLA import ESL_stats_from_raw_GESLA
        esl_statistics = ESL_stats_from_raw_GESLA(slr,cfg,0.1)

    elif cfg['esl_analysis']['input_source'].lower() not in ['gesla2','gesla3']:
        raise Exception('Cannot (yet) analyze ESLs of configured input source.')

    if cfg['output']['store_esl_statistics']:
        pickle.dump(esl_statistics,open(os.path.join(cfg['output']['output_dir'],'esl_statistics.pkl'),'wb'))
    
#read in pre-defined GPD parameters
#elif cfg['esl'][input_type'] == 'gpd_params'
#to be implemented

#read in pre-defined return curves    
elif cfg['esl_analysis']['input_type'] == 'return_curves':
    if cfg['esl_analysis']['input_source'].lower() =='coast-rp':
        coast_rp_coords = pd.read_pickle(os.path.join(cfg['esl_analysis']['input_dir'],'pxyn_coastal_points.xyn'))
        min_idx = [mindist(x,y,coast_rp_coords['lat'].values,coast_rp_coords['lon'].values, 0.1) for x,y in zip(slr.lat.values, slr.lon.values)]
        
        for i in np.arange(len(slr.locations)):
            this_id = str(int(coast_rp_coords.iloc[min_idx[0][0]].name))
            rc = pd.read_pickle(os.path.join(cfg['esl_analysis']['input_dir'],'rp_full_empirical_station_'+this_id+'.pkl'))
            rc = rc['rp'][np.isfinite(rc['rp'])]
            
            esl_statistics[slr.locations.values[i]] = {}
            esl_statistics[slr.locations.values[i]]['z_hist'] = rc.index.values
            esl_statistics[slr.locations.values[i]]['f_hist'] = 1/rc.values
        
    elif cfg['esl_analysis']['input_source']!='COAST-RP':
        raise Exception('Cannot (yet) open return curves of configured input source.')    
    
else:
    raise Exception('Input type not recognized.')
#### 
        
#### 3. Do the projections
if cfg['projecting']['project_AFs'] + cfg['projecting']['project_AF_timing'] > 0 :
    sites_output = []
    
    from multivariate_normal_gpd_samples_from_covmat import multivariate_normal_gpd_samples_from_covmat
    from GPD_Z_from_F import get_return_curve
    from compute_AF_timing import compute_AF_timing
    from compute_AF import compute_AF
    
    settings = cfg['projecting']['projection_settings']
    
    if slr.sea_level_change.units=='mm': #check sea-level change unit is meter
        slr['sea_level_change'] = slr['sea_level_change']/1000
        slr.sea_level_change.attrs['units'] = 'm'
    elif slr.sea_level_change.units=='cm':
        slr['sea_level_change'] = slr['sea_level_change']/100
        slr.sea_level_change.attrs['units'] = 'm'
        
        
    #determine reference frequencies
    if settings['refFreqs'] == 'diva': #find contemporary DIVA flood protection levels
        qlats = [slr.lat.sel(locations=k).values for k in esl_statistics.keys()]
        qlons = [slr.lon.sel(locations=k).values for k in esl_statistics.keys()]
        
        from find_diva_protection_levels import find_diva_protection_levels
        
        refFreqs = find_diva_protection_levels(qlons,qlats,"/Users/timhermans/Documents/Data/DIVA/cls.gpkg",15)
    
    elif settings['refFreqs'] == 'flopros': #find contemporary FLOPROS flood protection levels
        qlats = [slr.lat.sel(locations=k).values for k in esl_statistics.keys()]
        qlons = [slr.lon.sel(locations=k).values for k in esl_statistics.keys()]
        
        from find_flopros_protection_levels import find_flopros_protection_levels
        
        refFreqs = find_flopros_protection_levels(qlons,qlats,'/Users/timhermans/Documents/Data/FLOPROS/Tiggeloven/',15)    
    
    else:
        if np.isscalar(refFreqs): #use constant reference frequency for every location
            refFreqs = np.tile(refFreqs,len(esl_statistics))
        else: 
            raise Exception('Reference frequency must be "DIVA", "FLOPROS" or a constant.')
    
    #check if target_years in slr projections
    if cfg['projecting']['project_AFs']:
        target_years_ = np.array(str(settings['target_years']).split(',')).astype('int')
        target_years = np.intersect1d(target_years_,slr.years)
        
        if len(target_years)!=len(target_years_):
            warnings.warn('SLR projections do not contain all target years, continuing with a smaller set of target years.')
    
    #loop over sites in esl_statistics to do the projections:
    i=0    
    for site_id,stats in esl_statistics.items():
        refFreq = refFreqs[i] #get reference frequency for this site
        
        if ~np.isfinite(refFreq):
            print(settings['refFreqs']+' protection standard unavailable for site: '+site_id+', moving on to next site.')
            continue
        
        #get return curves
        if cfg['esl_analysis']['input_type'] == 'raw': #generate return curves
            scale_samples,shape_samples = multivariate_normal_gpd_samples_from_covmat(stats['scale'].iloc[0],stats['shape'].iloc[0],stats['cov'].iloc[0],cfg['global-options']['num_mc'])
            z_hist = get_return_curve(f,scale_samples,shape_samples,stats['loc'].iloc[0],stats['avg_extr_pyear'].iloc[0],'mhhw',stats['mhhw'].iloc[0])
            f_ = f
        
        #to-do implement input type gpd params
        
        elif cfg['esl_analysis']['input_type'] == 'return_curves': #load in return curves
            z_hist = stats['z_hist']
            f_ = stats['f_hist']
            
        #compute AFs:    
        if cfg['projecting']['project_AFs']:
            output = []
            for yr in target_years:
                z_fut,AF = compute_AF(f_,z_hist,slr.sel(locations=int(site_id)).sel(years=yr).sea_level_change.values,refFreq)

                output_ds = xr.Dataset(data_vars=dict(z_fut=(["qnt",'f'], np.quantile(z_fut,q=out_qnts,axis=-1)),AF=(["qnt"],np.quantile(AF,out_qnts))),
                        coords=dict(f=(["f"], f),qnt=(["qnt"], out_qnts),year=yr,site=site_id,),)
                output.append(output_ds)
                
            site_AFs = xr.concat(output,dim='year')
            site_AFs['refFreq'] = ([],refFreq)
     
        #compute AF timing
        if cfg['projecting']['project_AF_timing']: #if projecting timing of AFs
            
            
            annual_slr = slr.sel(locations=int(site_id)).interp(years=np.arange(slr.years[0],slr.years[-1]+1),method='quadratic') #interpolate SLR to annual timesteps
            
            if 'target_AFs' in settings:
                output = []
                
                target_AFs = np.array(str(settings['target_AFs']).split(',')).astype('float')
                
                for target_AF in target_AFs:
                    timing = compute_AF_timing(f_,z_hist,annual_slr.sea_level_change,refFreq,AF=target_AF)
                    
                    output_ds = xr.Dataset(data_vars=dict(af_timing=(["qnt"],np.quantile(timing,q=out_qnts,axis=-1).astype('int'))),
                            coords=dict(qnt=(["qnt"], out_qnts),target_AF=target_AF,site=site_id,),)
                    output.append(output_ds)
                af_timing = xr.concat(output,dim='target_AF')
                
            if 'target_freqs' in settings:
                output = []
                
                target_freqs = np.array(str(settings['target_freqs']).split(',')).astype('float')
            
                for target_freq in target_freqs:
                    timing = compute_AF_timing(f_,z_hist,annual_slr.sea_level_change,refFreq,AF=target_freq/refFreq)
                    
                    output_ds = xr.Dataset(data_vars=dict(freq_timing=(["qnt"],np.quantile(timing,q=out_qnts,axis=-1).astype('int'))),
                            coords=dict(qnt=(["qnt"], out_qnts),target_f=target_freq,site=site_id,),)
            
                    output.append(output_ds)
                freq_timing = xr.concat(output,dim='target_f')
            
            site_timing = xr.merge((af_timing,freq_timing))
            site_timing['refFreq'] = ([],refFreq)
        
        if cfg['projecting']['project_AFs'] & cfg['projecting']['project_AF_timing']:
            sites_output.append(xr.merge((site_AFs,site_timing)))
        elif cfg['projecting']['project_AFs'] & ~cfg['projecting']['project_AF_timing']:
            sites_output.append(site_AFs)
        else:
            sites_output.append(site_timing)
        i+=1
        
    output_ds = xr.concat(sites_output,dim='site')
    output_ds = output_ds.assign_coords({'lat':slr.lat.sel(locations=output_ds.site),'lon':slr.lon.sel(locations=output_ds.site)})
    output_ds.attrs['config'] = cfg
    output_ds.to_netcdf(os.path.join(cfg['output']['output_dir'],'esl_projections.nc'),mode='w')