#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:51:03 2023

@author: timhermans
"""
import sys
sys.path.insert(2, '/Users/timhermans/Documents/GitHub/projectESL/esl_analysis')
sys.path.insert(3, '/Users/timhermans/Documents/GitHub/projectESL/projecting')

import os
import yaml
import xarray as xr
import numpy as np
from utils import mindist
from multivariate_normal_gpd_samples_from_covmat import multivariate_normal_gpd_samples_from_covmat
from compute_AF import compute_AF
from GPD_Z_from_F import get_return_curve
import pandas as pd

with open('config.yml', 'r') as f: #load config
    cfg = yaml.safe_load(f)

f= 10**np.linspace(-6,2,num=2001)
f=np.append(f,np.arange(101,183))

#### 1. Get & manipulate SLR projections (to be loaded from cfg)
#slr = xr.open_mfdataset(("/Volumes/Naamloos/PhD_Data/AR6_projections/ar6-GESLA3-samples-total/wf_1f/ssp585/per_tg/Fort_Hamilton_New_York_1.nc",
#                         "/Volumes/Naamloos/PhD_Data/AR6_projections/ar6-GESLA3-samples-total/wf_1f/ssp585/per_tg/Aberdeen.nc"),concat_dim='locations',combine='nested').load()
slr = xr.open_mfdataset(("/Users/timhermans/Documents/GitHub/projectESL/Fort_Hamilton_New_York_1.nc"),concat_dim='locations',combine='nested').load()
#slr = xr.open_dataset(cfg['esl']['sl_projections']) #open sl projections

if 'locations' not in slr.dims:
    slr = slr.expand_dims('locations')

if 'samples' in slr.dims:
    if cfg['global-options']['num_mc']>len(slr.samples):
        raise Exception('Insufficient SLR samples for desired number of Monte Carlo samples.')
    else:
        idx = np.sort(np.random.choice(len(slr.samples), size=cfg['global-options']['num_mc'], replace=False))
    slr = slr.isel(samples = idx)

if slr.sea_level_change.units=='mm':
    slr['sea_level_change'] = slr['sea_level_change']/1000
    slr.sea_level_change.attrs['units'] = 'm'
    
####

#### 2. Get ESL information near queried locations
esl_statistics = {}

#derive GPD parameters from raw data
if (cfg['esl']['input_type'] == 'raw') & (cfg['esl']['input_source'].lower() in ['gesla2','gesla3']): #if using raw data from GESLA
    from ESL_stats_from_raw_GESLA import ESL_stats_from_raw_GESLA
    esl_statistics = ESL_stats_from_raw_GESLA(slr,cfg,0.1)

elif (cfg['esl']['input_type'] == 'raw') & (cfg['esl']['input_source'].lower() not in ['gesla2','gesla3']):
    raise Exception('Cannot (yet) analyze ESLs of configured input source.')

#open pre-defined GPD parameters
#to implement

#open pre-defined return curves    
if (cfg['esl']['input_type'] == 'return_curves') & (cfg['esl']['input_source']=='COAST-RP'):
    coast_rp_coords = pd.read_pickle(os.path.join('/Volumes/Naamloos/PhD_Data/COAST_RP/COAST_RP_full_dataset','pxyn_coastal_points.xyn'))
    
    min_idx = [mindist(x,y,coast_rp_coords['lat'].values,coast_rp_coords['lon'].values, 0.1) for x,y in zip(slr.lat.values, slr.lon.values)]
    
    for i in np.arange(len(slr.locations)):
        this_id = str(int(coast_rp_coords.iloc[min_idx[0][0]].name))
        rc = pd.read_pickle(os.path.join('/Volumes/Naamloos/PhD_Data/COAST_RP/COAST_RP_full_dataset','rp_full_empirical_station_'+this_id+'.pkl'))
        rc = rc['rp'][np.isfinite(rc['rp'])]
        
        esl_statistics[slr.locations.values[i]] = {}
        esl_statistics[slr.locations.values[i]]['z_hist'] = rc.index.values
        esl_statistics[slr.locations.values[i]]['f_hist'] = 1/rc.values
    
elif (cfg['esl']['input_type'] == 'return_curves') & (cfg['esl']['input_source']!='COAST-RP'):
    raise Exception('Cannot (yet) open return curves of of configured input source.')    
#### 
        
	
#### 3. Do the projections
#determine reference frequencies
if cfg['projecting']['refFreqs'] == 'diva': #find contemporary DIVA flood protection levels
    qlats = [slr.lat.sel(locations=k).values for k in esl_statistics.keys()]
    qlons = [slr.lon.sel(locations=k).values for k in esl_statistics.keys()]
    
    from find_diva_protection_levels import find_diva_protection_levels
    
    refFreqs = find_diva_protection_levels(qlons,qlats,"/Users/timhermans/Documents/Data/DIVA/cls.gpkg",15)

elif cfg['projecting']['refFreqs'] == 'flopros': #find contemporary FLOPROS flood protection levels
    qlats = [slr.lat.sel(locations=k).values for k in esl_statistics.keys()]
    qlons = [slr.lon.sel(locations=k).values for k in esl_statistics.keys()]
    
    from find_flopros_protection_levels import find_flopros_protection_levels
    
    refFreqs = find_flopros_protection_levels(qlons,qlats,'/Users/timhermans/Documents/Data/FLOPROS/Tiggeloven/',15)    

else:
    if ~np.isscalar(refFreqs):
        raise Exception('Reference frequency must be DIVA, FLOPROS or a scalar value.')
    else:
        refFreqs = np.tile(refFreqs,len(esl_statistics))

#project amplification factors relative to reference frequencies     
out_qnts = np.array(cfg['output']['quantiles'].split(',')).astype('float')
   
if cfg['output']['project_AFs']: #if projecting amplification factors
    target_years = np.array(str(cfg['output']['AF']['target_years']).split(',')).astype('int')
    
    try:
        slr_at_output_years = slr.sel(years=target_years) #select target years
    except:
        raise Exception('Could not select target years from sea-level projections.')

    #loop over queried sites with required information:
    i=0
    output = []
    for site_id,stats in esl_statistics.items():
        refFreq = refFreqs[i]
            
        if cfg['esl']['input_type'] == 'raw': #generate return curves
            scale_samples,shape_samples = multivariate_normal_gpd_samples_from_covmat(stats['scale'].iloc[0],stats['shape'].iloc[0],stats['cov'].iloc[0],cfg['global-options']['num_mc'])
            z_hist = get_return_curve(f,scale_samples,shape_samples,stats['loc'].iloc[0],stats['avg_extr_pyear'].iloc[0],'mhhw',stats['mhhw'].iloc[0])
            f_ = f
            
        elif cfg['esl']['input_type'] == 'return_curves': #load in return curves
            z_hist = stats['z_hist']
            f_ = stats['f_hist']
        
        
        for yr in target_years:
            site_output = []
            
            z_hist,z_fut,AF = compute_AF(f_,z_hist,slr_at_output_years.sel(locations=int(site_id)).sel(years=yr).sea_level_change.values,refFreq) #samples
        
            #to-do: store outputs
            ds = xr.Dataset(
                    data_vars=dict(
                        z_hist=(["qnt","f"], np.quantile(z_hist,q=out_qnts,axis=-1)),
                        z_fut=(["qnt",'f'], np.quantile(z_fut,q=out_qnts,axis=-1)),
                        AF=(["qnt"],np.quantile(AF,out_qnts))
                    ),
                    coords=dict(
                        f=(["f"], f),
                        qnt=(["qnt"], out_qnts),
                        year=yr,
                        site=site_id,
                    ),
                )
            site_output.append(ds)
        output.append(xr.concat(site_output,dim='year'))
        i+=1
    output_ds = xr.concat(output,dim='site')
    output_ds.attrs['config'] = cfg
    output_ds['lat'] = ('site',[slr.lat.sel(locations=k).values for k in output_ds.site.values])
    output_ds['lon'] = ('site',[slr.lon.sel(locations=k).values for k in output_ds.site.values])
    output_ds.set_coords(('lon','lat'))
    
#do quadratic interpolation to get years between decades before computing the timing    


if cfg['output']['project_AF_timing']: 
    from compute_AF_timing import compute_AF_timing
    annual_slr = slr.interp(years=np.arange(slr.years[0],slr.years[-1]+1),method='quadratic')
    
    i=0
    output = []
    for site_id,stats in esl_statistics.items():
        refFreq = refFreqs[i]
        
        i+=1
        
        if cfg['esl']['input_type'] == 'raw': #generate return curves
            scale_samples,shape_samples = multivariate_normal_gpd_samples_from_covmat(stats['scale'].iloc[0],stats['shape'].iloc[0],stats['cov'].iloc[0],cfg['global-options']['num_mc'])
            z_hist = get_return_curve(f,scale_samples,shape_samples,stats['loc'].iloc[0],stats['avg_extr_pyear'].iloc[0],'mhhw',stats['mhhw'].iloc[0])
            f_ = f
            
        elif cfg['esl']['input_type'] == 'return_curves': #load in return curves
            z_hist = stats['z_hist']
            f_ = stats['f_hist']
            
            
        #if AF='futFreq'
        #AF = futFreq/refFreq or AF
        
        timing = compute_AF_timing(f_,z_hist,annual_slr.sel(locations=int(site_id)).sea_level_change,refFreq,AF=.1) #AF=10, should set this in cfg, also have option to go to certain frequency? 1e1 for example
        
        
        #np.quantile(timing,out_qnts)
        
        #to-do: store outputs in a good way
        
    
'''   
slr.interp(years=np.arange(yr_ref,2151),method='quadratic')
timing = compute_AF_timing(f,z_hist,slr,refFreq,AF=10) #AF=10, should set this in cfg, also have option to go to certain frequency? 1e1 for example

#compute_AF_timing(f,z_hist,slr,refFreq,AF):
i_ref = np.argmin(np.abs(f-refFreq)) #find index of f closest to refFreq
i_ref_AF = np.argmin(np.abs(f-refFreq*10))

#f = np.repeat(f[:,None],len(slr),axis=1)
if (z_hist.ndim == 1):
    
    if np.isscalar(slr)==False:
        z_hist = np.repeat(z_hist[:,None],len(slr),axis=1)

req_slr = z_hist[i_ref] - z_hist[i_ref_AF]

#do quadratic interpolation to get years between decades

#try to vectorize this?
slr_minus_required = slr.sel(locations=int(site_id)).sea_level_change.values-np.repeat(req_slr[:,np.newaxis],len(slr.years),axis=1)
slr_minus_required[slr_minus_required<0] = 999
imin = np.nanargmin(slr_minus_required,axis=-1)

timing = slr.years[imin]

timing[timing==slr.years[-1]] = np.nan #end of timeseries or later -> cannot evaluate timing
#if imin=last index of SLR, set to nan
'''