#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:51:03 2023

Execution of projectESL according to user configuration in 'config.yml'.

@author: timhermans
"""
import os
import xarray as xr
import numpy as np
import warnings
from I_O import load_config, store_esl_params, get_refFreqs, open_sites_input
from I_O import get_gpd_params_from_Hermans2023,get_gpd_params_from_Kirezci2020, get_gpd_params_from_Vousdoukas2018, get_gum_amax_from_CoDEC, get_coast_rp_return_curves
from esl_analysis import ESL_stats_from_raw_GESLA, ESL_stats_from_gtsm_dmax, multivariate_normal_gpd_samples_from_covmat, get_return_curve_gpd, get_return_curve_gumbel
from projecting import compute_AFs, compute_AF_timing
from tqdm import tqdm


def get_ESL_statistics(cfg,sites):
    print('Extracting ESL information for queried sites...')
    esl_statistics = {} #initialize dictionary to hold ESL information
    
    if cfg['input']['input_type'] == 'raw': # Option A: derive GPD parameters from raw tide gauge data
        if cfg['input']['input_source'].lower() in ['gesla2','gesla3']: #if using raw data from GESLA
            esl_statistics = ESL_stats_from_raw_GESLA(sites,cfg,0.2)
            
            if cfg['output']['store_distParams']:
                store_esl_params(cfg,sites,esl_statistics) #store ESL statistics as netcdf
                print('Stored derived ESL parameters.')
                
        elif cfg['input']['input_source'].lower() in ['gtsm_dmax']:
            esl_statistics = ESL_stats_from_gtsm_dmax(sites,cfg)
            
            if cfg['output']['store_distParams']: 
                esl_statistics.attrs['config'] = str(cfg)
                esl_statistics.to_netcdf(os.path.join(cfg['output']['output_dir'],'esl_params.nc'),mode='w') #store ESL statistics as netcdf
                print('Stored derived ESL parameters.') 
        else:
            raise Exception('If input_type == "raw", "input_source" must be one of ["gesla2","gesla3","gtsm_dmax"].')
    
            
    elif cfg['input']['input_type'] == 'esl_params': # Option B: read in pre-defined distribution parameters
        if cfg['input']['input_source'] == 'codec_gumbel':
            esl_statistics = get_gum_amax_from_CoDEC(cfg,sites,esl_statistics) # Muis et al. (2020)
           
        elif cfg['input']['input_source'] == 'hermans2023':
            esl_statistics = get_gpd_params_from_Hermans2023(cfg,sites,esl_statistics) # Hermans et al. (2023)
        
        elif cfg['input']['input_source'] == 'kirezci2020':    
            esl_statistics = get_gpd_params_from_Kirezci2020(cfg,sites,esl_statistics) # Kirezci et al. (2021)
        
        elif cfg['input']['input_source'] == 'vousdoukas2018':    
            esl_statistics = get_gpd_params_from_Vousdoukas2018(cfg,sites,esl_statistics) # Vousdoukas et al. (2018)
            
    elif cfg['input']['input_type'] == 'return_curves': # Option C: read in pre-defined return curves 
        if cfg['input']['input_source'].lower() =='coast-rp':
            esl_statistics = get_coast_rp_return_curves(cfg,sites,esl_statistics)
        else:
            raise Exception('If input_type == "return_curves", "input_source" must be in ["coast-rp"].')   
            #others to be implemented
    else:
        raise Exception('Input type not recognized.')
    return esl_statistics
#### 

if __name__ == "__main__":
    
    cfg = load_config('../config.yml')
    
    sites = open_sites_input(cfg) #open queried sites
    #sites = sites.isel(locations = np.where(np.isfinite(sites.sea_level_change.isel(years=0,samples=0)))[0])#temporary, fix nans in nearest interpolation from full gridded samples
    sites = sites.isel(locations=np.arange(5))
    #### Step 1. Get ESL information at queried sites
    
    esl_statistics = get_ESL_statistics(cfg,sites)   
     
    #### Step 2. If going to compute AFs, load sea-level projections and get reference freqs
    if cfg['output']['output_AFs'] + cfg['output']['output_AF_timing'] > 0 : 
        if 'sea_level_change' not in sites:
            raise Exception('Cannot compute amplification factors without sea-level projections.')
        sites['sea_level_change'] = sites['sea_level_change'].load() #load into memory (not feasible if big; to-do is only load when needed, use dask)
        
        refFreqs = get_refFreqs(cfg,sites,esl_statistics) # generate reference frequencies for AFs at each site
        
        if cfg['output']['output_AFs']:
            target_years_ = np.array(str(cfg['projecting']['target_years']).split(',')).astype('int')  #check if target_years in slr projections
            target_years = np.intersect1d(target_years_,sites.years)
            
            if len(target_years)!=len(target_years_):
                warnings.warn('SLR projections do not contain all target years, continuing with a smaller set of target years.')
                
            if len(target_years)==0:
                raise Exception('"target_years" not included in SLR projections.')
    ####
     
    #### Step 3: Do the computations & generate the requested output for each site (possibly to be parallelized?)    
    out_qnts = np.array(cfg['output']['output_quantiles'].split(',')).astype('float') #output quantiles to evaluate results at
    
    f= 10**np.linspace(-6,2,num=1001) #input frequencies to compute return heights for:
    f=np.append(f,np.arange(101,183)) #add up to MHHW frequency
    
    if cfg['output']['output_AFs'] + cfg['output']['output_AF_timing'] + cfg['output']['store_RCs'] > 0: #if requesting at least one kind of output
        i=0    
        sites_output = [] #initialize output for all sites
        print('Computing requested output...')
        for site_id,stats in tqdm(esl_statistics.items()): #for each queried site for which we were able to extract ESL information
            site_rcs = None #initialize output for site
            site_AFs = None
            site_timing = None
            
            #compute/get return curves (to implement: modeling options below threshold)
            if cfg['input']['input_type'] == 'raw': # Option A: translate distribution parameters derived from tide gauge data to return curves
                scale = stats['scale'].iloc[0]
                shape = stats['shape'].iloc[0]
                
                scale_samples,shape_samples = multivariate_normal_gpd_samples_from_covmat(scale,shape,stats['cov'].iloc[0],cfg['general']['num_mc']) #generate scale/shape samples using covariance matrix
                z_hist_ce       = get_return_curve_gpd(f,scale,shape,stats['loc'].iloc[0],stats['avg_extr_pyear'].iloc[0],'mhhw',stats['mhhw'].iloc[0]) #return heights using central estimate scale/shape parameters
                z_hist_samples  = get_return_curve_gpd(f,scale_samples,shape_samples,stats['loc'].iloc[0],stats['avg_extr_pyear'].iloc[0],'mhhw',stats['mhhw'].iloc[0]) #return heights using scaleshape samples
          
            elif cfg['input']['input_type'] == 'esl_params': # Option B: translate predefined distribution parameters to return curves
                if cfg['input']['input_source'] == 'codec_gumbel':
                    z_hist_ce = get_return_curve_gumbel(f,stats['scale'],stats['loc'])
                    z_hist_samples = None #no uncertainty information available
                    
                elif cfg['input']['input_source'] == 'hermans2023':
                    z_hist_ce       = get_return_curve_gpd(f,stats['scale'],stats['shape'],stats['loc'],stats['avg_extr_pyear'])
                    z_hist_samples  = get_return_curve_gpd(f,stats['scale_samples'],stats['shape_samples'],stats['loc'],stats['avg_extr_pyear'])
    
                elif cfg['input']['input_source'] == 'kirezci2020' or cfg['input']['input_source'] == 'vousdoukas2018':
                    z_hist_ce = get_return_curve_gpd(f,stats['scale'],stats['shape'],stats['loc'],stats['avg_extr_pyear'])
                    z_hist_samples = None #no uncertainty information available
                    
            elif cfg['input']['input_type'] == 'return_curves':  # Option C: interpolate predefined return curves to f
                z_hist_ce = np.interp(f,stats['f_hist'],stats['z_hist'],left=np.nan,right=np.nan) #interpolation results in reasonably small differences
                z_hist_samples = None #no uncertainty information available
            
            #store return curves if requested    
            if cfg['output']['store_RCs']: #store return cures
               site_rcs = xr.Dataset(data_vars=dict(z_hist_ce=(["f"],z_hist_ce)),coords=dict(f=f,locations=site_id,),) #central estimate
               if z_hist_samples is not None: 
                   site_rcs['z_hist'] = (['qnt','f'],np.quantile(z_hist_samples,q=out_qnts,axis=-1)) #add samples evaluated at output quantiles
                   site_rcs = site_rcs.assign_coords({'qnt':out_qnts})
               try:    
                   site_rcs['vdatum'] = ([],stats['vdatum'].iloc[0])
               except:
                   pass
               
            #compute amplification factors
            if cfg['output']['output_AFs'] + cfg['output']['output_AF_timing'] > 0:
                refFreq = refFreqs[i] #get reference frequency for current site
                
                if ~np.isfinite(refFreq):
                    print(cfg['projecting']['refFreqs']+' protection standard unavailable for site: '+site_id+', moving on to next site.')
                else:
                    
                    #compute & store AFs:  
                    if cfg['output']['output_AFs']: 
                        output = []
                        
                        if z_hist_samples is not None:
                            z_hist = z_hist_samples
                        else:
                            z_hist = z_hist_ce
                            print('Warning: Computing z_fut & amplification factors without propagating uncertainty in historical return periods.')
                        
                        for yr in target_years:
                            AF_samples,maxAF,z_fut_samples = compute_AFs(f,z_hist,sites.sel(locations=site_id).sel(years=yr).sea_level_change.values,refFreq)
                            
                            output_ds = xr.Dataset(data_vars=dict(z_fut=(["qnt",'f'], np.quantile(z_fut_samples,q=out_qnts,axis=-1)),AF=(["qnt"],np.quantile(AF_samples,out_qnts))),
                                    coords=dict(f=(["f"], f),qnt=(["qnt"], out_qnts),year=yr,locations=site_id,),)
                             
                            output.append(output_ds)
                            
                        site_AFs = xr.concat(output,dim='year')
                        site_AFs['refFreq'] = ([],refFreq)
                        site_AFs['maxAF'] = ([],maxAF)
                 
                    #compute & store AF timing:
                    if cfg['output']['output_AF_timing']:
                        annual_slr = sites.sel(locations=site_id).interp(years=np.arange(sites.years[0],sites.years[-1]+1),method='quadratic') #interpolate SLR to annual timesteps
                        
                        if z_hist_samples is not None:
                            z_hist = z_hist_samples
                        else:
                            z_hist = z_hist_ce
                            print('Warning: Computing z_fut & amplification factor timings without propagating uncertainty in historical return periods.')
                            
                        if 'target_AFs' in cfg['projecting']: #if computing timing of AF relative to refFreq
                            output = []
                            
                            target_AFs = np.array(str(cfg['projecting']['target_AFs']).split(',')).astype('float')
                            
                            for target_AF in target_AFs:
                                timing = compute_AF_timing(f,z_hist,annual_slr.sea_level_change,refFreq,AF=target_AF)
                                
                                output_ds = xr.Dataset(data_vars=dict(af_timing=(["qnt"],np.quantile(timing,q=out_qnts,axis=-1).astype('int'))),
                                        coords=dict(qnt=(["qnt"], out_qnts),target_AF=target_AF,locations=site_id,),)
                                output.append(output_ds)
                            af_timing = xr.concat(output,dim='target_AF')
                            
                        if 'target_freqs' in cfg['projecting']: #if computing timing of refFreq going to target_freq (AF=target_freq/refFreq)
                            output = []
                            
                            target_freqs = np.array(str(cfg['projecting']['target_freqs']).split(',')).astype('float')
                        
                            for target_freq in target_freqs:
                                timing = compute_AF_timing(f,z_hist,annual_slr.sea_level_change,refFreq,AF=target_freq/refFreq)
                                
                                output_ds = xr.Dataset(data_vars=dict(freq_timing=(["qnt"],np.quantile(timing,q=out_qnts,axis=-1).astype('int'))),
                                        coords=dict(qnt=(["qnt"], out_qnts),target_f=target_freq,locations=site_id,),)
                        
                                output.append(output_ds)
                            freq_timing = xr.concat(output,dim='target_f')
                        
                        try:
                            site_timing = xr.merge((af_timing,freq_timing))
                        except:
                            try:
                                site_timing = af_timing
                            except:
                                site_timing = freq_timing
                                
                        site_timing['refFreq'] = ([],refFreq)
                    
            site_output = [k for k in [site_rcs,site_AFs,site_timing] if k!=None] #merge different requested outputs for current site
            if site_output!=[]:
                sites_output.append(xr.merge(site_output)) #append current site output to all sites output
            i+=1
        
        output_ds = xr.concat(sites_output,dim='locations',coords='minimal')
        output_ds = output_ds.assign_coords({'lat':sites.lat.sel(locations=output_ds.locations),'lon':sites.lon.sel(locations=output_ds.locations)})
        output_ds.attrs['config'] = str(cfg)
        output_ds.to_netcdf(os.path.join(cfg['output']['output_dir'],'projectESL_output_'+cfg['general']['run_name']+'.nc'),mode='w') #save file
    
    '''
    #temporary plot to check where coast rp is too far from tide gauges
    from utils import angdist
    import pandas as pd
    import matplotlib.pyplot as plt
    coast_rp_coords = pd.read_pickle(os.path.join(cfg['input']['paths']['coast-rp'],'pxyn_coastal_points.xyn'))
    angdists = [np.sort(angdist(x,y,coast_rp_coords['lat'].values,coast_rp_coords['lon'].values))[0] for x,y in zip(sites.lat.values, sites.lon.values)]
    
    gpd_params = xr.open_dataset(cfg['input']['paths']['vousdoukas2018'])
    
    angdists = [np.sort(angdist(x,y,gpd_params.lat.values,gpd_params.lon.values))[0] for x,y in zip(sites.lat.values, sites.lon.values)]
    
    
    idx = np.where(np.array(angdists)>0.2)
    import cartopy
    import cartopy.crs as ccrs
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    fig=plt.figure(figsize=(9,9.5)) #generate figure 
    gs = fig.add_gridspec(1,1)
    gs.update(hspace=.2)
    
    ax = plt.subplot(gs[0,0],projection=ccrs.Robinson(central_longitude=0))
    
    ax.add_feature(cartopy.feature.OCEAN, zorder=0,facecolor='grey')
    ax.add_feature(cartopy.feature.LAND, zorder=0, facecolor='grey')
    ax.scatter(gpd_params.lon.values,gpd_params.lat.values,c='blue',transform=ccrs.PlateCarree())
    sc=ax.scatter(sites.lon[idx],sites.lat[idx],c='red',transform=ccrs.PlateCarree(),zorder=3)
    
    sc.set_edgecolor('face')
    ax.coastlines(zorder=1)
    '''