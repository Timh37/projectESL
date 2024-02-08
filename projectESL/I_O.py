'''
@author: Tim Hermans
t(dot)h(dot)j(dot)hermans@uu(dot)nl
'''

import numpy as np
import xarray as xr
import pandas as pd
import yaml
import os
from projecting import find_flopros_protection_levels, find_diva_protection_levels
from utils import mindist

def load_config(cfgpath):
    '''load projectESL configuration from "cfgpath" '''
    with open(cfgpath, 'r') as f: #open configuration file
        return yaml.safe_load(f)

def open_sites_input(cfg):
    '''
    Open input sites input file.
    
    Input must be a netcdf or zarr file containing lon/lat coords with 'locations' dimension.
    
    Providing sea-level projections as variable 'sea_level_change' with dimensions 'locations', 'years','samples' is optional depending on requested output.
    
    Format following FACTS output.
    '''
    fn = cfg['input']['paths']['sites_input']
    if fn.endswith('.zarr'):
        sites = xr.open_dataset(fn,engine='zarr',chunks='auto')
    elif fn.endswith('.nc'):
        sites = xr.open_dataset(fn,chunks='auto')
    else:
        raise Exception('Sites input file format not recognized.')

    if 'locations' not in sites.dims:
        sites = sites.expand_dims('locations')
    sites['locations'] = sites.locations.astype('str') #to use them as keys for dictionary
    
    try:
        sites = sites.rename({'latitude':'lat','longitude':'lon'})
    except:
        pass
    
    if ('lon' not in sites) or ('lat' not in sites):
        raise Exception('Sites input must contain lon/lat coordinates.')
    
    if 'sea_level_change' in sites: #determine if file also contains sea-level projections:
        if 'samples' in sites.sea_level_change.dims: #randomly select "num_mc" samples
            if cfg['general']['num_mc']>len(sites.samples):
                raise Exception('Insufficient SLR samples for desired number of Monte Carlo samples.')
            else:
                idx = np.sort(np.random.choice(len(sites.samples), size=cfg['general']['num_mc'], replace=False))
            sites = sites.isel(samples = idx)
        else:
            print('Warning: Amplification factors will be computed without propagating uncertainty in sea-level projections.')
        
        if sites.sea_level_change.units!='m':
            if sites.sea_level_change.units=='mm':
                sites['sea_level_change'] = sites['sea_level_change']/1000
                sites.sea_level_change.attrs['units'] = 'm'
            elif sites.sea_level_change.units=='cm':
                sites['sea_level_change'] = sites['sea_level_change']/100
                sites.sea_level_change.attrs['units'] = 'm'
            else:
                raise Exception('Unit of sea-level projections not recognized.')
            
    return sites
    
def store_esl_params(cfg,sites,esl_statistics):
    '''stores ESL parameters derived from tide gauge records held in "esl_statistics" dictionary at queried sites as .nc file'''
    ds = xr.Dataset(
    data_vars=dict(
        loc=(['locations'], [v['loc'].values[0] for k,v in esl_statistics.items()]),
        scale=(['locations'], [v['scale'].values[0] for k,v in esl_statistics.items()]),
        shape=(['locations'], [v['shape'].values[0] for k,v in esl_statistics.items()]),
        cov=(['locations','i','j'], [v['cov'].values[0] for k,v in esl_statistics.items()]),
        avg_extr_pyear=(['locations'], [v['avg_extr_pyear'].values[0] for k,v in esl_statistics.items()]),
        mhhw=(['locations'], [v['mhhw'].values[0] for k,v in esl_statistics.items()]),
    ),
    coords=dict(
        lon=(['locations'], sites.sel(locations=list(esl_statistics.keys())).lon.values),
        lat=(['locations'], sites.sel(locations=list(esl_statistics.keys())).lat.values),
        locations=sites.sel(locations=list(esl_statistics.keys())).locations.values,
    ),
    )
    
    ds.attrs['config'] = str(cfg)
    return ds.to_netcdf(os.path.join(cfg['output']['output_dir'],'esl_params.nc'),mode='w')

def get_refFreqs(cfg,sites,esl_statistics):
    ''' determine reference frequencies for queried sites based on user options in config'''
    settings = cfg['projecting']
    if settings['refFreqs'] == 'diva': #find contemporary DIVA flood protection levels
        qlats = [sites.lat.sel(locations=k).values for k in esl_statistics.keys()]
        qlons = [sites.lon.sel(locations=k).values for k in esl_statistics.keys()]
        refFreqs = find_diva_protection_levels(qlons,qlats,cfg['input']['paths']['diva'],15) #set these paths in config
    
    elif settings['refFreqs'] == 'flopros': #find contemporary FLOPROS flood protection levels
        qlats = [sites.lat.sel(locations=k).values for k in esl_statistics.keys()]
        qlons = [sites.lon.sel(locations=k).values for k in esl_statistics.keys()]
        
        refFreqs = find_flopros_protection_levels(qlons,qlats,cfg['input']['paths']['flopros'],15)    
    
    elif np.isscalar(settings['refFreqs']): #use constant reference frequency for every location
            refFreqs = np.tile(settings['refFreqs'],len(esl_statistics))
    else: 
        raise Exception('Rquested reference frequency must be "diva", "flopros" or a constant.')
            
    return refFreqs


def get_gpd_params_from_Hermans2023(cfg,sites,esl_statistics):
    '''open GPD parameters from Hermans et al. (2023) NCLIM and find parameters nearest to queried sites'''
    gpd_params = xr.open_dataset(cfg['input']['paths']['hermans2023'])
    min_idx = [mindist(x,y,gpd_params.lat.values,gpd_params.lon.values, 0.2) for x,y in zip(sites.lat.values, sites.lon.values)]
    
    if len(gpd_params.nboot) > cfg['general']['num_mc']:
        isamps = np.random.randint(0,len(gpd_params.nboot),cfg['general']['num_mc'])
    elif len(gpd_params.nboot) < cfg['general']['num_mc']:
        isamps = np.hstack((gpd_params.nboot.values,np.random.randint(0,len(gpd_params.nboot),cfg['general']['num_mc']-len(gpd_params.nboot))))
    else:
        isamps = gpd_params.nboot.values
        
    for i in np.arange(len(sites.locations)):
        if min_idx[i] is not None:
            if np.isscalar(min_idx[i]) == False: #if multiple sites within radius, pick the first (we don't have information about series length here)
                min_idx[i] = min_idx[i][0]
            esl_statistics[sites.locations.values[i]] = {}
            esl_statistics[sites.locations.values[i]]['loc'] = gpd_params.isel(site=min_idx[i]).location.values
            esl_statistics[sites.locations.values[i]]['avg_extr_pyear'] = gpd_params.isel(site=min_idx[i]).avg_exceed.values
            esl_statistics[sites.locations.values[i]]['scale_samples'] = gpd_params.isel(site=min_idx[i]).scale_samples.values[isamps]
            esl_statistics[sites.locations.values[i]]['shape_samples'] = gpd_params.isel(site=min_idx[i]).shape_samples.values[isamps]
            esl_statistics[sites.locations.values[i]]['scale'] = gpd_params.isel(site=min_idx[i]).scale_samples.values
            esl_statistics[sites.locations.values[i]]['shape'] = gpd_params.isel(site=min_idx[i]).shape_samples.values
        else:
            print("Warning: No nearby ESL information found for {0}. Skipping site.".format(sites.locations[i].values))
            continue
        
    return esl_statistics


def get_gpd_params_from_Kirezci2020(cfg,sites,esl_statistics):
    from esl_analysis import infer_avg_extr_pyear
    '''open GPD parameters from Kirezci et al. (2020) and find parameters nearest to queried sites'''
    gpd_params = pd.read_csv(cfg['input']['paths']['kirezci2020'])
    
    min_idx = [mindist(x,y,gpd_params.lat.values,gpd_params.lon.values, 0.2) for x,y in zip(sites.lat.values, sites.lon.values)]

    for i in np.arange(len(sites.locations)):
        if min_idx[i] is not None:
            if np.isscalar(min_idx[i]) == False: #if multiple sites within radius, pick the first (we don't have information about series length here)
                min_idx[i] = min_idx[i][0]
            esl_statistics[sites.locations.values[i]] = {}
            esl_statistics[sites.locations.values[i]]['loc'] = gpd_params.iloc[min_idx[i]].GPDthresh
            esl_statistics[sites.locations.values[i]]['scale'] = gpd_params.iloc[min_idx[i]].GPDscale
            esl_statistics[sites.locations.values[i]]['shape'] = gpd_params.iloc[min_idx[i]].GPDshape
            esl_statistics[sites.locations.values[i]]['avg_extr_pyear'] = infer_avg_extr_pyear(gpd_params.iloc[min_idx[i]].GPDthresh,
                                                                                               gpd_params.iloc[min_idx[i]].GPDscale,
                                                                                               gpd_params.iloc[min_idx[i]].GPDshape,
                                                                                               gpd_params.iloc[min_idx[i]].TWL,
                                                                                               1/100)
        else:
            print("Warning: No nearby ESL information found for {0}. Skipping site.".format(sites.locations[i].values))
            continue
        
    return esl_statistics

def get_gpd_params_from_Vousdoukas2018(cfg,sites,esl_statistics):
    from esl_analysis import infer_avg_extr_pyear
    '''open GPD parameters from Vousdoukas et al. (2018) and find parameters nearest to queried sites'''
    gpd_params = xr.open_dataset(cfg['input']['paths']['vousdoukas2018'])
    
    min_idx = [mindist(x,y,gpd_params.lat.values,gpd_params.lon.values, 0.2) for x,y in zip(sites.lat.values, sites.lon.values)]

    for i in np.arange(len(sites.locations)):
        if min_idx[i] is not None:
            if np.isscalar(min_idx[i]) == False: #if multiple sites within radius, pick the first (we don't have information about series length here)
                min_idx[i] = min_idx[i][0]
            esl_statistics[sites.locations.values[i]] = {}
            esl_statistics[sites.locations.values[i]]['loc'] = gpd_params.isel(pt=min_idx[i]).thresholdParamGPD.item()
            esl_statistics[sites.locations.values[i]]['scale'] = gpd_params.isel(pt=min_idx[i]).scaleParamGPD.item()
            esl_statistics[sites.locations.values[i]]['shape'] = gpd_params.isel(pt=min_idx[i]).shapeParamGPD.item()
            esl_statistics[sites.locations.values[i]]['avg_extr_pyear'] = infer_avg_extr_pyear(gpd_params.isel(pt=min_idx[i]).thresholdParamGPD.item(),
                                                                                               gpd_params.isel(pt=min_idx[i]).scaleParamGPD.item(),
                                                                                               gpd_params.isel(pt=min_idx[i]).shapeParamGPD.item(),
                                                                                               gpd_params.isel(pt=min_idx[i]).returnlevelGPD.isel(return_period=12).item(),
                                                                                               1/100)
        else:
            print("Warning: No nearby ESL information found for {0}. Skipping site.".format(sites.locations[i].values))
            continue
        
    return esl_statistics

def get_gum_amax_from_CoDEC(cfg,sites,esl_statistics):
    '''open Gumbel parameters from Muis et al. (2020) and find parameters nearest to queried sites'''
    codec_gum = xr.open_dataset(cfg['input']['paths']['codec_gumbel'])
    min_idx = [mindist(x,y,codec_gum['station_y_coordinate'].values,codec_gum['station_x_coordinate'].values, 0.2)[0] for x,y in zip(sites.lat.values, sites.lon.values)]
    
    for i in np.arange(len(sites.locations)):
        if min_idx[i] is not None:
            if np.isscalar(min_idx[i]) == False:
                min_idx[i] = min_idx[i][0]
            esl_statistics[sites.locations.values[i]] = {}
            esl_statistics[sites.locations.values[i]]['loc'] = codec_gum.isel(stations=min_idx[i]).GUM.values[0]
            esl_statistics[sites.locations.values[i]]['scale'] = codec_gum.isel(stations=min_idx[i]).GUM.values[-1]
        else:
            print("Warning: No nearby ESL information found for {0}. Skipping site.".format(sites.locations[i].values))
            continue
        
    return esl_statistics


def get_coast_rp_return_curves(cfg,sites,esl_statistics):
    '''open return curves from Dulaart et al. (2021) and find curves nearest to queried sites'''
    coast_rp_coords = pd.read_pickle(os.path.join(cfg['input']['paths']['coast-rp'],'pxyn_coastal_points.xyn'))
    min_idx = [mindist(x,y,coast_rp_coords['lat'].values,coast_rp_coords['lon'].values, 0.2) for x,y in zip(sites.lat.values, sites.lon.values)]
    #use a slightly larger distance tolerance here because GTSM output is used at locations every 25 km along the global coastline (Dullart et al., 2021).
    
    for i in np.arange(len(sites.locations)):
        if min_idx[i] is not None:
            if np.isscalar(min_idx[i]) == False: #if multiple sites within radius, pick the first (we don't have information about series length here)
                min_idx[i] = min_idx[i][0]
            this_id = str(int(coast_rp_coords.iloc[min_idx[i]].name))
            if int(coast_rp_coords.iloc[min_idx[i]].name)<10000:
                this_id = '0'+this_id
           
            rc = pd.read_pickle(os.path.join(cfg['input']['paths']['coast-rp'],'rp_full_empirical_station_'+this_id+'.pkl'))
            rc =rc['rp'][np.isfinite(rc['rp'])]
            #in some cases coast rp RPs can be non-monotonically increasing, for low heights if tcs defined where etcs not defined; we don't want to use this part
            d = np.where(np.diff(rc)<0)[0]#find where not increasing
            try:
                rc = rc.iloc[d[0]+1::]
            except:
                pass
        
            esl_statistics[sites.locations.values[i]] = {}
            esl_statistics[sites.locations.values[i]]['z_hist'] = np.flip(rc.index.values)
            esl_statistics[sites.locations.values[i]]['f_hist'] = np.flip(1/rc.values) 
        else:
            print("Warning: No nearby ESL information found for {0}. Skipping site.".format(sites.locations[i].values))
            continue
    return esl_statistics

def open_CoDEC_waterlevels(cfg, sites):
    codec = xr.open_mfdataset(os.path.join(cfg['input']['paths']['codec_dmax'],'*.nc')) #assumes daily maxima in similar file structure as original CDS download
    codec_lats = codec.station_y_coordinate.values
    codec_lons = codec.station_x_coordinate.values
    
    min_idx = [mindist(x,y,codec_lats,codec_lons,0.2) for x,y in zip(sites.lat.values, sites.lon.values)]
    for k in np.arange(len(sites.locations)):
        if min_idx[k] is not None:
            min_idx[k] = min_idx[k][0]
        else:
            min_idx[k] = np.nan
            print("Warning: No nearby ESL information found for {0}. Skipping site.".format(sites.locations[k].values))
    codec = codec.isel(stations = np.array(min_idx)[np.isfinite(min_idx)])
    codec = codec.assign_coords({'locations':sites.locations[np.isfinite(min_idx)]})
        
    codec = codec.load() #load data into memory
    
    #do some cleaning up (not entirely sure why this is necessary, also seems to be the case in the original dataset)
    codec = codec.drop_isel(stations = np.where(np.isnan(codec.waterlevel).any(dim='time'))[0]) #remove stations containing nans in their timeseries
    codec = codec.where((codec.waterlevel.var(dim='time')>1e-5)).dropna(dim='stations') #remove weird stations with very low variance (erroneous, sea ice, internal seas?)

    return codec