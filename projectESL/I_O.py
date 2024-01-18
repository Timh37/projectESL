import numpy as np
import xarray as xr
import pandas as pd
import yaml
import os
from projecting import find_flopros_protection_levels, find_diva_protection_levels
from esl_analysis import infer_avg_extr_pyear
from utils import mindist
def load_config(cfgpath):
    with open(cfgpath, 'r') as f: #open configuration file
        return yaml.safe_load(f)
    
def store_esl_params(cfg,sites,esl_statistics):
    
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


def open_slr_projections(cfg):
    
    slr = xr.open_mfdataset((cfg['input']['paths']['slr']),concat_dim='locations',combine='nested').load()
    
    if 'locations' not in slr.dims:
        slr = slr.expand_dims('locations')
    
    if 'samples' in slr.dims:
        if cfg['general']['num_mc']>len(slr.samples):
            raise Exception('Insufficient SLR samples for desired number of Monte Carlo samples.')
        else:
            idx = np.sort(np.random.choice(len(slr.samples), size=cfg['general']['num_mc'], replace=False))
        slr = slr.isel(samples = idx)
    
    try:
        slr = slr.rename({'latitude':'lat','longitude':'lon'})
    except:
        pass
    
    if slr.sea_level_change.units=='mm': #check sea-level change unit is meter
        slr['sea_level_change'] = slr['sea_level_change']/1000
        slr.sea_level_change.attrs['units'] = 'm'
    elif slr.sea_level_change.units=='cm':
        slr['sea_level_change'] = slr['sea_level_change']/100
        slr.sea_level_change.attrs['units'] = 'm'
        
    return slr

def get_refFreqs(cfg,sites,esl_statistics):
    
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
        raise Exception('Reference frequency must be "diva", "flopros" or a constant.')
            
    return refFreqs


def get_gpd_params_from_Hermans2023(cfg,sites,esl_statistics):
    gpd_params = xr.open_dataset(cfg['input']['paths']['hermans2023'])
    min_idx = [mindist(x,y,gpd_params.lat.values,gpd_params.lon.values, 0.1) for x,y in zip(sites.lat.values, sites.lon.values)]
    
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
  
        else:
            print("Warning: No nearby ESL information found for {0}. Skipping site.".format(sites.locations[i].values))
            continue
        
    return esl_statistics


def get_gpd_params_from_Kirezci2020(cfg,sites,esl_statistics):
    gpd_params = pd.read_csv(cfg['input']['paths']['kirezci2020'])
    
    min_idx = [mindist(x,y,gpd_params.lat.values,gpd_params.lon.values, 0.1) for x,y in zip(sites.lat.values, sites.lon.values)]

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
    gpd_params = xr.open_dataset(cfg['input']['paths']['vousdoukas2018'])
    
    min_idx = [mindist(x,y,gpd_params.lat.values,gpd_params.lon.values, 0.1) for x,y in zip(sites.lat.values, sites.lon.values)]

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
    
    codec_gum = xr.open_dataset(cfg['input']['paths']['codec_gumbel'])
    min_idx = [mindist(x,y,codec_gum['station_y_coordinate'].values,codec_gum['station_x_coordinate'].values, 0.1)[0] for x,y in zip(sites.lat.values, sites.lon.values)]
    
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
    coast_rp_coords = pd.read_pickle(os.path.join(cfg['input']['paths']['coast-rp'],'pxyn_coastal_points.xyn'))
    min_idx = [mindist(x,y,coast_rp_coords['lat'].values,coast_rp_coords['lon'].values, 0.1)[0] for x,y in zip(sites.lat.values, sites.lon.values)]
    
    for i in np.arange(len(sites.locations)):
        if min_idx[i] is not None:
            if np.isscalar(min_idx[i]) == False: #if multiple sites within radius, pick the first (we don't have information about series length here)
                min_idx[i] = min_idx[i][0]
            this_id = str(int(coast_rp_coords.iloc[min_idx[i]].name))
            rc = pd.read_pickle(os.path.join(cfg['input']['paths']['coast-rp'],'rp_full_empirical_station_'+this_id+'.pkl'))
            rc = rc['rp'][np.isfinite(rc['rp'])]
            
            esl_statistics[sites.locations.values[i]] = {}
            esl_statistics[sites.locations.values[i]]['z_hist'] = np.flip(rc.index.values)
            esl_statistics[sites.locations.values[i]]['f_hist'] = np.flip(1/rc.values) 
        else:
            print("Warning: No nearby ESL information found for {0}. Skipping site.".format(sites.locations[i].values))
            continue
    return esl_statistics