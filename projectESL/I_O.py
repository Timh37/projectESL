import numpy as np
import xarray as xr
import pandas as pd
import yaml
import os
from projecting import find_flopros_protection_levels, find_diva_protection_levels
 
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
        lon=(['locations'], sites.lon.values),
        lat=(['locations'], sites.lat.values),
        locations=sites.locations.values,
    ),
    )
    
    ds.attrs['config'] = str(cfg)
    return ds.to_netcdf(os.path.join(cfg['output']['output_dir'],'esl_params.nc'),mode='w')


def open_slr_projections(cfg):
    
    slr = xr.open_mfdataset((os.path.join(cfg['general']['project_path'],'input',cfg['projecting']['slr_filename'])),concat_dim='locations',combine='nested').load()
    
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
    
    settings = cfg['projecting']['projection_settings']
    if settings['refFreqs'] == 'diva': #find contemporary DIVA flood protection levels
        qlats = [sites.lat.sel(locations=k).values for k in esl_statistics.keys()]
        qlons = [sites.lon.sel(locations=k).values for k in esl_statistics.keys()]
        refFreqs = find_diva_protection_levels(qlons,qlats,"/Users/timhermans/Documents/Data/DIVA/cls.gpkg",15) #set these paths in config
    
    elif settings['refFreqs'] == 'flopros': #find contemporary FLOPROS flood protection levels
        qlats = [sites.lat.sel(locations=k).values for k in esl_statistics.keys()]
        qlons = [sites.lon.sel(locations=k).values for k in esl_statistics.keys()]
        
        refFreqs = find_flopros_protection_levels(qlons,qlats,'/Users/timhermans/Documents/Data/FLOPROS/Tiggeloven/',15)    
    
    else:
        if np.isscalar(refFreqs): #use constant reference frequency for every location
            refFreqs = np.tile(refFreqs,len(esl_statistics))
        else: 
            raise Exception('Reference frequency must be "DIVA", "FLOPROS" or a constant.')
            
    return refFreqs