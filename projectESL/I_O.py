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

def load_config(path_to_config):
    '''load projectESL configuration from "cfgpath" '''
    with open(path_to_config, 'r') as f: #open configuration file
        return yaml.safe_load(f)

def open_input_locations(path_to_input,num_samps):
    '''
    Open input input_locations input file.
    
    Input must be a netcdf or zarr file containing lon/lat coords with 'locations' dimension.
    
    Providing sea-level projections as variable 'sea_level_change' with dimensions 'locations', 'years','samples' is optional depending on requested output.
    
    Format following FACTS output.
    '''
    fn = path_to_input
    if fn.endswith('.zarr'):
        input_locations = xr.open_dataset(fn,engine='zarr',chunks='auto')
    elif fn.endswith('.nc'):
        input_locations = xr.open_dataset(fn,chunks='auto')
    else:
        raise Exception('Input_locations input file format not recognized.')

    if 'locations' not in input_locations.dims:
        input_locations = input_locations.expand_dims('locations')
    input_locations['locations'] = input_locations.locations.astype('str') #to use them as keys for dictionary
    
    try:
        input_locations = input_locations.rename({'latitude':'lat','longitude':'lon'})
    except:
        pass
    
    if ('lon' not in input_locations) or ('lat' not in input_locations):
        raise Exception('Input_locations input must contain lon/lat coordinates.')
    
    #load coordinates into memory if not already
    input_locations['lon'] = (['locations'],input_locations['lon'].values)
    input_locations['lat'] = (['locations'],input_locations['lat'].values)
    
    if 'sea_level_change' in input_locations: #determine if file also contains sea-level projections:
        if 'samples' in input_locations.sea_level_change.dims: #randomly select "num_mc" samples
            if num_samps>len(input_locations.samples):
                raise Exception('Insufficient SLR samples for desired number of Monte Carlo samples.')
            else:
                idx = np.sort(np.random.choice(len(input_locations.samples), size=num_samps, replace=False))
            input_locations = input_locations.isel(samples = idx)
        else:
            raise Exception('SLR projections do not contain samples required for uncertainty propagation.')
        
        if input_locations.sea_level_change.units!='m':
            if input_locations.sea_level_change.units=='mm':
                input_locations['sea_level_change'] = input_locations['sea_level_change']/1000
                input_locations.sea_level_change.attrs['units'] = 'm'
            elif input_locations.sea_level_change.units=='cm':
                input_locations['sea_level_change'] = input_locations['sea_level_change']/100
                input_locations.sea_level_change.attrs['units'] = 'm'
            else:
                raise Exception('Unit of sea-level projections not recognized.')
        
    return input_locations

    
def esl_statistics_dict_to_ds(input_locations,esl_statistics):
    '''stores ESL parameters derived from tide gauge records held in "esl_statistics" dictionary at input_locations as .nc file'''
    ds = xr.Dataset(
    data_vars=dict(
        loc=(['locations'], [v['loc'].values[0] for k,v in esl_statistics.items()]),
        scale=(['locations'], [v['scale'].values[0] for k,v in esl_statistics.items()]),
        shape=(['locations'], [v['shape'].values[0] for k,v in esl_statistics.items()]),
        cov=(['locations','i','j'], [v['cov'].values[0] for k,v in esl_statistics.items()]),
        avg_extr_pyear=(['locations'], [v['avg_extr_pyear'].values[0] for k,v in esl_statistics.items()]),
        mhhw=(['locations'], [v['mhhw'].values[0] for k,v in esl_statistics.items()]), #to-do: this is not always in there
    ),
    coords=dict(
        lon=(['locations'], input_locations.sel(locations=list(esl_statistics.keys())).lon.values),
        lat=(['locations'], input_locations.sel(locations=list(esl_statistics.keys())).lat.values),
        locations=input_locations.sel(locations=list(esl_statistics.keys())).locations.values,
    ),
    )
   
    return ds

def save_esl_statistics(output_path,esl_statistics):
    return esl_statistics.to_netcdf(os.path.join(output_path,'esl_statistics.nc'),mode='w')

def get_refFreqs(refFreq_data,input_locations,esl_statistics,path_to_refFreqs=None):
    ''' determine reference frequencies for queried input_locations based on user options in config'''
    
    if refFreq_data == 'diva': #find contemporary DIVA flood protection levels
        qlats = [input_locations.lat.sel(locations=k).values for k in esl_statistics.keys()]
        qlons = [input_locations.lon.sel(locations=k).values for k in esl_statistics.keys()]
        refFreqs = find_diva_protection_levels(qlons,qlats,path_to_refFreqs,15) #set these paths in config
    
    elif refFreq_data == 'flopros': #find contemporary FLOPROS flood protection levels
        qlats = [input_locations.lat.sel(locations=k).values for k in esl_statistics.keys()]
        qlons = [input_locations.lon.sel(locations=k).values for k in esl_statistics.keys()]
        
        refFreqs = find_flopros_protection_levels(qlons,qlats,path_to_refFreqs,15)    
    
    elif np.isscalar(refFreq_data): #use constant reference frequency for every location
            refFreqs = np.tile(refFreq_data,len(esl_statistics))
    else: 
        raise Exception('Rquested reference frequency must be "diva", "flopros" or a constant.')
            
    return refFreqs

def open_gpd_parameters(input_data,data_path,input_locations,n_samples):
    esl_statistics = {}
    
    if input_data == 'hermans2023':
        
        gpd_params = xr.open_dataset(data_path)
        min_idx = [mindist(x,y,gpd_params.lat.values,gpd_params.lon.values, 0.2) for x,y in zip(input_locations.lat.values, input_locations.lon.values)]
        
        if len(gpd_params.nboot) > n_samples:
            isamps = np.random.randint(0,len(gpd_params.nboot),n_samples)
        elif len(gpd_params.nboot) < n_samples:
            isamps = np.hstack((gpd_params.nboot.values,np.random.randint(0,len(gpd_params.nboot),n_samples-len(gpd_params.nboot))))
        else:
            isamps = gpd_params.nboot.values
            
        for i in np.arange(len(input_locations.locations)):
            if min_idx[i] is not None:
                if np.isscalar(min_idx[i]) == False: #if multiple input_locations within radius, pick the first (we don't have information about series length here)
                    min_idx[i] = min_idx[i][0]
                esl_statistics[input_locations.locations.values[i]] = {}
                esl_statistics[input_locations.locations.values[i]]['loc'] = gpd_params.isel(site=min_idx[i]).location.values
                esl_statistics[input_locations.locations.values[i]]['avg_extr_pyear'] = gpd_params.isel(site=min_idx[i]).avg_exceed.values
                esl_statistics[input_locations.locations.values[i]]['scale_samples'] = gpd_params.isel(site=min_idx[i]).scale_samples.values[isamps]
                esl_statistics[input_locations.locations.values[i]]['shape_samples'] = gpd_params.isel(site=min_idx[i]).shape_samples.values[isamps]
                esl_statistics[input_locations.locations.values[i]]['scale'] = gpd_params.isel(site=min_idx[i]).scale_samples.values
                esl_statistics[input_locations.locations.values[i]]['shape'] = gpd_params.isel(site=min_idx[i]).shape_samples.values
            else:
                print("Warning: No nearby ESL information found for {0}. Skipping site.".format(input_locations.locations[i].values))
                continue
            
    elif input_data == 'kirezci2020':
        from esl_analysis import infer_avg_extr_pyear
        
        '''open GPD parameters from Kirezci et al. (2020) and find parameters nearest to queried input_locations'''
        gpd_params = pd.read_csv(data_path)
        
        min_idx = [mindist(x,y,gpd_params.lat.values,gpd_params.lon.values, 0.2) for x,y in zip(input_locations.lat.values, input_locations.lon.values)]
    
        for i in np.arange(len(input_locations.locations)):
            if min_idx[i] is not None:
                if np.isscalar(min_idx[i]) == False: #if multiple input_locations within radius, pick the first (we don't have information about series length here)
                    min_idx[i] = min_idx[i][0]
                esl_statistics[input_locations.locations.values[i]] = {}
                esl_statistics[input_locations.locations.values[i]]['loc'] = gpd_params.iloc[min_idx[i]].GPDthresh
                esl_statistics[input_locations.locations.values[i]]['scale'] = gpd_params.iloc[min_idx[i]].GPDscale
                esl_statistics[input_locations.locations.values[i]]['shape'] = gpd_params.iloc[min_idx[i]].GPDshape
                esl_statistics[input_locations.locations.values[i]]['avg_extr_pyear'] = infer_avg_extr_pyear(gpd_params.iloc[min_idx[i]].GPDthresh,
                                                                                                   gpd_params.iloc[min_idx[i]].GPDscale,
                                                                                                   gpd_params.iloc[min_idx[i]].GPDshape,
                                                                                                   gpd_params.iloc[min_idx[i]].TWL,
                                                                                                   1/100)
            else:
                print("Warning: No nearby ESL information found for {0}. Skipping site.".format(input_locations.locations[i].values))
                continue
            
    elif input_data == 'vousdoukas2018':
        from esl_analysis import infer_avg_extr_pyear
        '''open GPD parameters from Vousdoukas et al. (2018) and find parameters nearest to queried input_locations'''
        gpd_params = xr.open_dataset(data_path)
        
        min_idx = [mindist(x,y,gpd_params.lat.values,gpd_params.lon.values, 0.2) for x,y in zip(input_locations.lat.values, input_locations.lon.values)]
    
        for i in np.arange(len(input_locations.locations)):
            if min_idx[i] is not None:
                if np.isscalar(min_idx[i]) == False: #if multiple input_locations within radius, pick the first (we don't have information about series length here)
                    min_idx[i] = min_idx[i][0]
                esl_statistics[input_locations.locations.values[i]] = {}
                esl_statistics[input_locations.locations.values[i]]['loc'] = gpd_params.isel(pt=min_idx[i]).thresholdParamGPD.item()
                esl_statistics[input_locations.locations.values[i]]['scale'] = gpd_params.isel(pt=min_idx[i]).scaleParamGPD.item()
                esl_statistics[input_locations.locations.values[i]]['shape'] = gpd_params.isel(pt=min_idx[i]).shapeParamGPD.item()
                esl_statistics[input_locations.locations.values[i]]['avg_extr_pyear'] = infer_avg_extr_pyear(gpd_params.isel(pt=min_idx[i]).thresholdParamGPD.item(),
                                                                                                   gpd_params.isel(pt=min_idx[i]).scaleParamGPD.item(),
                                                                                                   gpd_params.isel(pt=min_idx[i]).shapeParamGPD.item(),
                                                                                                   gpd_params.isel(pt=min_idx[i]).returnlevelGPD.isel(return_period=12).item(),
                                                                                                   1/100)
            else:
                print("Warning: No nearby ESL information found for {0}. Skipping site.".format(input_locations.locations[i].values))
                continue
            
    elif input_data == 'codec_gumbel':
        '''open Gumbel parameters from Muis et al. (2020) and find parameters nearest to queried input_locations'''
        codec_gum = xr.open_dataset(data_path)
        min_idx = [mindist(x,y,codec_gum['station_y_coordinate'].values,codec_gum['station_x_coordinate'].values, 0.2)[0] for x,y in zip(input_locations.lat.values, input_locations.lon.values)]
        
        for i in np.arange(len(input_locations.locations)):
            if min_idx[i] is not None:
                if np.isscalar(min_idx[i]) == False:
                    min_idx[i] = min_idx[i][0]
                esl_statistics[input_locations.locations.values[i]] = {}
                esl_statistics[input_locations.locations.values[i]]['loc'] = codec_gum.isel(stations=min_idx[i]).GUM.values[0]
                esl_statistics[input_locations.locations.values[i]]['scale'] = codec_gum.isel(stations=min_idx[i]).GUM.values[-1]
            else:
                print("Warning: No nearby ESL information found for {0}. Skipping site.".format(input_locations.locations[i].values))
                continue
    return esl_statistics



def get_coast_rp_return_curves(input_dir,input_locations):
    '''open return curves from Dulaart et al. (2021) and find curves nearest to queried input_locations'''
    esl_statistics={}
    
    coast_rp_coords = pd.read_pickle(os.path.join(input_dir,'pxyn_coastal_points.xyn'))
    min_idx = [mindist(x,y,coast_rp_coords['lat'].values,coast_rp_coords['lon'].values, 0.2) for x,y in zip(input_locations.lat.values, input_locations.lon.values)]
    #use a slightly larger distance tolerance here because GTSM output is used at locations every 25 km along the global coastline (Dullart et al., 2021).
    
    for i in np.arange(len(input_locations.locations)):
        if min_idx[i] is not None:
            if np.isscalar(min_idx[i]) == False: #if multiple input_locations within radius, pick the first (we don't have information about series length here)
                min_idx[i] = min_idx[i][0]
            this_id = str(int(coast_rp_coords.iloc[min_idx[i]].name))
            if int(coast_rp_coords.iloc[min_idx[i]].name)<10000:
                this_id = '0'+this_id
           
            rc = pd.read_pickle(os.path.join(input_dir,'rp_full_empirical_station_'+this_id+'.pkl'))
            rc =rc['rp'][np.isfinite(rc['rp'])]
            #in some cases coast rp RPs can be non-monotonically increasing, for low heights if tcs defined where etcs not defined; we don't want to use this part
            d = np.where(np.diff(rc)<0)[0]#find where not increasing
            try:
                rc = rc.iloc[d[0]+1::]
            except:
                pass
        
            esl_statistics[input_locations.locations.values[i]] = {}
            esl_statistics[input_locations.locations.values[i]]['z_hist'] = np.flip(rc.index.values)
            esl_statistics[input_locations.locations.values[i]]['f_hist'] = np.flip(1/rc.values) 
        else:
            print("Warning: No nearby ESL information found for {0}. Skipping site.".format(input_locations.locations[i].values))
            continue
    return esl_statistics


def open_gtsm_waterlevels(gtsm_dir, input_locations):
    gtsm = xr.open_mfdataset(os.path.join(gtsm_dir,'*.nc')) #assumes daily maxima in similar file structure as original CDS download
    gtsm_lats = gtsm.station_y_coordinate.values
    gtsm_lons = gtsm.station_x_coordinate.values
    
    min_idx = [mindist(x,y,gtsm_lats,gtsm_lons,0.2) for x,y in zip(input_locations.lat.values, input_locations.lon.values)]
    for k in np.arange(len(input_locations.locations)):
        if min_idx[k] is not None:
            min_idx[k] = min_idx[k][0]
        else:
            min_idx[k] = np.nan
            print("Warning: No nearby ESL information found for {0}. Skipping site.".format(input_locations.locations[k].values))
    gtsm = gtsm.isel(stations = np.array(min_idx)[np.isfinite(min_idx)].astype('int'))
    gtsm = gtsm.rename({'stations':'locations'})
    gtsm['locations'] = input_locations.locations[np.isfinite(min_idx)]
        
    gtsm = gtsm.load() #load data into memory
    
    #do some cleaning up (not entirely sure why this is necessary, also seems to be the case in the original dataset)
    gtsm = gtsm.drop_isel(locations = np.where(np.isnan(gtsm.waterlevel).any(dim='time'))[0]) #remove stations containing nans in their timeseries
    gtsm = gtsm.where((gtsm.waterlevel.var(dim='time')>1e-5)).dropna(dim='locations') #remove weird stations with very low variance (erroneous, sea ice, internal seas?)

    return gtsm
