#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" .py
Created on Mon Oct 28 13:54:34 2019, adapted 20-04-22.
@author: Tim Hermans
tim(dot)hermans@nioz(dot)nl
"""
import numpy as np
import pandas as pd
from gesla import GeslaDataset
import os
from tqdm import tqdm
from copy import deepcopy

def resample_data(rawdata,resample_freq):
    h_means = rawdata.resample('H').mean() # Skipna=False doesn't work
    h_means.loc[h_means['use_flag']!=1,'sea_level'] = np.nan #set hours during which 1 or more observations are bad to nan
    h_means = h_means[np.isfinite(h_means['sea_level'])] #drop nans
    
    if resample_freq == 'H_mean':
        output = h_means
    elif resample_freq == 'D_mean':
        hours_per_day = h_means.resample('D').count()
        d_means = h_means.resample('D').mean()
        d_means = d_means[hours_per_day['sea_level']>=12] #only use days with 12 hourly means or more
        d_means = d_means[np.isfinite(d_means['sea_level'])] #drop nans
        output = d_means
    elif resample_freq == 'D_max':
        hours_per_day = h_means.resample('D').count()
        d_maxs = h_means.resample('D').max()
        d_maxs = d_maxs[hours_per_day['sea_level']>=12] #only use days with 12 hourly means or more
        d_maxs = d_maxs[np.isfinite(d_maxs['sea_level'])] #drop nans
        output = d_maxs
    else:
        raise('Unknown resample frequency.')
        
    output.attrs['resample_freq'] = resample_freq
    return output

def readmeta_GESLA2(filename):
    with open(filename,encoding = 'raw_unicode_escape') as myfile:
        head = [next(myfile) for x in range(9)]
    station_name = head[1][12:-1].strip()	 
    station_lat = float(head[4][11:-1].strip())
    station_lon = float(head[5][12:-1].strip()) 
    start_date = np.datetime64(head[7][18:-1].strip().replace('/','-'))
    end_date = np.datetime64(head[8][16:-1].strip().replace('/','-'))
	
    return (station_name, station_lat, station_lon, start_date, end_date)

def extract_GESLA2_locations(gesladir):
    # Get a list of the gesla database files
    geslafiles = os.listdir(gesladir)
    if '.DS_Store' in geslafiles:
        geslafiles.remove('.DS_Store')
    # Initialize the station variables
    station_names = []
    station_lats = []
    station_lons = []
    station_filenames = []
    
    # Loop over the gesla files
    for this_file in geslafiles:
	
        #Extract the station header information
        this_name, this_lat, this_lon, start_date, end_date = readmeta_GESLA2(os.path.join(gesladir, this_file))
        
        # ppend this information to appropriate lists
        station_names.append(this_name)	 
        station_lats.append(float(this_lat))
        station_lons.append(float(this_lon))
        station_filenames.append(this_file)
    
    return (station_names,station_lats,station_lons,station_filenames)

def extract_GESLA3_locations(metafile_path):
    
    meta = pd.read_csv(metafile_path)
    
    return (list(meta['SITE NAME']),list(meta['LATITUDE']),list(meta['LONGITUDE']),list(meta['FILE NAME']))
    

def get_data_starting_index(file_name):
    with open(file_name,errors='ignore') as fp:
        for i, line in enumerate(fp.readlines()):
                if line.startswith("#"):
                    continue
                else:
                    break
        return i

def open_GESLA3_files(cfg,types,fns=None):
    path_to_files = os.path.join(cfg['input']['paths']['gesla3'],'GESLA3.0_ALL')
    path_to_files = os.path.join(path_to_files,'') #append '/'
    
    meta_fn = os.path.join(cfg['input']['paths']['gesla3'],'GESLA3_ALL.csv')
    
    resample_freq = cfg['preprocessing']['resample_freq']
    min_yrs = cfg['preprocessing']['min_yrs']
    
    if not fns:
        fns = os.listdir(path_to_files) #get filenames
    g3object = GeslaDataset(meta_file=meta_fn,data_path=path_to_files) #create dataset class
    
    datasets = {}
    for fn in fns:
        data = g3object.file_to_pandas(fn) #data [0] + metadata [1]
        
        if data[1]['gauge_type'] not in types:
            continue
        if data[1]['number_of_years'] < min_yrs:
            continue
        
        rawdata = data[0]
        rawdata.loc[rawdata['use_flag']!=1,'sea_level'] = np.nan
        
        #compute MSL from raw data if referencing to MSL
        if cfg['preprocessing']['ref_to_msl']:
            if 'msl_period' in cfg['preprocessing']:
                years = cfg['preprocessing']['msl_period'].split(',')
                y0 = years[0]
                y1 = str(int(years[1])+1)
                
                msl = np.nanmean(rawdata[rawdata.index.to_series().between(y0,y1)]['sea_level'])
            else: 
                raise Exception('Could not reference to MSL because "msl_period" is undefined.')
            
            if np.isnan(msl):
                print('Warning: Could not reference observations to MSL for {0} because there are no observations available during "msl_period".'.format(fn))
            else:
                rawdata['sea_level'] = rawdata['sea_level'] - msl
        
        resampled_data  = resample_data(rawdata,resample_freq)
        
        #do not include data if shorter than minimum years of observations
        if (('H' in resample_freq) & (len(resampled_data)/365.25/24 < min_yrs)):
            continue
        if (('D' in resample_freq) & (len(resampled_data)/365.25 < min_yrs)):
            continue
        
        for k,v in data[1].items():
            resampled_data.attrs[k] = v
        #datasets.append(resampled_data.drop(columns=['qc_flag','use_flag']))
        datasets[fn] = resampled_data.drop(columns=['qc_flag','use_flag'])

    if len(datasets)==0:
        raise Exception('0 records exceeding record length requirement.')    
    return datasets


def open_GESLA2_files(cfg,fns=None):
    path_to_files = cfg['input']['paths']['gesla2']
    path_to_files = os.path.join(path_to_files,'') #append '/'
    
    resample_freq = cfg['preprocessing']['resample_freq']
    min_yrs = cfg['preprocessing']['min_yrs']
    
    if not fns:
        fns = os.listdir(path_to_files) #get filenames
    
    datasets = {}
    for fn in fns:
        metadata = readmeta_GESLA2(os.path.join(path_to_files,fn))
 
        if (metadata[-1] - metadata[-2])/np.timedelta64(1, 's')/(365.25*24*3600) < min_yrs:
            continue
    
        i = get_data_starting_index(os.path.join(path_to_files,fn))
        
        if 'national_tidal_centre' in fn:
            data = pd.read_csv(
                os.path.join(path_to_files,fn),
                skiprows=i,
                names=['date','time','sea_level','qc_flag','tu','use_flag'],
                sep="\s+",
                parse_dates=[[0, 1]],
                index_col=0,
            )
            
            data = data.drop(columns=['tu'])
        else:
            data = pd.read_csv(
                os.path.join(path_to_files,fn),
                skiprows=i,
                names=['date','time','sea_level','qc_flag','use_flag'],
                sep="\s+",
                parse_dates=[[0, 1]],
                index_col=0,
            )
        
        #set invalid or missing data to nan (see also GESLA2 definitions)
        data.loc[data['use_flag']!=1,'sea_level'] = np.nan
        data.loc[data['sea_level']<=-9.9,'sea_level'] = np.nan
        
        #compute MSL from raw data if referencing to MSL
        if cfg['preprocessing']['ref_to_msl']:
            if 'msl_period' in cfg['preprocessing']:
                years = cfg['preprocessing']['msl_period'].split(',')
                y0 = years[0]
                y1 = str(int(years[1])+1)
                
                msl = np.nanmean(data[data.index.to_series().between(y0,y1)]['sea_level'])
            else: 
                raise Exception('Could not reference to MSL because "msl_period" is undefined.')
            
            if np.isnan(msl):
                print('Warning: Could not reference observations to MSL for {0} because there are no observations available during "msl_period".'.format(fn))
            else:
                data['sea_level'] = data['sea_level'] - msl
            
        resampled_data = resample_data(data,resample_freq)
        
        #do not include data if shorter than minimum years of observations
        if (('H' in resample_freq) & (len(resampled_data)/365.25/24 < min_yrs)):
            continue
        if (('D' in resample_freq) & (len(resampled_data)/365.25 < min_yrs)):
            continue
        
        resampled_data.attrs['file_name'] = fn
        resampled_data.attrs['site_name'] = metadata[0]
        resampled_data.attrs['latitude'] = metadata[1]
        resampled_data.attrs['longitude'] = metadata[2]
        
        #datasets.append(resampled_data.drop(columns=['qc_flag','use_flag']))
        datasets[fn] = resampled_data.drop(columns=['qc_flag','use_flag'])
    
    if len(datasets)==0:
        raise Exception('0 records exceeding record length requirement.')    
    return datasets

def detrend_gesla_dfs(dfs):
    detrended_dfs = deepcopy(dfs)
    for k,df in detrended_dfs.items():
        x =  df.index.values.astype(np.int64) // 10 ** 9 #convert to seconds timestamp
        y =  df['sea_level'].values.astype('float64')
        lrcoefs = np.polyfit(x,y,1)
        trend = np.polyval(lrcoefs,x)
        
        #df['sea_level'] = df['sea_level'] - trend + lrcoefs[-1] #subtract trend without intercept
        df['sea_level'] = df['sea_level'] - trend + np.mean(trend) #subtract trend without changing the vertical datum (mean of timeseries)
    return detrended_dfs

def deseasonalize_gesla_dfs(dfs):
    deseasonalized_dfs = deepcopy(dfs)
    for k,df in deseasonalized_dfs.items():

        monthly_means_at_timesteps = df.groupby(df.index.month).transform('mean')['sea_level'].astype('float64') #contains mean of all timesteps in month for all years together at each timestep in that month
        
        df['sea_level'] = df['sea_level'] - monthly_means_at_timesteps + np.mean(monthly_means_at_timesteps) #subtract without changing the vertical datum (mean of timeseries)

    return deseasonalized_dfs

def subtract_amean_from_gesla_dfs(dfs):
    dfs_no_amean = deepcopy(dfs)
    for k,df in dfs_no_amean.items():
        annual_means_at_timesteps = df.groupby(df.index.year).transform('mean')['sea_level'].astype('float64')
        df['sea_level'] = df['sea_level'] - annual_means_at_timesteps + np.mean(annual_means_at_timesteps)  #subtract without changing the vertical datum (mean of timeseries)
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