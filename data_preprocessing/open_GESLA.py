#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" .py
Created on Mon Oct 28 13:54:34 2019, adapted 20-04-22.
@author: Tim Hermans
tim(dot)hermans@nioz(dot)nl
"""
import numpy as np
from datetime import datetime as dt
import pandas as pd
from gesla import GeslaDataset
import os
from collections import defaultdict
from tqdm import tqdm
def resample_data(rawdata,resample_freq):
    h_means = rawdata.shift(freq='30Min').resample('H').mean() #shift because some timestamps are reported as 1s before the hour
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

def open_GESLA3_files(path_to_files,meta_fn,types,resample_freq,min_yrs,fns=None):
    print('opening GESLA3 files')
    path_to_files = os.path.join(path_to_files,'') #append '/'
    
    if not fns:
        fns = os.listdir(path_to_files) #get filenames
    g3object = GeslaDataset(meta_file=meta_fn,data_path=path_to_files) #create dataset class
    
    datasets = {}
    for fn in tqdm(fns):
        data = g3object.file_to_pandas(fn) #data [0] + metadata [1]
        
        if data[1]['gauge_type'] not in types:
            continue
        if data[1]['number_of_years'] < min_yrs:
            continue
        
        rawdata = data[0]
        rawdata = rawdata[rawdata['use_flag']==1] #only use data with use flag 1 (qf=0-2) (see GESLA3 documentation)
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


def open_GESLA2_files(path_to_files,resample_freq,min_yrs,fns=None):
    print('opening GESLA3 files')
    path_to_files = os.path.join(path_to_files,'') #append '/'
    if not fns:
        fns = os.listdir(path_to_files) #get filenames
    
    datasets = {}
    for fn in tqdm(fns):
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
        
        #remove invalid or missing data (see also GESLA2 definitions)
        data = data[data['sea_level'] >-9.9] # remove missing data
        data = data[data['use_flag'] == 1] # only use data with use flag 1
        
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