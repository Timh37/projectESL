#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:51:03 2023

@author: timhermans
"""
import os
import yaml
import xarray as xr
import numpy as np
import numpy.matlib
import sys
sys.path.insert(1, '/Users/timhermans/Documents/GitHub/projectESL/data_preprocessing')
sys.path.insert(2, '/Users/timhermans/Documents/GitHub/projectESL/esl_analysis')
sys.path.insert(3, '/Users/timhermans/Documents/GitHub/projectESL/projecting')
from utils import mindist
from open_GESLA import extract_GESLA2_locations, extract_GESLA3_locations, open_GESLA2_files, open_GESLA3_files
from preprocess_GESLA import detrend_gesla_dfs, deseasonalize_gesla_dfs, subtract_amean_from_gesla_dfs
from multivariate_normal_gpd_samples_from_covmat import multivariate_normal_gpd_samples_from_covmat
from compute_AF import compute_AF
from analyze_GESLA_ESLs import pot_extremes_from_gesla_dfs, fit_gpd_to_gesla_extremes
from GPD_Z_from_F import get_return_curve
with open('config/config.yml', 'r') as f: #load config
    cfg = yaml.safe_load(f)

#settings (to be included in a config file)
f= 10**np.linspace(-6,2,num=2001)
f=np.append(f,np.arange(101,183))
max_dist=0.1 #11km    
subtract_amean= True
detrend=False
deseasonalize=False
resample_freq='H_mean'
refFreqs = 0.01
output_qnts = np.array([.05,.17,.5,.83,.95])


slr = xr.open_dataset(cfg['esl']['sl_projections']) #open sl projections

target_years = np.array(cfg['project_af']['target_years'].split(',')).astype('int')
try:
    slr = slr.sel(years=target_years) #select target years
except:
    raise Exception('Could not select target years from sea-level projections.')

try:
    slr = slr.expand_dims('locations')
except:
    pass

if 'samples' in slr.dims:
    num_mc = len(slr.samples)
else:
    num_mc = 1
    
#implement options to have samples of shape/scale without slr samples 

if slr.sea_level_change.units=='mm':
    slr['sea_level_change'] = slr['sea_level_change']/1000
    slr.sea_level_change.attrs['units'] = 'm'
    
#determine which ESL data to use
if (cfg['esl']['input_type'] == 'raw') & (cfg['esl']['input_source'] in ['gesla2','gesla3']):
    if cfg['esl']['input_source'] == 'gesla2':
        (station_names, station_lats, station_lons, station_filenames) = extract_GESLA2_locations('/Volumes/Naamloos/PhD_Data/GESLA2/private_14032017_public_110292018/')
    elif cfg['esl']['input_source'] == 'gesla3':
        (station_names, station_lats, station_lons, station_filenames) = extract_GESLA3_locations('/Volumes/Naamloos/PhD_Data/GESLA3/GESLA3_ALL.csv')

    min_idx = [mindist(x,y,station_lats,station_lons, max_dist) for x,y in zip(slr.lat.values, slr.lon.values)] #sites within maximum distance from point


    matched_filenames = []
    sites_with_esl = []
    for i in np.arange(len(slr.locations)): #loop over sl-projection sites
        if min_idx[i] is not None: #if nearby ESL data found, try to analyze that data
    
            matched_filenames.append([station_filenames[x] for x in min_idx[i]]) #esl stations within distance to sea-level projection site
            sites_with_esl.append(float(slr.locations[i])) #sea-level projections sites with nearby ESL data
    
    # if no locations with esl data are found, quit with error
    if not matched_filenames:
        raise Exception("No matches found within {} degrees for provided lat/lon list".format(max_dist))
      
    # Initialize station data dictionary to hold matched location data
    esl_params = {}
    	
    # Initialize variables to track the files that have been tested
    pass_files = {}
    fail_files = []
    	
    # Loop over the matched files
    for i in np.arange(len(sites_with_esl)): #loop over sea-level projection sites with matched ESL data
		
        # This ID
        this_id = sites_with_esl[i]
        this_id_passed = False
		
        # Loop over the esl files within the match radius for this location, use the first one for which data fulfills criteria
        for esl_file in matched_filenames[i]:
			
            # if this esl file was already tested and passed
            if np.isin(esl_file, list(pass_files.keys())):
                print("{0} already PASSED a previous check on data constraints. Mapping site ID {1} to site ID {2}.".format(esl_file, pass_files[esl_file], this_id))
                esl_params[this_id] = esl_params[pass_files[esl_file]]
                this_id_passed = True

            # if current sea-level projectio nlocation already has esl parameters, skip to the next
            if this_id_passed:
                continue            

            # This esl file has not been tested yet, try to analyze it
            try:
                #this tide gauge data
                if cfg['esl']['input_source'] == 'gesla2':
                    dfs = open_GESLA2_files('/Volumes/Naamloos/PhD_Data/GESLA2/private_14032017_public_110292018/',
                                        'H_mean',min_yrs=20,fns=[esl_file])

                elif cfg['esl']['input_source'] == 'gesla3':
                    dfs = open_GESLA3_files('/Volumes/Naamloos/PhD_Data/GESLA3/GESLA3.0_ALL','/Volumes/Naamloos/PhD_Data/GESLA3/GESLA3_ALL.csv',
                                        types=['Coastal'],resample_freq = 'D_max',min_yrs = 20,fns=[esl_file])

                #preproc options:
                if detrend:
                    dfs = detrend_gesla_dfs(dfs)
                if deseasonalize:
                    dfs = deseasonalize_gesla_dfs(dfs)
                if subtract_amean:
                    dfs = subtract_amean_from_gesla_dfs(dfs)

                extremes = pot_extremes_from_gesla_dfs(dfs,99.7,declus_method='iterative_descending',declus_window=3)
                gpd_params = fit_gpd_to_gesla_extremes(extremes)
                esl_params[this_id] = gpd_params
			
                # This file passed, add it to the pass file dictionary
                pass_files[esl_file] = this_id
                # This ID has data, set the flag to move onto the next ID
                this_id_passed = True
			
            except:
                # This file failed, add it to the fail list and continue with the next file
                print("{} did not pass the data constraints. Moving on to the next file.".format(esl_file))
                fail_files.append(esl_file)
                continue

        # Let the user know we didn't find a file that passes the data constraints
        if not this_id_passed:
            print("No locations within {0} degrees pass the data constraints for ID {1}.".format(max_dist, this_id))
	
#projection stage
for site_id,gpd in esl_params.items():
    if np.isscalar(refFreqs):
        refFreq = refFreqs
        
    scale_samples,shape_samples = multivariate_normal_gpd_samples_from_covmat(gpd['scale'].iloc[0],gpd['shape'].iloc[0],gpd['cov'].iloc[0],num_mc)
    z_hist = get_return_curve(f,scale_samples,shape_samples,gpd['loc'].iloc[0],gpd['avg_extr_pyear'].iloc[0],'mhhw',gpd['mhhw'].iloc[0])
    for yr in target_years:
        z_hist,z_fut,AF = compute_AF(f,z_hist,slr.sel(locations=site_id).sel(years=yr).sea_level_change.values,refFreq) #samples
        
        #to-do: store outputs