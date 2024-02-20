import numpy as np
import numpy.matlib
import os
import sys
sys.path.insert(1, '.../data_preprocessing')

from utils import mindist
from open_GESLA import extract_GESLA2_locations, extract_GESLA3_locations, open_GESLA2_files, open_GESLA3_files
from preprocess_GESLA import detrend_gesla_dfs, deseasonalize_gesla_dfs, subtract_amean_from_gesla_dfs
from analyze_GESLA_ESLs import pot_extremes_from_gesla_dfs, fit_gpd_to_gesla_extremes

def ESL_stats_from_raw_GESLA(queried,cfg,maxdist):
        
    settings = cfg['esl_analysis']['preproc_settings']

    #get GESLA locations
    if cfg['esl_analysis']['input_source'].lower() == 'gesla2':
        (station_names, station_lats, station_lons, station_filenames) = extract_GESLA2_locations(cfg['esl_analysis']['input_dir'])#extract_GESLA2_locations('/Volumes/Naamloos/PhD_Data/GESLA2/private_14032017_public_110292018/')
    elif cfg['esl_analysis']['input_source'].lower() == 'gesla3':
        (station_names, station_lats, station_lons, station_filenames) = extract_GESLA3_locations(os.path.join(cfg['esl_analysis']['input_dir'],'GESLA3_ALL.csv'))
    else:
        raise Exception('Data source not recognized.')
    
    min_idx = [mindist(x,y,station_lats,station_lons,maxdist) for x,y in zip(queried.lat.values, queried.lon.values)] #sites within maximum distance from queried points
    
    matched_filenames = []
    sites_with_esl = []
    for i in np.arange(len(queried.locations)): #loop over sl-projection sites
        if min_idx[i] is not None: #if nearby ESL data found, try to analyze that data
    
            matched_filenames.append([station_filenames[x] for x in min_idx[i]]) #esl stations within distance to sea-level projection site
            sites_with_esl.append(float(queried.locations[i])) #sea-level projections sites with nearby ESL data
    
    # if no locations with esl data are found, quit with error
    if not matched_filenames:
        raise Exception("No matches found within {} degrees for provided lat/lon list".format(maxdist))
      
    # Initialize variables to track the files that have been tested
    pass_files = {}
    fail_files = []
    esl_statistics = {}	
    
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
                esl_statistics[this_id] = esl_statistics[pass_files[esl_file]]
                this_id_passed = True
    
            # if current sea-level projectio nlocation already has esl parameters, skip to the next
            if this_id_passed:
                continue            
    
            # This esl file has not been tested yet, try to analyze it
            try:
                #this tide gauge data
                if cfg['esl_analysis']['input_source'].lower() == 'gesla2':
                    dfs = open_GESLA2_files(cfg['esl_analysis']['input_dir'],
                                        settings['resample_freq'],settings['min_yrs'],fns=[esl_file])
    
                elif cfg['esl_analysis']['input_source'].lower() == 'gesla3':
                    dfs = open_GESLA3_files(os.path.join(cfg['esl_analysis']['input_dir'],'GESLA3.0_ALL'),os.path.join(cfg['esl_analysis']['input_dir'],'GESLA3_ALL.csv'),
                                            ['Coastal'],settings['resample_freq'],settings['min_yrs'],fns=[esl_file])
    
                #preproc options:
                if settings['detrend']:
                    dfs = detrend_gesla_dfs(dfs)
                if settings['deseasonalize']:
                    dfs = deseasonalize_gesla_dfs(dfs)
                if settings['subtract_amean']:
                    dfs = subtract_amean_from_gesla_dfs(dfs)
    
                extremes = pot_extremes_from_gesla_dfs(dfs,settings['extremes_threshold'],settings['declus_method'],settings['declus_window'])
                gpd_params = fit_gpd_to_gesla_extremes(extremes)
                esl_statistics[this_id] = gpd_params
    			
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
            print("No locations within {0} degrees pass the data constraints for ID {1}.".format(maxdist, this_id))
    return esl_statistics