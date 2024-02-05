'''
@author: Tim Hermans
t(dot)h(dot)j(dot)hermans@uu(dot)nl
'''

import numpy as np
import xarray as xr
import os

def angdist(lat0, lon0, lat, lon):
	'''calculate angular distance between coordinates (lat0,lon0) and (lat,lon)'''
	# Convert the input from degrees to radians
	(lat0, lon0) = np.radians((lat0, lon0))
	(lat, lon) = np.radians((lat, lon))
	
	# Calculate the angle between the vectors
	temp = np.arctan2(np.sqrt((np.cos(lat)*np.sin(lon-lon0))**2 + \
	(np.cos(lat0)*np.sin(lat) - np.sin(lat0)*np.cos(lat) * np.cos(lon-lon0))**2),\
	(np.sin(lat0)*np.sin(lat) + np.cos(lat0)*np.cos(lat)*np.cos(lon-lon0)))
	
	# Convert the results from radians to degrees and return
	return(np.degrees(temp))


def mindist(qlat, qlon, lats, lons, limit=0.1):
	'''find indices of the smallest distances between (qlat,qlon) and (lats,lons) within angular distance limit "limit" '''
	# Calculate the angular distance
	dist = angdist(lats, lons, qlat, qlon)
	
	# If the minimum distance is beyond the limit, print a warning and return None
	if(np.amin(dist) > limit):
		return(None)
	
	else:
		
		# Perform an indirect sort of the distances
		sort_idx = np.argsort(dist)
		
		# Find which ones fall within the radius limit
		min_idx = sort_idx[np.flatnonzero(dist[sort_idx] <= limit)]
		
		return(min_idx)
    
    
def download_ar6_full_sample_projections_at_tgs(wf,ssp,out_dir): #~1.2Gb!
    #see https://github.com/Rutgers-ESSP/IPCC-AR6-Sea-Level-Projections
    ds_url = ('https://storage.googleapis.com/ar6-lsl-simulations-public-standard/'
         'tide-gauges/full_sample_workflows/'+wf+'/'+ssp+'/total-workflow.zarr'
       )
    ds = xr.open_dataset(ds_url, engine='zarr', chunks='auto')
    return ds.to_zarr(os.path.join(out_dir,'full_sample_total_'+wf+'_'+ssp+'.zarr'),mode='w')


def download_ar6_full_sample_projections_gridded(wf,ssp,out_dir): #~52Gb!
    #see https://github.com/Rutgers-ESSP/IPCC-AR6-Sea-Level-Projections
    ds_url = ('https://storage.googleapis.com/ar6-lsl-simulations-public-standard/'
         'gridded/full_sample_workflows/'+wf+'/'+ssp+'/total-workflow.zarr'
       )
    ds = xr.open_dataset(ds_url, engine='zarr', chunks='auto')
    return ds.to_zarr(os.path.join(out_dir,'full_sample_total_'+wf+'_'+ssp+'.zarr'),mode='w')

def add_ar6_full_sample_projections_to_sites(sites,slr_fn):
    
    ds = xr.open_dataset(slr_fn, engine='zarr', chunks='auto')
    ds_stacked = ds.stack(locations=['lon','lat'])
    
    min_idx = [np.argmin(angdist(x,y,ds_stacked.lat.values,ds_stacked.lon.values)) for x,y in zip(sites.lat.values, sites.lon.values)] #get nearest projections to sites
    
    ds_at_sites = ds_stacked.isel(locations=min_idx).drop('lat') #drop multiindex
    ds_at_sites['locations']=sites['locations'].values #assign locations from sites file
    
    sites['sea_level_change'] = ds_at_sites['sea_level_change'] #add SLC variable to sites file

    return sites