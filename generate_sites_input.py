#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 12:49:03 2024

@author: timhermans
"""

import xarray as xr
import pandas as pd
from projectESL.utils import add_ar6_full_sample_projections_to_locations
import os

def make_projectESL_input(location_names,location_lons,location_lats,path_to_sl=None,n_samples=None,period=None):
    
    #generate dataset with coordinates
    input_data = xr.Dataset(
        data_vars=dict(
        ),
        coords=dict(
            lon=(["locations"], location_lons),
            lat=(["locations"], location_lats),
            locations=location_names
        ),
    )
    
    input_data = input_data.drop_duplicates(dim='locations')
    
    if path_to_sl is not None:
        input_data = add_ar6_full_sample_projections_to_locations(input_data,path_to_sl,n_samples,period)
        input_data = input_data.chunk({'samples':5000,'years':5,'locations':100})
    
    
    return input_data


if __name__ == "__main__":
    
    
    #COAST-RP
    crp_points = pd.read_pickle('/Volumes/Naamloos/PhD_Data/COAST_RP/COAST_RP_full_dataset/pxyn_coastal_points.xyn')
    crp_points = crp_points[crp_points.station.str.startswith('i')]
    
    #input_data = make_projectESL_input(crp_points['station'],crp_points['lon'],crp_points['lat'],path_to_sl=None,n_samples=None,period=None)
    input_data = make_projectESL_input(crp_points['station'],crp_points['lon'],crp_points['lat'],
                                       '/Volumes/Naamloos/PhD_Data/AR6_projections/ar6-gridded-samples-total/full_sample_total_wf_1f_ssp245.zarr',
                                       10000,[2000,2150])
    input_data.to_zarr(os.path.join('input/full_sample_total_wf_1f_ssp245_at_coastal_COAST_RP.zarr'))
    
    
    #coastal GESLA3
    meta = pd.read_csv('/Volumes/Naamloos/PhD_Data/GESLA3/GESLA3_ALL.csv')
    meta = meta[(meta['GAUGE TYPE']=='Coastal') & (meta['OVERALL RECORD QUALITY']=='No obvious issues') & (meta['NUMBER OF YEARS'] >= 30) & (pd.to_datetime(meta['END DATE/TIME'],format='mixed').dt.year>=2000)].reset_index(drop=True) #only consider coastal gauges without issues with sufficient length

    input_data = make_projectESL_input(meta['FILE NAME'],meta['LONGITUDE'],meta['LATITUDE'],
                                       '/Volumes/Naamloos/PhD_Data/AR6_projections/ar6-gridded-samples-total/full_sample_total_wf_1f_ssp245.zarr',
                                       10000,[2000,2150])
    input_data.to_zarr(os.path.join('input/full_sample_total_wf_1f_ssp245_at_coastal_GESLA3.zarr'))
    
