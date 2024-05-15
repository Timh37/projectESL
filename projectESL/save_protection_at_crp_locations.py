#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 10:29:20 2024

@author: timhermans
"""

import xarray as xr

from I_O import find_diva_protection_levels, find_flopros_protection_levels


input_locations = xr.open_dataset('../input/full_sample_total_wf_1f_ssp245_at_coastal_COAST_RP.zarr',chunks='auto')

qlats = input_locations.lat.values
qlons = input_locations.lon.values

diva_at_crp = find_diva_protection_levels(qlons,qlats,'/Users/timhermans/Documents/Data/DIVA/cls_historical_protection_levels.gpkg',10)

#flopros_at_crp = find_flopros_protection_levels(qlons,qlats,'/Users/timhermans/Documents/Data/FLOPROS/Tiggeloven/',10)

#input_locations['protection'] = (('locations'),flopros_at_crp)
input_locations['protection'] = (('locations'),diva_at_crp)

input_locations[['protection']].to_netcdf('../input/diva_protection_estimates_at_coastrp_locations.nc',mode='w') 