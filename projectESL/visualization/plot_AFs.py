#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:07:28 2024

@author: timhermans
"""

import numpy as np
import xarray as xr
import os
import fnmatch
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ds = xr.open_dataset('/Users/timhermans/Documents/GitHub/projectESL/output/projectESL_output_test.nc')


fig=plt.figure(figsize=(9,9.5)) #generate figure  
gs = fig.add_gridspec(1,1)
gs.update(hspace=.2)

ax = plt.subplot(gs[0,0],projection=ccrs.Robinson(central_longitude=0))

#ax.add_feature(cartopy.feature.OCEAN, zorder=0,facecolor='grey')
#ax.add_feature(cartopy.feature.LAND, zorder=0, facecolor='grey')
sc=ax.scatter(ds.lon,ds.lat,c=ds.AF.isel(qnt=1,year=0),cmap='Reds',vmin=0,vmax=1000,transform=ccrs.PlateCarree(),zorder=3)

sc.set_edgecolor('face')
ax.coastlines(zorder=1)

cax=ax.inset_axes(bounds=(0, -.1,1,.075))
cb=fig.colorbar(sc, cax=cax,orientation='horizontal',label='AF [-]')
ax.set_title('AFs (median, 2100, SSP2-4.5, wf_1f)')


fig=plt.figure(figsize=(9,9.5)) #generate figure  
gs = fig.add_gridspec(1,1)
gs.update(hspace=.2)

ax = plt.subplot(gs[0,0],projection=ccrs.Robinson(central_longitude=0))

#ax.add_feature(cartopy.feature.OCEAN, zorder=0,facecolor='grey')
#ax.add_feature(cartopy.feature.LAND, zorder=0, facecolor='grey')
sc=ax.scatter(ds.lon,ds.lat,c=ds.af_timing.isel(qnt=1,target_AF=0),cmap='Reds',vmin=2020,vmax=2150,transform=ccrs.PlateCarree(),zorder=3)

sc.set_edgecolor('face')
ax.coastlines(zorder=1)

cax=ax.inset_axes(bounds=(0, -.1,1,.075))
cb=fig.colorbar(sc, cax=cax,orientation='horizontal',label='Year [-]')
ax.set_title('Timing AF=10xDIVA (median, SSP2-4.5, wf_1f)')

