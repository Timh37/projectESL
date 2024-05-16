#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 08:44:41 2024

compare return heights at the catchments of Jack Heslop

@author: timhermans
"""

import xarray as xr
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import matplotlib
import cmocean
plt.close('all')


ssps = ['ssp126','ssp245','ssp585']
years = [2050,2100]
afs = [20]
af_ds = xr.open_mfdataset('/Users/timhermans/Documents/GitHub/projectESL/output/TGs/ar6_1f_ssp*_gesla3_tgs_flopros/projectESL_output.nc',combine='nested',concat_dim='ssp')
af_ds = af_ds.load()

cmdict = cmocean.tools.get_dict(cmocean.cm.amp_r) #generate discrete colormap
cmap=matplotlib.colors.LinearSegmentedColormap('amp_discrete',cmdict,13).reversed()
cmap.set_over("cyan")

af_ds['ssp'] = ['ssp126','ssp245','ssp585']


fig=plt.figure(figsize=(11,8)) #generate figure 
gs = fig.add_gridspec(len(ssps),len(years))
gs.update(hspace=.1)

flag = 0
for y,year in enumerate(years):
    print(year)
    for s,ssp in enumerate(ssps):
        print(ssp)
        afs_to_plot = af_ds[['AF','maxAF']].sel(target_year=year,qnt=0.5,ssp=ssp).dropna(dim='locations')
        print('% between 1-10: '+str(100*np.sum(((afs_to_plot.AF>=1) & (afs_to_plot.AF<10))).values/len(afs_to_plot.AF)))
        print('% between 10-100: '+str(100*np.sum(((afs_to_plot.AF>=10) & (afs_to_plot.AF<100))).values/len(afs_to_plot.AF)))
        print('% between 100-1000: '+str(100*np.sum(((afs_to_plot.AF>=100))).values/len(afs_to_plot.AF)))
        print('median across tgs:' + str(np.nanmedian(afs_to_plot.AF)))
        max_af_reached = (afs_to_plot.AF==afs_to_plot.maxAF)

        ax = plt.subplot(gs[s,y],projection=ccrs.Robinson(central_longitude=0))
        ax.set_extent([-180, 180, -60, 80], crs=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.OCEAN, zorder=0,facecolor='grey')
        ax.add_feature(cartopy.feature.LAND, zorder=0, facecolor='grey')
        
        sc = ax.scatter(afs_to_plot.lon,afs_to_plot.lat,
                        c=afs_to_plot.AF,
                        transform=ccrs.PlateCarree(),s=35,cmap=cmocean.cm.matter,norm=matplotlib.colors.LogNorm(1,1e3,0),edgecolors='black')
        
        if max_af_reached.values.any():
            k=ax.scatter(afs_to_plot.lon[max_af_reached],afs_to_plot.lat[max_af_reached],
                            c=afs_to_plot.AF[max_af_reached],
                            edgecolors='cyan',
                            transform=ccrs.PlateCarree(),s=35,label='max. possible')
            flag=1
        ax.set_title(np.array([['(a)','(b)'],['(c)','(d)'],['(e)','(f)']])[s,y])
        if (s==len(ssps)-1):
            cax=ax.inset_axes(bounds=(0, -.1,1,.075))
            cb=fig.colorbar(sc, cax=cax,orientation='horizontal',label='Frequency amplification of $f_{FLOPROS}$ [-]',extend='both')
    ax.annotate(str(year),xy=(.283+.423*y,.9),xycoords='figure fraction',rotation=0,color='black',fontweight='bold')
    
for s,ssp in enumerate(ssps):   
    ax.annotate(''.join(('SSP',ssp[-3],'-',ssp[-2],'.',ssp[-1])),xy=(.1,.88-.27*s),xycoords='figure fraction',rotation=0,color=['blue','orange','red'][s],fontweight='bold')

if flag==1:
    ax.legend(loc='upper right',facecolor='grey')
        
        #ax.set_title(year)
        
        
        
fig=plt.figure(figsize=(11,8)) #generate figure 
gs = fig.add_gridspec(len(ssps),len(years))
gs.update(hspace=.1)

i=0
for a,af in enumerate(afs):
    print(af)
    for s,ssp in enumerate(ssps):
        print(ssp)
        timing_to_plot = af_ds[['AF_timing']].sel(target_AF=af,qnt=0.5,ssp=ssp).dropna(dim='locations')
        print('% before 2050: '+str(100*np.sum((timing_to_plot.AF_timing<2050)).values/len(timing_to_plot.AF_timing)))
        print('% 2050-2100: '+str(100*np.sum(((timing_to_plot.AF_timing<2100)&(timing_to_plot.AF_timing>=2050))).values/len(timing_to_plot.AF_timing)))
        print('% 2100-2150: '+str(100*np.sum(((timing_to_plot.AF_timing<2150)&(timing_to_plot.AF_timing>=2100))).values/len(timing_to_plot.AF_timing)))
        print('% >=2150: '+str(100*np.sum(((timing_to_plot.AF_timing>=2150))).values/len(timing_to_plot.AF_timing)))

        ax = plt.subplot(gs[s,a],projection=ccrs.Robinson(central_longitude=0))
        ax.set_extent([-180, 180, -60, 80], crs=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.OCEAN, zorder=0,facecolor='grey')
        ax.add_feature(cartopy.feature.LAND, zorder=0, facecolor='grey')
        
        sc = ax.scatter(afs_to_plot.lon,afs_to_plot.lat,
                        c=timing_to_plot.AF_timing,
                        transform=ccrs.PlateCarree(),s=35,cmap=cmap,vmin=2020,vmax=2150,edgecolors='black')
      
        ax.set_title(['(a)','(b)','(c)'][i])
        if (s==len(ssps)-1):
            cax=ax.inset_axes(bounds=(0, -.1,1,.075))
            cb=fig.colorbar(sc, cax=cax,orientation='horizontal',label='Timing of AF$_{FLOPROS}$=20 [yr]',extend='max')
        i+=1
for s,ssp in enumerate(ssps):   
    ax.annotate(''.join(('SSP',ssp[-3],'-',ssp[-2],'.',ssp[-1])),xy=(.1,.88-.27*s),xycoords='figure fraction',rotation=0,color=['blue','orange','red'][s],fontweight='bold')
    

cmap=matplotlib.colors.LinearSegmentedColormap('amp_discrete',cmdict,12)
fig=plt.figure(figsize=(11,8)) #generate figure 
gs = fig.add_gridspec(1,1)
gs.update(hspace=.1)

afs_to_plot = af_ds.refFreq.isel(ssp=0).dropna(dim='locations')
ax = plt.subplot(gs[0,0],projection=ccrs.Robinson(central_longitude=0))
ax.set_extent([-180, 180, -60, 80], crs=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.OCEAN, zorder=0,facecolor='grey')
ax.add_feature(cartopy.feature.LAND, zorder=0, facecolor='grey')

sc = ax.scatter(afs_to_plot.lon,afs_to_plot.lat,
                c=afs_to_plot,
                transform=ccrs.PlateCarree(),s=35,cmap=cmap,norm=matplotlib.colors.LogNorm(1e-3,1,0),edgecolors='black')


cax=ax.inset_axes(bounds=(0, -.1,1,.075))
cb=fig.colorbar(sc, cax=cax,orientation='horizontal',label='Estimated degree of flood protection $f_{FLOPROS}$ [1/yr]')
