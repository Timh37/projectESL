#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 09:14:59 2023

@author: timhermans
"""

import geopandas as geopd
from shapely.geometry import Point
import pandas as pd
from utils import angdist
import numpy as np
from tqdm import tqdm

def find_diva_protection_levels(qlons,qlats,diva_fn,maxdist):
    diva = geopd.read_file(diva_fn) #open diva geo file
    
    nearest_segments = []
    
    for qlat,qlon in tqdm(zip(qlats,qlons)): #loop over query coordinates
        p = Point(qlon,qlat) #put coords into point geometry
        
        #do a first filtering based on angular distance to segment/polygon center points (more efficient than shapely distance)
        angdists = angdist(qlat,qlon,diva.lati.values,diva.longi.values)
        nearby = np.where(angdists<5)[0]
        if len(nearby)==0:
            nearest_segments.append(np.nan)
            #add a warning?
            continue
        
        i_near = diva.iloc[nearby].index[np.argsort( [p.distance(k) for k in diva.iloc[nearby].geometry])[0:5]] #find the nearest 5 segments based on euclidean distances (default for shapely)
        
        #for these segments, refine the distance computation using a coordinate-specific appropriate reference system
        p_gdf = geopd.GeoDataFrame(geometry=[p], crs='epsg:4326') #put query point into cartesian geodataframe
        if qlat<0: #degermine appropriate epsg system
            e = 32701 + np.floor_divide((qlon+180),6)
            if qlat<-80:
                e = 32761
        else:
            e = 32601 + np.floor_divide((qlon+180),6)
            if qlat>80:
                e = 32661
            
        p_gdf.to_crs(epsg=e, inplace=True) #convert point to epsg system
        diva_ = diva.iloc[i_near].to_crs(epsg=e) #convert coordinates of 5 nearest diva segments to epsg system
        
        km_dists = np.array([p_gdf.distance(k).values[0] for k in diva_.geometry])/1000 #compute kilometric distances
        
        if np.min(km_dists>maxdist): #if nearest point on nearest segment is more than 15km away
            nearest_segments.append(np.nan)
            #add a warning?
            continue
    
        nearest_segments.append(diva_.index[np.argmin( km_dists )]) #append index of nearest diva segment
    
    protection_levels = []

    for k in nearest_segments:
        if np.isfinite(k):
            plevel = diva.protection_level_modelled.values[k] #in units of return period
            
            if plevel==0: #if no protection, assign protection of 1/2y (see Scussolini et al., 2016)
                plevel = 2
            protection_levels.append(1/plevel) #add protection level to list in units of return frequency
        else:
            protection_levels.append(np.nan)
    return np.array(protection_levels)


#visual test
if __name__ == "__main__":
    maxdist=15
    diva_fn = "/Users/timhermans/Documents/Data/DIVA/cls.gpkg"
    diva = geopd.read_file(diva_fn)
    g3_meta = pd.read_csv('/Volumes/Naamloos/PhD_Data/GESLA3/GESLA3_ALL.csv')
    qlats = g3_meta['LATITUDE'].values
    qlons = g3_meta['LONGITUDE'].values

    protection_levels = find_diva_protection_levels(qlons,qlats,diva_fn,maxdist)
    
    import matplotlib.pyplot as plt
    from cartopy import crs as ccrs
    
    fig=plt.figure(figsize=(8,8)) #generate figure  
    gs = fig.add_gridspec(1,1)
    crs = ccrs.Robinson(central_longitude=0)
    
    ax = plt.subplot(gs[0,0],projection=ccrs.Robinson(central_longitude=0))
    
    # This can be converted into a `proj4` string/dict compatible with GeoPandas
    crs_proj4 = crs.proj4_init
    df_ae = diva.to_crs(crs_proj4)
    
    # Here's what the plot looks like in GeoPandas
    df_ae.plot(column='protection_level_modelled',vmin=0,vmax=100,cmap='Reds',ax=ax)
    ax.scatter(qlons[np.isfinite(protection_levels)],qlats[np.isfinite(protection_levels)],c=1/np.array(protection_levels)[np.isfinite(protection_levels)],vmin=0,vmax=100,s=10,cmap='Reds',edgecolors='black',transform=ccrs.PlateCarree(),zorder=5)
