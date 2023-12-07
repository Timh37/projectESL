#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:09:51 2022

Computes kilometric distances between point (lon,lat) and vector of points (lons,lats) using lon/lat coordinates


@author: thermans
"""
import numpy as np

def compute_distances_to_point(lon,lat,lons,lats):
    
    distances = np.zeros(len(lons)) #initialize array
    
    distances = distances + 2*np.arcsin( np.sqrt(
            np.sin( (np.pi/180) * 0.5*(lats-lat) )**2 +
            np.cos((np.pi/180)*lat)*np.cos((np.pi/180)*lats)*np.sin((np.pi/180)*0.5*(lons-lon))**2) )
    
    return distances*6371