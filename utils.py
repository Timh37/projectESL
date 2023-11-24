import numpy as np

def angdist(lat0, lon0, lat, lon):
	
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
	
	# Calculate the angular distance
	dist = angdist(lats, lons, qlat, qlon)
	
	# If the minimum distance is beyond the limit, print a warning and return None
	if(np.amin(dist) > limit):
		print("Warning: No match for lat={0} lon={1}. Minimum distance to nearest tide gauge is {2:.4f} (limit={3}). Returning \"None\"".format(qlat, qlon, np.amin(dist), limit))
		return(None)
	
	else:
		
		# Perform an indirect sort of the distances
		sort_idx = np.argsort(dist)
		
		# Find which ones fall within the radius limit
		min_idx = sort_idx[np.flatnonzero(dist[sort_idx] <= limit)]
		
		return(min_idx)