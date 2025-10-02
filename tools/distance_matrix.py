from tools.DMT import VectorizedKeplerianOrbit
import pickle
from math import pi
import pandas as pd 
import numpy as np

"""
This file is used to process the given elset data into a distance matrix and save
to disk.
"""

def get_distance_matrix(df, save=False):
        
    line1 = df['line1'].values
    line2 = df['line2'].values
    
    print("Calculating orbits")
    orbits = VectorizedKeplerianOrbit(line1, line2)
    
    print("Calculating distances")
    distance_matrix = VectorizedKeplerianOrbit.DistanceMetric(orbits, orbits)
    if save:
        print("Saving distance matrix to disk...") 
        with open(f'data/distance_matrix.pkl', 'wb') as f:
            pickle.dump(distance_matrix, f)
    
    key = get_key(df, save=save)

    if _validate_matrix(distance_matrix):
        return distance_matrix, key

def _validate_matrix(distance_matrix):
    # check if distance matrix is square
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("Distance matrix is not square")
    
    # check if distance matrix is symmetric
    if not np.allclose(distance_matrix, distance_matrix.T):
        raise ValueError("Distance matrix is not symmetric")
    
    # check if diagonal is zero
    if not np.allclose(np.diag(distance_matrix), 0):
        raise ValueError("Distance matrix diagonal is not zero")
    
    print("Distance matrix is valid")
    return True
    
def get_key(df, save=False):
    """These dictionaries map satellite numbers to their
        index in the distance matrix and vice versa"""
        
    df = df['satNo'].unique()
    
    satNo_idx_dict = {}
    idx_satNo_dict = {}
    
    for i, satNo in enumerate(df):
        idx_satNo_dict[i] = satNo
        satNo_idx_dict[satNo] = i
    
    if save:
        # save both as pkl
        with open(f'data/satNo_idx_dict.pkl', 'wb') as f:
            pickle.dump(satNo_idx_dict, f)
        
        with open(f'data/idx_satNo_dict.pkl', 'wb') as f:
            pickle.dump(idx_satNo_dict, f)
            
    # return both dictionaries
    return {'satNo_idx_dict': satNo_idx_dict, 'idx_satNo_dict': idx_satNo_dict} 
        
if __name__ == '__main__':
    inclination_range = (0, 180)  # degrees
    apogee_range = (0, 2000)   # kilometers
    # get all satellites from celestrak data and create distance matrix
    celestrak_tles_path = 'data/3le'
    data = []
    
    with open(celestrak_tles_path, 'r') as f:
        lines = f.readlines()
    
    earth_radius = 6371000  # meters
    GM_earth = 3.986004418e14  # m^3/s^2
    seconds_in_day = 86400  # seconds per day
   
    for i in range(0, len(lines), 3):
        if i + 2 < len(lines):
            name = lines[i].strip()
            line1 = lines[i + 1].strip()
            line2 = lines[i + 2].strip()
           
            if not (line1.startswith('1 ') and line2.startswith('2 ')):
                continue
               
            sat_no = line1[2:7].strip()
           
            inclination = float(line2[8:16].strip())
           
            # Extract mean motion (revolutions per day)
            mean_motion = float(line2[52:63].strip())
           
            # Extract eccentricity (columns 27-33 in standard TLE format)
            eccentricity = float("0." + line2[26:33].strip())
           
            # Calculate angular velocity n in radians per second
            n = mean_motion * 2 * pi / seconds_in_day
           
            # Calculate semi-major axis a in meters
            a = (GM_earth / (n ** 2)) ** (1/3)
           
            # Calculate apogee in kilometers
            apogee = (a * (1 + eccentricity) - earth_radius) / 1000
           
            data.append({
                'satNo': sat_no,
                'line1': line1,
                'line2': line2,
                'inclination': inclination,
                'apogee': apogee
            })
    
    elset_df = pd.DataFrame(data)
    
    elset_df['satNo'] = (
        elset_df['satNo']
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )
    
    elset_df = elset_df[
        (elset_df['inclination'] >= inclination_range[0]) & (elset_df['inclination'] <= inclination_range[1]) &
        (elset_df['apogee'] >= apogee_range[0]) & (elset_df['apogee'] <= apogee_range[1])
    ].copy()

    distance_matrix, key = get_distance_matrix(elset_df, save=True)
