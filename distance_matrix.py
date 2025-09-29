from DMT import VectorizedKeplerianOrbit
import pickle
from math import pi
import pandas as pd 

"""
This file is used to process the given elset data into a distance matrix and save
to disk.
"""

def get_distance_matrix(df, save=False):
    
    """This will return the distance matrix upper triangle and the key for satno to index"""
    
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
    
    return distance_matrix, key
    
def get_key(df, save=False):
    
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
    
        print("Saved satNo to index and index to satNo dictionaries to disk.")
    
if __name__ == '__main__':
    # run this file to get the distance matrix for the entire catalog (have on standby for usage later)
    
    # get all satellites from celestrak data and create distance matrix
    celestrak_tles_path = 'data/3le'
    data = []
    with open(celestrak_tles_path, 'r') as f:
        lines = f.readlines()
   
    for i in range(0, len(lines), 3):
        if i + 2 < len(lines):
            name = lines[i].strip()
            line1 = lines[i + 1].strip()
            line2 = lines[i + 2].strip()
           
            if not (line1.startswith('1 ') and line2.startswith('2 ')):
                continue
               
            sat_no = line1[2:7].strip()
           
            data.append({
                'satNo': sat_no,
                'line1': line1,
                'line2': line2
            })
   
    elset_df = pd.DataFrame(data)
   
    elset_df['satNo'] = (
        elset_df['satNo']
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )
    
    distance_matrix, key = get_distance_matrix(elset_df, save=True)

