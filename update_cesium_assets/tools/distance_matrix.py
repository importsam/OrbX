
import numpy as np

from tools.DMT import VectorizedKeplerianOrbit
import os 
import pickle 

"""
This file is used to process the given elset data into a distance matrix and save
to disk.
"""
dmpath = "../globals/distance_matrix/distance_matrix.pkl"
keypath = "../globals/distance_matrix/key.pkl"

def get_distance_matrix(df=None):
        
    if os.path.exists(dmpath):
        # Load distance matrix
        with open(dmpath, "rb") as f:
            distance_matrix = pickle.load(f)

    else:
        
        if df is None: 
            raise ValueError("Elset dataframe missing for distance_matrix") 
        
        line1 = df['line1'].values
        line2 = df['line2'].values
        
        print("Calculating orbits")
        orbits = VectorizedKeplerianOrbit(line1, line2)
        
        print("Calculating distances")
        distance_matrix = VectorizedKeplerianOrbit.DistanceMetric(orbits, orbits)
    
    key = get_key(df)
    return distance_matrix, key
        
def get_key(df=None):
    
    """
    This will return two keys, basically tells you which satellite is at which index in the distance matrix
    """
    
    if os.path.exists(keypath):
        # Load distance matrix
        with open(keypath, "rb") as f:
            key = pickle.load(f)
            return key
        
    df = df['satNo'].unique()
    
    satNo_idx_dict = {}
    idx_satNo_dict = {}
    
    for i, satNo in enumerate(df):
        idx_satNo_dict[i] = satNo
        satNo_idx_dict[satNo] = i
        
    key = {
        "idx_satNo_dict": idx_satNo_dict,
        "satNo_idx_dict": satNo_idx_dict
    }
    
    return key
            


