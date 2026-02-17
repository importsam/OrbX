# import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sgp4.api import Satrec, jday
# from PIL import Image
# from graphviz import Graph
from orbit_finder.DMT import VectorizedKeplerianOrbit
import pickle
import json
import sys
"""
This file is used to process the given elset data into a distance matrix and save
to disk.
"""
    

def get_distance_matrix_leo(df):
    """
    Compute and save the distance matrix for only LEO satellites.
    LEO is defined as satellites with apogee altitude (after subtracting Earth's radius)
    less than or equal to 2000 km.
    """
    # Filter dataframe for LEO sats
    df_leo = df[df['apogee'] <= 2000].copy()
    
    if df_leo.empty:
        print("No LEO satellites found in the dataset.")
        return

    line1 = df_leo['line1'].values
    line2 = df_leo['line2'].values
    
    print("Calculating orbits for LEO satellites")
    orbits = VectorizedKeplerianOrbit(line1, line2)
    
    print("Calculating distances for LEO satellites")
    distance_matrix_leo = VectorizedKeplerianOrbit.DistanceMetric(orbits, orbits)
                
    print("Saving LEO distance matrix to disk...") 
    with open('distance_matrix_leo.pkl', 'wb') as f:
        pickle.dump(distance_matrix_leo, f)
        
    get_key(df_leo)


def get_distance_matrix(df):
    
    line1 = df['line1'].values
    line2 = df['line2'].values
    
    print("Calculating orbits")
    orbits = VectorizedKeplerianOrbit(line1, line2)
    
    print("Calculating distances")
    distance_matrix = VectorizedKeplerianOrbit.DistanceMetric(orbits, orbits)
                
    print("Saving distance matrix to disk...") 
    with open(f'data/distance_matrix.pkl', 'wb') as f:
        pickle.dump(distance_matrix, f)
        
    satNo_idx_dict, idx_satNo_dict = get_key(df)
    return distance_matrix, {"satNo_idx": satNo_idx_dict, "idx_satNo": idx_satNo_dict}
        
def get_key(df):
    
    df = df['satNo'].unique()
    
    satNo_idx_dict = {}
    idx_satNo_dict = {}
    
    for i, satNo in enumerate(df):
        idx_satNo_dict[i] = satNo
        satNo_idx_dict[satNo] = i
        
        # save both as pkl
    with open(f'data/satNo_idx_dict.pkl', 'wb') as f:
        pickle.dump(satNo_idx_dict, f)
        
    with open(f'data/idx_satNo_dict.pkl', 'wb') as f:
        pickle.dump(idx_satNo_dict, f)
            
    print("Saved satNo to index and index to satNo dictionaries to disk.")
    return satNo_idx_dict, idx_satNo_dict
    


