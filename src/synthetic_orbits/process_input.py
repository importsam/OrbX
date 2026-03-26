from DMT import VectorizedKeplerianOrbit
import pickle
import json
import sys

"""
This file is used to process the given elset data into a distance matrix and save
to disk.
"""
    
def get_distance_matrix(df):
    
    """This will return the distance matrix upper triangle and the key for satNo to index"""
    
    line1 = df['line1'].values
    line2 = df['line2'].values
    
    print("Calculating orbits")
    orbits = VectorizedKeplerianOrbit(line1, line2)
    
    print("Calculating distances")
    distance_matrix = VectorizedKeplerianOrbit.DistanceMetric(orbits, orbits)
                
    print("Saving distance matrix to disk...") 
    with open(f'data/distance_matrix.pkl', 'wb') as f:
        pickle.dump(distance_matrix, f)
        
    key = get_key(df)
    
    return distance_matrix, key

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
    


