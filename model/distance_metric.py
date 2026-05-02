import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta, timezone
# from PIL import Image
# from graphviz import Graph
from DMT import VectorizedKeplerianOrbit
import pickle
import json
import sys

"""
This file is used to process the given elset data into a distance matrix and save
to disk.
"""

def get_common_sats():
    
    """
    This file will return a df with the satellites unioned between results.csv and elsetcurrent from udl
    """

    with open('data/elset_current.json', 'r') as file:
        data = json.load(file)

    # put data into pd dataframe
    elsetcurrent_df = pd.DataFrame(data)

    # remove all duplicates of the same satNo
    elsetcurrent_df = elsetcurrent_df.drop_duplicates(subset='satNo', keep='first')

    # if line1, line2, or satNo is NaN, drop the row
    elsetcurrent_df = elsetcurrent_df.dropna(subset=['line1', 'line2', 'satNo'])

    # Remove trailing .0 and zero-pad to 5 digits
    elsetcurrent_df['satNo'] = (
        elsetcurrent_df['satNo']
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )

    # convert createdAt to datetime
    elsetcurrent_df['createdAt'] = pd.to_datetime(elsetcurrent_df['createdAt'], utc=True)

    # remove any outdated entries (1 month) - using UTC
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
    elsetcurrent_df = elsetcurrent_df[elsetcurrent_df['createdAt'] > cutoff_date]

    #load in results.csv into a dataframe
    results_df = pd.read_csv('data/results.csv')
    
    print(f"size of results_df: {results_df.shape}")

    results_df['NORAD_CAT_ID'] = (
        results_df['NORAD_CAT_ID']
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )

    union_df = pd.merge(results_df, elsetcurrent_df, left_on='NORAD_CAT_ID', right_on='satNo', how='inner')
    
    return union_df

def get_distance_matrix(df):
            
    line1 = df['line1'].values
    line2 = df['line2'].values
    
    print("Calculating orbits")
    orbits = VectorizedKeplerianOrbit(line1, line2)
    
    print("Calculating distances")
    distance_matrix = VectorizedKeplerianOrbit.DistanceMetric(orbits, orbits)
                
    return distance_matrix
        
def get_key(df):
    satNo_idx_dict = {}
    idx_satNo_dict = {}
    
    for i in range(len(df)):
        satNo = df['satNo'].iloc[i]
        idx_satNo_dict[i] = satNo
        satNo_idx_dict[satNo] = i
    
    return satNo_idx_dict, idx_satNo_dict
    
def uniq_score_df():
    df = get_common_sats()  # already has 'NORAD_CAT_ID', 'satNo', 'OBJECT_NAME', etc.
    
    distance_matrix = get_distance_matrix(df)
    satNo_idx_dict, idx_satNo_dict = get_key(df)
    
    avg_dist = np.mean(distance_matrix, axis=1)
    
    # DataFrame of scores mapped back to satNo
    scores_df = pd.DataFrame({
        'satNo': [idx_satNo_dict[i] for i in range(len(avg_dist))],
        'avg_dist': avg_dist
    })
    
    scores_df['uniqueness'] = (
        (scores_df['avg_dist'] - scores_df['avg_dist'].min()) /
        (scores_df['avg_dist'].max() - scores_df['avg_dist'].min())
    )
    
    # Merge scores back into the original df so it includes NORAD_CAT_ID etc.
    merged_df = pd.merge(df, scores_df, on='satNo', how='left')
    
    return merged_df
    

    
    
