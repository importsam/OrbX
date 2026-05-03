""" 
1. We want to isolate satellites from 300-700km based on calculated apogee.
2. Pass said sats into clustering.
3. Pass clusters and get both frechet and max separation orbits
4. Assign cluster label as a property to satellite objects in czml
5. Append the synthetic orbits to the czml.
"""
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import pandas as pd
from orbx.clustering.data_handling.TLEParser import TLEParser
from orbx.clustering import cluster
from orbx.synthetic_orbits.synthetic_orbit import synthetic_orbit


def spacetrack_parse_file(path='3le_1126'):
    
    df = pd.DataFrame()
    parser = TLEParser()
    
    with open(path, 'r') as f:
        lines = f.readlines()
    
    for i in range(0, len(lines), 3):
        if i + 2 >= len(lines):
            break
            
        name_line = lines[i].strip()
        name = name_line[2:].strip() if name_line.startswith('0 ') else name_line
        
        sat_obj = parser._parse_tle_group(
            lines[i+1].strip(), 
            lines[i+2].strip()
        )
        
        df = pd.concat([df, pd.DataFrame([{
            'satNo': sat_obj.sat_no,
            'name': name,
            'line1': sat_obj.line1,
            'line2': sat_obj.line2,
            'inclination': sat_obj.inclination,
            'apogee': sat_obj.apogee,
            'raan': sat_obj.raan,
            'argument_of_perigee': sat_obj.argument_of_perigee,
            'eccentricity': sat_obj.eccentricity,
            'mean_motion': sat_obj.mean_motion
        }])], ignore_index=True)

    return df

if __name__ == "__main__":
    
    df = spacetrack_parse_file('data/elset_current.text')
    
    df = df[(df['apogee'] >= 300) & (df['apogee'] <= 700)]
    
    print("Number of satellites in 300-700km: ", len(df))
    
    labels = cluster(df)
    
    df['label'] = labels
    
    # df.sample(100)
    # save as pkl 
    df.to_pickle('data/clustered_sats.pkl')
    
    print(df.head())
    
    # print number of clusters
    print(len(df['label'].unique()))
    
    df = df[df['label'] != -1]
    
    synthetic_df = synthetic_orbit(df, mode=['frechet', 'max_separation'], verbose=True)
    
    df.to_pickle('data/synthetic_orbits.pkl')
    
    