import os
import pandas as pd
from datetime import datetime
from sgp4.api import Satrec
import json
import datetime as dt
import numpy as np
from random import random

def getPos(dSeconds, satrec, julianDate):
    
    fraction = dSeconds / 86400
    
    days = 0
    if fraction > 1:
        days = int(fraction)
        fraction -= days
    
    return satrec.sgp4(julianDate + days, fraction)[1]

def get_posvcs(TLE_LINE1, TLE_LINE2, only_one_period = True):
    
    satrec = Satrec.twoline2rv(TLE_LINE1, TLE_LINE2)
    julianDate = satrec.jdsatepoch
    
    #getPos = lambda dSeconds: satrec.sgp4(julianDate, (dSeconds / 86400))[1]
    getLat = lambda position: np.degrees(np.arctan2(position[2], np.sqrt(position[0]**2 + position[1]**2)))
    getLon = lambda position: np.degrees(np.arctan2(position[1], position[0]))
    getAlt = lambda position: ((np.sqrt(position[0]**2 + position[1]**2 + position[2]**2) - 6371) * 1000)

    positions = []
    coord_list = []
    
    mean_motion = satrec.no_kozai # radians per minute
    periodInMinutes = np.pi * 2  / mean_motion
    periodInSeconds = int(periodInMinutes * 60)
    
    if only_one_period:
        time_limit = periodInSeconds
    else:
        time_limit = 86400
    
    stepSeconds = 600
    
    for dSeconds in range(0, time_limit + stepSeconds , stepSeconds):
        position = getPos(dSeconds, satrec, julianDate)
        positions.append(position)
        lat = getLat(position)
        lon = getLon(position)
        alt = getAlt(position)
        coord_list.extend([dSeconds, lon, lat, alt])
    
    if only_one_period:
        coord_list.extend([periodInSeconds, getLon(positions[0]), getLat(positions[0]), getAlt(positions[0])])
    
    return positions, coord_list

def build_czml(tle_df):

    epochTime = dt.datetime.now(dt.timezone.utc)
    endTime = epochTime + dt.timedelta(days=10)
    epochStr, endTimeStr = map(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S.%fZ'), [epochTime, endTime])
    czml = [{'id': 'document', 'version': '1.0'}]
    
    for i, row in tle_df.iterrows():
        tle_line1 = row['line1']
        tle_line2 = row['line2']
        satNo = row['satNo']
        try:
            _, coordinates = get_posvcs(tle_line1, tle_line2)
        except Exception as ex:
            print(f"Error processing TLE for {tle_line1}: {ex}")
            continue
        coords = [int(coord) if i % 4 == 0 else float(coord)
                    for i, coord in enumerate(coordinates)]
        
        czml.append({
            'id': satNo,
            'line1': tle_line1,
            'line2': tle_line2,
            'availability': f"{epochStr}/{endTimeStr}",
            'position': {
                'epoch': epochStr, 
                'cartographicDegrees': coords, 
                'interpolationDegree': 5,
                'interpolationAlgorithm': 'LAGRANGE'
            },
            'point': {
                    'color': {'rgba': [255, 255, 0, 255]}, 
                    'pixelSize': 2
            }
        })
            
    with open('cesium_model/data/output.czml', 'w') as file:
        json.dump(czml, file, indent=2, separators=(',', ': '))
      
def get_orbital_regimes() -> pd.DataFrame:
    """
    Get the orbital regimes from the CSV file
    """
    # Load the CSV file
    df = pd.read_csv('data/satcat.tsv', sep='\t', low_memory=False)

    # Things that are around the earth
    around_earth = df[df['Primary']=='Earth']

    # Remove rows with dashes/ negative values
    around_earth = around_earth[~around_earth['Perigee'].astype(str).str.contains('-')]
    around_earth.Perigee = around_earth.Perigee.astype(float)


    in_leo = around_earth[around_earth.OpOrbit.str.contains('LEO')].copy()
    in_geo = around_earth[around_earth.OpOrbit.str.contains('GEO')].copy()
    in_heo = around_earth[around_earth.OpOrbit.str.contains('HEO')].copy()
    in_meo = around_earth[around_earth.OpOrbit.str.contains('MEO')].copy()
    
    
    return in_leo, in_meo, in_heo, in_geo, 

if __name__ == '__main__':
    build_czml()