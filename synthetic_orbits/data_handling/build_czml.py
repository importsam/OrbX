from sgp4.api import Satrec, jday
import json
import datetime as dt
import numpy as np
from random import random
from skyfield.api import load
from synthetic_orbits.orbit_finder.get_optimum_orbit import calculate_average_epoch

ts = load.timescale()

def getPos(dSeconds, satrec, julianDate):
    
    fraction = dSeconds / 86400
    
    days = 0
    if fraction > 1:
        days = int(fraction)
        fraction -= days
    
    return satrec.sgp4(julianDate + days, fraction)[1]

def get_posvcs(TLE_LINE1, TLE_LINE2, epochStr, only_one_period=True):
    # Initialize satellite with skyfield
    satrec = Satrec.twoline2rv(TLE_LINE1, TLE_LINE2)
    
    # Convert epochStr to skyfield Time
    epoch_dt = dt.datetime.strptime(epochStr, '%Y-%m-%dT%H:%M:%S.%fZ')
    year = epoch_dt.year
    month = epoch_dt.month
    day = epoch_dt.day
    hour = epoch_dt.hour
    minute = epoch_dt.minute
    second = epoch_dt.second + epoch_dt.microsecond / 1e6  # Include microseconds
    
    # Convert epochStr to Julian date
    jd_epochStr, fr_epochStr = jday(year, month, day, hour, minute, second)
    
    mean_motion = satrec.no_kozai # radians per minute
    periodInMinutes = np.pi * 2  / mean_motion
    periodInSeconds = int(periodInMinutes * 60)
    
    if only_one_period:
        time_limit = periodInSeconds
    else:
        time_limit = 86400
    
    stepSeconds = 600
    
    positions = []
    coord_list = []
    
    getLat = lambda position: np.degrees(np.arctan2(position[2], np.sqrt(position[0]**2 + position[1]**2)))
    getLon = lambda position: np.degrees(np.arctan2(position[1], position[0]))
    getAlt = lambda position: ((np.sqrt(position[0]**2 + position[1]**2 + position[2]**2) - 6371) * 1000)
    
    for dSeconds in range(0, time_limit + stepSeconds , stepSeconds):
        delta_days = dSeconds / 86400.0
        jd_total = jd_epochStr + fr_epochStr + delta_days
        jd_int = int(jd_total)
        fr = jd_total - jd_int
        
        position = satrec.sgp4(jd_int, fr)[1]
        
        positions.append(position)
        lat = getLat(position)
        lon = getLon(position)
        alt = getAlt(position)
        coord_list.extend([dSeconds, lon, lat, alt])
        
        
    if only_one_period:
        lat0 = getLat(positions[0])
        lon0 = getLon(positions[0])
        alt0 = getAlt(positions[0])
        coord_list.extend([periodInSeconds, lon0, lat0, alt0])
    
    # if only_one_period:
    #     coord_list.extend([periodInSeconds, getLon(positions[0]), getLat(positions[0]), getAlt(positions[0])])
    
    return positions, coord_list

def build_czml(df):
    
    epochTime = calculate_average_epoch(df)
    
    endTime = epochTime + dt.timedelta(days=65)
    epochStr, endTimeStr = map(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S.%fZ'), [epochTime, endTime])
    czml = [{'id': 'document', 'version': '1.0'}]
    
    for i, row in df.iterrows():
        
        correlated = row['correlated']
       
        _, coordinates = get_posvcs(row['line1'], row['line2'], epochStr)
        coords = [int(coord) if i % 4 == 0 else float(coord) 
                  for i, coord in enumerate(coordinates)]
        
        # number of clusters given by max label
        num_clusters = len(set(df['label']))
        # color = get_random_color()
        
        if row['satNo'] == '99999':
            # synthetic orbit always has satNo 99999
            color = [255, 255, 255, 255]
        else:
            # keep inputs green
            color = [0, 255, 0, 255]
                
        czml.append({
            'id': str(row['satNo']),
            'name': str(row['name']),
            'availability': f"{epochStr}/{endTimeStr}",
            'position': {
                'epoch': epochStr, 
                'cartographicDegrees': coords, 
                'interpolationDegree': 3,
                'interpolationAlgorithm': 'LAGRANGE'
            },
            'properties': {
                'apogee': row['apogee'],
                'inclination': row['inclination'],
                'prop_correlated': row['correlated'],
                'prop_orbitColor': {"rgba": color}
            },
            'point': {
                'color': {
                    'rgba': color
                },
                'pixelSize': 2
            }
        })
        
        
            
    with open('orbX/output.czml', 'w') as file:
        json.dump(czml, file, indent=2, separators=(',', ': '))
    
if __name__ == '__main__':
    build_czml()