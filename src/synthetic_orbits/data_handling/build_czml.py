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


import colorsys

def get_cluster_colors(labels):
    """
    Generate a distinct, bright RGBA color for each unique cluster label.
    Uses HSV color space to space hues evenly, keeping high saturation and value.
    Noise label (-1) gets grey.
    """
    unique_labels = sorted(set(str(l).replace('.0', '') for l in labels))
    # Remove noise
    non_noise = [l for l in unique_labels if l != '-1']
    
    color_map = {}
    n = len(non_noise)
    
    for idx, label in enumerate(non_noise):
        # Space hues evenly across the spectrum, skip dark blues/near-black region
        hue = (idx / max(n, 1)) % 1.0
        # High saturation and value to avoid dark/muted colors
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        color_map[label] = [int(r * 255), int(g * 255), int(b * 255), 255]
    
    # Noise gets grey
    color_map['-1'] = [128, 128, 128, 255]
    
    return color_map


def build_czml(df):
    
    epochTime = calculate_average_epoch(df)
    endTime = epochTime + dt.timedelta(days=65)
    epochStr, endTimeStr = map(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S.%fZ'), [epochTime, endTime])
    
    # Pre-compute color map across all cluster labels in the dataframe
    color_map = get_cluster_colors(df['label'].tolist())
    
    czml = [{'id': 'document', 'version': '1.0'}]
    
    for i, row in df.iterrows():
        
        _, coordinates = get_posvcs(row['line1'], row['line2'], epochStr)
        coords = [int(coord) if idx % 4 == 0 else float(coord) 
                  for idx, coord in enumerate(coordinates)]
        
        label_str = str(row['label']).replace('.0', '')
        
        # Synthetic orbits override color to distinguish from real satellites
        if row['satNo'] == '99999':
            if row['name'] in ('MaximallySeparated', 'Maximally Separated'):
                color = [255, 255, 255, 255]   # white — maximally separated
            else:
                color = [255, 165, 0, 255]     # orange — Fréchet mean
        else:
            color = color_map.get(label_str, [0, 255, 0, 255])
        
        name = row['name']
        if name == 'MaximallySeparated':
            name = 'Maximally Separated'
        if name == 'Optimized':
            name = 'Fréchet mean'
                
        czml.append({
            'id': f"{row['satNo']}_{i}",
            'name': str(name),
            'availability': f"{epochStr}/{endTimeStr}",
            'position': {
                'epoch': epochStr, 
                'cartographicDegrees': coords, 
                'interpolationDegree': 5,
                'interpolationAlgorithm': 'LAGRANGE'
            },
            'properties': {
                'satNo': row['satNo'],
                'apogee': row['apogee'],
                'inclination': row['inclination'],
                'prop_correlated': row['correlated'],
                'label': label_str,
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