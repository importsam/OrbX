import numpy as np
from sgp4.api import Satrec
import os
import datetime as dt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CZMLConfig

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
    
    for dSeconds in range(0, time_limit + stepSeconds, stepSeconds):
        position = getPos(dSeconds, satrec, julianDate)
        positions.append(position)
        lat = getLat(position)
        lon = getLon(position)
        alt = getAlt(position)
        coord_list.extend([dSeconds, lon, lat, alt])
    
    if only_one_period:
        coord_list.extend([periodInSeconds, getLon(positions[0]), getLat(positions[0]), getAlt(positions[0])])
    
    return positions, coord_list

def density_to_color(density, min_density, max_density):
    """
    Convert density value to RGBA color using the same 'hot_r' colormap as the 2D plot
    """
    # Normalize density to 0-1 range
    norm = Normalize(vmin=min_density, vmax=max_density)
    normalized_density = norm(density)
    
    # Use hot_r colormap (same as in plot_KDE_2D)
    colormap = cm.get_cmap('hot_r')
    rgba = colormap(normalized_density)
    
    # Convert to 0-255 range for CZML
    rgba_255 = [int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255), 255]
    
    return rgba_255

def build_czml(df):

    print("building czml for live")
    
    epochTime = dt.datetime.now(dt.timezone.utc)
    endTime = epochTime + dt.timedelta(days=65)
    epochStr, endTimeStr = map(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S.%fZ'), [epochTime, endTime])

    czml = [{'id': 'document', 'version': '1.0'}]
    
    for _, row in df.iterrows():
        try:
            _, coordinates = get_posvcs(row['line1'], row['line2'])
    
            # Validate coordinates before adding them
            coords = []
            for i, coord in enumerate(coordinates):
                # Convert to appropriate type and validate
                try:
                    if i % 4 == 0:
                        coord_value = int(coord)
                    else:
                        coord_value = float(coord)
                    
                    # Check for NaN, Infinity values
                    if (isinstance(coord_value, float) and 
                        (np.isnan(coord_value) or np.isinf(coord_value))):
                        print(f"Invalid coordinate value found: {coord_value} for satellite {row['satNo']}")
                        # Skip this satellite
                        raise ValueError("Invalid coordinate detected")
                    
                    coords.append(coord_value)
                except (ValueError, TypeError) as e:
                    print(f"Error converting coordinate {coord}: {e}")
                    raise ValueError("Coordinate conversion error")
            
            czml.append({
                'id': str(row['satNo']),  # Ensure ID is a string
                'name': str(row['name']),  # Ensure name is a string
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
                    'density': row['density'],
                    # 'orbit_colour_r': point_color[0],
                    # 'orbit_colour_g': point_color[1],
                    # 'orbit_colour_b': point_color[2]
                },
                'point': {
                    'color': {'rgba': [255, 255, 0, 255]}, 
                    'pixelSize': 2
                }
            })
    
        except Exception as e:
            print(f"Error processing satellite {row.get('satNo', 'Unknown')}: {e}")
            continue
    
    output_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'output.czml')
    
    try:
        with open(output_file, 'w') as file:
            # Use simplejson if available for better NaN handling
            try:
                import simplejson as json_module
            except ImportError:
                import json as json_module
            
            json_module.dump(czml, file, indent=2, separators=(',', ': '))
            
        with open(output_file, 'r') as file:
            _ = json_module.load(file)
            print("CZML file validated successfully")
            
    except Exception as e:
        print(f"Error writing or validating CZML file: {e}")