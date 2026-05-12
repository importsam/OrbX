import numpy as np
from sgp4.api import Satrec
import os
import json

import datetime as dt


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

def build_czml_live(df):
    print("building czml for live")
    
    epochTime = dt.datetime.now(dt.timezone.utc)
    endTime = epochTime + dt.timedelta(days=65)
    epochStr, endTimeStr = map(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S.%fZ'), [epochTime, endTime])
                     
    czml = [{'id': 'document', 'version': '1.0'}]
    
    property_keys = [k for k in df.columns if k.startswith("prop_")]

    for _, row in df.iterrows():
        try:
            _, coordinates = get_posvcs(row['TLE_LINE1'], row['TLE_LINE2'])
            
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
                        print(f"Invalid coordinate value found: {coord_value} for satellite {row['NORAD_CAT_ID']}")
                        # Skip this satellite
                        raise ValueError("Invalid coordinate detected")
                    
                    coords.append(coord_value)
                except (ValueError, TypeError) as e:
                    print(f"Error converting coordinate {coord}: {e}")
                    raise ValueError("Coordinate conversion error")
            
            # Convert neighbours list into a dictionary with string keys
            neighbours = row.get("neighbours", [])
            neighbours_dict = {}
            for i, neighbour in enumerate(neighbours):
                if isinstance(neighbour, (str, int, float, bool, type(None))):
                    neighbours_dict[str(i+1)] = neighbour
                else:
                    neighbours_dict[str(i+1)] = str(neighbour)

            raw_cluster_label = row.get("label", row.get("cluster_label"))
            cluster_label = str(raw_cluster_label)

            synthetic_type = row.get("synthetic_type", None)
            if synthetic_type is None or (isinstance(synthetic_type, float) and np.isnan(synthetic_type)):
                synthetic_type = "None"

            additional_properties = {
                'uniqueness_range': row.get('uniqueness_range', 'none'),
                'neighbours': neighbours_dict,
                'cluster_label': cluster_label,
                'synthetic_type': synthetic_type,
            }
            
            # Convert properties and check for valid JSON values
            cleaned_properties = {}
            for key in property_keys:
                prop_value = row.get(key)
                if prop_value is None:
                    cleaned_properties[key[5:]] = "None"
                elif isinstance(prop_value, (str, int, bool)):
                    cleaned_properties[key[5:]] = prop_value
                elif isinstance(prop_value, float):
                    if np.isnan(prop_value) or np.isinf(prop_value):
                        cleaned_properties[key[5:]] = "None"
                    else:
                        cleaned_properties[key[5:]] = prop_value
                else:
                    cleaned_properties[key[5:]] = str(prop_value)

            if synthetic_type == 'frechet':
                point_style = {'color': {'rgba': [0, 255, 128, 255]}, 'pixelSize': 6}
            elif synthetic_type == 'max_separation':
                point_style = {'color': {'rgba': [255, 80, 80, 255]}, 'pixelSize': 6}
            else:
                point_style = {'color': {'rgba': [255, 255, 0, 255]}, 'pixelSize': 2}
            
            czml.append({
                'id': str(row['NORAD_CAT_ID']),  # Ensure ID is a string
                'name': str(row['OBJECT_NAME']),  # Ensure name is a string
                'availability': f"{epochStr}/{endTimeStr}",
                'position': {
                    'epoch': epochStr, 
                    'cartographicDegrees': coords, 
                    'interpolationDegree': 3,
                    'interpolationAlgorithm': 'LAGRANGE'
                },
                'properties': {**cleaned_properties, **additional_properties},
                'point': point_style
            })
            
        except Exception as e:
            # Log the error and continue with the next satellite
            print(f"Error processing satellite {row.get('NORAD_CAT_ID', 'Unknown')}: {e}")
            continue
    
    # Write file with error checking
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
            
        # Validate the file works
        with open(output_file, 'r') as file:
            _ = json_module.load(file)
            print("CZML file validated successfully")
            
    except Exception as e:
        print(f"Error writing or validating CZML file: {e}")