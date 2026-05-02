import os
import pandas as pd
from query_spacetrack import query_udl
import numpy as np
from DMT import VectorizedKeplerianOrbit
from dev.build_czml import build_czml_dev
from live.build_czml import build_czml_live

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

def get_satellites_info(run_from_scratch=True):
    
    # let this refresh the elsets every time it gets called
    
    print('Getting orbital regimes')
    LEO, MEO, HEO, GEO = get_orbital_regimes()
    print("Done")
    
    # load in satcat.tsv into a dataframe
    satcat = pd.read_csv('data/satcat.tsv', sep='\t', low_memory=False)
        
    if run_from_scratch:
        if os.path.exists("data/elset_current.text"):
            os.remove("data/elset_current.text")
        query_udl()
    
    if not os.path.exists("data/elset_current.text"):
        print("File not found data/elset_current.text, QUITTING")
        return False
    
    print('Reading current Elsets')
    with open("data/elset_current.text", "r") as f:
        lines = [l.strip() for l in f.readlines()]
        
    print('Done')
        
    tles = []
    for i in range(0, len(lines), 3):
        name = lines[i][2:]
        tle_line1 = lines[i+1]
        tle_line2  = lines[i+2]
        NORAD_ID = str(tle_line1[2:2+5]).replace(" ", "0")
        
        # if the NORAD_ID is not in the Satcat col of satcat, skip
        if NORAD_ID not in satcat['Satcat'].values:
            print(f"Skipping {NORAD_ID} as it is not in the satcat")
            continue
                
        if NORAD_ID in LEO.Satcat.values:
            orbital_regime = "LEO"
        elif NORAD_ID in MEO.Satcat.values:
            orbital_regime = "MEO"
        elif NORAD_ID in HEO.Satcat.values:
            orbital_regime = "HEO"
        elif NORAD_ID in GEO.Satcat.values:
            orbital_regime = "GEO"
        else:
            continue
        
        uniqueness_score = -1.0 
        tles.append([NORAD_ID, 
                     name, 
                     tle_line1, 
                     tle_line2,  
                     orbital_regime, 
                     uniqueness_score,
                     0])
        
    tle_df = pd.DataFrame(tles, columns=["NORAD_CAT_ID", 
                                         "OBJECT_NAME",
                                         "TLE_LINE1", 
                                         "TLE_LINE2",
                                         "prop_orbit_class",
                                         "prop_uniqueness",
                                         "prop_rank"])
    
    # remove any 'TO BE ASSIGNED' satellites
    tle_df = tle_df[~tle_df['OBJECT_NAME'].astype(str).str.contains('TBA - TO BE ASSIGNED')]
    return tle_df

def compute_neighbour_lookup(top_n=10):
    
    """
    Load satellites from TLE file, compute the distance matrix for all satellites,
    and return a dictionary where each key (satNo) maps to a list of its 10 nearest neighbour satNos,
    sorted from smallest to largest distance.
    """
    
    # Load TLE data.
    with open("data/elset_current.text", "r") as f:
        lines = [l.strip() for l in f.readlines()]
    
    tles = []
    for i in range(0, len(lines), 3):
        name = lines[i][2:]
        tle_line1 = lines[i+1]
        tle_line2 = lines[i+2]
        # Extract NORAD ID from the TLE line, pad with zeroes as necessary.
        satNo = str(tle_line1[2:2+5]).replace(" ", "0")
        tles.append([satNo, name, tle_line1, tle_line2])
        
    df = pd.DataFrame(tles, columns=["satNo", "name", "TLE_LINE1", "TLE_LINE2"])
    
    # Remove any 'TO BE ASSIGNED' satellites.
    df = df[~df['name'].astype(str).str.contains('TBA - TO BE ASSIGNED')]
    
    # Compute the orbits and full distance matrix.
    line1 = df['TLE_LINE1'].values
    line2 = df['TLE_LINE2'].values
    print("Calculating orbits")
    orbits = VectorizedKeplerianOrbit(line1, line2)
    
    print("Calculating distances")
    distance_matrix = np.array(VectorizedKeplerianOrbit.DistanceMetric(orbits, orbits))
    
    # Build neighbour lookup dictionary.
    neighbour_lookup = {}
    for idx, satNo in enumerate(df['satNo']):
        distances = distance_matrix[idx]
        # Sort indices by distance, ignoring self (assumes self-distance == 0)
        sorted_indices = np.argsort(distances)
        neighbour_indices = [j for j in sorted_indices if j != idx][:top_n]
        
        # Build a list of neighbour satNos (ordered from smallest to largest distance).
        neighbours = [df.iloc[neighbour_idx]['satNo'] for neighbour_idx in neighbour_indices]
        
        neighbour_lookup[satNo] = neighbours
        
    return neighbour_lookup

def scoring_main(from_strach=True):    
    sats_df = get_satellites_info(from_strach)

    orbital_regimes = ["LEO", "MEO", "HEO", "GEO", "GTO", "DSO", "CLO", "EEO", "HCO", "PCO", "SSE"]

    results = pd.DataFrame()

    for regime in orbital_regimes:
        print(f"Number of satellites in {regime}: {sats_df[sats_df.prop_orbit_class==regime].shape[0]}")

        current_regime = sats_df[sats_df.prop_orbit_class == regime].copy()
                
        # if current_regime is empty continue
        if len(current_regime) < 100:
            print(f"Skipping {regime} as it has less than 100 satellites")
            continue
        
        lines1 = current_regime.TLE_LINE1.values
        lines2 = current_regime.TLE_LINE2.values
        
        orbits = VectorizedKeplerianOrbit(lines1, lines2)
        print("Calculating distances")
        
        distances = VectorizedKeplerianOrbit.DistanceMetric(orbits, orbits)
        distances = np.array(distances)
        
        def k_nearest_mean(dist_row):
            k = 100
            sorted_row = np.sort(dist_row)
            # Exclude the first element (self-distance, assumed 0) and take the next k distances
            return np.mean(sorted_row[1:k+1])
        
        scores = np.apply_along_axis(k_nearest_mean, 1, distances)
        
        mean_scores = np.mean(scores)
        var_scores = np.var(scores)
        
        valid_idx = np.square(scores - mean_scores) / var_scores < 2.71
        scores = scores[valid_idx]
        current_regime = current_regime[valid_idx]
        
        current_regime['prop_uniqueness'] = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        current_regime['prop_rank'] = current_regime['prop_uniqueness'].rank(ascending=False).astype(int)
        
        # Set default uniqueness_range to "none"
        current_regime["uniqueness_range"] = "none"
        # Sort by prop_rank so that the highest uniqueness (lowest rank number) come first
        current_regime.sort_values("prop_rank", inplace=True)
        # Get the indices for the top 5 and bottom 5
        top_five_indices = current_regime.head(5).index
        bottom_five_indices = current_regime.tail(5).index
        # Set uniqueness_range for top 5 as "most" and bottom 5 as "least"
        current_regime.loc[top_five_indices, "uniqueness_range"] = "most"
        current_regime.loc[bottom_five_indices, "uniqueness_range"] = "least"
        
        results = pd.concat([results, current_regime])
        
        print(f"Mean distances score for {regime}: {np.mean(scores)}")
        print(f"Variation in distances score for {regime}: {np.var(scores)}")
    
    neighbour_lookup = compute_neighbour_lookup(top_n=20)  # Get more than needed
    
    # Create a set of all valid satellite IDs in the final results
    valid_sat_ids = set(results['NORAD_CAT_ID'])
    
    # Function to get valid neighbors
    def get_valid_neighbors(sat_id):
        # Get all potential neighbors
        all_neighbors = neighbour_lookup.get(sat_id, [])
        # Filter to only include neighbors that exist in the final results
        valid_neighbors = [n for n in all_neighbors if n in valid_sat_ids and n != sat_id]
        
        # If we don't have enough valid neighbors, add random ones from the same regime
        if len(valid_neighbors) < 10:
            regime = results.loc[results['NORAD_CAT_ID'] == sat_id, 'prop_orbit_class'].iloc[0]
            # Get other satellites in the same regime
            same_regime_sats = results[
                (results['prop_orbit_class'] == regime) & 
                (results['NORAD_CAT_ID'] != sat_id)
            ]['NORAD_CAT_ID'].tolist()
            
            # Add satellites not already in neighbors
            additional_needed = 10 - len(valid_neighbors)
            additional = [s for s in same_regime_sats if s not in valid_neighbors][:additional_needed]
            valid_neighbors.extend(additional)
            
            # If still not enough, add from any regime
            if len(valid_neighbors) < 10:
                all_sats = results[results['NORAD_CAT_ID'] != sat_id]['NORAD_CAT_ID'].tolist()
                additional_needed = 10 - len(valid_neighbors)
                additional = [s for s in all_sats if s not in valid_neighbors][:additional_needed]
                valid_neighbors.extend(additional)
        
        # Return exactly 10 neighbors (or all we have if less than 10)
        return valid_neighbors[:10]
    
    # Apply the function to get exactly 10 neighbors where possible
    results["neighbours"] = results["NORAD_CAT_ID"].apply(get_valid_neighbors)
    
    # Save results
    results.to_pickle("data/satellites_with_scores.pkl")

if __name__ == '__main__':
    scoring_main()
    # read in the results
    