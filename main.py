import pandas as pd
from math import pi
from sklearn.cluster import AffinityPropagation
import numpy as np
from distance_matrix import get_distance_matrix
from matplotlib import pyplot as plt


def celestrak_elsets_to_df():
    data = []
    celestrak_tles_path = 'data/3le'
    with open(celestrak_tles_path, 'r') as f:
        lines = f.readlines()
   
    earth_radius = 6371000  # meters
    GM_earth = 3.986004418e14  # m^3/s^2
    seconds_in_day = 86400  # seconds per day
   
    for i in range(0, len(lines), 3):
        if i + 2 < len(lines):
            name = lines[i].strip()
            line1 = lines[i + 1].strip()
            line2 = lines[i + 2].strip()
           
            if not (line1.startswith('1 ') and line2.startswith('2 ')):
                continue
               
            sat_no = line1[2:7].strip()
           
            inclination = float(line2[8:16].strip())
           
            # Extract mean motion (revolutions per day)
            mean_motion = float(line2[52:63].strip())
           
            # Extract eccentricity (columns 27-33 in standard TLE format)
            eccentricity = float("0." + line2[26:33].strip())
           
            # Calculate angular velocity n in radians per second
            n = mean_motion * 2 * pi / seconds_in_day
           
            # Calculate semi-major axis a in meters
            a = (GM_earth / (n ** 2)) ** (1/3)
           
            # Calculate apogee in kilometers
            apogee = (a * (1 + eccentricity) - earth_radius) / 1000
           
            data.append({
                'satNo': sat_no,
                'line1': line1,
                'line2': line2,
                'inclination': inclination,
                'apogee': apogee
            })
   
    elset_df = pd.DataFrame(data)
   
    elset_df['satNo'] = (
        elset_df['satNo']
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )
   
    return elset_df

def compute_clusters(distance_matrix, damping):
    
    affinity_clustering = AffinityPropagation(affinity='precomputed', damping=damping)
    affinity_clustering.fit(np.exp(-distance_matrix/np.var(distance_matrix)))

    # Get the cluster labels
    labels = affinity_clustering.labels_

    return labels

def plot(df):
    
    if df.empty:
        print("No data available to plot.")
        return
   
    inclination = df['inclination'].values
    apogee = df['apogee'].values
    
    inc_min, inc_max = inclination.min(), inclination.max()
    apo_min, apo_max = apogee.min(), apogee.max()
   
    inc_grid, apo_grid = np.mgrid[inc_min:inc_max:150j, apo_min:apo_max:150j]

    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.set_xlabel('Inclination (degrees)')
    ax.set_ylabel('Apogee (km)')

    plt.tight_layout()
    fig.savefig(f'data/orbital_density_2D_({apo_min}, {apo_max}),({inc_min}, {inc_max}).png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    # Only loading in sats from leo in celestrak current catalog
    
    tle_df = celestrak_elsets_to_df()
    
    # get the distance matrix and compute clusters
    distance_matrix, _ = get_distance_matrix(tle_df)
    labels = compute_clusters(distance_matrix, 0.95)
    
    # get cluster counts for logging
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique_labels, label_counts))
    print("\nCluster counts:")
    print(cluster_counts)
    
    # Add cluster information directly to the DataFrame
    tle_df['label'] = labels.astype(int)
    tle_df['correlated'] = False

    plot(tle_df, ranges={'inclination': (0, 180), 'apogee': (0, 2000)})