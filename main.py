import random
import pandas as pd
from math import pi
from sklearn.cluster import AffinityPropagation
import numpy as np
from distance_matrix import get_distance_matrix
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score
import pickle

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


def compute_clusters_affinity(distance_matrix, damping):
    affinity_clustering = AffinityPropagation(affinity='precomputed', damping=damping)
    affinity_clustering.fit(np.exp(-distance_matrix/np.var(distance_matrix)))

    # Get the cluster labels
    labels = affinity_clustering.labels_
    
    def compute_silhouette(distance_matrix, labels):
        
        distance_matrix_copy = distance_matrix.copy()
        np.fill_diagonal(distance_matrix_copy, 0)
        
        # Only compute silhouette if we have more than 1 cluster
        if len(set(labels)) > 1:
            score = silhouette_score(distance_matrix_copy, labels, metric="precomputed")
            print(f"\nSilhouette score: {score:.4f}")
            return score
        else:
            print("\nOnly one cluster, silhouette score = N/A")
            return None
    
    compute_silhouette(distance_matrix, labels)
    
    return labels

def check_matrix(distance_matrix):
    # Check if the distance matrix is symmetric
    if not np.allclose(distance_matrix, distance_matrix.T):
        raise ValueError("Distance matrix is not symmetric")
    
    # Check if the diagonal elements are zero
    if not np.allclose(np.diag(distance_matrix), 0):
        raise ValueError("Diagonal elements of the distance matrix are not zero")
    
    print("Distance matrix is valid.")

def plot(df):
    if df.empty:
        print("No data available to plot.")
        return

    # get unique clusters
    unique_labels = sorted(df['label'].unique())

    # assign a random RGB colour for each cluster
    cluster_colors = {
        label: f"rgb({random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)})"
        for label in unique_labels
    }

    fig = go.Figure()

    for label in unique_labels:
        cluster_df = df[df['label'] == label]
        fig.add_trace(go.Scatter(
            x=cluster_df['inclination'],
            y=cluster_df['apogee'],
            mode='markers',
            marker=dict(size=5, opacity=0.6, color=cluster_colors[label]),
            name=f"Cluster {label}",
            text=cluster_df['satNo'],  # hover text
            hovertemplate=(
                "SatNo: %{text}<br>"
                "Inclination: %{x}<br>"
                "Apogee: %{y} km<br>"
                "Cluster: " + str(label) + "<extra></extra>"
            )
        ))

    fig.update_layout(
        title="Satellite clustering",
        xaxis_title="Inclination (degrees)",
        yaxis_title="Apogee (km)",
        template="plotly_white",
        width=900,
        height=700,
        legend_title="Clusters"
    )

    # save to HTML
    output_path = "data/orbit_scatter.html"
    fig.write_html(output_path, include_plotlyjs="cdn")

def main():
    # Set the inclination and apogee ranges for filtering
    inclination_range = (0, 180)  # degrees
    apogee_range = (0, 2000)   # kilometers
    
    tle_df = celestrak_elsets_to_df()
    
    # Filter the DataFrame based on inclination and apogee ranges
    tle_df = tle_df[
        (tle_df['inclination'] >= inclination_range[0]) & (tle_df['inclination'] <= inclination_range[1]) &
        (tle_df['apogee'] >= apogee_range[0]) & (tle_df['apogee'] <= apogee_range[1])
    ].copy()
    
    print(f"# of sats in df: {len(tle_df)}")

    # get the distance matrix and compute clusters
    # distance_matrix, key = get_distance_matrix(tle_df)

    
    # load distance matrix 

    with open("data/distance_matrix.pkl", "rb") as f:
        distance_matrix = pickle.load(f)
    with open("data/idx_satNo_dict.pkl", "rb") as f:
        idx_satNo_dict = pickle.load(f)
    with open("data/satNo_idx_dict.pkl", "rb") as f:
        satNo_idx_dict = pickle.load(f)
        
    key = {'idx_satNo_dict': idx_satNo_dict, 'satNo_idx_dict': satNo_idx_dict}
    
    check_matrix(distance_matrix)

    satNos_in_order = [key['idx_satNo_dict'][i] for i in range(len(key['idx_satNo_dict']))]
    tle_df = tle_df.set_index("satNo").loc[satNos_in_order].reset_index()
    
    labels = compute_clusters_affinity(distance_matrix, 0.95)
    
    # Add cluster information directly to the DataFrame
    tle_df['label'] = labels.astype(int)
    
    plot(tle_df)

if __name__ == '__main__':
    main()