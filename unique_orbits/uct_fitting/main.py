
import pickle

import pandas as pd
import numpy as np
from tools.distance_matrix import get_distance_matrix
from unique_orbits.uct_fitting.orbit_finder.get_optimum_orbit import get_optimum_orbit, get_maximally_separated_orbit
from unique_orbits.uct_fitting.data_handling.build_czml import build_czml
from unique_orbits.uct_fitting.data_handling.ionop_czml import ionop_czml
from app import SatelliteClusteringApp
from tools.DMT import VectorizedKeplerianOrbit
from tle_parser import TLEParser
from configs import ClusterConfig, OrbitalConstants, PathConfig
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.graph_objects as go

"""
This file is used to process the given elset data into a distance matrix and save
to disk.
"""

class UCTFitting:
    def __init__(self, cluster_config: ClusterConfig):
        self.cluster_config = cluster_config
        self.tle_parser = TLEParser("Space-Track")
        self.app = SatelliteClusteringApp(cluster_config)
        
    def load_tle_dataframe_for_file(self) -> pd.DataFrame:

        df = self.tle_parser.df
        # filter by inclination and apogee range
        df = df[
            (df["inclination"] >= self.cluster_config.inclination_range[0])
            & (df["inclination"] <= self.cluster_config.inclination_range[1])
            & (df["apogee"] >= self.cluster_config.apogee_range[0])
            & (df["apogee"] <= self.cluster_config.apogee_range[1])
        ].copy()

        # print(
        #     f"Loaded {len(df)} satellites in range - inc: {self.cluster_config.inclination_range}, apogee: {self.cluster_config.apogee_range}"
        # )

        # Get or compute the distance matrix
        distance_matrix, key = get_distance_matrix(df)
        df = self.app._reorder_dataframe(df, key)
        
        # tag dataset for downstream UI filtering
        df['dataset'] = "dataset_name"

        with open("data/cluster_results/hdbscan_obj.pkl", "rb") as f:
            hdbscan_obj = pickle.load(f)

        labels = hdbscan_obj.labels
        
        df['label'] = labels.astype(int)
        df['correlated'] = False
        
        df = self.filter_clusters(df.copy())
        
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        cluster_counts = dict(zip(unique_labels, label_counts))
        # print("\nCluster counts:")
        # print(cluster_counts)

        # Append optimized orbit for this dataset
        # df = get_optimum_orbit(df)
        
        df, diagnostics = get_maximally_separated_orbit(df, return_diagnostics=True)
        print("Void orbit diagnostics:")
        print(f"  NN distance: {diagnostics['r_star']:.6f}")
        print(f"  Percentile: {diagnostics['percentile_vs_cluster']:.1f}%")
        print(f"  Ratio to median: {diagnostics['ratio_to_median_spacing']:.2f}×")
        
        return df

    def filter_clusters(self, df: pd.DataFrame, target_size: int = 15) -> pd.DataFrame:
        labels = df['label'].to_numpy()
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        cluster_counts = dict(zip(unique_labels, label_counts))

        # choose the first non-noise cluster with exactly target_size points
        selected = [
            lbl for lbl, cnt in cluster_counts.items()
            if lbl != -1 and cnt == target_size
        ]
        
        if not selected:
            raise ValueError(f"No cluster with exactly {target_size} points found")

        chosen_label = selected[0]
        return df[df['label'] == chosen_label].copy()

    def test_keplerians(self, df):
        # Select only the row where satNo is '99999'
        optimum_orbit_df = df[df['satNo'] == '99999']
        
        # Get non-optimum orbits and take only 3 random rows
        other_df = df[df['satNo'] != '99999']
        if len(other_df) > 3:
            other_df = other_df.sample(n=3)
        
        if optimum_orbit_df.empty:
            print("No optimum orbit (satNo 99999) found in the dataframe")
            return df
        
        # Combine optimum orbit with the selected rows
        test_rows = pd.concat([optimum_orbit_df, other_df], ignore_index=True)
        
        # Extract Keplerian elements using VectorizedKeplerianOrbit class
        line1_array = np.array(test_rows['line1'].tolist())
        line2_array = np.array(test_rows['line2'].tolist())
        
        # Create orbit object to access Keplerian elements
        orbit = VectorizedKeplerianOrbit(line1_array, line2_array)
        
        # Write test orbits and their Keplerian elements to a file
        with open("data/test_orbits.txt", "w") as f:
            for idx, row in test_rows.iterrows():
                f.write(f"Orbit {idx+1}\n")
                f.write(f"Satellite Number: {row['satNo']}\n")
                f.write(f"Line 1: {row['line1']}\n")
                f.write(f"Line 2: {row['line2']}\n")
                
                # Write Keplerian elements
                f.write("Keplerian Elements:\n")
                f.write(f"  Semi-major axis (a): {orbit.a[idx]:.6f} km\n")
                f.write(f"  Eccentricity (e): {orbit.e[idx]:.6f}\n")
                f.write(f"  Inclination (i): {np.rad2deg(orbit.i[idx]):.6f} deg\n")
                f.write(f"  Argument of Perigee (ω): {np.rad2deg(orbit.omega[idx]):.6f} deg\n")
                f.write(f"  RAAN (Ω): {np.rad2deg(orbit.raan[idx]):.6f} deg\n")
                f.write("-" * 50 + "\n")
        
        return test_rows
    

    def graph_tsne(self, df: pd.DataFrame, name: str = "tsne_avg_synthOrb_cluster"):
        """
        Diagnostic t-SNE plot showing a cluster and its synthetic void orbit.
        """

        df = df.copy()

        # --- identify synthetic orbit ---
        void_mask = df['satNo'] == '99999'
        if void_mask.sum() != 1:
            print("No unique synthetic orbit found for t-SNE plot.")
            return

        real_mask = ~void_mask

        # --- build Keplerian distance matrix ---
        line1 = df['line1'].values
        line2 = df['line2'].values
        orbits = VectorizedKeplerianOrbit(line1, line2)
        D = VectorizedKeplerianOrbit.DistanceMetric(orbits, orbits)

        # --- t-SNE on distance matrix ---
        tsne = TSNE(
            n_components=2,
            metric="precomputed",
            perplexity=min(10, len(df) - 1),
            init="random",
            random_state=42
        )
        
        D = np.asarray(D)

        # kill tiny negative roundoff
        D[D < 0] = 0.0

        # force exact zeros on diagonal
        np.fill_diagonal(D, 0.0)
        X_2d = tsne.fit_transform(D)

        # ================================
        # Plotly (interactive)
        # ================================
        fig = go.Figure()

        # real cluster orbits
        fig.add_trace(go.Scatter(
            x=X_2d[real_mask, 0],
            y=X_2d[real_mask, 1],
            mode="markers",
            name="Cluster orbits",
            marker=dict(
                size=6,
                opacity=0.7,
                color="rgba(100,100,100,0.6)"
            ),
            text=df.loc[real_mask, 'satNo'],
            hovertemplate="SatNo: %{text}<extra></extra>"
        ))

        # synthetic orbit
        fig.add_trace(go.Scatter(
            x=X_2d[void_mask, 0],
            y=X_2d[void_mask, 1],
            mode="markers",
            name="Synthetic void orbit",
            marker=dict(
                symbol="star",
                size=18,
                color="crimson",
                line=dict(width=2, color="black")
            ),
            text=["Synthetic orbit (99999)"],
            hovertemplate="%{text}<extra></extra>"
        ))

        fig.update_layout(
            title="t-SNE diagnostic: cluster vs synthetic void orbit",
            xaxis_title="t-SNE component 1",
            yaxis_title="t-SNE component 2",
            template="plotly_white",
            width=800,
            height=650,
            legend=dict(itemsizing="constant")
        )

        out_html = f"data/tsne_{name}.html"
        fig.write_html(str(out_html), include_plotlyjs="cdn")
        print(f"Saved t-SNE void diagnostic to {out_html}")

        # ================================
        # Matplotlib (paper-ready)
        # ================================
        plt.figure(figsize=(7, 6), dpi=150)

        plt.scatter(
            X_2d[real_mask, 0],
            X_2d[real_mask, 1],
            s=12,
            alpha=0.6
        )

        plt.scatter(
            X_2d[void_mask, 0],
            X_2d[void_mask, 1],
            s=160,
            marker="*",
            edgecolor="black",
            linewidth=1.2,
            zorder=5
        )

        plt.title("Synthetic Orbital Candidate in Cluster")
        plt.xlabel("t-SNE component 1")
        plt.ylabel("t-SNE component 2")
        plt.tight_layout()

        out_png = f"data/tsne_{name}.png"
        plt.savefig(out_png)
        plt.close()

        print(f"Saved t-SNE PNG to {out_png}")

    def czml_main(self):

        df = self.load_tle_dataframe_for_file()
        self.graph_tsne(df.copy())
        
        if df is None or df.empty:
            print("No TLE data loaded. Quitting.")
            return
        
        try:
            self.test_keplerians(df.copy())
        except Exception as e:
            print(f"Test keplerians failed: {e}")
        
