
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
import os
import random
"""
This file is used to process the given elset data into a distance matrix and save
to disk.
"""

import pandas as pd
import numpy as np

from unique_orbits.uct_fitting.orbit_finder.get_optimum_orbit import get_optimum_orbit, get_maximally_separated_orbit

from app import SatelliteClusteringApp

from tle_parser import TLEParser
from configs import ClusterConfig


import os

"""
This file is used to process the given elset data into a distance matrix and save
to disk.
"""

from unique_orbits.uct_fitting.orbit_finder.frechet_orbit_finder import get_optimum_orbit
from unique_orbits.uct_fitting.orbit_finder.void_orbit_finder import (
    get_maximally_separated_orbit,
)



class SyntheticOrbits:
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
        df = get_optimum_orbit(df)
    
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

        plt.title("Frechet Mean Synthetic Orbit in Cluster")
        plt.xlabel("t-SNE component 1")
        plt.ylabel("t-SNE component 2")
        plt.tight_layout()

        out_png = f"data/tsne_{name}.png"
        plt.savefig(out_png)
        plt.close()

        print(f"Saved t-SNE PNG to {out_png}")

    def czml_main(self):

        # 1) Load all clusters with labels, no synthetic orbits
        df_all = self.load_hdbscan_labeled_dataframe()
        if df_all is None or df_all.empty:
            print("No TLE data loaded. Quitting.")
            return

        # 2) Run optimiser on every cluster and print diagnostics
        results_df = self.evaluate_optimizer_all_clusters(df_all.copy(), min_cluster_size=2)

        # 3) Optionally save summary to CSV for later analysis
        os.makedirs("data", exist_ok=True)
        results_df.to_csv("data/frechet_optimizer_summary.csv", index=False)
        print("Saved optimiser summary to data/frechet_optimizer_summary.csv")
        
    def evaluate_optimizer_all_clusters(self, df, min_cluster_size=2):
        """
        Run Fréchet-mean optimisation on every cluster that meets the size threshold.
        Prints per-cluster diagnostics (from get_optimum_orbit) to the console and
        returns a summary DataFrame.
        """
        results = []

        for label, df_cluster in df.groupby("label"):
            if label == -1:
                continue  # skip noise

            N = len(df_cluster)
            if N < min_cluster_size:
                continue

            print(f"\n=== Cluster label {label}, N = {N} ===")
            try:
                diagnostics = get_optimum_orbit(
                    df_cluster.copy(),   # isolate mutations if return_diagnostics=False later
                    return_diagnostics=True
                )
                # diagnostics already contains N, but we can overwrite to be explicit
                diagnostics["label"] = label
                diagnostics["N"] = N
                results.append(diagnostics)

            except Exception as e:
                print(f"Optimizer failed for cluster {label}: {e}")
                results.append({
                    "label": label,
                    "N": N,
                    "error": str(e),
                    "success": False,
                })

        return pd.DataFrame(results)

        
        
    def load_hdbscan_labeled_dataframe(self) -> pd.DataFrame:
        """
        Loader for EXPERIMENT #2 ONLY.
        Loads all clusters, no filtering, no synthetic orbits.
        """

        df = self.tle_parser.df

        df = df[
            (df["inclination"] >= self.cluster_config.inclination_range[0]) &
            (df["inclination"] <= self.cluster_config.inclination_range[1]) &
            (df["apogee"] >= self.cluster_config.apogee_range[0]) &
            (df["apogee"] <= self.cluster_config.apogee_range[1])
        ].copy()

        distance_matrix, key = get_distance_matrix(df)
        df = self.app._reorder_dataframe(df, key)

        with open("data/cluster_results/hdbscan_obj.pkl", "rb") as f:
            hdbscan_obj = pickle.load(f)

        df["label"] = hdbscan_obj.labels.astype(int)
        df["correlated"] = False
        df["dataset"] = "dataset_name"

        return df



    def evaluate_void_over_cluster_sizes(self,
        df,
        min_cluster_size=2
    ):
        """
        Experiment #2:
        For each cluster size, evaluate ONE representative cluster.
        """

        clusters = self.select_one_cluster_per_size(
            df,
            min_cluster_size=min_cluster_size
        )

        cluster_sizes = []
        percentiles = []
        ratios = []

        for N in sorted(clusters.keys()):
            diagnostics = self.evaluate_void_for_cluster(
                clusters[N]
            )

            cluster_sizes.append(N)
            percentiles.append(diagnostics["percentile_vs_cluster"])
            ratios.append(diagnostics["ratio_to_median_spacing"])

        return cluster_sizes, percentiles, ratios



    def plot_void_performance(self, cluster_sizes, percentiles, ratios,
                            out_path="data/void_performance_vs_cluster_size.png"):
        cluster_sizes = np.asarray(cluster_sizes)
        percentiles = np.asarray(percentiles)
        ratios = np.asarray(ratios)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        fig, ax1 = plt.subplots(figsize=(7, 5), dpi=150)

        # Left y-axis: percentile
        color1 = "tab:blue"
        ax1.set_xlabel("Cluster size (N)")
        ax1.set_ylabel("Void NN percentile (%)", color=color1)
        ax1.plot(cluster_sizes, percentiles, marker="o", color=color1,
                label="Percentile vs cluster size")
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_ylim(0, 100)

        # Right y-axis: ratio R*/median
        ax2 = ax1.twinx()
        color2 = "tab:red"
        ax2.set_ylabel(r"$R^\ast$ / median NN spacing", color=color2)
        ax2.plot(cluster_sizes, ratios, marker="s", linestyle="--",
                color=color2, label="Radius ratio vs cluster size")
        ax2.tick_params(axis="y", labelcolor=color2)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved void performance plot to {out_path}")
        
        
    def evaluate_void_all_clusters(self, df, min_cluster_size=2):
        cluster_sizes = []
        percentiles = []
        ratios = []
        labels = []

        for label, df_cluster in df.groupby("label"):
            if label == -1:
                continue
            N = len(df_cluster)
            if N < min_cluster_size:
                continue

            _, diagnostics = get_maximally_separated_orbit(
                df_cluster.copy(),
                return_diagnostics=True
            )

            ratio = diagnostics["ratio_to_median_spacing"]
            if not np.isfinite(ratio):
                # skip degenerate clusters with median NN ~ 0
                continue

            cluster_sizes.append(N)
            percentiles.append(diagnostics["percentile_vs_cluster"])
            ratios.append(ratio)
            labels.append(label)

        return np.array(cluster_sizes), np.array(percentiles), np.array(ratios), np.array(labels)


    def select_one_cluster_per_size(self, df, min_cluster_size=2, random_state=10):
        """
        Returns a dict: {cluster_size: df_cluster}
        Selects one random cluster for each cluster size (among all with that size).
        """
        rng = np.random.default_rng(random_state)

        # group by label, collect sizes
        label_groups = {label: grp.copy()
                        for label, grp in df.groupby("label")
                        if label != -1}  # skip noise

        # map size -> list of labels with that size
        size_to_labels = {}
        for label, grp in label_groups.items():
            N = len(grp)
            if N < min_cluster_size:
                continue
            size_to_labels.setdefault(N, []).append(label)

        clusters = {}
        for N, labels in size_to_labels.items():
            # pick one label at random among those with size N
            chosen_label = rng.choice(labels)
            clusters[N] = label_groups[chosen_label]

        return clusters

    def evaluate_void_for_cluster(self,
        df_cluster
    ):
        """
        Runs void-orbit optimization on a single cluster
        and returns diagnostics only.
        """

        _, diagnostics = get_maximally_separated_orbit(
            df_cluster.copy(),      # critical: isolate mutation
            return_diagnostics=True
        )

        return diagnostics


    def run_frechet_all_clusters(self, min_cluster_size=2):
        df_all = self.load_hdbscan_labeled_dataframe()
        if df_all is None or df_all.empty:
            print("No TLE data loaded.")
            return

        results = []
        for label, df_cluster in df_all.groupby("label"):
            if label == -1:
                continue
            N = len(df_cluster)
            if N < min_cluster_size:
                continue

            print(f"\n=== Fréchet mean for cluster {label} (N={N}) ===")
            try:
                diagnostics = get_optimum_orbit(df_cluster.copy(), return_diagnostics=True)
                diagnostics["label"] = label
                diagnostics["N"] = N
                results.append(diagnostics)
            except Exception as e:
                print(f"Fréchet optimisation failed for label {label}: {e}")
                results.append({"label": label, "N": N, "error": str(e), "success": False})

        results_df = pd.DataFrame(results)
        os.makedirs("data", exist_ok=True)
        results_df.to_csv("data/frechet_optimizer_summary.csv", index=False)
        print("Saved Fréchet optimiser summary to data/frechet_optimizer_summary.csv")
        return results_df

    def run_void_all_clusters(self, min_cluster_size=2):
        df_all = self.load_hdbscan_labeled_dataframe()
        if df_all is None or df_all.empty:
            print("No TLE data loaded.")
            return

        cluster_sizes = []
        percentiles = []
        ratios = []
        labels = []

        for label, df_cluster in df_all.groupby("label"):
            if label == -1:
                continue
            N = len(df_cluster)
            if N < min_cluster_size:
                continue

            print(f"\n=== Void orbit for cluster {label} (N={N}) ===")
            _, diagnostics = get_maximally_separated_orbit(df_cluster.copy(), return_diagnostics=True)
            ratio = diagnostics["ratio_to_median_spacing"]
            if not np.isfinite(ratio):
                continue

            cluster_sizes.append(N)
            percentiles.append(diagnostics["percentile_vs_cluster"])
            ratios.append(ratio)
            labels.append(label)

        cluster_sizes = np.array(cluster_sizes)
        percentiles = np.array(percentiles)
        ratios = np.array(ratios)
        labels = np.array(labels)

        return cluster_sizes, percentiles, ratios, labels

    def run_orbit_generator(self, mode="frechet_single"):
        """
        mode:
          - "frechet_single": one cluster + Frechet orbit + t-SNE/CZML
          - "frechet_all": run Frechet optimiser over all clusters (no plots)
          - "void_all": run void optimiser over all clusters + performance plot
        """
        
        if mode == "frechet_single":
            df = self.load_tle_dataframe_for_file()  # filtered + single cluster + Frechet orbit
            if df is None or df.empty:
                print("No data.")
                return
            self.graph_tsne(df.copy(), name="tsne_frechet_cluster")
            # build_czml(df, ...)
            return

        if mode == "frechet_all":
            self.run_frechet_all_clusters(min_cluster_size=2)
            return

        if mode == "void_all":
            df_all = self.load_hdbscan_labeled_dataframe()
            cs, pct, rat, lbl = self.run_void_all_clusters(min_cluster_size=2)
            self.plot_void_performance(cs, pct, rat)
            return

        print(f"Unknown mode '{mode}'")