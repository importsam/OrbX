
import pickle
import time
import pandas as pd
import numpy as np
from tools.distance_matrix import get_distance_matrix
from synthetic_orbits.orbit_finder.get_optimum_orbit import get_optimum_orbit, get_maximally_separated_orbit
from synthetic_orbits.data_handling.build_czml import build_czml
from synthetic_orbits.data_handling.ionop_czml import ionop_czml
from app import SatelliteClusteringApp
from tools.DMT import VectorizedKeplerianOrbit
from tle_parser import TLEParser
from configs import ClusterConfig
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import pandas as pd
import numpy as np

from synthetic_orbits.orbit_finder.frechet_orbit_finder import get_optimum_orbit
from synthetic_orbits.orbit_finder.max_separation_orbit_finder import get_maximally_separated_orbit

from app import SatelliteClusteringApp

from tle_parser import TLEParser
from configs import ClusterConfig

import os

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
    

    def graph_tsne(self, df: pd.DataFrame, name: str = "tsne_synthOrb_cluster", mode: str = "max_separation"):
        """
        Diagnostic t-SNE plot showing a cluster and its synthetic orbit
        (either Fréchet mean or max_separation, depending on `mode`).
        """

        if mode == "frechet":
            orbit_label = "Fréchet mean orbit"
            title = "Fréchet Mean Synthetic Orbit in Cluster"
            plotly_title = "t-SNE diagnostic: cluster vs Fréchet mean orbit"
        else:
            orbit_label = "Max NN orbit"
            title = "Maximally Separated Synthetic Orbit in Cluster"
            plotly_title = "t-SNE diagnostic: cluster vs synthetic max_separation orbit"

        df = df.copy()

        # --- identify synthetic orbit ---
        max_separation_mask = df["satNo"] == "99999"
        if max_separation_mask.sum() != 1:
            print("No unique synthetic orbit found for t-SNE plot.")
            return

        real_mask = ~max_separation_mask

        # --- build Keplerian distance matrix ---
        line1 = df["line1"].values
        line2 = df["line2"].values
        orbits = VectorizedKeplerianOrbit(line1, line2)
        D = VectorizedKeplerianOrbit.DistanceMetric(orbits, orbits)

        # --- t-SNE on distance matrix ---
        tsne = TSNE(
            n_components=2,
            metric="precomputed",
            perplexity=min(10, len(df) - 1),
            init="random",
            random_state=42,
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

        # synthetic orbit
        fig.add_trace(
            go.Scatter(
                x=X_2d[max_separation_mask, 0],
                y=X_2d[max_separation_mask, 1],
                mode="markers",
                name=orbit_label,
                marker=dict(
                    symbol="star",
                    size=14,
                    color="crimson",
                    line=dict(width=2, color="black"),
                ),
                text=[f"Synthetic orbit (99999) - {orbit_label}"],
                hovertemplate="%{text}<extra></extra>",
            )
        )

        # real cluster orbits
        fig.add_trace(
            go.Scatter(
                x=X_2d[real_mask, 0],
                y=X_2d[real_mask, 1],
                mode="markers",
                name="Cluster orbits",
                marker=dict(
                    size=6,
                    opacity=0.7,
                    color="rgba(100,100,100,0.6)",
                ),
                text=df.loc[real_mask, "satNo"],
                hovertemplate="SatNo: %{text}<extra></extra>",
            )
        )

        fig.update_layout(
            title=plotly_title,
            xaxis_title="t-SNE component 1",
            yaxis_title="t-SNE component 2",
            template="plotly_white",
            width=800,
            height=650,
            legend=dict(itemsizing="constant"),
        )

        out_html = f"data/{name}.html"
        fig.write_html(str(out_html), include_plotlyjs="cdn")
        print(f"Saved t-SNE diagnostic to {out_html}")

        # ================================
        # Matplotlib (static PNG)
        # ================================
        plt.figure(figsize=(7, 6), dpi=150)

        # real/input orbits (background)
        real_handle = plt.scatter(
            X_2d[real_mask, 0],
            X_2d[real_mask, 1],
            s=12,
            alpha=0.6,
            color="tab:blue",
            label="Input orbits",
        )

        # synthetic orbit (foreground star)
        max_separation_handle = plt.scatter(
            X_2d[max_separation_mask, 0],
            X_2d[max_separation_mask, 1],
            s=130,
            marker="*",
            edgecolor="black",
            linewidth=1.2,
            color="crimson",
            zorder=5,
            label=orbit_label,
        )

        plt.title(title)
        plt.xlabel("t-SNE component 1")
        plt.ylabel("t-SNE component 2")

        # Explicit legend order: synthetic first, then real orbits
        plt.legend(
            [max_separation_handle, real_handle],
            [orbit_label, "Input orbits"],
            loc="upper right",
        )

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



    def evaluate_max_separation_over_cluster_sizes(self,
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
            diagnostics = self.evaluate_max_separation_for_cluster(
                clusters[N]
            )

            cluster_sizes.append(N)
            percentiles.append(diagnostics["percentile_vs_cluster"])
            ratios.append(diagnostics["ratio_to_median_spacing"])

        return cluster_sizes, percentiles, ratios



    def plot_max_separation_performance(self, cluster_sizes, percentiles, ratios,
                            out_path="data/max_separation_performance_vs_cluster_size.png"):
        cluster_sizes = np.asarray(cluster_sizes)
        percentiles = np.asarray(percentiles)
        ratios = np.asarray(ratios)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        fig, ax1 = plt.subplots(figsize=(7, 5), dpi=150)

        # Left y-axis: percentile
        color1 = "tab:blue"
        ax1.set_xlabel("Cluster size (N)")
        ax1.set_ylabel("max_separation NN percentile (%)", color=color1)
        ax1.plot(cluster_sizes, percentiles, marker="o", color=color1,
                label="Percentile vs cluster size")
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_ylim(0, 100)

        # Right y-axis: ratio R*/median
        ax2 = ax1.twinx()
        color2 = "tab:red"
        ax2.set_ylabel(r"$r(o_{max}) / median NN spacing", color=color2)
        ax2.plot(cluster_sizes, ratios, marker="s", linestyle="--",
                color=color2, label="Radius ratio vs cluster size")
        ax2.tick_params(axis="y", labelcolor=color2)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved max_separation performance plot to {out_path}")
        
        
    def evaluate_max_separation_all_clusters(self, df, min_cluster_size=2, random_state=42):
        """
        For each cluster size N (>= min_cluster_size), randomly pick ONE cluster
        of that size, run max_separation optimisation on it, and return arrays of:
        cluster_sizes, percentiles, ratios, labels
        """
        # 1) select one cluster per size
        clusters = self.select_one_cluster_per_size(
            df,
            min_cluster_size=min_cluster_size,
            random_state=random_state,
        )

        cluster_sizes = []
        percentiles = []
        ratios = []
        labels = []

        # 2) run max_separation optimiser on that single representative cluster for each size
        for N in sorted(clusters.keys()):
            df_cluster = clusters[N]

            _, diagnostics = get_maximally_separated_orbit(
                df_cluster.copy(),
                return_diagnostics=True,
            )

            ratio = diagnostics["ratio_to_median_spacing"]
            if not np.isfinite(ratio):
                # skip degenerate clusters with median NN ~ 0
                continue

            cluster_sizes.append(N)
            percentiles.append(diagnostics["percentile_vs_cluster"])
            ratios.append(ratio)

            # record which label we used for this N (use the cluster's label)
            label = int(df_cluster["label"].iloc[0])
            labels.append(label)

        return (
            np.array(cluster_sizes),
            np.array(percentiles),
            np.array(ratios),
            np.array(labels),
        )



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

    def evaluate_max_separation_for_cluster(self,
        df_cluster
    ):
        """
        Runs max_separation-orbit optimization on a single cluster
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
        synthetic_rows = []

        for label, df_cluster in df_all.groupby("label"):
            if label == -1:
                continue
            N = len(df_cluster)
            if N < min_cluster_size:
                continue

            print(f"\n=== Fréchet mean for cluster {label} (N={N}) ===")
            try:
                # 1) diagnostics-only call
                diagnostics = get_optimum_orbit(df_cluster.copy(), return_diagnostics=True)
                diagnostics["label"] = label
                diagnostics["N"] = N
                results.append(diagnostics)

                # 2) mutation call that appends synthetic orbit to this cluster
                df_with_opt = get_optimum_orbit(df_cluster.copy(), return_diagnostics=False)
                optimum_row = df_with_opt.iloc[-1]  # last row is the synthetic orbit

                # keep same schema as df_all plus label
                synthetic_rows.append(
                    {
                        "satNo": optimum_row["satNo"],
                        "name": optimum_row["name"],
                        "line1": optimum_row["line1"],
                        "line2": optimum_row["line2"],
                        "inclination": optimum_row.get("inclination"),
                        "apogee": optimum_row.get("apogee"),
                        "raan": optimum_row.get("raan"),
                        "argument_of_perigee": optimum_row.get("argument_of_perigee"),
                        "eccentricity": optimum_row.get("eccentricity"),
                        "mean_motion": optimum_row.get("mean_motion"),
                        "label": label,
                        "correlated": optimum_row.get("correlated", True),
                        "dataset": optimum_row.get("dataset", "frechet_synthetic"),
                    }
                )
            except Exception as e:
                print(f"Fréchet optimisation failed for label {label}: {e}")
                results.append(
                    {"label": label, "N": N, "error": str(e), "success": False}
                )

        # --- write summary CSV as before ---
        results_df = pd.DataFrame(results)
        os.makedirs("data", exist_ok=True)
        results_df.to_csv("data/frechet_optimizer_summary.csv", index=False)
        print("Saved Fréchet optimiser summary to data/frechet_optimizer_summary.csv")

        # --- build df_all_with_synthetics ---
        if synthetic_rows:
            synthetics_df = pd.DataFrame(synthetic_rows)
            # ensure same column order as df_all where possible
            cols = list(df_all.columns)
            extra_cols = [c for c in synthetics_df.columns if c not in cols]
            synthetics_df = synthetics_df[cols + extra_cols]
            df_all_with_synth = pd.concat([df_all, synthetics_df], ignore_index=True)
        else:
            df_all_with_synth = df_all.copy()

        return df_all_with_synth

    def run_max_separation_all_clusters(self, min_cluster_size=2):
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

            print(f"\n=== max_separation orbit for cluster {label} (N={N}) ===")
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

    def load_single_cluster_with_max_separation(
        self,
        target_size: int = 16,
        n_samples: int = 5000,
    ) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
        """
        1) Load TLEs in the configured inc/apogee range.
        2) Attach HDBSCAN labels.
        3) Select one cluster with exactly target_size points.
        4) Run max-min (max_separation) optimisation on that cluster, append satNo=99999,
        and return the augmented df plus diagnostics.
        """
        # base: same as load_tle_dataframe_for_file, but without Frechet
        df = self.tle_parser.df

        df = df[
            (df["inclination"] >= self.cluster_config.inclination_range[0])
            & (df["inclination"] <= self.cluster_config.inclination_range[1])
            & (df["apogee"] >= self.cluster_config.apogee_range[0])
            & (df["apogee"] <= self.cluster_config.apogee_range[1])
        ].copy()

        # build distance matrix and reorder to match the matrix
        distance_matrix, key = get_distance_matrix(df)
        df = self.app._reorder_dataframe(df, key)

        df["dataset"] = "dataset_name"

        with open("data/cluster_results/hdbscan_obj.pkl", "rb") as f:
            hdbscan_obj = pickle.load(f)

        labels = hdbscan_obj.labels
        df["label"] = labels.astype(int)
        df["correlated"] = False

        # filter to one cluster with exactly target_size points
        df_cluster = self.filter_clusters(df.copy(), target_size=target_size)

        # run max_separation optimiser on that one cluster
        df_with_max_separation, diagnostics = get_maximally_separated_orbit(
            df_cluster.copy(),
            n_samples=n_samples,
            return_diagnostics=True,
        )

        return df_with_max_separation, diagnostics

    def filter_clusters(self, df: pd.DataFrame, target_size: int = 16) -> pd.DataFrame:
        labels = df["label"].to_numpy()
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        cluster_counts = dict(zip(unique_labels, label_counts))

        selected = [
            lbl for lbl, cnt in cluster_counts.items()
            if lbl != -1 and cnt == target_size
        ]

        if not selected:
            raise ValueError(f"No cluster with exactly {target_size} points found")

        chosen_label = selected[0]
        return df[df["label"] == chosen_label].copy()

    def run_max_separation_single(
        self,
        target_size: int = 16,
        n_samples: int = 5000,
    ):
        df_with_max_separation, diagnostics = self.load_single_cluster_with_max_separation(
            target_size=target_size,
            n_samples=n_samples,
        )

        if df_with_max_separation is None or df_with_max_separation.empty:
            print("No data / no suitable cluster for max_separation_single.")
            return

        print("max_separation diagnostics:", diagnostics)
        self.graph_tsne(df_with_max_separation.copy(), name=f"max_separation_cluster_N{target_size}")


    def run_max_separation_all_size_15(
        self,
        n_samples: int = 5000,
        out_dir: str = "data/max_separation_single_clusters",
    ):
        """
        For every HDBSCAN cluster with exactly 15 members:
        - run max_separation optimisation (maximally separated orbit),
        - append the max_separation orbit (satNo=99999) to that cluster,
        - run t-SNE plotting,
        - save the satellite names for that cluster to a text file.
        """
        os.makedirs(out_dir, exist_ok=True)

        # Load all labelled TLEs in the configured inc/apogee range
        df_all = self.load_hdbscan_labeled_dataframe()
        if df_all is None or df_all.empty:
            print("No TLE data loaded.")
            return

        # count cluster sizes
        labels = df_all["label"].to_numpy()
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        cluster_counts = dict(zip(unique_labels, label_counts))

        # select all non-noise clusters with exactly 15 members
        target_size = 15
        selected_labels = [
            lbl for lbl, cnt in cluster_counts.items() if lbl != -1 and cnt == target_size
        ]

        if not selected_labels:
            print(f"No clusters found with exactly {target_size} points.")
            return

        print(f"Found {len(selected_labels)} clusters with N={target_size}.")

        for lbl in selected_labels:
            df_cluster = df_all[df_all["label"] == lbl].copy()
            print(f"\n=== Processing cluster label {lbl} (N={len(df_cluster)}) ===")

            # run max_separation optimiser for this cluster
            df_with_max_separation, diagnostics = get_maximally_separated_orbit(
                df_cluster.copy(),
                n_samples=n_samples,
                return_diagnostics=True,
            )

            # t-SNE plot (HTML + PNG) for this cluster
            tsne_name = f"max_separation_cluster_label{lbl}_N{target_size}"
            self.graph_tsne(df_with_max_separation.copy(), name=tsne_name, mode="max_separation")

            # write satellite names (including max_separation orbit) to an identifying file
            names_path = os.path.join(
                out_dir, f"cluster_label{lbl}_N{target_size}_names.txt"
            )
            with open(names_path, "w") as f:
                f.write(f"Cluster label: {lbl}\n")
                f.write(f"Cluster size (without max_separation): {target_size}\n")
                f.write(f"Total rows in file (including max_separation): {len(df_with_max_separation)}\n\n")
                for idx, row in df_with_max_separation.iterrows():
                    f.write(
                        f"{idx:04d}  satNo={row.get('satNo', ''):<8}  "
                        f"name={row.get('name', '')}\n"
                    )

            print(f"Saved satellite names for cluster {lbl} to {names_path}")
            print("max_separation diagnostics:", diagnostics)


    def run_cesium_generator(self):
        """
        Build a dataframe that, for the top 10 largest clusters (by size),
        includes both the Fréchet-mean synthetic orbit and the maximally
        separated synthetic orbit appended to each cluster.

        Returns
        -------
        pd.DataFrame
            DataFrame containing only the top-10 clusters, each augmented with
            their max-separation and Fréchet synthetic orbits.
        """
        # 1) Load all labelled TLEs in the configured inc/apogee range
        df_clusters = self.load_hdbscan_labeled_dataframe()
        if df_clusters is None or df_clusters.empty:
            print("No TLE data loaded.")
            return

        # 2) Keep only non-noise, then the top-10 labels by cluster size
        df_clusters = df_clusters.copy()
        non_noise = df_clusters[df_clusters["label"] != -1]
        if non_noise.empty:
            print("No non-noise clusters found.")
            return

        cluster_sizes = non_noise["label"].value_counts()
        top_labels = cluster_sizes.head(10).index
        df_clusters = df_clusters[df_clusters["label"].isin(top_labels)].copy()

        if df_clusters.empty:
            print("No clusters left after filtering to top 10.")
            return

        print(f"Using top {len(top_labels)} clusters: {list(top_labels)}")

        augmented_clusters = []

        # 3) For each of these clusters, append max_separation and Fréchet orbits
        for label, df_cluster in df_clusters.groupby("label"):
            N = len(df_cluster)
            print(f"\n=== Processing cluster {label} (N={N}) ===")

            # --- Maximally separated synthetic orbit ---
            try:
                df_with_max_sep, diag_max = get_maximally_separated_orbit(
                    df_cluster.copy(),
                    return_diagnostics=True,
                )
                # Stamp label onto any newly appended synthetic rows
                # synthetic_mask = (df_with_max_sep["satNo"] == "99999") & \
                #                 (~df_cluster.index.isin(df_with_max_sep.index) | 
                #                 df_with_max_sep["label"].isna())
                # df_with_max_sep.loc[df_with_max_sep["satNo"] == "99999", "label"] = label

            except Exception as e:
                print(f"  max_separation failed for label {label}: {e}")
                df_with_max_sep = df_cluster.copy()

            # --- Fréchet mean synthetic orbit ---
            try:
                df_with_frechet = get_optimum_orbit(
                    df_with_max_sep.copy(),
                    return_diagnostics=False,
                )
                # Stamp label onto any newly appended synthetic rows
                # df_with_frechet.loc[df_with_frechet["satNo"] == "99999", "label"] = label

            except Exception as e:
                print(f"  Fréchet optimisation failed for label {label}: {e}")
                df_with_frechet = df_with_max_sep.copy()

            augmented_clusters.append(df_with_frechet)

        # 4) Concatenate all augmented clusters into a single dataframe
        df_augmented = pd.concat(augmented_clusters, ignore_index=True)
        
        # print columns:
        print("Columns in augmented dataframe:", df_augmented.columns.tolist())
        
        # print first row 
        print("First row of augmented dataframe:")
        print(df_augmented.iloc[0])

        print(
            f"Built augmented dataframe with {len(df_augmented)} rows "
            f"from {len(top_labels)} clusters (original + max_separation + Fréchet)."
        )
        
        # stick tape solution. Need to get the kep elements for the synthetic orbits
        ######################
        
        # for each synthetic orbit
        
        for idx in df_augmented.index:
            if df_augmented.loc[idx, "satNo"] == "99999":
                print(f"Parsing synthetic orbit at index {idx} for Keplerian elements.")
                
                sat_obj = self.tle_parser._parse_tle_group(
                    "Synthetic Orbit",
                    df_augmented.iloc[idx]['line1'],
                    df_augmented.iloc[idx]['line2']
                )
                
                df_augmented.loc[idx, "inclination"] = sat_obj.inclination
                df_augmented.loc[idx, "apogee"] = sat_obj.apogee
                df_augmented.loc[idx, "raan"] = sat_obj.raan
                df_augmented.loc[idx, "argument_of_perigee"] = sat_obj.argument_of_perigee
                df_augmented.loc[idx, "eccentricity"] = sat_obj.eccentricity
                df_augmented.loc[idx, "mean_motion"] = sat_obj.mean_motion     
                df_augmented.loc[idx, "label"] = "N/A"
                
        ######################
        
        build_czml(df_augmented)
        # wait 2 seconds for it to finish writing the file before we try to read it again
        time.sleep(2)
        ionop_czml()
        # return df_augmented
        
        
    # THIS IS THE MAIN FUNCTION!!!!!!

    def run_orbit_generator(self, mode="max_separation_single"):
        """
        mode:
        - "frechet_single": one cluster + Frechet orbit + t-SNE/CZML
        - "frechet_all": run Frechet optimiser over all clusters (no plots)
        - "max_separation_single": one cluster + maximally separated orbit + t-SNE
        - "max_separation_all": run max_separation optimiser over all clusters + performance plot
        """

        if mode == "frechet_single":
            df = self.load_tle_dataframe_for_file()
            if df is None or df.empty:
                print("No data.")
                return
            self.graph_tsne(df.copy(), name="tsne_frechet_cluster")
            print(df.head(10))
            return

        if mode == "frechet_all":
            df = self.run_frechet_all_clusters(min_cluster_size=2)
            df.to_pickle("data/frechet_all_synth.pkl")
            print("saved frechet all as pkl")
            return

        if mode == "max_separation_single":
            # choose whatever defaults you like, or pass them in from the caller
            self.run_max_separation_single(target_size=15, n_samples=5000)
            return

        if mode == "max_separation_all":
            df_all = self.load_hdbscan_labeled_dataframe()
            cs, pct, rat, lbl = self.run_max_separation_all_clusters(min_cluster_size=2)
            self.plot_max_separation_performance(cs, pct, rat)
            return
        
        

        print(f"Unknown mode '{mode}'")
