import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from tools.density_estimation import DensityEstimator
import numpy as np
import pandas as pd
from pathlib import Path
from clustering_algs.cluster_wrapper import ClusterWrapper
import pickle 
from tools.distance_matrix import get_distance_matrix
from tle_parser import TLEParser
from tools.density_estimation import DensityEstimator
from tools.DMT import VectorizedKeplerianOrbit

class Analysis:
    def __init__(self, output_dir: str = "data/analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.density_estimator = DensityEstimator()
        self.cluster_wrapper = ClusterWrapper()
        self.tle_parser = TLEParser("Space-Track")
        
    def cluster_size_summary(self, labels: np.ndarray) -> dict:
        """
        Compute size structure + coverage/noise stats for a single clustering.

        Returns a dict with:
            - n_total
            - n_clusters
            - n_noise
            - frac_noise
            - frac_clustered
            - sizes (list of cluster sizes)
            - size_stats (min / q1 / median / q3 / max)
            - n_tiny_clusters (size in [2, 3])
        """
        labels = np.asarray(labels)
        n_total = labels.shape[0]

        # Noise is label -1
        noise_mask = labels == -1
        n_noise = int(noise_mask.sum())
        frac_noise = n_noise / n_total if n_total > 0 else np.nan
        frac_clustered = 1.0 - frac_noise

        # Cluster labels (excluding noise)
        cluster_labels = labels[~noise_mask]
        unique_clusters = np.unique(cluster_labels)
        n_clusters = unique_clusters.size

        # Sizes per cluster
        counts = Counter(cluster_labels)
        sizes = np.array([counts[c] for c in unique_clusters], dtype=int)

        if sizes.size > 0:
            size_stats = {
                "min": int(np.min(sizes)),
                "q1": float(np.percentile(sizes, 25)),
                "median": float(np.median(sizes)),
                "q3": float(np.percentile(sizes, 75)),
                "max": int(np.max(sizes)),
            }
        else:
            size_stats = {
                "min": 0,
                "q1": 0.0,
                "median": 0.0,
                "q3": 0.0,
                "max": 0,
            }

        # Tiny clusters (e.g. size 2–3)
        tiny_mask = (sizes >= 2) & (sizes <= 3)
        n_tiny_clusters = int(tiny_mask.sum())

        return {
            "n_total": n_total,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "frac_noise": frac_noise,
            "frac_clustered": frac_clustered,
            "sizes": sizes,
            "size_stats": size_stats,
            "n_tiny_clusters": n_tiny_clusters,
        }

    def plot_cluster_size_distributions(
        self,
        sizes_dict: dict,
        algorithm_order=None,
        log_x: bool = True,
        save_name: str = "cluster_size_distributions.png",
    ):
        """
        Make side-by-side histogram + boxplot of cluster sizes for multiple algorithms.

        sizes_dict: {algorithm_name: 1D array-like of cluster sizes}
        algorithm_order: list to control plotting order; if None, dict order is used.
        """
        if algorithm_order is None:
            algorithm_order = list(sizes_dict.keys())

        # Filter out algorithms with no clusters
        algorithm_order = [
            name for name in algorithm_order
            if len(sizes_dict.get(name, [])) > 0
        ]
        if not algorithm_order:
            print("No non-empty clusterings to plot.")
            return

        # Use a modern color palette
        colors = plt.cm.Set2(np.linspace(0, 1, len(algorithm_order)))
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(10, 9),
            constrained_layout=True,
        )
        
        fig.patch.set_facecolor('white')

        # --- Histogram of cluster sizes per algorithm ---
        ax_hist = axes[0]
        for i, name in enumerate(algorithm_order):
            sizes = np.asarray(sizes_dict[name])
            ax_hist.hist(
                sizes,
                bins=30,
                alpha=0.65,
                label=name,
                color=colors[i],
                edgecolor='white',
                linewidth=0.5,
            )

        if log_x:
            ax_hist.set_xscale("log")

        ax_hist.set_xlabel("Cluster Size (log)", fontsize=11, fontweight='semibold')
        ax_hist.set_ylabel("Number of Clusters", fontsize=11, fontweight='semibold')
        ax_hist.legend(title="Algorithm", framealpha=0.95, shadow=True, fontsize=10)
        ax_hist.set_title("Cluster Size Distribution", fontsize=13, fontweight='bold', pad=15)
        ax_hist.grid(True, alpha=0.3, linestyle='--')
        ax_hist.spines['top'].set_visible(False)
        ax_hist.spines['right'].set_visible(False)

        # --- Boxplot of cluster sizes per algorithm ---
        ax_box = axes[1]
        data = [np.asarray(sizes_dict[name]) for name in algorithm_order]
        
        bp = ax_box.boxplot(
            data,
            labels=algorithm_order,
            showfliers=True,
            patch_artist=True,
            notch=True,
            widths=0.6,
            boxprops=dict(linewidth=1.5, alpha=0.8),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            medianprops=dict(linewidth=2, color='darkred'),
            flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5),
        )
        
        # Color each box
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        if log_x:
            ax_box.set_yscale("log")
            ax_box.set_ylabel("Cluster Size (log)", fontsize=11, fontweight='semibold')
        else:
            ax_box.set_ylabel("Cluster Size", fontsize=11, fontweight='semibold')

        ax_box.set_xlabel("Algorithm", fontsize=11, fontweight='semibold')
        ax_box.set_title("Cluster Size Distribution (Median, IQR, Range)", fontsize=13, fontweight='bold', pad=15)
        ax_box.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax_box.spines['top'].set_visible(False)
        ax_box.spines['right'].set_visible(False)
        ax_box.tick_params(axis='x', rotation=15)

        # Save
        out_path = self.output_dir / save_name
        fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved cluster size distributions to {out_path}")
            
    def cluster_mean_densities(
            self,
            labels: np.ndarray,
            densities: np.ndarray,   # <— change: pass densities directly
        ) -> dict:
        """
        Compute mean density per cluster (excluding noise).
        """

        labels = np.asarray(labels)
        densities = np.asarray(densities)

        mask = labels != -1
        labels_n = labels[mask]
        dens_n = densities[mask]

        if labels_n.size == 0:
            return {
                "cluster_ids": np.array([], dtype=int),
                "cluster_sizes": np.array([], dtype=int),
                "cluster_mean_density": np.array([], dtype=float),
            }

        unique_clusters = np.unique(labels_n)
        cluster_ids = []
        cluster_sizes = []
        cluster_mean_density = []

        for cid in unique_clusters:
            cid_mask = labels_n == cid
            cluster_ids.append(cid)
            cluster_sizes.append(cid_mask.sum())
            cluster_mean_density.append(dens_n[cid_mask].mean())

        return {
            "cluster_ids": np.asarray(cluster_ids),
            "cluster_sizes": np.asarray(cluster_sizes),
            "cluster_mean_density": np.asarray(cluster_mean_density),
        }
        
    def plot_hdbscan_cluster_sizes(
        self,
        hdbscan_sizes,
        log_x: bool = True,
        save_name: str = "hdbscan_cluster_sizes.png",
        grouped: bool = False,
    ):
        """
        Plot HDBSCAN cluster sizes.

        If grouped=True, color bars by size class:
            Micro: 2–4
            Minor: 5–19
            Major: 20–99
            Mega:  100+
        and show a legend for these classes.
        """
        sizes = np.asarray(hdbscan_sizes)
        if sizes.size == 0:
            print("No HDBSCAN clusters to plot.")
            return

        plt.style.use("seaborn-v0_8-darkgrid")
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        fig.patch.set_facecolor("white")

        # Base histogram
        counts, bin_edges, patches = ax.hist(
            sizes,
            bins=30,
            alpha=0.9,
            color="#4C72B0",
            edgecolor="white",
            linewidth=0.5,
        )  # [web:37][web:40]

        if grouped:
            class_ranges = {
                "Micro (2–4)":  (2, 5),
                "Minor (5–19)": (5, 20),
                "Major (20–49)": (20, 50),
                "Mega (50+)":   (50, np.inf),
            }

            class_colors = {
                "Micro (2–4)":  "#66c2a5",
                "Minor (5–19)": "#fc8d62",
                "Major (20–49)": "#8da0cb",
                "Mega (50+)":   "#e78ac3",
            }

            # --- print counts per class ---
            sizes_arr = np.asarray(sizes)
            print("HDBSCAN cluster counts by size class:")
            for label, (lo, hi) in class_ranges.items():
                mask = (sizes_arr >= lo) & (sizes_arr < hi)
                print(f"  {label}: {mask.sum()} clusters")

            # --- color bins by class ---
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            for center, patch in zip(bin_centers, patches):
                # default (if outside all ranges)
                patch.set_facecolor("#cccccc")
                for label, (lo, hi) in class_ranges.items():
                    if lo <= center < hi:
                        patch.set_facecolor(class_colors[label])
                        patch.set_label(label)
                        break

            handles, labels = ax.get_legend_handles_labels()
            seen = set()
            uniq_handles, uniq_labels = [], []
            for h, lab in zip(handles, labels):
                if lab not in seen:
                    seen.add(lab)
                    uniq_handles.append(h)
                    uniq_labels.append(lab)
            ax.legend(
                uniq_handles,
                uniq_labels,
                title="Cluster Class",
                loc="upper right",
                framealpha=0.95,
            )


            # For each bin, decide which class it belongs to by its bin center
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            for center, patch in zip(bin_centers, patches):
                # Default color if no class matches (e.g. size < 2)
                patch.set_facecolor("#cccccc")
                for label, (lo, hi) in class_ranges.items():
                    if lo <= center <= hi:
                        patch.set_facecolor(class_colors[label])
                        patch.set_label(label)  # will be deduplicated by legend
                        break

            # Legend in top-right corner
            handles, labels = ax.get_legend_handles_labels()
            # Deduplicate labels while preserving order
            seen = set()
            uniq_handles, uniq_labels = [], []
            for h, lab in zip(handles, labels):
                if lab not in seen:
                    seen.add(lab)
                    uniq_handles.append(h)
                    uniq_labels.append(lab)
            ax.legend(
                uniq_handles,
                uniq_labels,
                title="Cluster Class",
                loc="upper right",
                framealpha=0.95,
            )  # [web:56][web:59][web:62]

        if log_x:
            ax.set_xscale("log")

        ax.set_xlabel("Cluster Size (log)" if log_x else "Cluster Size",
                      fontsize=11, fontweight="semibold")
        ax.set_ylabel("Number of Clusters", fontsize=11, fontweight="semibold")
        ax.set_title("HDBSCAN Cluster Size Distribution",
                    fontsize=13, fontweight="bold", pad=15)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        out_path = self.output_dir / save_name
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved HDBSCAN cluster size histogram to {out_path}")

                    
    def get_variance(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        df = df.copy()

        if "label" not in df.columns:
            raise ValueError("DataFrame must contain a 'label' column.")

        distance_matrix, key = get_distance_matrix(df)
        satno_to_idx_raw = key["satNo_idx_dict"]
        satno_to_idx = {str(k): int(v) for k, v in satno_to_idx_raw.items()}


        variances = {}

        for label, df_cluster in df.groupby("label"):
            if label == -1:
                continue

            syn_mask = (df_cluster["name"] == "Optimized") | (df_cluster["satNo"] == "99999")
            if not syn_mask.any():
                print(f"No synthetic orbit found for cluster {label}; skipping variance.")
                continue

            syn_row = df_cluster.loc[syn_mask].iloc[0]
            syn_satno = str(syn_row["satNo"])
            if syn_satno not in satno_to_idx:
                print(f"Synthetic satNo {syn_satno} not in key; skipping cluster {label}.")
                continue
            syn_idx = satno_to_idx[syn_satno]

            members = df_cluster.loc[~syn_mask]
            if members.empty:
                print(f"No real members in cluster {label} after excluding synthetic; skipping.")
                continue

            member_indices = []
            for _, row in members.iterrows():
                s = str(row["satNo"])
                idx = satno_to_idx.get(s)
                if idx is not None:
                    member_indices.append(idx)
                else:
                    print(f"satNo {s} not found in key; skipping this member in cluster {label}.")

            if not member_indices:
                print(f"No valid members with indices for cluster {label}; skipping.")
                continue

            member_indices = np.asarray(member_indices, dtype=int)

            dists = distance_matrix[member_indices, syn_idx]
            dists_sq = dists**2

            var_label = float(np.mean(dists_sq))
            variances[label] = var_label
            print(
                f"Cluster {label}: variance (mean squared distance to synthetic) = {var_label:.6e}"
            )

        df["cluster_variance"] = np.nan
        for label, var in variances.items():
            df.loc[df["label"] == label, "cluster_variance"] = var

        return df, variances

            
        
    def plot_size_vs_variance(
        self,
        sizes: np.ndarray,
        variances: np.ndarray,
        save_name: str = "cluster_size_vs_variance.png",
    ):
        plt.style.use("seaborn-v0_8-darkgrid")
        fig, ax = plt.subplots(figsize=(8, 6))

        sizes = np.asarray(sizes)
        variances = np.asarray(variances)

        ax.scatter(
            sizes,
            variances,
            alpha=0.7,
            s=25,
            color="C0",
            edgecolor="none",
        )
        
        ax.set_xlabel("Cluster size")
        ax.set_ylabel(r"Within-cluster variance $\sigma_c^2$")
        ax.set_title(r"Within-cluster variance vs. cluster size")

        out_path = self.output_dir / save_name
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved size-variance plot to {out_path}")


    def plot_variance_from_existing_frechet(self, pkl_path="data/frechet_all_synth.pkl"):
        df = pd.read_pickle(pkl_path)
        df_var, variances = self.get_variance(df)

        cluster_sizes = []
        cluster_vars = []

        for label, df_cluster in df_var.groupby("label"):
            if label == -1:
                continue

            syn_mask = (df_cluster["name"] == "Optimized") | (df_cluster["satNo"] == "99999")
            members = df_cluster.loc[~syn_mask]

            if members.empty:
                continue

            size = len(members)
            var = members["cluster_variance"].iloc[0]
            if np.isnan(var):
                continue

            cluster_sizes.append(size)
            cluster_vars.append(var)

        cluster_sizes = np.asarray(cluster_sizes, dtype=float)
        cluster_vars = np.asarray(cluster_vars, dtype=float)

        var_items = sorted(variances.items(), key=lambda x: x[0])
        var_df = pd.DataFrame(var_items, columns=["label", "within_cluster_variance"])
        out_csv = self.output_dir / "cluster_variances.csv"
        var_df.to_csv(out_csv, index=False)
        print(f"Saved cluster variances to {out_csv}")

        self.plot_size_vs_variance(cluster_sizes, cluster_vars)
