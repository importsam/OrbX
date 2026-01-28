import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter


class Analysis:
    def __init__(self, output_dir: str = "data/analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(7, 8),
            constrained_layout=True,
        )

        # --- Histogram of cluster sizes per algorithm ---
        ax_hist = axes[0]
        for name in algorithm_order:
            sizes = np.asarray(sizes_dict[name])
            ax_hist.hist(
                sizes,
                bins="auto",
                alpha=0.5,
                label=name,
            )

        if log_x:
            ax_hist.set_xscale("log")

        ax_hist.set_xlabel("Cluster size (number of orbits)")
        ax_hist.set_ylabel("Count of clusters")
        ax_hist.legend(title="Algorithm")
        ax_hist.set_title("Cluster size distribution")

        # --- Boxplot of cluster sizes per algorithm ---
        ax_box = axes[1]
        data = [np.asarray(sizes_dict[name]) for name in algorithm_order]
        ax_box.boxplot(
            data,
            labels=algorithm_order,
            showfliers=True,
        )
        if log_x:
            ax_box.set_yscale("log")
            ax_box.set_ylabel("Cluster size (log scale)")
        else:
            ax_box.set_ylabel("Cluster size")

        ax_box.set_title("Cluster sizes (median, IQR, whiskers)")

        # Save
        out_path = self.output_dir / save_name
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved cluster size distributions to {out_path}")
