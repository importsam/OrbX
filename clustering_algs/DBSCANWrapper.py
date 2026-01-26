import numpy as np
from sklearn.cluster import DBSCAN
from metrics.quality_metrics import QualityMetrics
from models import ClusterResult
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


class DBSCANClusterer:

    def __init__(self):
        # Broad sweep, but selection will prefer small values
        self.min_samples_range = range(2, 20)

        # Domain expectations
        self.target_cluster_range = (200, 500)
        self.target_cluster_center = 350
        self.max_noise_fraction = 0.6

        self.quality_metrics = QualityMetrics()

    def run(self, distance_matrix: np.ndarray, X: np.ndarray):
        return self.fit(distance_matrix, X)

    def _evaluate(self, X, distance_matrix, eps, min_samples):
        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="precomputed",
            n_jobs=-1
        )

        labels = model.fit_predict(distance_matrix)
        n_clusters = len(set(labels) - {-1})

        if n_clusters < 2:
            return -1.0, labels

        try:
            score = self.quality_metrics.dbcv_score_wrapper(X, labels)
            return score, labels
        except Exception:
            return -1.0, labels

    def _feasible(self, n_clusters, noise_count, n_total):
        if not (self.target_cluster_range[0] <= n_clusters <= self.target_cluster_range[1]):
            return False
        if noise_count / n_total > self.max_noise_fraction:
            return False
        return True

    def fit(self, distance_matrix: np.ndarray, X: np.ndarray) -> ClusterResult:
        best_score = -np.inf
        best_labels = None
        best_params = None

        n_total = distance_matrix.shape[0]

        # ---- eps generation: VERY small, log-spaced ----
        dist_vals = distance_matrix[distance_matrix > 0]

        low, high = np.percentile(dist_vals, [0.001, 5])
        eps_values = np.geomspace(low, high, 80)

        sweep_results = []

        for min_samples in tqdm(
            self.min_samples_range,
            desc="DBSCAN sweep",
            unit="min_samples"
        ):
            for eps in eps_values:
                score, labels = self._evaluate(X, distance_matrix, eps, min_samples)
                n_clusters = len(set(labels) - {-1})
                noise_count = (labels == -1).sum()

                sweep_results.append(
                    (min_samples, eps, score, n_clusters, noise_count)
                )

                # --- soft domain-aware selection ---
                if score < 0:
                    continue

                penalty = abs(n_clusters - self.target_cluster_center) / self.target_cluster_center
                adjusted_score = score - 0.5 * penalty

                if self._feasible(n_clusters, noise_count, n_total):
                    if adjusted_score > best_score:
                        best_score = adjusted_score
                        best_labels = labels
                        best_params = (eps, min_samples)

        if best_labels is None:
            raise RuntimeError("DBSCAN failed to find a domain-feasible clustering")

        # ---- Diagnostics plots ----
        output_dir = Path("data")
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(
            sweep_results,
            columns=["min_samples", "eps", "dbcv", "clusters", "noise"]
        )

        fig, ax = plt.subplots(figsize=(9, 5))

        for ms in sorted(df.min_samples.unique()):
            sub = df[df.min_samples == ms]
            ax.plot(
                sub.eps,
                sub.clusters,
                alpha=0.5,
                label=f"min_samples={ms}"
            )

        ax.set_xscale("log")
        ax.set_xlabel("eps (log scale)")
        ax.set_ylabel("Number of clusters")
        ax.set_title("DBSCAN: clusters vs eps (log-scaled)")
        ax.legend(ncol=2, fontsize=8)

        output_path = output_dir / "dbscan_eps_vs_clusters.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close(fig)

        # ---- Final reporting ----
        print(f"Saved plot to {output_path}")
        print(
            f"Best DBSCAN params → eps={best_params[0]:.6f}, "
            f"min_samples={best_params[1]}"
        )
        print(
            f"DBSCAN found {len(set(best_labels) - {-1})} clusters "
            f"(noise points: {(best_labels == -1).sum()})"
        )
        print(f"Adjusted selection score: {best_score:.4f}")

        return ClusterResult(
            best_labels,
            len(set(best_labels) - {-1}),
            (best_labels == -1).sum(),
            best_score,
            self.quality_metrics.s_dbw_score_wrapper(X, best_labels),
        )
