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
        self.min_samples_range = [2]
        self.min_samples = 2
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

        # =========================
        # Stage 1: coarse eps sweep
        # =========================
        dist_vals = distance_matrix[distance_matrix > 0]

        low, high = np.percentile(dist_vals, [0.1, 3.0])
        low = max(low, 0.15)
        high = min(high, 4.0)

        coarse_eps = np.geomspace(low, high, 80)

        coarse_results = []

        for eps in tqdm(coarse_eps, desc="DBSCAN coarse eps sweep"):
            score, labels = self._evaluate(X, distance_matrix, eps, self.min_samples)
            n_clusters = len(set(labels) - {-1})
            noise_count = (labels == -1).sum()

            coarse_results.append(
                (eps, score, n_clusters, noise_count)
            )

        coarse_df = pd.DataFrame(
            coarse_results,
            columns=["eps", "dbcv", "clusters", "noise"]
        )

        # =========================
        # Find eps focus region
        # =========================
        focus = coarse_df[
            coarse_df.clusters.between(
                self.target_cluster_range[0],
                self.target_cluster_range[1]
            )
        ]

        if focus.empty:
            raise RuntimeError("No eps values produced feasible cluster counts")

        eps_center = focus.eps.median()

        # =========================
        # Stage 2: focused eps sweep
        # =========================
        eps_low = eps_center / 1.5
        eps_high = eps_center * 1.5

        eps_values = np.geomspace(eps_low, eps_high, 120)

        sweep_results = []

        for eps in tqdm(eps_values, desc="DBSCAN focused eps sweep"):
            score, labels = self._evaluate(X, distance_matrix, eps, self.min_samples)
            n_clusters = len(set(labels) - {-1})
            noise_count = (labels == -1).sum()

            sweep_results.append(
                (2, eps, score, n_clusters, noise_count)
            )

            if score < 0:
                continue

            penalty = abs(n_clusters - self.target_cluster_center) / self.target_cluster_center
            adjusted_score = score - 0.5 * penalty

            if self._feasible(n_clusters, noise_count, n_total):
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_labels = labels
                    best_params = (eps, 2)

        if best_labels is None:
            raise RuntimeError("DBSCAN failed to find a domain-feasible clustering")

        # =========================
        # Diagnostics plot (focused)
        # =========================
        # output_dir = Path("data")
        # output_dir.mkdir(parents=True, exist_ok=True)

        # df = pd.DataFrame(
        #     sweep_results,
        #     columns=["min_samples", "eps", "dbcv", "clusters", "noise"]
        # )

        # fig, ax1 = plt.subplots(figsize=(9, 5))

        # # ---- Left axis: number of clusters ----
        # ax1.plot(
        #     df.eps,
        #     df.clusters,
        #     alpha=0.7,
        #     label="Clusters",
        #     color="tab:blue"
        # )
        # ax1.set_xscale("log")
        # ax1.set_xlabel("eps (log scale)")
        # ax1.set_ylabel("Number of clusters", color="tab:blue")
        # ax1.tick_params(axis="y", labelcolor="tab:blue")

        # # Vertical reference line
        # ax1.axvline(
        #     eps_center,
        #     color="gray",
        #     linestyle="--",
        #     alpha=0.6,
        #     label="eps center"
        # )

        # # ---- Right axis: DBCV score ----
        # ax2 = ax1.twinx()
        # ax2.plot(
        #     df.eps,
        #     df.dbcv,
        #     alpha=0.7,
        #     label="DBCV",
        #     color="tab:orange"
        # )
        # ax2.set_ylabel("DBCV score", color="tab:orange")
        # ax2.tick_params(axis="y", labelcolor="tab:orange")

        # # ---- Combined legend ----
        # lines_1, labels_1 = ax1.get_legend_handles_labels()
        # lines_2, labels_2 = ax2.get_legend_handles_labels()
        # ax1.legend(
        #     lines_1 + lines_2,
        #     labels_1 + labels_2,
        #     loc="best",
        #     fontsize=9
        # )

        # ax1.set_title("DBSCAN (focused): clusters & DBCV vs eps")

        # output_path = output_dir / "dbscan_eps_clusters_dbcv_focused.png"
        # plt.tight_layout()
        # plt.savefig(output_path, dpi=200)
        # plt.close(fig)


        # =========================
        # Final reporting
        # =========================
        print(f"Saved plot to {output_path}")
        print(
            f"Best DBSCAN params → eps={best_params[0]:.6f}, min_samples=2"
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

