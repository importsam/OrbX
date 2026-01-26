import numpy as np
from sklearn.cluster import HDBSCAN
from tqdm import tqdm
import matplotlib.pyplot as plt
from metrics.quality_metrics import QualityMetrics
from models import ClusterResult
from pathlib import Path


class HDBSCANClusterer:

    def __init__(
        self,
        min_cluster_size_range=[2],
        min_samples_range=range(2, 50),
    ):
        self.min_cluster_size_range = min_cluster_size_range
        self.min_samples_range = min_samples_range
        self.quality_metrics = QualityMetrics()

    def run(self, distance_matrix: np.ndarray, X: np.ndarray):
        return self.fit(distance_matrix, X)

    def _evaluate(self, X, distance_matrix, min_cluster_size, min_samples):

        print(f"Running HDBSCAN: mcs={min_cluster_size}, ms={min_samples}", flush=True)

        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="precomputed",
            cluster_selection_method="eom",
            n_jobs=-1,
        )

        labels = clusterer.fit_predict(distance_matrix)

        unique_clusters = set(labels) - {-1}
        if len(unique_clusters) < 2:
            print("!!!HDBSCAN found less than 2 clusters, SET SCORE TO -1.0!!!")
            return -1.0, labels

        try:
            print("Started DBCV calculation...", flush=True)
            score = self.quality_metrics.dbcv_score_wrapper(X, labels)
            print(
                f"min_cluster_size={min_cluster_size}, "
                f"min_samples={min_samples}, "
                f"score={score:.4f}"
            )
            return score, labels

        except Exception:
            print("!!!DBCV calculation failed, SET SCORE TO -1.0!!!")
            print(
                "Params: min_cluster_size={min_cluster_size}, min_samples={min_samples}"
            )
            return -1.0, labels

    def fit(self, distance_matrix: np.ndarray, X: np.ndarray) -> ClusterResult:
        best_score = -np.inf
        best_labels = None
        best_min_cluster_size = None
        best_min_samples = None

        # Storage for plotting
        min_samples_values = []
        dbcv_scores = []
        num_clusters = []

        for min_cluster_size in tqdm(
            self.min_cluster_size_range,
            desc="HDBSCAN min_cluster_size sweep",
            unit="config",
        ):
            for min_samples in self.min_samples_range:
                score, labels = self._evaluate(
                    X,
                    distance_matrix,
                    min_cluster_size,
                    min_samples,
                )

                # acceptance = QualityMetrics.is_clustering_acceptable(labels.copy())
                # if not acceptance["acceptable"]:
                #     print(f"Rejected ({acceptance['fail_reasons']})")
                #     continue

                # ---- Collect metrics ----
                min_samples_values.append(min_samples)
                dbcv_scores.append(score)
                num_clusters.append(len(set(labels) - {-1}))

                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_min_cluster_size = min_cluster_size
                    best_min_samples = min_samples

                print("Clusters:", num_clusters[-1])

        if best_labels is None:
            raise RuntimeError("HDBSCAN failed to find a valid clustering")

        print(
            f"Best HDBSCAN params → "
            f"min_cluster_size={best_min_cluster_size}, "
            f"min_samples={best_min_samples}"
        )
        print(
            f"HDBSCAN found {len(set(best_labels) - {-1})} clusters "
            f"(noise points: {(best_labels == -1).sum()})"
        )
        print(f"Best score DBCV: {best_score:.4f}")

        cluster_result_obj = ClusterResult(
            best_labels,
            len(set(best_labels)),
            (best_labels == -1).sum(),
            best_score,
            self.quality_metrics.s_dbw_score_wrapper(X, best_labels),
        )

        return cluster_result_obj


    def plot_dbcv_vs_min_samples(self, min_samples_values, dbcv_scores, num_clusters, best_min_cluster_size):

        output_dir = Path("data")
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, ax1 = plt.subplots()

        # Left axis: DBCV (blue)
        ax1.plot(
            min_samples_values,
            dbcv_scores,
            color="tab:blue",
            linewidth=2,
        )
        ax1.set_xlabel("min_samples")
        ax1.set_ylabel("DBCV score", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        # Right axis: number of clusters (orange)
        ax2 = ax1.twinx()
        ax2.plot(
            min_samples_values,
            num_clusters,
            color="tab:orange",
            linewidth=2,
        )
        ax2.set_ylabel("Number of clusters", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        fig.suptitle(
            f"HDBSCAN: DBCV & #Clusters vs min_samples (min_cluster_size={best_min_cluster_size})"
        )

        output_path = output_dir / "hdbscan_dbcv_vs_clusters.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close(fig)


        print(f"Saved plot to {output_path}")