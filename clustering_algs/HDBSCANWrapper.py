import numpy as np
from sklearn.cluster import HDBSCAN
from tqdm import tqdm
from metrics.quality_metrics import QualityMetrics
from models import ClusterResult

class HDBSCANClusterer:

    def __init__(
        self,
        min_cluster_size_range=[2],
        min_samples_range=range(2, 11),
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
            n_jobs=1,
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
            print("Params: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
            return -1.0, labels

    def fit(self, distance_matrix: np.ndarray, X: np.ndarray) -> ClusterResult:
        best_score = -np.inf
        best_labels = None
        best_min_cluster_size = None
        best_min_samples = None

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

                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_min_cluster_size = min_cluster_size
                    best_min_samples = min_samples

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

        cluster_result_obj = ClusterResult(best_labels, len(set(best_labels)), 
                                           (best_labels == -1).sum(), best_score,
                                           self.quality_metrics.s_dbw_score_wrapper(X, best_labels))

        return cluster_result_obj