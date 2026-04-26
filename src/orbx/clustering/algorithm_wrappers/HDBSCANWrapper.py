import numpy as np
from sklearn.cluster import HDBSCAN
from tqdm import tqdm
import matplotlib.pyplot as plt
from metrics.quality_metrics import QualityMetrics
from models import ClusterResult
from pathlib import Path
import pandas as pd

class HDBSCANWrapper:

    def __init__(
        self,
        min_cluster_size_range=[2],
        min_samples_range=[3],
    ):
        self.min_cluster_size_range = min_cluster_size_range
        self.min_samples_range = min_samples_range
        self.quality_metrics = QualityMetrics()

    def run(self, distance_matrix: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
        # return labels, best_score
        return self.fit(distance_matrix, X)

    def _evaluate(self, X, distance_matrix, min_cluster_size, min_samples):

        print(f"Running HDBSCAN: ms={min_samples}", flush=True)

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
                f"min_samples={min_samples}, "
                f"score={score:.4f}"
            )
            return score, labels

        except Exception:
            print("!!!DBCV calculation failed, SET SCORE TO -1.0!!!")
            print(
                "Params: min_samples={min_samples}"
            )
            return -1.0, labels

    def fit(self, distance_matrix: np.ndarray, X: np.ndarray) -> ClusterResult:
        best_score = -np.inf
        best_labels = None
        best_min_samples = None

        # Storage for plotting
        min_samples_values = []
        dbcv_scores = []
        num_clusters = []


        for min_samples in self.min_samples_range:
            score, labels = self._evaluate(
                X,
                distance_matrix,
                2,
                min_samples,
            )

            # acceptance = QualityMetrics.is_clustering_acceptable(labels.copy())
            # if not acceptance["acceptable"]:
            #     print(f"Rejected ({acceptance['fail_reasons']})")
            #     continue

            min_samples_values.append(min_samples)
            dbcv_scores.append(score)
            num_clusters.append(len(set(labels) - {-1}))

            if score > best_score:
                best_score = score
                best_labels = labels
                best_min_samples = min_samples

            print("Clusters:", num_clusters[-1])

        if best_labels is None:
            raise RuntimeError("HDBSCAN failed to find a valid clustering")

        print(
            f"Best HDBSCAN params → "
            f"min_samples={best_min_samples}"
        )
        print(
            f"HDBSCAN found {len(set(best_labels) - {-1})} clusters "
            f"(noise points: {(best_labels == -1).sum()})"
        )
        print(f"Best score DBCV: {best_score:.4f}")


        # cluster_df = pd.DataFrame({"cluster": best_labels})
        # cluster_result_obj = ClusterResult(df=cluster_df, dbcv_score=best_score)

        return best_labels, best_score
