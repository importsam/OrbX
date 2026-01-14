import numpy as np
from sklearn.cluster import HDBSCAN
import dbcv
from tqdm import tqdm


class HDBSCANClusterer:

    def __init__(self, min_cluster_size=4, min_samples_range=range(2, 15)):
        self.min_cluster_size = min_cluster_size
        self.min_samples_range = min_samples_range

    def run(self, distance_matrix: np.ndarray, X: np.ndarray):
        return self.fit(distance_matrix, X)

    def _evaluate(self, X, distance_matrix, min_samples):
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=min_samples,
            metric='precomputed',
            cluster_selection_method='eom',
            n_jobs=-1
        )

        labels = clusterer.fit_predict(distance_matrix)

        unique_clusters = set(labels) - {-1}
        if len(unique_clusters) < 2:
            return -1.0, labels

        try:
            score = dbcv.dbcv(X, labels)
            return score, labels
        except Exception:
            return -1.0, labels

    def fit(self, distance_matrix: np.ndarray, X: np.ndarray):
        best_score = -np.inf
        best_labels = None
        best_min_samples = None

        for min_samples in tqdm(
            self.min_samples_range,
            desc="HDBSCAN min_samples sweep",
            unit="config"
        ):
            score, labels = self._evaluate(X, distance_matrix, min_samples)

            if score > best_score:
                best_score = score
                best_labels = labels
                best_min_samples = min_samples

        if best_labels is None:
            raise RuntimeError("HDBSCAN failed to find a valid clustering")

        print(
            f"Best HDBSCAN params → "
            f"min_cluster_size={self.min_cluster_size}, "
            f"min_samples={best_min_samples}"
        )
        print(
            f"HDBSCAN found {len(set(best_labels) - {-1})} clusters "
            f"(noise points: {(best_labels == -1).sum()})"
        )
        print(f"Best DBCV score: {best_score:.4f}")

        return best_labels
