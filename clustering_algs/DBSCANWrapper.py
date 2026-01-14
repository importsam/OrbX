import numpy as np
from sklearn.cluster import DBSCAN
from kneed import KneeLocator
import dbcv
from tqdm import tqdm

class DBSCANClusterer:

    def __init__(self, min_samples_range=range(2, 12)):
        self.min_samples_range = min_samples_range

    def run(self, distance_matrix: np.ndarray, X: np.ndarray):
        return self.fit(distance_matrix, X)

    def _evaluate(self, X, distance_matrix, eps, min_samples):
        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='precomputed',
            n_jobs=-1
        )

        labels = model.fit_predict(distance_matrix)

        unique_clusters = set(labels) - {-1}
        if len(unique_clusters) < 2:
            return -1.0, labels

        try:
            score = dbcv.dbcv(X, labels)
            return score, labels
        except Exception:
            return -1.0, labels

    def _find_optimal_eps(self, distance_matrix, min_samples):
        k = min_samples - 1
        k_distances = []

        for i in range(distance_matrix.shape[0]):
            row = np.sort(distance_matrix[i][distance_matrix[i] != 0])
            k_distances.append(row[k - 1] if len(row) >= k else row[-1])

        k_distances = np.sort(k_distances)

        knee = KneeLocator(
            range(len(k_distances)),
            k_distances,
            curve="convex",
            direction="increasing"
        )

        if knee.elbow is None:
            return np.percentile(k_distances, 90)

        return k_distances[knee.elbow]

    def fit(self, distance_matrix: np.ndarray, X: np.ndarray):
        best_score = -np.inf
        best_labels = None
        best_params = None

        for min_samples in tqdm(
            self.min_samples_range,
            desc="DBSCAN min_samples sweep",
            unit="config"
        ):
            eps = self._find_optimal_eps(distance_matrix, min_samples)
            score, labels = self._evaluate(X, distance_matrix, eps, min_samples)

            if score > best_score:
                best_score = score
                best_labels = labels
                best_params = (eps, min_samples)

        if best_labels is None:
            raise RuntimeError("DBSCAN failed to find a valid clustering")

        print(
            f"Best DBSCAN params → eps={best_params[0]:.4f}, "
            f"min_samples={best_params[1]}"
        )
        print(
            f"DBSCAN found {len(set(best_labels) - {-1})} clusters "
            f"(noise points: {(best_labels == -1).sum()})"
        )
        print(f"Best DBCV score: {best_score:.4f}")

        return best_labels
