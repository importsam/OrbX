import numpy as np
from sklearn.cluster import HDBSCAN

class HDBSCANWrapper:

    def __init__(self):
        pass

    def run(self, distance_matrix: np.ndarray, X: np.ndarray, min_samples: int = 3, min_cluster_size: int = 2) -> np.ndarray:
        # return labels, best_score
        return self.fit(distance_matrix, X, min_samples, min_cluster_size)

    def fit(self, distance_matrix: np.ndarray, X: np.ndarray, min_samples: int = 3, min_cluster_size: int = 2) -> np.ndarray:
    
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
        n_clusters = len(unique_clusters)
        if n_clusters < 2:
            n_orbits = len(labels)
            n_noise = int(np.sum(labels == -1))
            raise ValueError(
                f"HDBSCAN produced {n_clusters} cluster(s) from {n_orbits} orbit(s) "
                f"({n_noise} marked as noise). Adjust your dataset or tune the HDBSCAN parameters."
            )

        return labels
