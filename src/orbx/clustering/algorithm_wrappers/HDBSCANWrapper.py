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
        if len(unique_clusters) < 2:
            raise RuntimeError("HDBSCAN found less than 2 clusters")
        
        return labels
