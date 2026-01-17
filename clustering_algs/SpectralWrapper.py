from sklearn.cluster import SpectralClustering
import numpy as np

class SpectralWrapper:
    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters

    def run(self, distance_matrix: np.ndarray, X: np.ndarray) -> np.ndarray:
        model = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42
        )
    
        labels = model.fit_predict(distance_matrix)
        return labels