import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score

class SatelliteClusterer:

    def compute_clusters_affinity(self, distance_matrix: np.ndarray, damping: float = 0.95):
        print("Clustering has begun...\n")
        
        affinity_clustering = AffinityPropagation(affinity='precomputed', damping=damping, verbose=True)
        affinity_clustering.fit(np.exp(-distance_matrix/np.var(distance_matrix)))

        # Get the cluster labels
        labels = affinity_clustering.labels_
        
        silhouette = self._compute_silhouette(distance_matrix, labels)
        
        return labels, silhouette

    def _compute_silhouette(self, distance_matrix: np.ndarray, labels: np.ndarray) -> float:
            
        distance_matrix_copy = distance_matrix.copy()
        np.fill_diagonal(distance_matrix_copy, 0)
        
        # Only compute silhouette if we have more than 1 cluster
        if len(set(labels)) > 1:
            score = silhouette_score(distance_matrix_copy, labels, metric="precomputed")
            print(f"\nSilhouette score: {score:.4f}")
        else:
            print("\nOnly one cluster, silhouette score = N/A")
            
        return score