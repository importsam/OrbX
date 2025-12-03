import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score

class SatelliteClusterer:

    def compute_clusters_affinity(self, distance_matrix: np.ndarray, damping: float = 0.95):
        print("Clustering has begun...\n")
                
        normalizer = np.median(distance_matrix)
        
        if normalizer == 0:
            normalizer = 1.0
        
        similarity_matrix = np.exp(-distance_matrix/normalizer)
        
        median_sim = np.median(similarity_matrix)
        
        print(f"Median similarity: {median_sim:.2f}\n")
        
        affinity_clustering = AffinityPropagation(
            affinity='precomputed',
            damping=damping,
            verbose=False,
            max_iter=500,
            convergence_iter=15,
            random_state=42,
            preference=10
        )
        
        try:
            affinity_clustering.fit(similarity_matrix)
            labels = affinity_clustering.labels_
            n_clusters = len(set(labels))
            
            if n_clusters > 1 and n_clusters < len(distance_matrix):
                
                distance_matrix_copy = distance_matrix.copy()
                
                np.fill_diagonal(distance_matrix_copy, 0)
                
                score = silhouette_score(distance_matrix_copy, labels, metric="precomputed")

                print(f"Clusters: {n_clusters:4d} | Silhouette: {score:.4f}")
                
            else:
                score = np.nan
            
                print(f"Clusters: {n_clusters:4d} | Silhouette: N/A")
                
        except Exception as e:
            print(f"Failed: {str(e)[:50]}")

        return labels, score

