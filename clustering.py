import numpy as np
from sklearn.cluster import AffinityPropagation, DBSCAN, HDBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

from kneed import KneeLocator
import matplotlib.pyplot as plt

class SatelliteClusterer:

    def compute_clusters_affinity(self, distance_matrix: np.ndarray, damping: float = 0.95):
        
        # normalizer = np.median(distance_matrix)
        
        normalizer = 1347.81296
        
        print(f"Clustering has begun... norm = {normalizer}\n")
        
        if normalizer == 0:
            normalizer = 1.0
        
        similarity_matrix = np.exp(-distance_matrix/normalizer)
        
        affinity_clustering = AffinityPropagation(
            affinity='precomputed',
            damping=damping,
            max_iter=500,
            convergence_iter=15,
            random_state=42
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

    def compute_clusters_dbscan(self, distance_matrix: np.ndarray):
        
        def evaluate_dbscan(distance_matrix, eps, min_samples):
            
            """
            Evaluate DBSCAN clustering using silhouette score.
            
            Parameters:
            - distance_matrix: Precomputed distance matrix
            - eps: Maximum distance between two samples for one to be considered as in the neighborhood of the other
            - min_samples: Number of samples in a neighborhood for a point to be a core point
            
            Returns:
            - Silhouette score (or -1 if clustering is invalid), cluster labels
            """
            
            db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
            labels = db.fit_predict(distance_matrix)
            if len(set(labels)) > 1 and len(set(labels)) < len(labels):  # Valid clustering
                score = self.robust_silhouette_score(distance_matrix, labels)
            else:
                score = -1  # Invalid clustering (all noise or one cluster)
            return score, labels
    
        def find_optimal_eps(distance_matrix, min_samples, graph=False):
            """
            Find the optimal eps using the elbow method with kneed library.
            
            Parameters:
            - distance_matrix: Precomputed distance matrix
            - min_samples: Number of samples in a neighborhood for a point to be a core point
            
            Returns:
            - Optimal eps value
            """
            k = min_samples - 1
            k_distances = []
            for i in range(distance_matrix.shape[0]):
                distances = distance_matrix[i, np.arange(distance_matrix.shape[0]) != i]
                sorted_distances = np.sort(distances)
                k_dist = sorted_distances[k-1] if len(sorted_distances) >= k else sorted_distances[-1]
                k_distances.append(k_dist)
            k_distances_sorted = np.sort(k_distances)
            kneedle = KneeLocator(range(len(k_distances_sorted)), k_distances_sorted, S=1.0, curve='convex', direction='increasing')
            optimal_eps = k_distances_sorted[kneedle.elbow]
            
            if graph:
                # Plot k-distance graph for verification
                plt.figure(figsize=(8, 6))
                plt.plot(range(len(k_distances_sorted)), k_distances_sorted)
                plt.axvline(x=kneedle.elbow, color='r', linestyle='--', label=f'Elbow at eps={optimal_eps:.4f}')
                plt.xlabel('Points sorted by distance')
                plt.ylabel(f'{k}-th Nearest Neighbor Distance')
                plt.title(f'K-Distance Plot (min_samples={min_samples})')
                plt.legend()
                plt.savefig(f'k_distance_min_samples_{min_samples}.png')
                plt.close()
            
            return optimal_eps
    
        min_samples_range = range(2, 41)
        best_score = -1
        best_params = None
        best_labels = None
        
        for min_samples in min_samples_range:
            print(f"Evaluating min_samples={min_samples}...")
            eps = find_optimal_eps(distance_matrix, min_samples)
            score, labels = evaluate_dbscan(distance_matrix, eps, min_samples)
            print(f"  eps={eps:.4f}, silhouette_score={score:.4f}")
            if score > best_score:
                best_score = score
                best_params = (eps, min_samples)
                best_labels = labels
        
        if best_params is None:
            print("No valid clustering found. Adjust min_samples range or check data.")
            exit()
        
        eps, min_samples = best_params
        labels = best_labels
        print(f"Best parameters: eps={eps:.4f}, min_samples={min_samples}, silhouette_score={best_score:.4f}")
        
        outliers = np.where(labels == -1)[0].tolist()
        print(f"Number of outliers (orbits not in clusters): {len(outliers)}")
        print(f"Outlier satellite numbers: {outliers}")
        
        return labels, best_score

        
    def compute_clusters_hdbscan(self, distance_matrix: np.ndarray):
        
        def evaluate_hdbscan(distance_matrix, min_cluster_size):

            hdb = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_cluster_size, 
                metric='precomputed',
                cluster_selection_method='eom',
                allow_single_cluster=False
            )
            labels = hdb.fit_predict(distance_matrix)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters > 1 and n_clusters < len(distance_matrix):

                distance_matrix_copy = distance_matrix.copy()
                np.fill_diagonal(distance_matrix_copy, 0)
                score = self.robust_silhouette_score(distance_matrix_copy, labels)
            else:
                score = -1  # Invalid clustering
            return score, labels
        
        min_cluster_size_range = range(2, 41)
        best_score = -1
        best_params = None
        best_labels = None
        
        for min_cluster_size in min_cluster_size_range:
            print(f"Evaluating min_cluster_size={min_cluster_size}...")
            score, labels = evaluate_hdbscan(distance_matrix, min_cluster_size)
            print(f"  silhouette_score={score:.4f}")
            if score > best_score:
                best_score = score
                best_params = min_cluster_size
                best_labels = labels
        
        if best_params is None:
            print("No valid clustering found. Adjust min_cluster_size range or check data.")
            # Return all as noise as fallback
            labels = np.full(distance_matrix.shape[0], -1)
            score = -1
        else:
            labels = best_labels
            print(f"Best parameters: min_cluster_size={best_params}, silhouette_score={best_score:.4f}")
        
        # Report outliers
        outliers = np.where(labels == -1)[0].tolist()
        print(f"Number of outliers (orbits not in clusters): {len(outliers)}")
        print(f"Outlier satellite numbers: {outliers}")
        
        return labels, best_score
    
    def compute_clusters_agglomerative(self, distance_matrix: np.ndarray):
        print("Running Agglomerative Clustering...")
        
        threshold_guess = np.percentile(distance_matrix, 0.35) 
        
        agg = AgglomerativeClustering(
            n_clusters=None, 
            metric='precomputed',
            linkage='complete',
            distance_threshold=threshold_guess
        )
        
        labels = agg.fit_predict(distance_matrix)
        n_clusters = len(set(labels))
        
        if n_clusters > 1 and n_clusters < len(distance_matrix):
            # Note: Agglomerative doesn't produce noise (-1), so standard silhouette is fine
            np.fill_diagonal(distance_matrix, 0)
            score = silhouette_score(distance_matrix, labels, metric="precomputed")
            print(f"Agglomerative | Clusters: {n_clusters} | Silhouette: {score:.4f}")
        else:
            score = -1
            
        return labels, score


    def robust_silhouette_score(self, distance_matrix, labels):
        # Filter out noise points (-1)
        core_mask = labels != -1
        
        # Check if we have at least 2 clusters AND valid points remaining
        if np.sum(core_mask) > 0 and len(set(labels[core_mask])) > 1:
            # subset the matrix and labels
            core_dist_matrix = distance_matrix[np.ix_(core_mask, core_mask)]
            core_labels = labels[core_mask]
            
            return silhouette_score(core_dist_matrix, core_labels, metric='precomputed')
        return -1.0 # Invalid result