from doctest import master
from clustering_algs.affinity_propagation import AffinityPropagationWrapper
from clustering_algs.OPTICSWrapper import OPTICSWrapper
from clustering_algs.star_clustering.star_clustering import StarCluster
from clustering_algs.agglomerative_clustering import AgglomerativeClustererWrapper
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
import dbcv

import numpy as np

class ClusterWrapper:
    def __init__(self):
        self.affinity_propagation = AffinityPropagationWrapper()
        self.optics = OPTICSWrapper()
        self.star_cluster = StarCluster()
        self.agglomerative = AgglomerativeClustererWrapper()
        
    def run_all(self, distance_matrix: np.ndarray, X: np.ndarray) -> None:
        
        # affinity_labels = self.affinity_propagation.run(distance_matrix.copy())
        # print(f"Affinity Propagation found {len(set(affinity_labels))} clusters")
        # self.quality_metrics(X, affinity_labels)

        optics_labels = self.optics.run_X(distance_matrix.copy())
        print(f"OPTICS found {len(set(optics_labels))} clusters")
        self.quality_metrics(X, optics_labels)
        
        # agglomerative_labels = self.agglomerative.run_X(X.copy())
        # print(f"Agglomerative Clustering found {len(set(agglomerative_labels))} clusters")
        # self.quality_metrics(X, agglomerative_labels)
        
        # gmm = GaussianMixture(n_components=285, init_params='k-means++')
        # labels = gmm.fit_predict(X)
        # print(f"GMM found {len(set(labels))} clusters")
        # self.quality_metrics(X, labels)
        
        # self.star_cluster.fit(X)
        # star_labels = self.star_cluster.labels_
        # print(f"Star Clustering found {len(set(star_labels))} clusters")
        # self.quality_metrics(X, star_labels)

        
        """
        WCSS - use elbow method to determine optimal number of clusters.
        For multiple metrics we can use multivariable optimisation to find global maxima - hopefully
        """
        
    def quality_metrics(self, X: np.ndarray, labels: np.ndarray) -> dict:
        """Compute clustering quality metrics"""
        
        try:
            
            ch_score = calinski_harabasz_score(X, labels)
            silhouette_avg = silhouette_score(X, labels)
            db_score = davies_bouldin_score(X, labels)
            dbcv_score = dbcv.dbcv(X, labels)
            
            print(f"Calinski-Harabasz Score: {ch_score}")
            print(f"Silhouette Score: {silhouette_avg}")
            print(f"Davies-Bouldin Score: {db_score}")
            print(f"DBCV Score: {dbcv_score}")
            
            return {
                'Calinski-Harabasz': ch_score,
                'Silhouette Score': silhouette_avg,
                'Davies-Bouldin': db_score,
                'DBCV': dbcv_score
            }
            
        except Exception as e:
            print(f"Error computing quality metrics: {e}")
            return {}