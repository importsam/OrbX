from clustering_algs.affinity_propagation import AffinityPropagationWrapper
from clustering_algs.OPTICSWrapper import OPTICSWrapper
from clustering_algs.StarCluster import StarCluster
from clustering_algs.agglomerative_clustering import AgglomerativeClustererWrapper
import numpy as np

class ClusterWrapper:
    def __init__(self):
        self.affinity_propagation = AffinityPropagationWrapper()
        self.optics = OPTICSWrapper()
        self.star_cluster = StarCluster()
        self.agglomerative = AgglomerativeClustererWrapper()
        
    def run_all(self, distance_matrix: np.ndarray, X: np.ndarray) -> None:
        affinity_labels_X = self.affinity_propagation.run_X(X.copy())

        print(f"Affinity Propagation found {len(set(affinity_labels_X))} clusters")

        optics_labels = self.optics.run_X(distance_matrix.copy())
        print(f"OPTICS found {len(set(optics_labels))} clusters")
        
        self.star_cluster.fit(distance_matrix.copy())
        star_labels = self.star_cluster.labels_
        print(f"Star Clustering found {len(set(star_labels))} clusters")
        
        agglomerative_labels = self.agglomerative.run(distance_matrix.copy())
        print(f"Agglomerative Clustering found {len(set(agglomerative_labels))} clusters")
        
        """
        WCSS - use elbow method to determine optimal number of clusters.
        For multiple metrics we can use multivariable optimisation to find global maxima - hopefully
        """
        