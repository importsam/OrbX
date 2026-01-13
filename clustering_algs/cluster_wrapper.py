from doctest import master
from clustering_algs.affinity_propagation import AffinityPropagationWrapper
from clustering_algs.OPTICSWrapper import OPTICSWrapper
from clustering_algs.star_clustering.star_clustering import StarCluster
from clustering_algs.agglomerative_clustering import AgglomerativeClustererWrapper
from metrics.quality_metrics import QualityMetrics
from sklearn.mixture import GaussianMixture


import numpy as np

class ClusterWrapper:
    def __init__(self):
        self.affinity_propagation = AffinityPropagationWrapper()
        self.optics = OPTICSWrapper()
        self.star_cluster = StarCluster()
        self.agglomerative = AgglomerativeClustererWrapper()
        self.quality_metrics = QualityMetrics()
        
    def run_all(self, distance_matrix: np.ndarray, X: np.ndarray) -> None:
        
        affinity_labels = self.affinity_propagation.run(distance_matrix.copy())
        print(f"Affinity Propagation found {len(set(affinity_labels))} clusters\n")
        self.quality_metrics.quality_metrics(X, distance_matrix, affinity_labels)

        # optics_labels = self.optics.run(distance_matrix.copy())
        # print(f"OPTICS found {len(set(optics_labels))} clusters")
        # self.quality_metrics.quality_metrics(X, optics_labels)

        # agglomerative_labels = self.agglomerative.run(distance_matrix.copy())
        # print(f"Agglomerative Clustering found {len(set(agglomerative_labels))} clusters")
        # self.quality_metrics.quality_metrics(X, agglomerative_labels)
        
        # gmm = GaussianMixture(n_components=285, init_params='k-means++')
        # labels = gmm.fit_predict(X)
        # print(f"GMM found {len(set(labels))} clusters")
        # self.quality_metrics.quality_metrics(X, labels)
        
        # self.star_cluster.fit(X)
        # star_labels = self.star_cluster.labels_
        # print(f"Star Clustering found {len(set(star_labels))} clusters")
        # self.quality_metrics.quality_metrics(X, star_labels)
        
    def run_affinity(self, distance_matrix: np.ndarray, X) -> np.ndarray:
        affinity_labels = self.affinity_propagation.run(distance_matrix.copy())
        print(f"Affinity Propagation found {len(set(affinity_labels))} clusters")
        self.quality_metrics.quality_metrics(X, distance_matrix, affinity_labels)
        
        return affinity_labels
    
    def run_optics(self, distance_matrix: np.ndarray, X) -> np.ndarray:
        optics_labels = self.optics.run(distance_matrix.copy())
        print(f"OPTICS found {len(set(optics_labels))} clusters")
        self.quality_metrics.quality_metrics(X, optics_labels)
        
        return optics_labels