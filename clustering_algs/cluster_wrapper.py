from doctest import master
from clustering_algs.affinity_propagation import AffinityPropagationWrapper
from clustering_algs.OPTICSWrapper import OPTICSWrapper
from clustering_algs.star_clustering.star_clustering import StarCluster
from clustering_algs.agglomerative_clustering import AgglomerativeClustererWrapper
from clustering_algs.DBSCANWrapper import DBSCANClusterer
from clustering_algs.HDBSCANWrapper import HDBSCANClusterer
from clustering_algs.KmeansWrapper import KMeansWrapper
from metrics.quality_metrics import QualityMetrics

import numpy as np

class ClusterWrapper:
    def __init__(self):
        self.affinity_propagation = AffinityPropagationWrapper()
        self.optics = OPTICSWrapper()
        self.star_cluster = StarCluster()
        self.agglomerative = AgglomerativeClustererWrapper()
        self.dbscan = DBSCANClusterer()
        self.hdbscan = HDBSCANClusterer()
        self.quality_metrics = QualityMetrics()
        self.kmeans = KMeansWrapper()
        
    def run_all(self, distance_matrix: np.ndarray, X: np.ndarray) -> None:
        
        affinity_labels = self.affinity_propagation.run(distance_matrix.copy())
        print(f"Affinity Propagation found {len(set(affinity_labels))} clusters\n")
        self.quality_metrics.quality_metrics(X, distance_matrix, affinity_labels)

        optics_labels = self.optics.run(distance_matrix.copy(), X.copy())
        print(f"OPTICS found {len(set(optics_labels))} clusters")
        self.quality_metrics.quality_metrics(X, distance_matrix, optics_labels)
        
        dbscan_labels = self.dbscan.run(distance_matrix.copy(), X.copy())
        print(f"DBSCAN found {len(set(dbscan_labels))} clusters")
        self.quality_metrics.quality_metrics(X, distance_matrix, dbscan_labels)

        hdbscan_labels = self.hdbscan.run(distance_matrix.copy(), X.copy())
        print(f"HDBSCAN found {len(set(hdbscan_labels) - {-1})} clusters")
        self.quality_metrics.quality_metrics(X, distance_matrix, hdbscan_labels)

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
        
    def run_all_optimizer(self, distance_matrix: np.ndarray, X: np.ndarray) -> None:
        """
        What this will do is run all clustering algs with hyperparameter optimization
        against some elected quality metric
        """
        
        affinity_labels = self.affinity_propagation.run_pref_optimization(distance_matrix.copy(), X.copy())
        print(f"Affinity Propagation found {len(set(affinity_labels))} clusters\n")
        self.quality_metrics.quality_metrics(X, distance_matrix, affinity_labels)

        optics_labels = self.optics.run_pref_optimization(distance_matrix.copy(), X.copy())
        print(f"OPTICS found {len(set(optics_labels))} clusters")
        self.quality_metrics.quality_metrics(X, distance_matrix, optics_labels)
        
        dbscan_labels = self.dbscan.run(distance_matrix.copy(), X.copy())
        print(f"DBSCAN found {len(set(dbscan_labels))} clusters")
        self.quality_metrics.quality_metrics(X, distance_matrix, dbscan_labels)
        
        hdbscan_labels = self.hdbscan.run(distance_matrix.copy(), X.copy())
        print(f"HDBSCAN found {len(set(hdbscan_labels) - {-1})} clusters")
        self.quality_metrics.quality_metrics(X, distance_matrix, hdbscan_labels)
        
        return {
            "affinity": affinity_labels,
            "optics": optics_labels,
            "dbscan": dbscan_labels,
            "hdbscan": hdbscan_labels
        }

    def run_affinity(self, distance_matrix: np.ndarray, X: np.ndarray) -> np.ndarray:
        affinity_labels = self.affinity_propagation.run(distance_matrix.copy())
        print(f"Affinity Propagation found {len(set(affinity_labels))} clusters")
        self.quality_metrics.quality_metrics(X, distance_matrix, affinity_labels)
        
        return affinity_labels
    
    def run_optics(self, distance_matrix: np.ndarray, X: np.ndarray) -> np.ndarray:
        optics_labels = self.optics.run(distance_matrix.copy())
        print(f"OPTICS found {len(set(optics_labels))} clusters")
        self.quality_metrics.quality_metrics(X, distance_matrix, optics_labels)

        return optics_labels
    
    def run_dbscan(self, distance_matrix: np.ndarray, X: np.ndarray) -> np.ndarray:
        dbscan_labels = self.dbscan.run(distance_matrix.copy(), X.copy())
        print(f"DBSCAN found {len(set(dbscan_labels))} clusters")
        self.quality_metrics.quality_metrics(X, distance_matrix, dbscan_labels)

        return dbscan_labels
    
    def run_hdbscan(self, distance_matrix: np.ndarray, X: np.ndarray) -> np.ndarray:
        hdbscan_labels = self.hdbscan.run(distance_matrix.copy(), X.copy())
        print(f"HDBSCAN found {len(set(hdbscan_labels) - {-1})} clusters")
        self.quality_metrics.quality_metrics(X, distance_matrix, hdbscan_labels)

        return hdbscan_labels
    
    def run_kmeans(self, distance_matrix: np.ndarray, X) -> np.ndarray:
        kmeans_labels = self.kmeans.run(distance_matrix.copy(), X.copy())
        print(f"KMeans found {len(set(kmeans_labels))} clusters")
        self.quality_metrics.quality_metrics(X, distance_matrix, kmeans_labels)

        return kmeans_labels