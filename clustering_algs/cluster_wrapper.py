from doctest import master

import numpy as np

from clustering_algs.affinity_propagation import AffinityPropagationWrapper
from clustering_algs.agglomerative_clustering import AgglomerativeClustererWrapper
from clustering_algs.DBSCANWrapper import DBSCANClusterer
from clustering_algs.HDBSCANWrapper import HDBSCANClusterer
from clustering_algs.KmeansWrapper import KMeansWrapper
from clustering_algs.OPTICSWrapper import OPTICSWrapper
from clustering_algs.SpectralWrapper import SpectralWrapper
from clustering_algs.star_clustering.star_clustering import StarCluster
from metrics.quality_metrics import QualityMetrics
from models import ClusterResult


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
        self.spectral = SpectralWrapper()

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

    def run_all_optimizer(self, distance_matrix: np.ndarray, X: np.ndarray):
        """
        What this will do is run all clustering algs with hyperparameter optimization
        against some elected quality metric

        Returns:
            dict of ClusterResult objects for each clustering algorithm
        """

        # affinity_results = self.affinity_propagation.run_pref_optimization(distance_matrix.copy(), X.copy())
        # if affinity_results:
        #   print(f"Affinity Propagation found {len(set(affinity_results.labels))} clusters\n")
        #  self.quality_metrics.quality_metrics(X, distance_matrix, affinity_results.labels)

        # optics_results = self.optics.run_pref_optimization(
        #     distance_matrix.copy(), X.copy()
        # )
        # print(f"OPTICS found {len(set(optics_results.labels))} clusters")
        # self.quality_metrics.quality_metrics(X, distance_matrix, optics_results.labels)

        # dbscan_results = self.dbscan.run(distance_matrix.copy(), X.copy())
        # if dbscan_results:
        #     print(f"DBSCAN found {len(set(dbscan_results.labels))} clusters")
        #     self.quality_metrics.quality_metrics(
        #         X, distance_matrix, dbscan_results.labels
        #     )

        hdbscan_results = self.hdbscan.run(distance_matrix.copy(), X.copy())
        print(f"HDBSCAN found {len(set(hdbscan_results.labels) - {-1})} clusters")
        self.quality_metrics.quality_metrics(X, distance_matrix, hdbscan_results.labels)

        return {
            # "affinity_results": affinity_results,
            # "optics_results": optics_results,
            # "dbscan_results": dbscan_results,
            "hdbscan_results": hdbscan_results,
        }

    def run_affinity(self, distance_matrix: np.ndarray, X: np.ndarray) -> np.ndarray:
        affinity_labels = self.affinity_propagation.run(
            distance_matrix.copy(), X.copy()
        )
        print(f"Affinity Propagation found {len(set(affinity_labels))} clusters")
        self.quality_metrics.quality_metrics(X, distance_matrix, affinity_labels)

        return affinity_labels

    def run_optics(self, distance_matrix: np.ndarray, X: np.ndarray) -> np.ndarray:
        optics_results = self.optics.run(distance_matrix.copy(), X.copy())
        print(f"OPTICS found {len(set(optics_results.labels))} clusters")
        self.quality_metrics.quality_metrics(X, distance_matrix, optics_results.labels)

        return optics_results.labels
    
    def run_dbscan(self, distance_matrix: np.ndarray, X: np.ndarray) -> np.ndarray:
        dbscan_results = self.dbscan.run(distance_matrix.copy(), X.copy())
        print(f"DBSCAN found {len(set(dbscan_results.labels))} clusters")
        self.quality_metrics.quality_metrics(X, distance_matrix, dbscan_results.labels)

        return dbscan_results.labels
    
    def run_hdbscan(self, distance_matrix: np.ndarray, X: np.ndarray) -> np.ndarray:
        hdbscan_result = self.hdbscan.run(distance_matrix.copy(), X.copy())
        print(f"HDBSCAN found {len(set(hdbscan_result.labels) - {-1})} clusters")
        self.quality_metrics.quality_metrics(X, distance_matrix, hdbscan_result.labels)

        return hdbscan_result.labels
    def run_kmeans(self, distance_matrix: np.ndarray, X) -> np.ndarray:
        kmeans_labels = self.kmeans.run(distance_matrix.copy(), X.copy())
        print(f"KMeans found {len(set(kmeans_labels))} clusters")
        self.quality_metrics.quality_metrics(X, distance_matrix, kmeans_labels)

        return kmeans_labels

    def run_spectral(self, distance_matrix: np.ndarray, X: np.ndarray) -> np.ndarray:

        spectral_labels = self.spectral.run(distance_matrix.copy(), X.copy())
        print(f"Spectral Clustering found {len(set(spectral_labels))} clusters")
        self.quality_metrics.quality_metrics(X, distance_matrix, spectral_labels)

        return spectral_labels
