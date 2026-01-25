from typing import Any, Dict, List

import dbcv
import numpy as np
from s_dbw import S_Dbw
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from viasckde import viasckde_score


class QualityMetrics:

    def __init__(self):
        pass

    def quality_metrics(
        self, X: np.ndarray, distance_matrix: np.ndarray, labels: np.ndarray
    ) -> dict:
        """Compute clustering quality metrics"""

        try:
            ch_score = self.calinski_harabasz_score_wrapper(X, labels)
            silhouette = self.silhouette_score_wrapper(distance_matrix, labels)
            db_score = self.davies_bouldin_score_wrapper(X, labels)
            dbcv_score = self.dbcv_score_wrapper(X, labels)
            s_Dbw_score = self.s_dbw_score_wrapper(X, labels)
            viasckde = self.viasckde_score_wrapper(X, labels)

            print("Clustering Quality Metrics:")
            print(f"Primary - DBCV Score: {dbcv_score}")
            print(f"Secondary - S_Dbw Score: {s_Dbw_score}\n")
            print(f"Sanity - Viasckde Score: {viasckde}\n")
            print(f"Calinski-Harabasz Score: {ch_score}")
            print(f"Silhouette Score: {silhouette}")
            print(f"Davies-Bouldin Score: {db_score}")

            return {
                "Calinski-Harabasz": ch_score,
                "Silhouette Score": silhouette,
                "Davies-Bouldin": db_score,
                "DBCV": dbcv_score,
                "S_Dbw": s_Dbw_score,
                "Viasckde": viasckde,
            }

        except Exception as e:
            print(f"Error computing quality metrics: {e}")
            return {}

    def silhouette_score_wrapper(
        self, distance_matrix: np.ndarray, labels: np.ndarray
    ) -> float:
        """Compute Silhouette Score"""
        return silhouette_score(distance_matrix, labels, metric="precomputed")

    def calinski_harabasz_score_wrapper(
        self, X: np.ndarray, labels: np.ndarray
    ) -> float:
        """Compute Calinski-Harabasz Score"""
        return calinski_harabasz_score(X, labels)

    def davies_bouldin_score_wrapper(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Compute Davies-Bouldin Score"""
        return davies_bouldin_score(X, labels)

    def dbcv_score_wrapper(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Compute DBCV Score

        We have to remove any duplicate points for dbcv to work properly
        """

        X_unique, unique_idx = np.unique(X, axis=0, return_index=True)

        labels_unique = labels[unique_idx]

        return dbcv.dbcv(X_unique, labels_unique)

    def s_dbw_score_wrapper(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Compute S_Dbw Score"""
        return S_Dbw(
            X,
            labels,
            centers_id=None,
            method="Tong",
            alg_noise="filter",
            centr="mean",
            nearest_centr=True,
            metric="euclidean",
        )

    def viasckde_score_wrapper(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Compute Viasckde Score"""
        return viasckde_score(X, labels)

    def is_clustering_acceptable(
        labels: np.ndarray,
        *,
        min_n_clusters: int = 20,
        max_n_clusters: int = 1000,
        max_cluster_size_fraction: float = 0.25,
        min_noise_fraction: float = 0.00,
        max_noise_fraction: float = 1.00,
    ) -> Dict[str, Any]:
        """
        Evaluate whether a clustering solution is acceptable for OrbX / STM use.

        Parameters
        ----------
        labels : np.ndarray
            1D array of cluster labels for each point. Noise should be encoded as -1.
        min_n_clusters : int, optional
            Minimum acceptable number of clusters (excluding noise).
        max_n_clusters : int, optional
            Maximum acceptable number of clusters (excluding noise).
        max_cluster_size_fraction : float, optional
            Maximum allowed fraction of total points in any single cluster.
            Helps reject over-merged solutions (e.g. one giant cluster).
        min_noise_fraction : float, optional
            Minimum acceptable fraction of points labeled as noise.
            Helps reject solutions that force everything into clusters.
        max_noise_fraction : float, optional
            Maximum acceptable fraction of points labeled as noise.
            Helps reject over-pruned solutions.

        Returns
        -------
        Dict[str, Any]
            {
                "acceptable": bool,
                "n_points": int,
                "n_clusters": int,
                "noise_fraction": float,
                "max_cluster_size_fraction": float,
                "fail_reasons": List[str]
            }
        """

        result = {
            "acceptable": True,
            "n_points": int(len(labels)),
            "n_clusters": 0,
            "noise_fraction": 0.0,
            "max_cluster_size_fraction": 0.0,
            "fail_reasons": [],
        }

        n_points = len(labels)
        if n_points == 0:
            result["acceptable"] = False
            result["fail_reasons"].append("no_points")
            return result

        # Noise mask and fraction
        noise_mask = labels == -1
        n_noise = int(noise_mask.sum())
        noise_fraction = n_noise / n_points
        result["noise_fraction"] = noise_fraction

        # Cluster labels (exclude noise)
        cluster_labels = labels[~noise_mask]
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        n_clusters = len(unique_clusters)
        result["n_clusters"] = int(n_clusters)

        # Max cluster size fraction
        if n_clusters > 0:
            max_cluster_size = counts.max()
            max_cluster_size_fraction_observed = max_cluster_size / n_points
        else:
            max_cluster_size_fraction_observed = 0.0
        result["max_cluster_size_fraction"] = max_cluster_size_fraction_observed

        # --- Apply filters ---

        # 1. Cluster count range
        if n_clusters < min_n_clusters:
            result["acceptable"] = False
            result["fail_reasons"].append("too_few_clusters")
        if n_clusters > max_n_clusters:
            result["acceptable"] = False
            result["fail_reasons"].append("too_many_clusters")

        # 2. Max cluster size fraction (avoid one huge cluster)
        if max_cluster_size_fraction_observed > max_cluster_size_fraction:
            result["acceptable"] = False
            result["fail_reasons"].append("cluster_too_large")

        # 3. Noise fraction range
        if noise_fraction < min_noise_fraction:
            result["acceptable"] = False
            result["fail_reasons"].append("too_little_noise")
        if noise_fraction > max_noise_fraction:
            result["acceptable"] = False
            result["fail_reasons"].append("too_much_noise")

        return result
