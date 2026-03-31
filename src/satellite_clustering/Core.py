from unittest import result

import pandas as pd
from src.satellite_clustering.Schema import Schema
from tools.distance_matrix import get_distance_matrix
from src.satellite_clustering.clustering.cluster_wrapper import ClusterWrapper
from tools.density_estimation import DensityEstimator
from models import ClusterResult
from src.satellite_clustering.data_handling.DataHandler import DataHandler

class Core:
    def __init__(self, cluster_config):
        self.schema = Schema()
        self.cluster_wrapper = ClusterWrapper()
        self.density_estimator = DensityEstimator()
        self.data_handler = DataHandler(cluster_config)

    def _cluster(self, df: pd.DataFrame, algorithm: str = "hdbscan") -> ClusterResult:

        distance_matrix, key = get_distance_matrix(df)
        df = self._reorder_dataframe(df, key)
        X = self.data_handler.get_points(df)
        
        cluster_result = self.run_algorithm(distance_matrix, X)
        
        cluster_result.df['density'] = self.density_estimator.density(distance_matrix)
        
        return cluster_result
    
    def _reorder_dataframe(self, df: pd.DataFrame, key: dict) -> pd.DataFrame:
        """Reorder dataframe to match key order (this is just overly cautious)"""
        idx_satNo = key["idx_satNo_dict"]

        satNos_in_order = [idx_satNo[i] for i in range(len(idx_satNo))]
        return df.set_index("satNo").loc[satNos_in_order].reset_index()
    
    def run_algorithm(self, distance_matrix, X) -> ClusterResult:
        
        return self.cluster_wrapper.run_hdbscan(distance_matrix, X)