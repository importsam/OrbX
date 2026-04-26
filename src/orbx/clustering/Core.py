import numpy as np
import pandas as pd
from .Schema import Schema
from tools.distance_matrix import get_distance_matrix
from .clustering.cluster_wrapper import ClusterWrapper
from tools.density_estimation import DensityEstimator
from models import ClusterResult
from .data_handling.DataHandler import DataHandler

class Core:
    def __init__(self):
        self.schema = Schema()
        self.cluster_wrapper = ClusterWrapper()
        self.density_estimator = DensityEstimator()
        self.data_handler = DataHandler()


    """
    The cluster function will take in a dataframe of TLEs and return a ClusterResult.
    """
    def cluster(self, df: pd.DataFrame, algorithm: str = "hdbscan") -> ClusterResult:

        """
        Needs to take the input df
        which supposedly has only the line1 line2 cols, and then 
        extract the keplerian elements. These must then be concatenated to the
        original df
        """
        
        df = self.data_handler.tle_to_keplerian(df)
        
        distance_matrix, key = get_distance_matrix(df)
        df = self._reorder_dataframe(df, key)
        X = self.data_handler.get_points(df)
        
        labels, best_score = self.run_algorithm(distance_matrix, X)
        
        density_df = self.density_estimator.density(distance_matrix)
        
        cluster_result = ClusterResult(labels=labels, density_df=density_df, dbcv_score=best_score)
        
        return cluster_result
    
    def _reorder_dataframe(self, df: pd.DataFrame, key: dict) -> pd.DataFrame:
        """Reorder dataframe to match key order (this is just overly cautious)"""
        idx_satNo = key["idx_satNo_dict"]

        satNos_in_order = [idx_satNo[i] for i in range(len(idx_satNo))]
        return df.set_index("satNo").loc[satNos_in_order].reset_index()
    
    def run_algorithm(self, distance_matrix, X) -> tuple[np.ndarray, float]:
        
        labels, best_score = self.cluster_wrapper.run_hdbscan(distance_matrix, X)
        return labels, best_score