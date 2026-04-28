import numpy as np
import pandas as pd
from orbx.clustering.Schema import Schema
from orbx.tools.distance_matrix import get_distance_matrix
from orbx.clustering.algorithm_wrappers.cluster_wrapper import ClusterWrapper
from orbx.clustering.data_handling.DataHandler import DataHandler

class Core:
    def __init__(self):
        self.schema = Schema()
        self.cluster_wrapper = ClusterWrapper()
        self.data_handler = DataHandler()

    """
    The cluster function will take in a dataframe of TLEs and return a ClusterResult.
    """
    def cluster(self, df: pd.DataFrame, min_samples: int = 3, min_cluster_size: int = 2) -> np.ndarray:

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
        
        labels = self.cluster_wrapper.run_hdbscan(distance_matrix, X, min_samples, min_cluster_size)
        
        return labels
    
    def _reorder_dataframe(self, df: pd.DataFrame, key: dict) -> pd.DataFrame:
        """Reorder dataframe to match key order (this is just overly cautious)"""
        idx_satNo = key["idx_satNo_dict"]

        satNos_in_order = [idx_satNo[i] for i in range(len(idx_satNo))]
        return df.set_index("satNo").loc[satNos_in_order].reset_index()