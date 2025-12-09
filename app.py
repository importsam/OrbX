import pandas as pd
from configs import PathConfig, ClusterConfig
from tle_parser import TLEParser
from tools.distance_matrix import get_distance_matrix
from clustering import SatelliteClusterer
from graph import Grapher
import os

class SatelliteClusteringApp:
    def __init__(self, cluster_config: ClusterConfig):
        
        """Getting full TLE catalog from Space-Track
        can be either Space-Track, Celestrak or UDL"""
        # celestrak and UDL parsers not implemented yet
        self.tle_parser = TLEParser("Space-Track") 
        self.cluster_config = cluster_config
        self.path_config = PathConfig()
        self.clusterer = SatelliteClusterer()
        self.graph = Grapher()
    
    """use_cached - if using cached distance matrix and dictionaries"""
    def run(self, use_cached: bool = False):
        # 1. Get the satellite data into a dataframe 
        df = self.tle_parser.df
        # filter by inclination and apogee range
        df = df[
            (df['inclination'] >= self.cluster_config.inclination_range[0]) &
            (df['inclination'] <= self.cluster_config.inclination_range[1]) &
            (df['apogee'] >= self.cluster_config.apogee_range[0]) &
            (df['apogee'] <= self.cluster_config.apogee_range[1])
        ].copy()

        print(f"Loaded {len(df)} satellites in range - inc: {self.cluster_config.inclination_range}, apogee: {self.cluster_config.apogee_range}")

        # 2. Get or compute the distance matrix
        distance_matrix, key = get_distance_matrix(df)
        df = self._reorder_dataframe(df, key)
        
        # 3. clustering 
        # labels, silhouette = self.clusterer.compute_clusters_affinity(distance_matrix, damping=0.95) # affinity propagation
        labels, silhouette = self.clusterer.compute_clusters_agglomerative(distance_matrix)

        
        df['label'] = labels
        
        # save the current dataframe with clusters 
        os.makedirs(self.path_config.output_dataframe, exist_ok=True)

        # Build the save path
        cluster_save_path = (
            f"{self.path_config.output_dataframe}/"
            f"clusters_inc_{self.cluster_config.inclination_range[0]}-"
            f"{self.cluster_config.inclination_range[1]}_"
            f"apogee_{self.cluster_config.apogee_range[0]}-"
            f"{self.cluster_config.apogee_range[1]}_"
            "silhouette_"
            f"{silhouette:.3f}.pkl"
        )

        # Save the dataframe
        df.to_pickle(cluster_save_path)
        
        # 4. plot 
        self.graph.plot_clusters(df, self.path_config.output_plot)

    def _reorder_dataframe(self, df: pd.DataFrame, key: dict) -> pd.DataFrame:
        """Reorder dataframe to match key order (this is just overly cautious)"""
        idx_satNo = key['idx_satNo_dict']
        
        satNos_in_order = [idx_satNo[i] for i in range(len(idx_satNo))]
        return df.set_index("satNo").loc[satNos_in_order].reset_index()
