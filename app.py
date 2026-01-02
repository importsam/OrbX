import pandas as pd
from configs import PathConfig, ClusterConfig
from tle_parser import TLEParser
from tools.distance_matrix import get_distance_matrix
from graph import Grapher
from clustering_algs.cluster_wrapper import ClusterWrapper

class SatelliteClusteringApp:
    def __init__(self, cluster_config: ClusterConfig):
        
        """Getting full TLE catalog from Space-Track
        can be either Space-Track, Celestrak or UDL"""
        # celestrak and UDL parsers not implemented yet
        self.tle_parser = TLEParser("Space-Track") 
        self.cluster_config = cluster_config
        self.path_config = PathConfig()
        self.graph = Grapher()
        self.cluster_wrapper = ClusterWrapper()
    
    """use_cached - if using cached distance matrix and dictionaries"""
    def run(self, use_cached: bool = False):
        # Get the satellite data into a dataframe 
        df = self.tle_parser.df
        # filter by inclination and apogee range
        df = df[
            (df['inclination'] >= self.cluster_config.inclination_range[0]) &
            (df['inclination'] <= self.cluster_config.inclination_range[1]) &
            (df['apogee'] >= self.cluster_config.apogee_range[0]) &
            (df['apogee'] <= self.cluster_config.apogee_range[1])
        ].copy()

        print(f"Loaded {len(df)} satellites in range - inc: {self.cluster_config.inclination_range}, apogee: {self.cluster_config.apogee_range}")

        # Get or compute the distance matrix
        distance_matrix, key = get_distance_matrix(df)
        df = self._reorder_dataframe(df, key)
        
        # Clustering 
        """
        So here I want to use all the clustering algs and do comparative analysis of performance.
        """
        # init the clustering algs
        self.cluster_wrapper.run_all(distance_matrix)
        
        # # plot 
        # self.graph.plot_clusters(df, self.path_config.output_plot)

    def _reorder_dataframe(self, df: pd.DataFrame, key: dict) -> pd.DataFrame:
        """Reorder dataframe to match key order (this is just overly cautious)"""
        idx_satNo = key['idx_satNo_dict']
        
        satNos_in_order = [idx_satNo[i] for i in range(len(idx_satNo))]
        return df.set_index("satNo").loc[satNos_in_order].reset_index()
    
