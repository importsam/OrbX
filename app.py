import pandas as pd
from SatelliteClustering.update_cesium_assets.ionop_czml import ionop_czml
from configs import PathConfig, ClusterConfig, OrbitalConstants
from tle_parser import TLEParser
from tools.distance_matrix import get_distance_matrix
from graph import Grapher
from clustering_algs.cluster_wrapper import ClusterWrapper
from tools.density_estimation import DensityEstimator
import numpy as np
import sys
from pathlib import Path

# Add update_cesium_assets to path to import build_czml
sys.path.append(str(Path(__file__).parent / 'update_cesium_assets' / 'live'))
from build_czml import build_czml

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
        self.orbital_constants = OrbitalConstants()
        self.density_estimator = DensityEstimator()
        
    def run_metrics(self):
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
        orbit_points = self.get_points(df)
        df = self._reorder_dataframe(df, key)
        
        # Clustering 
        """
        So here I want to use all the clustering algs and do comparative analysis of performance.
        """
        # init the clustering algs
        self.cluster_wrapper.run_all(distance_matrix, orbit_points)

    def run_graphs(self):
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
        distance_matrix, key = get_distance_matrix(df.copy())
        orbit_points = self.get_points(df.copy())
        df = self._reorder_dataframe(df.copy(), key.copy())
        df = self.density_estimator.assign_density(df.copy(), distance_matrix.copy())
        
        # Clustering 
        """
        So here I want to use all the clustering algs and do comparative analysis of performance.
        """
        
        # If you want to run without optimzation for each alg and then graph
        # affinity_labels = self.cluster_wrapper.run_affinity(distance_matrix, orbit_points)
        # optics_labels = self.cluster_wrapper.run_optics(distance_matrix, orbit_points)
        # dbscan_labels = self.cluster_wrapper.run_dbscan(distance_matrix, orbit_points)
        # hdbscan_labels = self.cluster_wrapper.run_hdbscan(distance_matrix, orbit_points)
        
        # if you want to run optimzation for each alg and then graph
        labels_dict = self.cluster_wrapper.run_all_optimizer(distance_matrix.copy(), orbit_points.copy())
        affinity_labels = labels_dict["affinity"]
        optics_labels = labels_dict["optics"]
        dbscan_labels = labels_dict["dbscan"]
        hdbscan_labels = labels_dict["hdbscan"]
        
        # plot tsne graphs
        self.graph.plot_tsne(orbit_points, df, labels=affinity_labels, name="affinity")
        self.graph.plot_tsne(orbit_points, df, labels=optics_labels, name="optics")
        self.graph.plot_tsne(orbit_points, df, labels=dbscan_labels, name="dbscan")
        self.graph.plot_tsne(orbit_points, df, labels=hdbscan_labels, name="hdbscan")
        
        # plot UMAP graphs
        self.graph.plot_umap(orbit_points, df, labels=affinity_labels, name="affinity")
        self.graph.plot_umap(orbit_points, df, labels=optics_labels, name="optics")
        self.graph.plot_umap(orbit_points, df, labels=dbscan_labels, name="dbscan")
        self.graph.plot_umap(orbit_points, df, labels=hdbscan_labels, name="hdbscan")
        
        # Plot clusters in apogee/inclination space  
        df_opt = df.copy()
        df_opt['label'] = optics_labels
        self.graph.plot_clusters(df_opt, self.path_config.output_plot / "optics_clusters.html")
        
        # now for affinity
        df_aff = df.copy()
        df_aff['label'] = affinity_labels
        self.graph.plot_clusters(df_aff, self.path_config.output_plot / "affinity_clusters.html")
        
        # now for dbscan
        df_db = df.copy()
        df_db['label'] = dbscan_labels
        self.graph.plot_clusters(df_db, self.path_config.output_plot / "dbscan_clusters.html")
        
        df_hdb = df.copy()
        df_hdb['label'] = hdbscan_labels
        self.graph.plot_clusters(df_hdb, self.path_config.output_plot / "hdbscan_clusters.html")
        
        # Generate CZML for Cesium visualization
        print("\nGenerating CZML for Cesium visualization...")
        self.run_cesium(df.copy(), distance_matrix.copy())
    
    def run_cesium(self, df: pd.DataFrame = None, distance_matrix: np.ndarray = None):
        """
        Generate CZML file for Cesium visualization from clustering dataframe.
        
        Args:
            df: Dataframe with satellite data. If None, will use filtered dataframe.
            distance_matrix: Distance matrix. If None, will be computed from df.
        """
        # If df not provided, get the filtered dataframe
        if df is None:
            df = self.tle_parser.df
            # filter by inclination and apogee range
            df = df[
                (df['inclination'] >= self.cluster_config.inclination_range[0]) &
                (df['inclination'] <= self.cluster_config.inclination_range[1]) &
                (df['apogee'] >= self.cluster_config.apogee_range[0]) &
                (df['apogee'] <= self.cluster_config.apogee_range[1])
            ].copy()
        
        # Get or compute distance matrix if needed
        if distance_matrix is None:
            distance_matrix, key = get_distance_matrix(df.copy())
            df = self._reorder_dataframe(df.copy(), key.copy())
        else:
            # If distance_matrix provided but no key, we still need to ensure proper ordering
            _, key = get_distance_matrix(df.copy())
            df = self._reorder_dataframe(df.copy(), key.copy())
        
        # Ensure density is assigned if not present
        if 'density' not in df.columns:
            df = self.density_estimator.assign_density(df.copy(), distance_matrix.copy())
        
        # Ensure name column exists (fallback to satNo if not present)
        if 'name' not in df.columns:
            df['name'] = df['satNo']
        
        print(f"Generating CZML for {len(df)} satellites...")
        
        # Build CZML file
        build_czml(df)
        
        ACCESSTOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI5OTMwYjJlMS0yYjBhLTQwMmMtYjJkZi1mZWZiY2RiYTNmN2UiLCJpZCI6MjQwODIwLCJpYXQiOjE3MzgzMDM2ODl9.h1pXOgujWRPoS6ZFc5wL-l5_XJnSyUsPZym3ssZj7TQ'
  
        try:
            ionop_czml(ACCESSTOKEN)

        except Exception as e:
            print(f"Error: {e}")
        
        print(f"CZML file generated successfully at update_cesium_assets/live/data/output.czml")
    
    def _reorder_dataframe(self, df: pd.DataFrame, key: dict) -> pd.DataFrame:
        """Reorder dataframe to match key order (this is just overly cautious)"""
        idx_satNo = key['idx_satNo_dict']
        
        satNos_in_order = [idx_satNo[i] for i in range(len(idx_satNo))]
        return df.set_index("satNo").loc[satNos_in_order].reset_index()
    
    def get_points(self, df: pd.DataFrame):
        """Takes in a dataframe of Satellite objects. Converts each to a point in the 5D manifold embedded in 6D.
        This is so the raw data can be passed into clustering algs, quality metrics, etc.
        
        Args:
            df (pd.DataFrame): _description_

        Returns:
            _type_: _description_
        """
        
        # Convert degrees → radians
        i = np.deg2rad(df['inclination'].values)
        Omega = np.deg2rad(df['raan'].values)
        omega = np.deg2rad(df['argument_of_perigee'].values)
        e = df['eccentricity'].values
        n = df['mean_motion'].values  # rev/day

        # Constants
        MU = self.orbital_constants.GM_EARTH  # m^3/s^2

        # Semi-major axis from mean motion
        n_rad = 2 * np.pi * n / 86400.0
        a = (MU / n_rad**2) ** (1/3)

        # Semi-latus rectum
        p = a * (1 - e**2)
        sqrt_p = np.sqrt(p)

        # Angular momentum vector u
        u = np.column_stack([
            sqrt_p * np.sin(i) * np.sin(Omega),
            -sqrt_p * np.sin(i) * np.cos(Omega),
            sqrt_p * np.cos(i)
        ])

        # LRL vector v
        v = np.column_stack([
            e * sqrt_p * (np.cos(omega) * np.cos(Omega) - np.cos(i) * np.sin(omega) * np.sin(Omega)),
            e * sqrt_p * (np.cos(omega) * np.sin(Omega) + np.cos(i) * np.sin(omega) * np.cos(Omega)),
            e * sqrt_p * (np.sin(i) * np.sin(omega))
        ])

        X = np.hstack([u, v])

        return X