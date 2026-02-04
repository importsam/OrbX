from tools.distance_matrix import get_distance_matrix
import pickle 
from graph import Grapher
from tle_parser import TLEParser
import pandas as pd 
import numpy as np 
from configs import ClusterConfig, OrbitalConstants, PathConfig


class DataHandler:
    def __init__(self, inclination_range=[0,180], apogee_range=[300,700]):
        self.inclination_range = inclination_range
        self.apogee_range = apogee_range
        self.graphing = Grapher()
        self.tle_parser = TLEParser("Space-Track")
        self.orbital_constants = OrbitalConstants
        
    def load_data(self):
        df = self.tle_parser.df
        df = df[
            (df["inclination"] >= self.inclination_range[0])
            & (df["inclination"] <= self.inclination_range[1])
            & (df["apogee"] >= self.apogee_range[0])
            & (df["apogee"] <= self.apogee_range[1])
        ].copy()

        print(
            f"Loaded {len(df)} satellites in range - inc: {self.inclination_range}, apogee: {self.apogee_range}"
        )

        # Get or compute the distance matrix
        distance_matrix, key = get_distance_matrix(df)

        # Reorder df to match distance_matrix
        orbit_points = self.get_points(df)
        df = self._reorder_dataframe(df, key)
        
        # load in the clustering labels 
        with open("data/cluster_results/dbscan_obj.pkl", "rb") as f:
            dbscan_obj = pickle.load(f)
            
        with open("data/cluster_results/hdbscan_obj.pkl", "rb") as f:
            hdbscan_obj = pickle.load(f)
            
        with open("data/cluster_results/optics_obj.pkl", "rb") as f:
            optics_obj = pickle.load(f)
        
        data_dict = {
            "orbit_df": df,
            "X": orbit_points,
            "distance_matrix": distance_matrix, 
            "key": key,
            "hdbscan_obj": hdbscan_obj,
            "dbscan_obj": dbscan_obj,
            "optics_obj": optics_obj
        }
        
        return data_dict
    
    def _reorder_dataframe(self, df: pd.DataFrame, key: dict) -> pd.DataFrame:
        """Reorder dataframe to match key order (this is just overly cautious)"""
        idx_satNo = key["idx_satNo_dict"]

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
        i = np.deg2rad(df["inclination"].values)
        Omega = np.deg2rad(df["raan"].values)
        omega = np.deg2rad(df["argument_of_perigee"].values)
        e = df["eccentricity"].values
        n = df["mean_motion"].values  # rev/day

        # Constants
        MU = self.orbital_constants.GM_EARTH  # m^3/s^2

        # Semi-major axis from mean motion
        n_rad = 2 * np.pi * n / 86400.0
        a = (MU / n_rad**2) ** (1 / 3)

        # Semi-latus rectum
        p = a * (1 - e**2)
        sqrt_p = np.sqrt(p)

        # Angular momentum vector u
        u = np.column_stack(
            [
                sqrt_p * np.sin(i) * np.sin(Omega),
                -sqrt_p * np.sin(i) * np.cos(Omega),
                sqrt_p * np.cos(i),
            ]
        )

        # LRL vector v
        v = np.column_stack(
            [
                e
                * sqrt_p
                * (
                    np.cos(omega) * np.cos(Omega)
                    - np.cos(i) * np.sin(omega) * np.sin(Omega)
                ),
                e
                * sqrt_p
                * (
                    np.cos(omega) * np.sin(Omega)
                    + np.cos(i) * np.sin(omega) * np.cos(Omega)
                ),
                e * sqrt_p * (np.sin(i) * np.sin(omega)),
            ]
        )

        X = np.hstack([u, v])

        return X