import pandas as pd
import numpy as np
import pickle
from .tle_parser import TLEParser
from configs import OrbitalConstants
from tools.distance_matrix import get_distance_matrix
from configs import ClusterConfig
from models import Satellite
from math import pi
from pathlib import Path
from math import pi
import pandas as pd
from models import Satellite
from configs import OrbitalConstants, PathConfig
class DataHandler:
    
    def __init__(self, cluster_config):
        self.cluster_config = cluster_config
        self.tle_parser = TLEParser("Space-Track")
        self.orbital_constants = OrbitalConstants()

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

    def run_metrics(self):
            # Get the satellite data into a dataframe
            df = self.tle_parser.df
            # filter by inclination and apogee range
            df = df[
                (df["inclination"] >= self.cluster_config.inclination_range[0])
                & (df["inclination"] <= self.cluster_config.inclination_range[1])
                & (df["apogee"] >= self.cluster_config.apogee_range[0])
                & (df["apogee"] <= self.cluster_config.apogee_range[1])
            ].copy()

            print(
                f"Loaded {len(df)} satellites in range - inc: {self.cluster_config.inclination_range}, apogee: {self.cluster_config.apogee_range}"
            )

            # Get or compute the distance matrix
            distance_matrix, key = get_distance_matrix(df)
            orbit_points = self.get_points(df)
            df = self._reorder_dataframe(df, key)

            # Clustering
            """
            So here I want to use all the clustering algs and do comparative analysis of performance.
            """
            # init the clustering algs
            cluster_result_dict = self.cluster_wrapper.run_hdbscan(
                distance_matrix, orbit_points
            )
        
    def run_experiment(self):
        # Get the satellite data into a dataframe
        df = self.tle_parser.df
        df = df[
            (df["inclination"] >= self.cluster_config.inclination_range[0])
            & (df["inclination"] <= self.cluster_config.inclination_range[1])
            & (df["apogee"] >= self.cluster_config.apogee_range[0])
            & (df["apogee"] <= self.cluster_config.apogee_range[1])
        ].copy()

        print(
            f"Loaded {len(df)} satellites in range - inc: {self.cluster_config.inclination_range}, apogee: {self.cluster_config.apogee_range}"
        )

        # Get or compute the distance matrix
        distance_matrix, key = get_distance_matrix(df)

        # Reorder df to match distance_matrix
        orbit_points = self.get_points(df)
        df = self._reorder_dataframe(df, key)

        # SINGLE SOURCE OF TRUTH: compute densities once, aligned with df
        densities = self.density_estimator.density(distance_matrix)
        df["density"] = densities
        # Clustering
        """
        So here I want to use all the clustering algs and do comparative analysis of performance.
        """
        # init the clustering algs
        # cluster_result_dict = self.cluster_wrapper.run_all_optimizer(
        #     distance_matrix, orbit_points
        # )
        
        # load in results 
        with open("data/cluster_results/dbscan_obj.pkl", "rb") as f:
            dbscan_obj = pickle.load(f)
            
        with open("data/cluster_results/hdbscan_obj.pkl", "rb") as f:
            hdbscan_obj = pickle.load(f)
            
        with open("data/cluster_results/optics_obj.pkl", "rb") as f:
            optics_obj = pickle.load(f)
            
        cluster_result_dict = {
            "dbscan_results": dbscan_obj,
            "hdbscan_results": hdbscan_obj,
            "optics_results": optics_obj,
        }

        # self.process_post_clustering(cluster_result_dict, df, distance_matrix)
        # self.analysis_graphs(cluster_result_dict, df, distance_matrix)
        
        hdbscan_labels = cluster_result_dict["hdbscan_results"].labels
        df["cluster"] = hdbscan_labels
        
        
        self.run_cesium(df=df, distance_matrix=distance_matrix, key=key)

        return None
    
    
    def tle_to_keplerian(self, input_df) -> pd.DataFrame:
        
        # Pre-allocate lists for keplerian elements
        sat_nos = []
        inclinations = []
        apogees = []
        raans = []
        arguments_of_perigee = []
        eccentricities = []
        mean_motions = []
        
        # for each row, compute the keplerian elements and add to df
        for index, row in input_df.iterrows():
            sat_obj = self._parse_tle_group(
                row['line1'],
                row['line2']
            )
            
            sat_nos.append(sat_obj.sat_no)
            inclinations.append(sat_obj.inclination)
            apogees.append(sat_obj.apogee)
            raans.append(sat_obj.raan)
            arguments_of_perigee.append(sat_obj.argument_of_perigee)
            eccentricities.append(sat_obj.eccentricity)
            mean_motions.append(sat_obj.mean_motion)
        
        # Add as new columns
        input_df['satNo'] = sat_nos
        input_df['inclination'] = inclinations
        input_df['apogee'] = apogees
        input_df['raan'] = raans
        input_df['argument_of_perigee'] = arguments_of_perigee
        input_df['eccentricity'] = eccentricities
        input_df['mean_motion'] = mean_motions
        
        return input_df
    
    def _parse_tle_group(self, line1: str, line2: str) -> Satellite:
        """Parse a single TLE group into a Satellite object"""
        if not (line1.startswith('1 ') and line2.startswith('2 ')):
            raise ValueError("Invalid TLE format")

        sat_no = line1[2:7].strip()

        inclination = float(line2[8:16].strip())
        mean_motion = float(line2[52:63].strip())
        eccentricity = float("0." + line2[26:33].strip())
        raan = float(line2[17:25].strip())
        argument_of_perigee = float(line2[34:42].strip())
        apogee = self._calculate_apogee(mean_motion, eccentricity)

        return Satellite(
            sat_no=sat_no,
            line1=line1,
            line2=line2,
            inclination=inclination,
            apogee=apogee,
            raan=raan,
            argument_of_perigee=argument_of_perigee,
            eccentricity=eccentricity,
            mean_motion=mean_motion
        )
    
    def _calculate_apogee(self, mean_motion: float, eccentricity: float) -> float:
        """Calculate apogee in kilometers from mean motion and eccentricity"""
        n = mean_motion * 2 * pi / self.orbital_constants.SECONDS_IN_DAY
        a = (self.orbital_constants.GM_EARTH / (n ** 2)) ** (1/3)
        apogee_m = a * (1 + eccentricity) - self.orbital_constants.EARTH_RADIUS_M
        return apogee_m / 1000  # Convert to km


