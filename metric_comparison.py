import numpy as np 
import pandas as pd 
import pickle 
import os
from tools.DMT import VectorizedKeplerianOrbit
from tools.distance_matrix import get_distance_matrix
from tle_parser import TLEParser
from app import SatelliteClusteringApp
from configs import ClusterConfig

class MetricComparison:
    def __init__(self):
        self.tle_parser = TLEParser("Space-Track")
        self.app = SatelliteClusteringApp(ClusterConfig)
        
    def load_data(self):
        df = self.tle_parser.df
        df = df[
            (df["inclination"] >= 0)
            & (df["inclination"] <= 180)
            & (df["apogee"] >= 300)
            & (df["apogee"] <= 700)
        ].copy()



        # Get or compute the distance matrix
        distance_matrix, key = get_distance_matrix(df)

        # Reorder df to match distance_matrix
        df = self.app._reorder_dataframe(df, key)
        
        return df
    
    @staticmethod
    def keplerian_from_tle(df: pd.DataFrame) -> np.ndarray:
        """
        Return an array of shape (N, 6) with [a, e, i, omega, raan, M] for each row.
        Here M is set to 0 (or you can extract mean anomaly if available).
        """
        line1_array = df["line1"].to_numpy()
        line2_array = df["line2"].to_numpy()
        orbit = VectorizedKeplerianOrbit(line1_array, line2_array)

        # a [km], e [-], i/omega/raan [rad], M = 0 as placeholder
        keps = np.column_stack([
            orbit.a,
            orbit.e,
            orbit.i,
            orbit.omega,
            orbit.raan,
            np.zeros_like(orbit.a),
        ])
        return keps
        
    def main(self):
        
        df = self.load_data()
        
        # load in cluster labels 
        hdbscan_labels = pickle.load(open("data/cluster_results/hdbscan_labels.pkl", "rb"))
        
        # take 3 orbits from same cluster
        df['labels'] = hdbscan_labels
        
        # pick a random cluster with at least 3 members
        cluster_sizes = df['labels'].value_counts()
        valid_clusters = cluster_sizes[cluster_sizes >= 3].index.tolist()
        chosen_cluster = np.random.choice(valid_clusters)
        
        cluster_df = df[df['labels'] == chosen_cluster].reset_index(drop=True)
        
        selected_orbits = cluster_df.sample(n=3, random_state=42).reset_index(drop=True)
        
    def dmt_pairwise_D(self, df: pd.DataFrame) -> np.ndarray:
        line1 = df["line1"].values
        line2 = df["line2"].values
        orbits = VectorizedKeplerianOrbit(line1, line2)
        D = VectorizedKeplerianOrbit.DistanceMetric(orbits, orbits)
        return np.asarray(D)
        
        
    @staticmethod
    def pairwise_euclidean(points: np.ndarray) -> np.ndarray:

        
        
        return D
            
            
    def main(self,
            labels_path="data/cluster_results/hdbscan_obj.pkl",
            n_sample_orbits=3,
            out_dir="data/metric_comparison"):
        os.makedirs(out_dir, exist_ok=True)

        # 1) load filtered df
        df = self.load_data()

        # 2) load HDBSCAN object and attach labels
        with open(labels_path, "rb") as f:
            hdbscan_obj = pickle.load(f)

        # depending on how it was saved, use .labels_ or .labels
        if hasattr(hdbscan_obj, "labels_"):
            labels = hdbscan_obj.labels_
        else:
            labels = hdbscan_obj.labels

        if len(labels) != len(df):
            raise ValueError("Length of HDBSCAN labels does not match dataframe length.")

        df["label"] = labels

        # 3) choose one cluster with at least n_sample_orbits orbits
        cluster_sizes = df["label"].value_counts()
        valid_clusters = cluster_sizes[cluster_sizes >= n_sample_orbits].index.tolist()
        if not valid_clusters:
            raise ValueError(f"No cluster with at least {n_sample_orbits} members.")

        chosen_cluster = np.random.choice(valid_clusters)
        print(f"Chosen cluster label: {chosen_cluster}")

        cluster_df = df[df["label"] == chosen_cluster].reset_index(drop=True)

        # 4) select n_sample_orbits orbits from this cluster
        selected_df = cluster_df.sample(n=n_sample_orbits, random_state=42).reset_index(drop=True)
        print(f"Selected satNos: {selected_df['satNo'].tolist()}")

        # 5) build distance matrices
        keps = self.keplerian_from_tle(selected_df)
        D_kep = self.pairwise_euclidean(keps)
        D_dmt = self.dmt_pairwise_D(selected_df)

        # 6) write CSVs
        N = len(selected_df)
        rows_kep, rows_dmt = [], []
        for i in range(N):
            for j in range(N):
                rows_kep.append({
                    "i": i,
                    "j": j,
                    "satNo_i": selected_df.loc[i, "satNo"],
                    "satNo_j": selected_df.loc[j, "satNo"],
                    "distance": D_kep[i, j],
                })
                rows_dmt.append({
                    "i": i,
                    "j": j,
                    "satNo_i": selected_df.loc[i, "satNo"],
                    "satNo_j": selected_df.loc[j, "satNo"],
                    "distance": D_dmt[i, j],
                })

        df_kep = pd.DataFrame(rows_kep)
        df_dmt = pd.DataFrame(rows_dmt)

        kep_path = os.path.join("data/pairwise_distances_keplerian.csv")
        dmt_path = os.path.join("data/pairwise_distances_dmt.csv")

        df_kep.to_csv(kep_path, index=False)
        df_dmt.to_csv(dmt_path, index=False)

        print(f"Wrote Keplerian distance CSV to {kep_path}")
        print(f"Wrote DMT distance CSV to {dmt_path}")

        return df_kep, df_dmt

    
    
if __name__ == "__main__":
    mc = MetricComparison()
    mc.main() 
    
