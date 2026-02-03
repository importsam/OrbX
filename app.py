import os
import sys
from pathlib import Path

from metrics import analysis
import numpy as np
import pandas as pd
import pickle
        
from clustering_algs.cluster_wrapper import ClusterWrapper
from configs import ClusterConfig, OrbitalConstants, PathConfig
from graph import Grapher
from models import ClusterResult
from tle_parser import TLEParser
from tools.density_estimation import DensityEstimator
from tools.distance_matrix import get_distance_matrix
from models import ClusterResult
update_cesium_assets_path = Path(__file__).parent / "update_cesium_assets"
sys.path.append(str(update_cesium_assets_path))
sys.path.append(str(update_cesium_assets_path / "live"))
from build_czml import build_czml
from ionop_czml import ionop_czml
from metrics.analysis import Analysis
from sklearn.cluster import HDBSCAN
from tools.density_estimation import DensityEstimator
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
        self.analysis = Analysis()

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

        self.process_post_clustering(cluster_result_dict, df, distance_matrix)
        self.analysis_graphs(cluster_result_dict, df, distance_matrix)

        return None
        
    def process_post_clustering(self, cluster_result_dict, df, distance_matrix):
        optics_results = cluster_result_dict["optics_results"]
        dbscan_results = cluster_result_dict["dbscan_results"]
        hdbscan_results = cluster_result_dict["hdbscan_results"]

        results_list = [
            ("OPTICS", optics_results),
            ("DBSCAN", dbscan_results),
            ("HDBSCAN", hdbscan_results),
        ]

        valid_results = [(name, result) for name, result in results_list if result is not None]
        # ... print DBCV ranking ...

        # Save CSVs as before (size-ranked)
        for name, result in valid_results:
            self.save_cluster_characterisation(
                df=df.copy(),
                result=result,
                out_path=f"data/cluster_characterisation_{name}.csv",
                top_k=50,
            )

        # HDBSCAN only: top 20 by mean density
        if hdbscan_results is None:
            print("\nNo HDBSCAN result available for density ranking.")
            return

        # Full per-cluster stats (unsorted)
        hdbscan_clusters_df = self.save_cluster_characterisation(
            df=df.copy(),
            result=hdbscan_results,
            out_path="data/cluster_characterisation_HDBSCAN.csv",
            # top_k=50,
        )

        print("\nTop 20 HDBSCAN clusters ranked by mean density:")
        hdbscan_top20 = hdbscan_clusters_df.sort_values(
            by="Mean Density", ascending=False
        ).head(20)

        for _, row in hdbscan_top20.iterrows():
            print(
                f"Cluster {int(row['Cluster ID'])} | "
                f"Tier={row['Tier']} | "
                f"Size={int(row['Size'])} | "
                f"Mean density={row['Mean Density']:.6e} | "
                f"Alt range={row['Min Altitude (km)']:.1f}-{row['Max Altitude (km)']:.1f} km"
            )

    def save_cluster_characterisation(
        self,
        df: pd.DataFrame,
        result: ClusterResult,
        out_path: str = "data/cluster_characterisation.csv",
        top_k: int = None,
    ):
        labels = result.labels

        if len(df) != len(labels):
            raise ValueError(
                f"Label / dataframe mismatch: df={len(df)}, labels={len(labels)}"
            )

        df = df.copy()
        df["cluster_id"] = labels

        # Drop noise
        df = df[df["cluster_id"] != -1]

        # df["density"] is already present from run_experiment

        cluster_rows = []
        for cluster_id, g in df.groupby("cluster_id"):
            size = len(g)
            tier = self.cluster_tier(size)
            if tier == "Ignore":
                continue

            altitude_min = g["apogee"].min()
            altitude_max = g["apogee"].max()

            cluster_rows.append(
                {
                    "Cluster ID": int(cluster_id),
                    "Tier": tier,
                    "Size": size,
                    "Min Altitude (km)": altitude_min,
                    "Max Altitude (km)": altitude_max,
                    "Mean Density": g["density"].mean(),
                }
            )

        if not cluster_rows:
            raise RuntimeError("No valid clusters found for characterisation")

        clusters_df = pd.DataFrame(cluster_rows)

        # CSV stays as you had it (size-ranked)
        if top_k is None:
            top_k = len(clusters_df)
        clusters_df_csv = clusters_df.sort_values(by="Size", ascending=False).head(top_k)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        clusters_df_csv.to_csv(out_path, index=False)

        print(f"\nSaved cluster characterisation → {out_path}")
        print(f"Clusters saved: {len(clusters_df_csv)}")

        # Return the full unsorted frame for further in-memory use
        return clusters_df

    def analysis_graphs(self, cluster_result_dict, df, distance_matrix):

        hdbscan_result = cluster_result_dict["hdbscan_results"]

        hdb_sd = self.analysis.cluster_mean_densities(
            hdbscan_result.labels,
            df["density"].to_numpy(),
        )

        # Build DataFrame from exactly what is plotted
        hdb_df = pd.DataFrame({
            "Cluster ID": hdb_sd["cluster_ids"],
            "Size": hdb_sd["cluster_sizes"],
            "Mean Density": hdb_sd["cluster_mean_density"],
        })

        # Find all clusters with mean density >= 8e-11
        high = hdb_df[hdb_df["Mean Density"] >= 8e-11].sort_values(
            by="Mean Density", ascending=False
        )

        print("\nAll HDBSCAN clusters with mean density >= 8e-11:")
        for _, row in high.iterrows():
            print(
                f"Cluster {int(row['Cluster ID'])} | "
                f"Size={int(row['Size'])} | "
                f"Mean density={row['Mean Density']:.6e}"
            )

        size_density_dict = {
            "HDBSCAN": (hdb_sd["cluster_sizes"], hdb_sd["cluster_mean_density"]),
        }
        
        # Print top 10 clusters by size
        print("\nTop 10 HDBSCAN clusters by size:")
        top_10 = hdb_df.nlargest(10, "Size")
        for _, row in top_10.iterrows():
            print(
                f"Cluster {int(row['Cluster ID'])} | "
                f"Size={int(row['Size'])} | "
                f"Mean density={row['Mean Density']:.6e}"
            )
        
        self.analysis.plot_size_vs_density(size_density_dict)

        # dbscan_result = cluster_result_dict["dbscan_results"]
        # hdbscan_result = cluster_result_dict["hdbscan_results"]
        # optics_result = cluster_result_dict["optics_results"]

        # dbscan_stats = self.analysis.cluster_size_summary(dbscan_result.labels)
        # hdbscan_stats = self.analysis.cluster_size_summary(hdbscan_result.labels)
        # optics_stats = self.analysis.cluster_size_summary(optics_result.labels)

        # sizes_dict = {
        #     "DBSCAN": dbscan_stats["sizes"],
        #     "HDBSCAN": hdbscan_stats["sizes"],
        #     "OPTICS": optics_stats["sizes"],
        # }

        # self.analysis.plot_cluster_size_distributions(sizes_dict, log_x=True)

    def cluster_tier(self, size: int) -> str:
        if size >= 100:
            return "Mega"
        elif size >= 20:
            return "Major"
        elif size >= 5:
            return "Minor"
        elif size >= 2:
            return "Micro"
        else:
            return "Ignore"

    def run_graphs(self):
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
        # results_dict = self.cluster_wrapper.run_all_optimizer(
        #     distance_matrix.copy(), orbit_points.copy()
        # )
        # affinity_labels = labels_dict["affinity"]
        
        hdbscan_results = pickle.load(open("data/cluster_results/hdbscan_obj.pkl", "rb"))
        hdbscan_labels = hdbscan_results.labels
        optics_results = pickle.load(open("data/cluster_results/optics_obj.pkl", "rb"))
        optics_labels = optics_results.labels
        dbscan_results = pickle.load(open("data/cluster_results/dbscan_obj.pkl", "rb"))
        dbscan_labels = dbscan_results.labels
        
        

        # plot tsne graphs
        # self.graph.plot_tsne(orbit_points, df, labels=affinity_labels, name="affinity")
        self.graph.plot_tsne(orbit_points, df, labels=optics_labels, name="OPTICS")
        self.graph.plot_tsne(orbit_points, df, labels=dbscan_labels, name="DBSCAN")
        self.graph.plot_tsne(orbit_points, df, labels=hdbscan_labels, name="HDBSCAN")

        # plot UMAP graphs
        # self.graph.plot_umap(orbit_points, df, labels=affinity_labels, name="affinity")
        # self.graph.plot_umap(orbit_points, df, labels=optics_labels, name="optics")
        # self.graph.plot_umap(orbit_points, df, labels=dbscan_labels, name="dbscan")
        # self.graph.plot_umap(orbit_points, df, labels=hdbscan_labels, name="hdbscan")

        # Plot clusters in apogee/inclination space
        # df_opt = df.copy()
        # df_opt["label"] = optics_labels
        # self.graph.plot_clusters(
        #     df_opt, self.path_config.output_plot / "optics_clusters.html"
        # )

        # # now for affinity
        # df_aff = df.copy()
        # df_aff["label"] = affinity_labels
        # self.graph.plot_clusters(
        #     df_aff, self.path_config.output_plot / "affinity_clusters.html"
        # )

        # # now for dbscan
        # df_db = df.copy()
        # df_db["label"] = dbscan_labels
        # self.graph.plot_clusters(
        #     df_db, self.path_config.output_plot / "dbscan_clusters.html"
        # )

        # df_hdb = df.copy()
        # df_hdb["label"] = hdbscan_labels
        # self.graph.plot_clusters(
        #     df_hdb, self.path_config.output_plot / "hdbscan_clusters.html"
        # )

        # Generate CZML for Cesium visualization
        # print("\nGenerating CZML for Cesium visualization...")
        # self.run_cesium(df.copy(), distance_matrix.copy())

    def run_graphs_supervised(self):
        
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
        hdbscan_labels = self.cluster_wrapper.run_hdbscan(distance_matrix, orbit_points)

        # if you want to run optimzation for each alg and then graph
        # results_dict = self.cluster_wrapper.run_all_optimizer(
        #     distance_matrix.copy(), orbit_points.copy()
        # )
        
        # affinity_labels = labels_dict["affinity"]
        # optics_labels = labels_dict["optics"]
        # dbscan_labels = labels_dict["dbscan"]
        # hdbscan_result = results_dict["hdbscan_results"]
        # hdbscan_labels = hdbscan_result.labels

        # plot tsne graphs
        # self.graph.plot_tsne(orbit_points, df, labels=affinity_labels, name="affinity")
        # self.graph.plot_tsne(orbit_points, df, labels=optics_labels, name="optics")
        # self.graph.plot_tsne(orbit_points, df, labels=dbscan_labels, name="dbscan")
        
        print("Starting HDBSCAN supervised TSNE...")
        self.graph.plot_tsne_supervised(orbit_points, df, labels=hdbscan_labels, name="hdbscan")

        # plot UMAP graphs
        # self.graph.plot_umap(orbit_points, df, labels=affinity_labels, name="affinity")
        # self.graph.plot_umap(orbit_points, df, labels=optics_labels, name="optics")
        # self.graph.plot_umap(orbit_points, df, labels=dbscan_labels, name="dbscan")
        # self.graph.plot_umap(orbit_points, df, labels=hdbscan_labels, name="hdbscan")

        # Plot clusters in apogee/inclination space
        # df_opt = df.copy()
        # df_opt["label"] = optics_labels
        # self.graph.plot_clusters(
        #     df_opt, self.path_config.output_plot / "optics_clusters.html"
        # )

        # # now for affinity
        # df_aff = df.copy()
        # df_aff["label"] = affinity_labels
        # self.graph.plot_clusters(
        #     df_aff, self.path_config.output_plot / "affinity_clusters.html"
        # )

        # # now for dbscan
        # df_db = df.copy()
        # df_db["label"] = dbscan_labels
        # self.graph.plot_clusters(
        #     df_db, self.path_config.output_plot / "dbscan_clusters.html"
        # )

        df_hdb = df.copy()
        df_hdb["label"] = hdbscan_labels
        self.graph.plot_clusters(
            df_hdb, self.path_config.output_plot / "hdbscan_clusters.html"
        )

        # Generate CZML for Cesium visualization
        # print("\nGenerating CZML for Cesium visualization...")
        # self.run_cesium(df.copy(), distance_matrix.copy())

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
            print(f"Starting with {len(df)} satellites from TLE parser")
            # filter by inclination and apogee range
            df = df[
                (df["inclination"] >= self.cluster_config.inclination_range[0])
                & (df["inclination"] <= self.cluster_config.inclination_range[1])
                & (df["apogee"] >= self.cluster_config.apogee_range[0])
                & (df["apogee"] <= self.cluster_config.apogee_range[1])
            ].copy()
            print(
                f"After filtering: {len(df)} satellites in range - inc: {self.cluster_config.inclination_range}, apogee: {self.cluster_config.apogee_range}"
            )
        else:
            print(f"Received dataframe with {len(df)} satellites")

        # Get or compute distance matrix if needed
        if distance_matrix is None:
            distance_matrix, key = get_distance_matrix(df.copy())
            df = self._reorder_dataframe(df.copy(), key.copy())
        else:
            # If distance_matrix provided, create key from the current df to ensure they match
            _, key = get_distance_matrix(df.copy())
            df = self._reorder_dataframe(df.copy(), key.copy())

        if "density" not in df.columns:
            df = self.density_estimator.assign_density(
                df.copy(), distance_matrix.copy()
            )

        if "name" not in df.columns:
            df["name"] = df["satNo"]

        orbit_points = self.get_points(df.copy())

        def assign_cluster_labels(
            df: pd.DataFrame, labels: np.ndarray, label_name: str = "cluster"
        ) -> pd.DataFrame:
            df = df.copy()
            df[label_name] = labels
            return df

        # SELECT CLUSTERING ALGORITHM HERE  
        labels = self.cluster_wrapper.run_optics(distance_matrix.copy(), orbit_points.copy())
        df = assign_cluster_labels(df, labels)

        # KEEP ONLY TOP AND BOTTOM 5 CLUSTERS BY SIZE, exlcuding noise
        cluster_sizes = df[df["cluster"] != -1]["cluster"].value_counts()
        top_clusters = cluster_sizes.head(5).index.tolist()
        bottom_clusters = cluster_sizes.tail(5).index.tolist()
        selected_clusters = top_clusters + bottom_clusters
        df = df[df["cluster"].isin(selected_clusters)].copy()
        
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        def cluster_colors(labels, cmap_name="tab20"):
            unique = sorted(set(labels))
            cmap = cm.get_cmap(cmap_name, len(unique))

            color_map = {}
            for i, lab in enumerate(unique):
                if lab == -1:
                    color_map[lab] = [150, 150, 150, 255]  # noise = gray
                else:
                    rgba = cmap(i)
                    color_map[lab] = [
                        int(rgba[0] * 255),
                        int(rgba[1] * 255),
                        int(rgba[2] * 255),
                        255,
                    ]
            return color_map

        color_map = cluster_colors(df["cluster"])

        df["cluster_color"] = df["cluster"].map(color_map)
        df[["orbit_colour_r", "orbit_colour_g", "orbit_colour_b", "orbit_colour_a"]] = (
            pd.DataFrame(df["cluster_color"].tolist(), index=df.index)
        )

        # only keep the first couple of clusters

        # df = df[df['cluster'].isin([0, 1, 2])]
        # Build CZML file
        build_czml(df)

        ACCESSTOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI5OTMwYjJlMS0yYjBhLTQwMmMtYjJkZi1mZWZiY2RiYTNmN2UiLCJpZCI6MjQwODIwLCJpYXQiOjE3MzgzMDM2ODl9.h1pXOgujWRPoS6ZFc5wL-l5_XJnSyUsPZym3ssZj7TQ"

        try:
            ionop_czml(ACCESSTOKEN)

        except Exception as e:
            print(f"Error: {e}")

        print(
            f"CZML file generated successfully at update_cesium_assets/live/data/output.czml"
        )

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

    def save_obj(self):
        """ This will run everything per normal, then save the label data from each alg as a pkl
        file in data/ for quicker retrieval later.
        """
        
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
        
        # load labels from pickle files
        with open("data/cluster_results/dbscan_labels.pkl", "rb") as f:
            dbscan_labels = pickle.load(f)
        
        dbscan_obj = ClusterResult(
            dbscan_labels,
            len(set(dbscan_labels)),
            (dbscan_labels == -1).sum(),
            self.cluster_wrapper.quality_metrics.dbcv_score_wrapper(orbit_points, dbscan_labels),
            self.cluster_wrapper.quality_metrics.s_dbw_score_wrapper(orbit_points, dbscan_labels)
        )
        
        with open("data/cluster_results/hdbscan_labels.pkl", "rb") as f:
            hdbscan_labels = pickle.load(f)
        
        hdbscan_obj = ClusterResult(
            hdbscan_labels,
            len(set(hdbscan_labels)),
            (hdbscan_labels == -1).sum(),
            self.cluster_wrapper.quality_metrics.dbcv_score_wrapper(orbit_points, hdbscan_labels),
            self.cluster_wrapper.quality_metrics.s_dbw_score_wrapper(orbit_points, hdbscan_labels)
        )

        with open("data/cluster_results/optics_labels.pkl", "rb") as f:
            optics_labels = pickle.load(f)
        
        optics_obj = ClusterResult(
            optics_labels,
            len(set(optics_labels)),
            (optics_labels == -1).sum(),
            self.cluster_wrapper.quality_metrics.dbcv_score_wrapper(orbit_points, optics_labels),
            self.cluster_wrapper.quality_metrics.s_dbw_score_wrapper(orbit_points, optics_labels)
        )
        
        with open("data/cluster_results/dbscan_obj.pkl", "wb") as f:
            pickle.dump(dbscan_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open("data/cluster_results/hdbscan_obj.pkl", "wb") as f:
            pickle.dump(hdbscan_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open("data/cluster_results/optics_obj.pkl", "wb") as f:
            pickle.dump(optics_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                
    def data_labels_csv(self, X):
        """
        This will save the orbits as points (X) as a csv, along side the labels from each alg.
        This is for the stability analysis, these csvs will be read in by an R script.
        """
        
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
        
        # load in the objects 
        dbscan_results = pickle.load(open("data/cluster_results/dbscan_obj.pkl", "rb"))
        optics_results = pickle.load(open("data/cluster_results/optics_obj.pkl", "rb"))
        hdbscan_results = pickle.load(open("data/cluster_results/hdbscan_obj.pkl", "rb"))
        
        labels_dbscan = dbscan_results.labels
        labels_optics = optics_results.labels
        labels_hdb = hdbscan_results.labels
        
        pd.DataFrame(X).to_csv("X.csv", index=False)
        pd.DataFrame({"hdbscan": labels_hdb,
                    "optics": labels_optics,
                    "dbscan": labels_dbscan}).to_csv("data/analysis/stability_pairs/labels.csv", index=False)
        
    def bootstrap_cluster_stability(self):
        """
        Perform bootstrap resampling to assess cluster stability.

        Saves bootstrap clustering results to CSV files in 'stability_pairs' directory.
        Each file contains two columns: 'ref' (reference labels) and 'boot' (bootstrap labels).
        """
        
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
        X = self.get_points(df)
        df = self._reorder_dataframe(df, key)
        
        hdbscan_results = pickle.load(open("data/cluster_results/hdbscan_obj.pkl", "rb"))

        rng = np.random.default_rng(42)
        B = 100
        sample_frac = 0.7

        # X: (n, d) from your pipeline
        # distance_matrix: (n, n) used for clustering
        # hdbscan_clusterer: your configured HDBSCAN object
        labels_ref = hdbscan_results.labels
        n = X.shape[0]

        out_dir = Path("data/analysis/stability_pairs")
        out_dir.mkdir(exist_ok=True)

        for b in range(B):
            idx = rng.choice(n, size=int(sample_frac * n), replace=False)
            X_b = X[idx]
            D_b = distance_matrix[np.ix_(idx, idx)]

            labels_boot = HDBSCAN(
                min_cluster_size=2,
                min_samples=3,
                metric="precomputed",
            ).fit_predict(D_b)

            df = pd.DataFrame({
                "ref": labels_ref[idx],
                "boot": labels_boot
            })
            
            df.to_csv(out_dir / f"hdbscan_boot_{b:03d}.csv", index=False)
      
    def load_data(self):
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
            
    def supervised_clustering(self):
        """ 
        Supervised validation: check whether Starlink shell-1 satellites
        (≈53° / 550 km) that are in the same orbital planes (RAAN bins)
        are also clustered together by HDBSCAN.
        """        

        data_dict = self.load_data()
        df = data_dict["orbit_df"].copy()
        hdbscan_obj = data_dict["hdbscan_obj"]

        labels_hdb = hdbscan_obj.labels
        if len(labels_hdb) != len(df):
            raise ValueError(
                f"HDBSCAN label / dataframe mismatch: df={len(df)}, labels={len(labels_hdb)}"
            )
        df["hdb_label"] = labels_hdb

        df["is_starlink"] = df["name"].str.startswith("STARLINK")
        df["alt_km"] = df["apogee"]

        df["shell"] = [
            self.classify_shell(i, h)
            for i, h in zip(df["inclination"].to_numpy(), df["alt_km"].to_numpy())
        ]

        shell1 = df[
            (df["is_starlink"]) &
            (df["shell"] == "Starlink_shell_53deg_550km") &
            (df["hdb_label"] != -1)
        ].copy()

        if shell1.empty:
            print("No Starlink shell-1 satellites found in current filter.")
            return

        print(f"Found {len(shell1)} Starlink shell-1 sats (53° / ~550 km, non-noise).")

        # ---- plane IDs from RAAN bins ----
        bin_width = 5.0
        bins = np.arange(0.0, 360.0 + bin_width, bin_width)
        shell1["raan_bin"] = np.digitize(shell1["raan"].to_numpy(), bins)
        shell1["plane_id"] = shell1["raan_bin"] - 1  # 0-based

        # ---- plane_summary ----
        plane_summary_rows = []
        for plane_id, g in shell1.groupby("plane_id"):
            plane_size = len(g)
            clusters = g["hdb_label"].to_numpy()
            unique_clusters, counts = np.unique(clusters, return_counts=True)
            max_cluster_sats = int(counts.max())
            plane_summary_rows.append({
                "plane_id": int(plane_id),
                "n_sats": plane_size,
                "n_clusters": len(unique_clusters),
                "cluster_ids": list(np.sort(unique_clusters)),
                "max_cluster_sats": max_cluster_sats,
                "max_cluster_frac": max_cluster_sats / plane_size,
            })

        plane_summary = pd.DataFrame(plane_summary_rows).sort_values(
            by="n_sats", ascending=False
        )

        print("\nStarlink shell-1 planes (RAAN bins) and their HDBSCAN cluster counts:")
        for _, row in plane_summary.iterrows():
            print(
                f"Plane {row['plane_id']:2d} | "
                f"N_sats={row['n_sats']:3d} | "
                f"N_clusters={row['n_clusters']:2d} | "
                f"max_cluster_frac={row['max_cluster_frac']:.3f} | "
                f"Clusters={row['cluster_ids']}"
            )

        # ---- cluster_summary ----
        cluster_summary_rows = []
        for cluster_id, g in shell1.groupby("hdb_label"):
            n_sats = len(g)
            planes = g["plane_id"].to_numpy()
            unique_planes, counts = np.unique(planes, return_counts=True)
            max_plane_sats = int(counts.max())
            cluster_summary_rows.append({
                "cluster_id": int(cluster_id),
                "n_sats": n_sats,
                "n_planes": len(unique_planes),
                "plane_ids": list(np.sort(unique_planes)),
                "max_plane_sats": max_plane_sats,
                "max_plane_frac": max_plane_sats / n_sats,
                "mean_inc": float(g["inclination"].mean()),
                "mean_alt": float(g["alt_km"].mean()),
            })

        cluster_summary = pd.DataFrame(cluster_summary_rows).sort_values(
            by="n_sats", ascending=False
        )

        print("\nHDBSCAN clusters restricted to Starlink shell-1 and their plane spans:")
        for _, row in cluster_summary.iterrows():
            print(
                f"Cluster {row['cluster_id']:4d} | "
                f"N_sats={row['n_sats']:3d} | "
                f"N_planes={row['n_planes']:2d} | "
                f"max_plane_frac={row['max_plane_frac']:.3f} | "
                f"Planes={row['plane_ids']} | "
                f"⟨i⟩={row['mean_inc']:.3f} deg | "
                f"⟨alt⟩={row['mean_alt']:.1f} km"
            )

        # ---- global fractions for your “tightened story” ----

        # 1) “X% of satellites lie in clusters where ≥80% of members share the same RAAN plane.”
        tight_clusters = cluster_summary[cluster_summary["max_plane_frac"] >= 0.8]
        n_sats_in_tight_clusters = int(tight_clusters["n_sats"].sum())
        total_shell1_sats = int(shell1.shape[0])
        frac_sats_in_tight_clusters = n_sats_in_tight_clusters / total_shell1_sats

        print(
            f"\nShell-1 Starlinks: {frac_sats_in_tight_clusters*100:.1f}% of satellites "
            f"({n_sats_in_tight_clusters}/{total_shell1_sats}) lie in HDBSCAN clusters "
            f"where ≥80% of members share the same RAAN plane."
        )

        # 2) “Y of Z RAAN planes, a single HDBSCAN cluster contains ≥70% of that plane’s satellites.”
        plane_good = plane_summary[plane_summary["max_cluster_frac"] >= 0.7]
        Y = int(plane_good.shape[0])
        Z = int(plane_summary.shape[0])

        print(
            f"For Starlink shell-1, {Y} of {Z} RAAN planes "
            f"({Y/Z*100:.1f}%) have at least one HDBSCAN cluster containing "
            f"≥70% of that plane's satellites."
        )

        # ---- save summaries ----
        out_dir = Path("data/analysis/starlink_supervised")
        out_dir.mkdir(parents=True, exist_ok=True)

        plane_summary.to_csv(out_dir / "shell1_plane_vs_hdbscan.csv", index=False)
        cluster_summary.to_csv(out_dir / "shell1_hdbscan_vs_planes.csv", index=False)

        print("\nSaved supervised Starlink shell-1 summaries to:")
        print(f"  {out_dir / 'shell1_plane_vs_hdbscan.csv'}")
        print(f"  {out_dir / 'shell1_hdbscan_vs_planes.csv'}")

        
        
    @staticmethod
    def classify_shell(i_deg: float, alt_km: float) -> str:
        """
        Very simple shell classifier based on inclination and altitude.

        For now we only care about the 53° / ~550 km shell (Starlink 'shell 1/4').
        """
        if 52.5 <= i_deg <= 53.5 and 530.0 <= alt_km <= 570.0:
            return "Starlink_shell_53deg_550km"
        return "other"

        