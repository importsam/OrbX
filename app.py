import os
import sys
from pathlib import Path

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
        
        self.process_post_clustering(cluster_result_dict, df)
        
        self.analysis_graphs(cluster_result_dict)

        return None

    def process_post_clustering(self, cluster_result_dict, df):
        """
        Process the clustering results from each algorithm and prepare for visualization.

        Args:
            cluster_result_dict: dict of ClusterResult objects from clustering algorithms
        """

        # affinity_results = cluster_result_dict["affinity_results"]
        optics_results = cluster_result_dict["optics_results"]
        dbscan_results = cluster_result_dict["dbscan_results"]
        hdbscan_results = cluster_result_dict["hdbscan_results"]

        # rank based on DBCV score
        results_list = [
            # ("Affinity Propagation", affinity_results),
            ("OPTICS", optics_results),
            ("DBSCAN", dbscan_results),
            ("HDBSCAN", hdbscan_results),
        ]

        valid_results = [
            (name, result) for name, result in results_list if result is not None
        ]

        if not valid_results:
            raise RuntimeError("No clustering algorithm produced a valid result")

        valid_results.sort(key=lambda x: x[1].dbcv_score, reverse=True)

        print("\nClustering Results Ranked by DBCV Score:")

        for name, result in valid_results:
            print(
                f"{name}: "
                f"Clusters={result.n_clusters}, "
                f"Noise={result.n_noise}, "
                f"DBCV Score={result.dbcv_score:.4f}, "
                f"S_Dbw Score={result.s_Dbw_score:.4f}"
            )

        skipped = [name for name, result in results_list if result is None]
        if skipped:
            print("\nSkipped algorithms (no acceptable clustering found):")
            for name in skipped:
                print(f" - {name}")

        """
        What we need this to do now is to characterise what the clusters look like inside. 
        There are four characterisations of clusters:
        Mega cluster - 100+ sats
        Major cluster - 20-100
        Minor cluster - 5-20 sats
        Micro cluster - 2-5 sats
        
        For these, I need it to rank the top 50 and save in a csv the following:
        Cluster ID, Tier, Size, Altitude Range
        """

        for name, result in valid_results:

            self.save_cluster_characterisation(
                df=df,
                result=result,
                out_path=f"data/cluster_characterisation_{name}.csv",
                top_k=50,
            )

    def analysis_graphs(self, cluster_result_dict):
        
        dbscan_result = cluster_result_dict["dbscan_results"]
        hdbscan_result = cluster_result_dict["hdbscan_results"]
        optics_result = cluster_result_dict["optics_results"]

        dbscan_stats = self.analysis.cluster_size_summary(dbscan_result.labels)
        hdbscan_stats = self.analysis.cluster_size_summary(hdbscan_result.labels)
        optics_stats = self.analysis.cluster_size_summary(optics_result.labels)

        sizes_dict = {
            "DBSCAN": dbscan_stats["sizes"],
            "HDBSCAN": hdbscan_stats["sizes"],
            "OPTICS": optics_stats["sizes"],
        }

        self.analysis.plot_cluster_size_distributions(sizes_dict, log_x=True)

    def save_cluster_characterisation(
        self,
        df: pd.DataFrame,
        result: ClusterResult,
        out_path: str = "data/cluster_characterisation.csv",
        top_k: int = 50,
    ):

        labels = result.labels

        if len(df) != len(labels):
            raise ValueError(
                f"Label / dataframe mismatch: df={len(df)}, labels={len(labels)}"
            )

        df = df.copy()
        df["cluster_id"] = labels

        # drop noise
        df = df[df["cluster_id"] != -1]

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
                }
            )

        if not cluster_rows:
            raise RuntimeError("No valid clusters found for characterisation")

        clusters_df = pd.DataFrame(cluster_rows)

        # Rank by size
        clusters_df = clusters_df.sort_values(by="Size", ascending=False).head(top_k)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        clusters_df.to_csv(out_path, index=False)

        print(f"\nSaved cluster characterisation → {out_path}")
        print(f"Clusters saved: {len(clusters_df)}")

        return clusters_df

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