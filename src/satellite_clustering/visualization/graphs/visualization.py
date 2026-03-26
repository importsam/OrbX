

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

        # self.analysis.plot_cluster_size_distributions(sizes_dict, log_x=True)
        self.analysis.plot_hdbscan_cluster_sizes(hdbscan_stats["sizes"], log_x=False, grouped=True)
        
        
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