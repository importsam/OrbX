from app import SatelliteClusteringApp
from configs import ClusterConfig

if __name__ == '__main__':

    cluster_config = ClusterConfig(
        apogee_range=(300, 700),
        inclination_range=(80, 100),
        damping=0.95
    )
    
    # nothing is cached
    app = SatelliteClusteringApp(cluster_config)
    app.run_apo_inc_clustering_graphs()