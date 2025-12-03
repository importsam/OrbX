from app import SatelliteClusteringApp
from configs import ClusterConfig

if __name__ == '__main__':

    cluster_config = ClusterConfig(
        inclination_range=(60, 120),
        apogee_range=(1000, 2000),
        damping=0.95
    )
    
    # nothing is cached
    app = SatelliteClusteringApp(cluster_config)
    app.run()