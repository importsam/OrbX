from app import SatelliteClusteringApp
from configs import ClusterConfig
from unique_orbits.uct_fitting.synthetic_orbits import SyntheticOrbits

if __name__ == "__main__":

    cluster_config = ClusterConfig(
        apogee_range=(300, 700),
        # inclination_range=(80, 100),
        damping=0.95,
    )
    
    # app = SatelliteClusteringApp(cluster_config)
    # app.run_experiment()
    
    syn_orbits = SyntheticOrbits(cluster_config)
    syn_orbits.run_orbit_generator()