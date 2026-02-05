from app import SatelliteClusteringApp
from configs import ClusterConfig
from unique_orbits.uct_fitting.synthetic_orbits import SyntheticOrbits
from sparse_orbits import SparseOrbits
from metrics.analysis import Analysis

if __name__ == "__main__":

    cluster_config = ClusterConfig(
        apogee_range=(300, 700),
        # inclination_range=(80, 100),
        damping=0.95,
    )

    # sparse = SparseOrbits()
    # sparse.graph_tsne_with_isolated()
    
    # app = SatelliteClusteringApp(cluster_config)
    # # app.supervised_clustering()
    # app.run_experiment()
    
    syn_orbits = SyntheticOrbits(cluster_config)
    syn_orbits.run_void_all_size_15()
    
    # analysis = Analysis()
    # analysis.plot_variance_from_existing_frechet()
    
    
    