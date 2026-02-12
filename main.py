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
    # app.run_experiment()



    """
        mode:
        - "frechet_single": one cluster + Frechet orbit + t-SNE/CZML
        - "frechet_all": run Frechet optimiser over all clusters (no plots)
        - "max_separation_single": one cluster + maximally separated orbit + t-SNE (png and html)
        - "max_separation_all": run max_separation optimiser over all clusters + performance plot
    """    

    syn_orbits = SyntheticOrbits(cluster_config)
    syn_orbits.run_orbit_generator(mode="max_separation_single")
    
    # analysis = Analysis()
    # analysis.plot_variance_from_existing_frechet()
    
    