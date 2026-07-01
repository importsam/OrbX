import numpy as np
import pytest

from orbx.clustering.algorithm_wrappers.HDBSCANWrapper import HDBSCANWrapper


def test_hdbscan_wrapper_raises_when_too_few_clusters():
    n_orbits = 5
    distance_matrix = np.ones((n_orbits, n_orbits)) * 1000.0
    np.fill_diagonal(distance_matrix, 0.0)
    orbit_points = np.zeros((n_orbits, 6))

    with pytest.raises(ValueError, match="HDBSCAN produced 0 cluster"):
        HDBSCANWrapper().fit(
            distance_matrix,
            orbit_points,
            min_samples=5,
            min_cluster_size=5,
        )
