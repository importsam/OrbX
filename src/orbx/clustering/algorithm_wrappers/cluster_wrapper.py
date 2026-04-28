import numpy as np
from .HDBSCANWrapper import HDBSCANWrapper

class ClusterWrapper:
    def __init__(self):
        self.hdbscan = HDBSCANWrapper()
   
    def run_hdbscan(self, distance_matrix: np.ndarray, X: np.ndarray, min_samples: int = 3, min_cluster_size: int = 2) -> np.ndarray:
        labels = self.hdbscan.run(distance_matrix.copy(), X.copy(), min_samples, min_cluster_size)
        print(f"HDBSCAN found {len(set(labels) - {-1})} clusters")

        return labels
