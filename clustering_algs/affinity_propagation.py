from sklearn.cluster import AffinityPropagation
import numpy as np

class AffinityPropagationWrapper:

    def __init__(self):
        self.damping = 0.95

    def run(self, distance_matrix: np.ndarray) -> np.ndarray:
        print("Running Affinity Propagation...")

        normalizer = 2 * (np.std(distance_matrix) ** 2)
        similarity_matrix = np.exp(-distance_matrix / normalizer)
        
        model = AffinityPropagation(
            affinity='precomputed',
            damping=self.damping,
            max_iter=500,
            convergence_iter=15,
            random_state=42
        )

        model.fit(similarity_matrix)
        return model.labels_
    
    def run_X(self, X: np.ndarray) -> np.ndarray:
        print("Running Affinity Propagation on raw data...")

        model = AffinityPropagation(
            damping=self.damping,
            max_iter=500,
            convergence_iter=15,
            random_state=42
        )

        model.fit(X)
        return model.labels_
