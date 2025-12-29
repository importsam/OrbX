from sklearn.cluster import AffinityPropagation
import numpy as np

class AffinityPropagation:

    def __init__(self, damping=0.95):
        self.damping = damping

    def fit(self, distance_matrix: np.ndarray):
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
