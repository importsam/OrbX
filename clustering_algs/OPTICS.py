from sklearn.cluster import OPTICS
import numpy as np

class OPTICS:
    def __init__(self, min_samples=5, max_eps=np.inf):
        self.min_samples = min_samples
        self.max_eps = max_eps

    def fit(self, distance_matrix):
        model = OPTICS(
            min_samples=self.min_samples,
            max_eps=self.max_eps,
            metric='precomputed'
        )
        
        labels = model.fit_predict(distance_matrix)
        return labels