from sklearn.cluster import AgglomerativeClustering
import numpy as np

class AgglomerativeClustererWrapper:

    def __init__(self):
        self.linkage = 'complete'
    def run(self, distance_matrix):
        distance_threshold = np.percentile(distance_matrix, 90)
        
        model = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage=self.linkage,
            distance_threshold=distance_threshold
        )
        
        return model.fit_predict(distance_matrix)
