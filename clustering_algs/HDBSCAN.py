from sklearn.cluster import HDBSCAN

class HDBSCANClusterer:

    def __init__(self, min_cluster_size=5):
        self.min_cluster_size = min_cluster_size

    def fit(self, distance_matrix):
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_cluster_size,
            metric='precomputed',
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(distance_matrix)
        return labels
