from sklearn.cluster import AgglomerativeClustering

class AgglomerativeClusterer:

    def __init__(self, linkage='complete'):
        self.linkage = linkage

    def fit(self, distance_matrix):
        model = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage=self.linkage
        )
        return model.fit_predict(distance_matrix)
