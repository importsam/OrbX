from sklearn.cluster import DBSCAN

class DBSCANClusterer:

    def __init__(self, min_samples_range=range(2, 41)):
        self.min_samples_range = min_samples_range

    def _evaluate(self, distance_matrix, eps, min_samples):
        model = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        labels = model.fit_predict(distance_matrix)

        if len(set(labels)) <= 1 or len(set(labels)) == len(labels):
            return -1, labels

        score = silhouette_score(distance_matrix, labels, metric="precomputed")
        return score, labels

    def _find_optimal_eps(self, distance_matrix, min_samples):
        k = min_samples - 1
        k_distances = []

        for i in range(distance_matrix.shape[0]):
            row = np.sort(distance_matrix[i][distance_matrix[i] != 0])
            k_distances.append(row[k - 1] if len(row) >= k else row[-1])

        k_distances = np.sort(k_distances)
        knee = KneeLocator(
            range(len(k_distances)),
            k_distances,
            curve="convex",
            direction="increasing"
        )

        return k_distances[knee.elbow]

    def fit(self, distance_matrix):
        best_score = -1
        best_labels = None
        best_params = None

        for min_samples in self.min_samples_range:
            eps = self._find_optimal_eps(distance_matrix, min_samples)
            score, labels = self._evaluate(distance_matrix, eps, min_samples)

            if score > best_score:
                best_score = score
                best_labels = labels
                best_params = (eps, min_samples)

        print(f"Best DBSCAN params → eps={best_params[0]:.4f}, min_samples={best_params[1]}")
        print(f"Silhouette score: {best_score:.4f}")

        return best_labels, best_score
