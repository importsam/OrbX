from sklearn.cluster import AffinityPropagation
import numpy as np
from sklearn.metrics import silhouette_score
# Add tqdm progress bar for preference sweep
from tqdm import tqdm
        
class AffinityPropagationWrapper:

    def __init__(self):
        self.damping = 0.95
        self.preference = -0.0003
        
    def run(self, distance_matrix: np.ndarray) -> np.ndarray:
        print("Running Affinity Propagation (preference sweep)...")
        
        normaliser = np.std(distance_matrix) ** 2
        similarity_matrix = (-distance_matrix / normaliser)

        model = AffinityPropagation(
            affinity='precomputed',
            damping=self.damping,
            preference=self.preference,
            max_iter=500,
            random_state=42
        )

        labels = model.fit_predict(similarity_matrix)

        return labels

    def run_pref_test(self, distance_matrix: np.ndarray) -> np.ndarray:
        print("Running Affinity Propagation (parallel preference sweep)...")

        normaliser = np.std(distance_matrix) ** 2
        similarity_matrix = -distance_matrix / normaliser

        pref_min = np.min(similarity_matrix)
        pref_med = np.median(similarity_matrix)
        preferences = np.linspace(pref_min, pref_med, 25)

        results = Parallel(
            n_jobs=2,          # use all CPU cores
            backend="loky"      # safe for sklearn
        )(
            delayed(_test_preference)(
                pref, similarity_matrix, distance_matrix, self.damping
            )
            for pref in preferences
        )

        results = [r for r in results if r is not None]

        if not results:
            raise RuntimeError("Affinity Propagation failed to find a valid clustering")

        best_pref, best_labels, best_k, best_score = max(
            results, key=lambda x: x[3]
        )

        print(
            f"Selected preference={best_pref:.4f}, "
            f"clusters={best_k}, silhouette={best_score:.3f}"
        )

        return best_labels


from joblib import Parallel, delayed
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score

def _test_preference(pref, similarity_matrix, distance_matrix, damping):
    model = AffinityPropagation(
        affinity='precomputed',
        damping=damping,
        preference=pref,
        max_iter=1000,
        random_state=42
    )

    labels = model.fit_predict(similarity_matrix)
    n_clusters = len(np.unique(labels))

    if n_clusters < 2:
        return None

    score = silhouette_score(
        distance_matrix,
        labels,
        metric='precomputed'
    )

    return pref, labels, n_clusters, score
