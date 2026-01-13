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
        
        # 1. Build similarity matrix
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
        print("Running Affinity Propagation (preference sweep)...")
        
        # 1. Build similarity matrix
        normaliser = np.std(distance_matrix) ** 2
        similarity_matrix = (-distance_matrix / normaliser)

        # 2. Preference sweep (lower = more clusters)
        pref_min = np.min(similarity_matrix)
        pref_med = np.median(similarity_matrix)
        preferences = np.linspace(pref_min, pref_med, 25)

        best_score = -1
        best_labels = None
        best_info = None

        for pref in tqdm(preferences, desc="Testing preferences"):
            model = AffinityPropagation(
                affinity='precomputed',
                damping=self.damping,
                preference=pref,
                max_iter=1000,
                random_state=42
            )

            labels = model.fit_predict(similarity_matrix)
            n_clusters = len(np.unique(labels))

            # Skip degenerate solutions
            if n_clusters < 2:
                continue

            score = silhouette_score(
                distance_matrix,
                labels,
                metric='precomputed'
            )

            if score > best_score:
                best_score = score
                best_labels = labels
                best_info = (pref, n_clusters, score)

        if best_labels is None:
            raise RuntimeError("Affinity Propagation failed to find a valid clustering")

        pref, k, score = best_info
        print(f"Selected preference={pref:.4f}, clusters={k}, silhouette={score:.3f}")

        return best_labels
