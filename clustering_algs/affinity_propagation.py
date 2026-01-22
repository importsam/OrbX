from enum import unique
from sklearn.cluster import AffinityPropagation
import numpy as np
from sklearn.metrics import silhouette_score
# Add tqdm progress bar for preference sweep
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
from sklearn.cluster import AffinityPropagation
from metrics.quality_metrics import QualityMetrics 
from models import ClusterResult
class AffinityPropagationWrapper:
    
    
    """
        The problem here is that preference needs to be defined more uniquely for each 
        orbit I feel. try using the density score for each as a preference.
    
    """

    def __init__(self):
        self.damping = 0.95
        self.preference = -10
        self.quality_metrics = QualityMetrics()
        
    def run(self, distance_matrix: np.ndarray, X: np.ndarray) -> np.ndarray:
        print("Running Affinity Propagation (preference sweep)...")
        
        # normaliser = np.std(distance_matrix) ** 2
        # similarity_matrix = (-distance_matrix / normaliser)
        
        similarity_matrix = -distance_matrix ** 2

        model = AffinityPropagation(
            affinity='euclidean',
            damping=self.damping,
            preference=self.preference,
            max_iter=500,
            random_state=42
        )
    
        labels = model.fit_predict(X)

        return labels

    def run_pref_optimization(self, distance_matrix: np.ndarray, X: np.ndarray) -> ClusterResult:
        print("Running Affinity Propagation (parallel preference sweep)...")

        similarity_matrix = -distance_matrix ** 2

        """-- this was mega clusters (15 for 2500 points)"""
        pref_min = np.min(similarity_matrix)
        pref_med = np.median(similarity_matrix)
        preferences = np.linspace(pref_min, pref_med, 25)
        
        # similarity_matrix is full (n x n), symmetric with diagonal (self-similarities)
        sim_vals = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]

        p_min = np.percentile(sim_vals, 5)   # quite low
        p_med = np.median(sim_vals)
        p_max = np.percentile(sim_vals, 95)  # quite high

        # To encourage *more* clusters, stay toward the lower end
        preferences = np.linspace(p_min - (p_med - p_min), p_med, 40)


        results = Parallel(
            n_jobs=2,          # use all CPU cores
            backend="loky"      # safe for sklearn
        )(
            delayed(_test_preference)(
                pref, similarity_matrix, X, self.damping, self.quality_metrics
            )
            for pref in preferences
        )

        results = [r for r in results if r is not None]

        if not results:
            print("Affinity Propagation: no acceptable clustering found")
            return None

        # We want the highest score, so we use max()
        best_pref, best_labels, best_k, best_score = max(
            results, key=lambda x: x[3]
        )

        print(
            f"Selected preference={best_pref:.4f}, "
            f"clusters={best_k}, best DBCV score={best_score:.3f}"
        )

        cluster_result_obj = ClusterResult(best_labels, len(set(best_labels)), 
                                           (best_labels == -1).sum(), best_score,
                                           self.quality_metrics.s_dbw_score_wrapper(X, best_labels))

        return cluster_result_obj

def _test_preference(pref, similarity_matrix, X, damping, quality_metrics):
    
    model = AffinityPropagation(
        affinity='precomputed',
        damping=damping,
        preference=pref,
        max_iter=1000,
        random_state=42
    )

    labels = model.fit_predict(similarity_matrix)
    n_clusters = len(np.unique(labels))

    acceptance = QualityMetrics.is_clustering_acceptable(labels.copy())
    
    if not acceptance["acceptable"]:
        print(
            f"Rejected ({acceptance['fail_reasons']})"
        )
        return None

    if n_clusters < 2:
        print("!!!Affinity Propagation found less than 2 clusters, skipping preference!!!")
        return None

    score = quality_metrics.dbcv_score_wrapper(X, labels)

    return pref, labels, n_clusters, score
