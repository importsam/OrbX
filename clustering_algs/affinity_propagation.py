import numpy as np
from sklearn.cluster import AffinityPropagation
from metrics.quality_metrics import QualityMetrics
from models import ClusterResult
from tools.density_estimation import DensityEstimator


class AffinityPropagationWrapper:
    def __init__(self):
        self.damping = 0.95
        self.quality_metrics = QualityMetrics()
        self.density_estimator = DensityEstimator()

    def run(self, distance_matrix: np.ndarray, X: np.ndarray) -> ClusterResult:
        print("Running Affinity Propagation with preference sweep...")

        normalizer = np.std(distance_matrix) ** 2
        similarity_matrix = -distance_matrix / normalizer

        sim_values = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]

        pref_min = sim_values.min()
        pref_med = np.median(sim_values)
        pref_max = sim_values.max()

        # search range: from more negative than median up toward max
        pref_max_safe = np.percentile(sim_values, 95)  # Don't go all the way to max
        pref_grid = np.linspace(pref_min, pref_max_safe, 15)

        best_score = -np.inf
        best_pref = None
        best_labels = None

        for pref in pref_grid:
            model = AffinityPropagation(
                affinity="precomputed",
                damping=self.damping,
                preference=pref,
                max_iter=1000,
                random_state=42,
            )
            labels = model.fit_predict(similarity_matrix)
            n_clusters = len(set(labels))
            
            # Skip if we hit the explosion zone
            if n_clusters > 0.5 * len(distance_matrix):  # More than 50% are exemplars
                print(f"[pref sweep] pref={pref:.3e}, clusters={n_clusters} - SKIPPED (too many)")
                break  # Stop searching higher preferences
            
            print(f"[pref sweep] pref={pref:.3e}, clusters={n_clusters}")

            acceptance = QualityMetrics.is_clustering_acceptable(labels.copy())
            if not acceptance["acceptable"]:
                continue

            dbcv_score = self.quality_metrics.dbcv_score_wrapper(X, labels)

            if dbcv_score > best_score:
                best_score = dbcv_score
                best_pref = pref
                best_labels = labels

        if best_labels is None:
            print("No acceptable scalar preference found.")
            return None

        print(
            f"Best scalar preference: {best_pref:.3e}, "
            f"DBCV={best_score:.4f}, "
            f"clusters={len(set(best_labels))}"
        )

        # if you still want density-based per-point preferences,
        # re-center them around best_pref with a small adjustment:
        eps = 1e-12
        densities = self.density_estimator.density(
            distance_matrix,
            self.density_estimator.radius
        )
        dens_min, dens_max = densities.min(), densities.max()
        dens_norm = (densities - dens_min) / (dens_max - dens_min + eps)

        # small adjustment scale, not huge
        sim_spread = np.percentile(sim_values, 75) - np.percentile(sim_values, 25)
        adjust_scale = 0.1 * sim_spread

        preferences = best_pref + adjust_scale * (dens_norm - 0.5)

        print(
            f"Preference stats (final): "
            f"min={preferences.min():.3e}, "
            f"max={preferences.max():.3e}, "
            f"median={np.median(preferences):.3e}"
        )

        model = AffinityPropagation(
            affinity="precomputed",
            damping=self.damping,
            preference=preferences,
            max_iter=1000,
            random_state=42,
        )
        labels = model.fit_predict(similarity_matrix)

        n_clusters = len(set(labels))
        noise_count = (labels == -1).sum()

        print(f"Affinity Propagation (final) found {n_clusters} clusters")

        acceptance = QualityMetrics.is_clustering_acceptable(labels.copy())
        if not acceptance["acceptable"] or n_clusters < 2:
            print("Final AP result not acceptable.")
            return None

        dbcv_score = self.quality_metrics.dbcv_score_wrapper(X, labels)
        print(f"DBCV score: {dbcv_score:.4f}")

        return ClusterResult(
            labels,
            n_clusters,
            noise_count,
            dbcv_score,
            self.quality_metrics.s_dbw_score_wrapper(X, labels),
        )

    def debug_pref_sweep(self, distance_matrix, damping=0.95):

        similarity_matrix = -distance_matrix ** 2
        sim_values = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]

        pref_min = sim_values.min()
        pref_med = np.median(sim_values)
        pref_max = sim_values.max()

        print(f"sim stats: min={pref_min:.3e}, med={pref_med:.3e}, max={pref_max:.3e}")

        for pref in np.linspace(pref_min, pref_max, 10):
            model = AffinityPropagation(
                affinity="precomputed",
                damping=damping,
                preference=pref,
                max_iter=1000,
                random_state=42,
            )
            labels = model.fit_predict(similarity_matrix)
            n_clusters = len(set(labels))
            print(f"pref={pref:.3e}, clusters={n_clusters}")
