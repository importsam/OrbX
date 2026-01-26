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

    def run(self, distance_matrix: np.ndarray, X: np.ndarray, 
            target_min_clusters=200, target_max_clusters=500) -> ClusterResult:
        print(f"Running Affinity Propagation targeting {target_min_clusters}-{target_max_clusters} clusters...")

        normalizer = np.std(distance_matrix) ** 2
        similarity_matrix = -distance_matrix / normalizer

        sim_values = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]

        pref_min = sim_values.min()
        pref_med = np.median(sim_values)
        pref_max = sim_values.max()

        # Phase 1: Coarse search to find the range
        print("\n=== Phase 1: Coarse search ===")
        pref_max_safe = np.percentile(sim_values, 95)
        pref_grid_coarse = np.linspace(pref_min, pref_max_safe, 20)

        cluster_counts = []
        pref_values = []

        for pref in pref_grid_coarse:
            model = AffinityPropagation(
                affinity="precomputed",
                damping=self.damping,
                preference=pref,
                max_iter=1000,
                random_state=42,
            )
            labels = model.fit_predict(similarity_matrix)
            n_clusters = len(set(labels))
            
            # Stop if explosion
            if n_clusters > 0.5 * len(distance_matrix):
                print(f"[coarse] pref={pref:.3e}, clusters={n_clusters} - STOPPED")
                break
            
            cluster_counts.append(n_clusters)
            pref_values.append(pref)
            print(f"[coarse] pref={pref:.3e}, clusters={n_clusters}")

        # Find preferences that bracket the target range
        cluster_counts = np.array(cluster_counts)
        pref_values = np.array(pref_values)
        
        # Find where we cross into the target range
        below_target = cluster_counts < target_min_clusters
        in_target = (cluster_counts >= target_min_clusters) & (cluster_counts <= target_max_clusters)
        above_target = cluster_counts > target_max_clusters
        
        if in_target.any():
            # We found some in range, narrow search around them
            in_range_indices = np.where(in_target)[0]
            idx_min = max(0, in_range_indices[0] - 1)
            idx_max = min(len(pref_values) - 1, in_range_indices[-1] + 1)
            pref_search_min = pref_values[idx_min]
            pref_search_max = pref_values[idx_max]
        elif above_target.any():
            # All tested values give too many clusters, search lower
            first_above = np.where(above_target)[0][0]
            if first_above > 0:
                pref_search_max = pref_values[first_above]
                pref_search_min = pref_values[max(0, first_above - 2)]
            else:
                pref_search_min = pref_min
                pref_search_max = pref_values[0]
        else:
            # All values give too few clusters, search higher
            pref_search_min = pref_values[-1]
            pref_search_max = min(pref_values[-1] + (pref_values[-1] - pref_values[-2]), pref_max_safe)

        # Phase 2: Fine-grained search in the identified range
        print(f"\n=== Phase 2: Fine search in [{pref_search_min:.3e}, {pref_search_max:.3e}] ===")
        pref_grid_fine = np.linspace(pref_search_min, pref_search_max, 25)

        best_score = -np.inf
        best_pref = None
        best_labels = None
        candidates = []

        for pref in pref_grid_fine:
            model = AffinityPropagation(
                affinity="precomputed",
                damping=self.damping,
                preference=pref,
                max_iter=1000,
                random_state=42,
            )
            labels = model.fit_predict(similarity_matrix)
            n_clusters = len(set(labels))
            
            if n_clusters > 0.5 * len(distance_matrix):
                print(f"[fine] pref={pref:.3e}, clusters={n_clusters} - STOPPED")
                break
            
            in_range = target_min_clusters <= n_clusters <= target_max_clusters
            status = "✓ IN RANGE" if in_range else ""
            print(f"[fine] pref={pref:.3e}, clusters={n_clusters} {status}")

            acceptance = QualityMetrics.is_clustering_acceptable(labels.copy())
            if not acceptance["acceptable"]:
                continue

            # Only consider candidates in target range
            if not in_range:
                continue

            dbcv_score = self.quality_metrics.dbcv_score_wrapper(X, labels)
            candidates.append({
                'pref': pref,
                'labels': labels,
                'n_clusters': n_clusters,
                'dbcv': dbcv_score
            })

            if dbcv_score > best_score:
                best_score = dbcv_score
                best_pref = pref
                best_labels = labels

        if best_labels is None:
            print(f"\nNo acceptable clustering found in target range {target_min_clusters}-{target_max_clusters}")
            # Fall back to best outside range if available
            if candidates:
                print("Using best available clustering outside target range")
                best_candidate = max(candidates, key=lambda x: x['dbcv'])
                best_pref = best_candidate['pref']
                best_labels = best_candidate['labels']
                best_score = best_candidate['dbcv']
            else:
                return None

        n_clusters_base = len(set(best_labels))
        print(f"\n=== Best base clustering ===")
        print(f"Preference: {best_pref:.3e}, DBCV={best_score:.4f}, clusters={n_clusters_base}")

        # Phase 3: Apply density-based adjustment (but keep it minimal)
        print("\n=== Phase 3: Density adjustment ===")
        eps = 1e-12
        densities = self.density_estimator.density(
            distance_matrix,
            self.density_estimator.radius
        )
        dens_min, dens_max = densities.min(), densities.max()
        dens_norm = (densities - dens_min) / (dens_max - dens_min + eps)

        # Use much smaller adjustment to stay in target range
        sim_spread = np.percentile(sim_values, 75) - np.percentile(sim_values, 25)
        
        # Calculate adjustment that won't push us too far
        # Aim to stay within target range
        max_cluster_increase = target_max_clusters - n_clusters_base
        if max_cluster_increase < 50:
            # Very close to upper bound, use tiny adjustment
            adjust_scale = 0.02 * sim_spread
        else:
            adjust_scale = 0.05 * sim_spread  # Reduced from 0.1

        preferences = best_pref + adjust_scale * (dens_norm - 0.5)

        print(f"Adjustment scale: {adjust_scale:.3e}")
        print(f"Preference range: [{preferences.min():.3e}, {preferences.max():.3e}]")

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

        print(f"\n=== Final result ===")
        print(f"Clusters: {n_clusters} (target: {target_min_clusters}-{target_max_clusters})")
        
        in_range = target_min_clusters <= n_clusters <= target_max_clusters
        if not in_range:
            print(f"WARNING: Final clustering outside target range!")
            # If density adjustment pushed us out, revert to base clustering
            if abs(n_clusters - n_clusters_base) > 100:
                print("Reverting to base clustering without density adjustment")
                labels = best_labels
                n_clusters = n_clusters_base

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