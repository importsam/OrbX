from sklearn.cluster import OPTICS
import numpy as np
from metrics.quality_metrics import QualityMetrics
from models import ClusterResult
import pickle

class OPTICSWrapper:
    def __init__(self):
        self.min_samples_range = [3]

        self.xi_values = np.geomspace(0.005, 0.2, 10)

        self.max_eps = np.inf
        self.quality_metrics = QualityMetrics()

    """
    Grid search over (min_samples, xi)
    """
    def run(self, distance_matrix, X) -> ClusterResult:
        best_score = -np.inf
        best_params = None
        best_labels = None

        n_total = distance_matrix.shape[0]

        for min_samples in self.min_samples_range:
            for xi in self.xi_values:
                try:
                    model = OPTICS(
                        min_samples=min_samples,
                        max_eps=self.max_eps,
                        metric="precomputed",
                        cluster_method="xi",
                        xi=xi,
                        n_jobs=-1,
                    )

                    labels = model.fit_predict(distance_matrix)

                    n_clusters = len(set(labels) - {-1})
                    noise_count = (labels == -1).sum()

                    # acceptance = QualityMetrics.is_clustering_acceptable(labels.copy())
                    # if not acceptance["acceptable"]:
                    #     print(
                    #         f"Rejected: min_samples={min_samples}, xi={xi:.4f} "
                    #         f"({acceptance['fail_reasons']})"
                    #     )
                    #     continue

                    score = self.quality_metrics.dbcv_score_wrapper(X, labels)

                    print(
                        f"min_samples={min_samples:2d}, "
                        f"xi={xi:.4f}, "
                        f"clusters={n_clusters:4d}, "
                        f"noise={noise_count:4d}, "
                        f"DBCV={score:.4f}"
                    )

                    if score > best_score:
                        best_score = score
                        best_params = (min_samples, xi)
                        best_labels = labels

                except Exception as e:
                    print(
                        f"Error: min_samples={min_samples}, xi={xi:.4f} → {e}"
                    )
                    continue

        if best_labels is None:
            raise RuntimeError("OPTICS failed to find a valid clustering")

        best_min_samples, best_xi = best_params

        print(
            f"\nBest OPTICS params → min_samples={best_min_samples}, "
            f"xi={best_xi:.4f}, "
            f"DBCV={best_score:.4f}"
        )

        print(
            f"OPTICS found {len(set(best_labels) - {-1})} clusters "
            f"(noise points: {(best_labels == -1).sum()})"
        )

        # Save the labels as pkl  
        with open("data/cluster_results/optics_labels.pkl", "wb") as f:
            pickle.dump(best_labels, f)

        return ClusterResult(
            best_labels,
            len(set(best_labels) - {-1}),
            (best_labels == -1).sum(),
            best_score,
            self.cluster_wrapper.quality_metrics.s_dbw_score_wrapper(X, best_labels)
        )
