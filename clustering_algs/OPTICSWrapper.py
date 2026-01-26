from sklearn.cluster import OPTICS
import numpy as np
from metrics.quality_metrics import QualityMetrics
from models import ClusterResult

class OPTICSWrapper:
    def __init__(self):
        self.min_samples_range = range(2, 10)
        self.max_eps = np.inf
        self.quality_metrics = QualityMetrics()
    
    
    """
    By default now, this will do the grid search
    """
    def run(self, distance_matrix, X) -> ClusterResult:
        best_score = -np.inf
        best_min_samples = None

        for min_samples in self.min_samples_range:
            try:
                model = OPTICS(
                    min_samples=min_samples,
                    max_eps=self.max_eps,
                    metric='precomputed'
                )
                
                labels = model.fit_predict(distance_matrix)
                
                acceptance = QualityMetrics.is_clustering_acceptable(labels.copy())
                
                if not acceptance["acceptable"]:
                    print(
                        f"Rejected ({acceptance['fail_reasons']})"
                    )
                    return None

                score = self.quality_metrics.dbcv_score_wrapper(X, labels)

                print(f"Min Samples: {min_samples}, DBCV Score: {score}")

                # Higher score is better clustering
                if score > best_score:
                    best_score = score
                    best_min_samples = min_samples
                    
            except Exception as e:
                print(f"Error with Min Samples {min_samples}: {e}")
                continue
        
        print(f"\nBest Min Samples: {best_min_samples}, Best DBCV Score: {best_score}")
        
        # Run again with best settings
        model = OPTICS(
            min_samples=best_min_samples,
            max_eps=self.max_eps,
            metric='precomputed'
        )
        
        best_labels = model.fit_predict(distance_matrix)
        
        cluster_result_obj = ClusterResult(best_labels, len(set(best_labels)), 
                                           (best_labels == -1).sum(), best_score,
                                           self.quality_metrics.s_dbw_score_wrapper(X, best_labels))

        return cluster_result_obj