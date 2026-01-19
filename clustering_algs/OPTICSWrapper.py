from sklearn.cluster import OPTICS
import numpy as np
from metrics.quality_metrics import QualityMetrics

class OPTICSWrapper:
    def __init__(self):
        self.min_samples = 5
        self.max_eps = np.inf
        self.quality_metrics = QualityMetrics()

    def run(self, distance_matrix: np.ndarray, X: np.ndarray) -> np.ndarray:
        model = OPTICS(
            min_samples=self.min_samples,
            max_eps=self.max_eps,
            metric='precomputed',
            n_jobs=-1
        )
    
        labels = model.fit_predict(distance_matrix)
        return labels
    
    def run_pref_optimization(self, distance_matrix, X):
        best_score = np.inf
        best_min_samples = None
        
        for min_samples in range(2, 30):
            try:
                model = OPTICS(
                    min_samples=min_samples,
                    max_eps=self.max_eps,
                    metric='precomputed'
                )
                
                labels = model.fit_predict(distance_matrix)
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
        
        return model.fit_predict(distance_matrix)
    
    # def run_parameter_test(self, distance_matrix, X):
    #     for min_samples in range(2, 30):
    #         try:
    #             model = OPTICS(
    #                 min_samples=min_samples,
    #                 max_eps=self.max_eps,
    #                 metric='precomputed'
    #             )
                
    #             labels = model.fit_predict(distance_matrix)

    #             dbcv_score = self.quality_metrics.quality_metrics(X, labels)
                
    #             print(f"Min Samples: {min_samples}, DBCV Score: {dbcv_score}")
    #         except Exception as e:
    #             print(f"Error with Min Samples {min_samples}: {e}")
    #             continue
                
    # def run_X(self, X):
    #     model = OPTICS(
    #         min_samples=self.min_samples,
    #         max_eps=self.max_eps,
    #         metric='euclidean'
    #     )
        
    #     labels = model.fit_predict(X)
    #     return labels