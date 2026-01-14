from sklearn.cluster import OPTICS
import numpy as np
from metrics.quality_metrics import QualityMetrics

class OPTICSWrapper:
    def __init__(self):
        self.min_samples = 5
        self.max_eps = np.inf
        self.quality_metrics = QualityMetrics()

    def run(self, distance_matrix):
        model = OPTICS(
            min_samples=self.min_samples,
            max_eps=self.max_eps,
            metric='precomputed',
            n_jobs=-1
        )
    
        labels = model.fit_predict(distance_matrix)
        return labels
    
    def run_parameter_test(self, distance_matrix, X):
        for min_samples in range(2, 13):
            try:
                model = OPTICS(
                    min_samples=min_samples,
                    max_eps=self.max_eps,
                    metric='precomputed'
                )
                
                labels = model.fit_predict(distance_matrix)

                dbcv_score = self.quality_metrics.quality_metrics(X, labels)
                
                print(f"Min Samples: {min_samples}, DBCV Score: {dbcv_score}")
            except Exception as e:
                print(f"Error with Min Samples {min_samples}: {e}")
                continue
                
    # def run_X(self, X):
    #     model = OPTICS(
    #         min_samples=self.min_samples,
    #         max_eps=self.max_eps,
    #         metric='euclidean'
    #     )
        
    #     labels = model.fit_predict(X)
    #     return labels