from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
import numpy as np
import dbcv

class QualityMetrics:
    
    @staticmethod
    def quality_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
        """Compute clustering quality metrics"""
        
        try:
            
            ch_score = calinski_harabasz_score(X, labels)
            silhouette_avg = silhouette_score(X, labels)
            db_score = davies_bouldin_score(X, labels)
            dbcv_score = dbcv.dbcv(X, labels)
            
            print(f"Calinski-Harabasz Score: {ch_score}")
            print(f"Silhouette Score: {silhouette_avg}")
            print(f"Davies-Bouldin Score: {db_score}")
            print(f"DBCV Score: {dbcv_score}\n")
            
            return {
                'Calinski-Harabasz': ch_score,
                'Silhouette Score': silhouette_avg,
                'Davies-Bouldin': db_score,
                'DBCV': dbcv_score
            }
            
        except Exception as e:
            print(f"Error computing quality metrics: {e}")
            return {}