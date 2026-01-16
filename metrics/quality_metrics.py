from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
import numpy as np
import dbcv
from s_dbw import S_Dbw
from viasckde import viasckde_score

class QualityMetrics:
    
    @staticmethod
    def quality_metrics(X: np.ndarray, distance_matrix: np.ndarray, labels: np.ndarray) -> dict:
        """Compute clustering quality metrics"""
        
        try:
            
            ch_score = calinski_harabasz_score(X, labels)
            silhouette = silhouette_score(distance_matrix, labels, metric='precomputed')
            db_score = davies_bouldin_score(X, labels)
            dbcv_score = dbcv.dbcv(X, labels)
            s_Dbw_score = S_Dbw(X, labels, centers_id=None, method='Tong', alg_noise='filter', centr='mean', nearest_centr=True, metric='euclidean')
            viasckde = viasckde_score(X, labels)
            
            print(f"Calinski-Harabasz Score: {ch_score}")
            print(f"Silhouette Score: {silhouette}")
            print(f"Davies-Bouldin Score: {db_score}")
            print(f"DBCV Score: {dbcv_score}\n")
            print(f"S_Dbw Score: {s_Dbw_score}\n")
            print(f"Viasckde Score: {viasckde}\n")
            
            return {
                'Calinski-Harabasz': ch_score,
                'Silhouette Score': silhouette,
                'Davies-Bouldin': db_score,
                'DBCV': dbcv_score,
                'S_Dbw': s_Dbw_score,
                'Viasckde': viasckde
            }
            
        except Exception as e:
            print(f"Error computing quality metrics: {e}")
            return {}