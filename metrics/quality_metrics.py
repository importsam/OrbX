from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
import numpy as np
from metrics.DBCV import DBCV
from s_dbw import S_Dbw
from viasckde import viasckde_score

class QualityMetrics:
    
    def __init__(self):
        pass
    
    def quality_metrics(self, X: np.ndarray, distance_matrix: np.ndarray, labels: np.ndarray) -> dict:
        """Compute clustering quality metrics"""
        
        try:
            
            ch_score = self.calinski_harabasz_score_wrapper(X, labels)
            silhouette = self.silhouette_score_wrapper(distance_matrix, labels)
            db_score = self.davies_bouldin_score_wrapper(X, labels)
            dbcv_score = self.dbcv_score_wrapper(X, labels)
            s_Dbw_score = self.s_dbw_score_wrapper(X, labels)
            viasckde = self.viasckde_score_wrapper(X, labels)
            
            print("Clustering Quality Metrics:")
            print(f"Primary - DBCV Score: {dbcv_score}")
            print(f"Secondary - S_Dbw Score: {s_Dbw_score}\n")
            print(f"Sanity - Viasckde Score: {viasckde}\n")
            print(f"Calinski-Harabasz Score: {ch_score}")
            print(f"Silhouette Score: {silhouette}")
            print(f"Davies-Bouldin Score: {db_score}")
            
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
        
    def silhouette_score_wrapper(self, distance_matrix: np.ndarray, labels: np.ndarray) -> float:
        """Compute Silhouette Score"""
        return silhouette_score(distance_matrix, labels, metric='precomputed')
    
    def calinski_harabasz_score_wrapper(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Compute Calinski-Harabasz Score"""
        return calinski_harabasz_score(X, labels)
    
    def davies_bouldin_score_wrapper(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Compute Davies-Bouldin Score"""
        return davies_bouldin_score(X, labels)
    
    def dbcv_score_wrapper(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Compute DBCV Score"""
        return DBCV(X, labels)
    
    def s_dbw_score_wrapper(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Compute S_Dbw Score"""
        return S_Dbw(X, labels, centers_id=None, method='Tong', alg_noise='filter', centr='mean', nearest_centr=True, metric='euclidean')
    
    def viasckde_score_wrapper(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Compute Viasckde Score"""
        return viasckde_score(X, labels)
    
    