from dataclasses import dataclass
import numpy as np

@dataclass
class Satellite:
    """Represents a satellite with its orbital parameters"""
    sat_no: str
    line1: str
    line2: str
    inclination: float
    apogee: float
    raan: float
    argument_of_perigee: float
    eccentricity: float
    mean_motion: float
    
    def __post_init__(self):
        # Normalize satellite number
        self.sat_no = str(self.sat_no).replace('.0', '').zfill(5)
        
        
@dataclass 
class ClusterResult:
    """Holds the best result of a clustering operation"""
    labels: np.ndarray
    n_clusters: int
    n_noise: int
    dbcv_score: float = None
    s_Dbw_score: float = None