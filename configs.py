from dataclasses import dataclass
from pathlib import Path

@dataclass
class OrbitalConstants:
    EARTH_RADIUS_M = 6371000  # meters
    GM_EARTH = 3.986004418e14  # m^3/s^2
    SECONDS_IN_DAY = 86400

@dataclass
class ClusterConfig:
    inclination_range: tuple[float, float] = (0, 180)
    apogee_range: tuple[float, float] = (0, 2000)
    damping: float = 0.95
    
@dataclass
class PathConfig:
    spacetrack_data: Path = Path('data/3le_1126')
    celestrak_data: Path = None
    udl_data: Path = None
    distance_matrix: Path = Path('data/distance_matrix.pkl')
    idx_satNo_dict: Path = Path('data/idx_satNo_dict.pkl')
    satNo_idx_dict: Path = Path('data/satNo_idx_dict.pkl')
    output_plot: Path = Path('data/')
    output_dataframe: Path = Path('data/clustering_results')