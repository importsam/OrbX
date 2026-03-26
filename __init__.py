import pandas as pd

from satellite_clustering.Schema import validate
from satellite_clustering.Core import Core
from models import ClusterResult 

def cluster(df: pd.DataFrame, algorithm: str = "hdbscan") -> ClusterResult:
    df = validate(df)     
    # add option for n_jobs (currently -1, so all available cores)
    return Core()._cluster(df, algorithm)