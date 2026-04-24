import pandas as pd

from orbx.clustering.Schema import validate
from orbx.clustering.Core import Core
from models import ClusterResult 

def cluster(df: pd.DataFrame, algorithm: str = "hdbscan") -> ClusterResult:
    df = validate(df)     
    # add option for n_jobs (currently -1, so all available cores)
    return Core()._cluster(df, algorithm)