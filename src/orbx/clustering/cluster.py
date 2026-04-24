import pandas as pd
import numpy as np
from .Core import Core
from models import ClusterResult


def cluster(
    df: pd.DataFrame,
    algorithm: str = "hdbscan"   
    ) -> ClusterResult:
    
    cluster_result = Core().cluster(df, algorithm)
    
    return cluster_result