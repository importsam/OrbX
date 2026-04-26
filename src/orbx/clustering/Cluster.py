import pandas as pd
import io
import warnings
from contextlib import redirect_stdout

from .Core import Core
from models import ClusterResult


def cluster(
    df: pd.DataFrame,
    algorithm: str = "hdbscan",
    verbose: bool = False,
    ) -> ClusterResult:

    if verbose:
        cluster_result = Core().cluster(df, algorithm)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            with redirect_stdout(io.StringIO()):
                cluster_result = Core().cluster(df, algorithm)
    
    return cluster_result