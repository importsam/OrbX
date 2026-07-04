import pandas as pd
import io
import warnings
from contextlib import redirect_stdout
import numpy as np
from .Core import Core
from .Schema import Schema


def cluster(
    df: pd.DataFrame,
    min_samples: int = 3,
    min_cluster_size: int = 2,
    verbose: bool = False,
    ) -> np.ndarray:

    """
    Arguments
    ----------
    df: pandas.DataFrame
        This should contain the TLEs you want to cluster (at a minimum). Your df should contain
        a "line1" and "line2" in each row, corresponding to a satellite.
    min_samples: int, default 3
        HDBSCAN minimum samples parameter. Refer to HDBSCAN documentation for more details.
    min_cluster_size: int, default 2
        HDBSCAN minimum cluster size parameter. Refer to HDBSCAN documentation for more details.
    verbose: bool, default False
        If ``True``, print internal clustering progress and allow warnings.
        If ``False`` (default), suppress stdout and ``FutureWarning`` noise.
    """



    # Handle input validation 
    """ 
    Core().cluster() is expecting a dataframe with the columns "line1" and "line2"
    """
    Schema().validate(df)

    if verbose:
        labels = Core().cluster(df, min_samples, min_cluster_size)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            with redirect_stdout(io.StringIO()):
                labels = Core().cluster(df, min_samples, min_cluster_size)
    
    return labels