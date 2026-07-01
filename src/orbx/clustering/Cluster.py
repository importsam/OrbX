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