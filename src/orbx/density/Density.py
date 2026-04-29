import contextlib
import io

import pandas as pd

from orbx.synthetic_orbits.orbit_finder.frechet_orbit_finder import frechet_orbit
from orbx.synthetic_orbits.orbit_finder.DMT import VectorizedKeplerianOrbit

def _cluster_density(df_cluster: pd.DataFrame, verbose: bool = False) -> float:
    """Calculate variance-style density for one cluster DataFrame."""
    cluster_size = len(df_cluster)
    if cluster_size < 2:
        return 0.0

    if verbose:
        frechet_orbit_result = frechet_orbit(df_cluster)
    else:
        # Suppress verbose optimizer logs unless explicitly requested.
        with contextlib.redirect_stdout(io.StringIO()):
            frechet_orbit_result = frechet_orbit(df_cluster)
    if frechet_orbit_result is None:
        raise ValueError("Unable to compute Fréchet mean orbit for cluster.")

    if isinstance(frechet_orbit_result, pd.DataFrame):
        frechet_mean_orbit = frechet_orbit_result.iloc[-1]
    else:
        frechet_mean_orbit = frechet_orbit_result

    if "line1" not in df_cluster.columns or "line2" not in df_cluster.columns:
        raise ValueError("df_cluster must include 'line1' and 'line2' columns.")
    if "line1" not in frechet_mean_orbit or "line2" not in frechet_mean_orbit:
        raise ValueError("Fréchet mean orbit must include 'line1' and 'line2'.")

    mean_orbit = VectorizedKeplerianOrbit(
        pd.Series([frechet_mean_orbit["line1"]]).values,
        pd.Series([frechet_mean_orbit["line2"]]).values,
    )
    cluster_orbits = VectorizedKeplerianOrbit(
        df_cluster["line1"].values,
        df_cluster["line2"].values,
    )

    distances = VectorizedKeplerianOrbit.DistanceMetric(mean_orbit, cluster_orbits).ravel()
    total_squared_distance = float((distances**2).sum())

    variance = total_squared_distance / (cluster_size - 1)
    return float(variance)

def density(df: pd.DataFrame, label_column: str = "labels", verbose: bool = False) -> pd.DataFrame:
    """
    Compute density per label group and return a summary DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing orbit rows and a label column.
    label_column : str
        Name of the cluster label column. Defaults to "labels".
    verbose : bool
        If True, print optimizer diagnostics while computing Fréchet means.
    """
    required_columns = {label_column, "line1", "line2"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    results = []
    for label, group in df.groupby(label_column, dropna=False):
        cluster_density = _cluster_density(group, verbose=verbose)
        results.append({"label": label, "density": cluster_density})

    return pd.DataFrame(results)