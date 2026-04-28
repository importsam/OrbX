import pandas as pd

from orbx.synthetic_orbits.orbit_finder.frechet_orbit_finder import frechet_orbit
from orbx.synthetic_orbits.orbit_finder.DMT import VectorizedKeplerianOrbit

def density(df_cluster: pd.DataFrame) -> float:
    """
    Calculate internal cluster density as variance around the Fréchet mean orbit.

    Parameters
    ----------
    df_cluster : pd.DataFrame
        DataFrame containing all orbit rows for one cluster.
    """
    cluster_size = len(df_cluster)
    if cluster_size < 2:
        return 0.0  # Variance of 1 item is 0 (or return float('nan'))

    # Calculate the mean
    frechet_orbit_result = frechet_orbit(df_cluster)
    
    if frechet_orbit_result is None:
        raise ValueError("Unable to compute Fréchet mean orbit for cluster.")
        
    # Extract the single mean orbit row safely
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