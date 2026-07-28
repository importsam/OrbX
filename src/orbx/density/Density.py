import contextlib
import io

import pandas as pd
from tqdm import tqdm

from orbx.clustering.Schema import Schema
from orbx.synthetic_orbits.orbit_finder.DMT import VectorizedKeplerianOrbit


def _cluster_density(df_cluster: pd.DataFrame, verbose: bool = False) -> float:
    """Variance-style dispersion around the Fréchet mean (smaller = tighter)."""
    cluster_size = len(df_cluster)
    if cluster_size < 2:
        return 0.0

    from orbx.synthetic_orbits.orbit_finder.frechet_orbit_finder import frechet_orbit

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

    # DistanceMetric returns q, which is already a squared distance.
    squared_distances = VectorizedKeplerianOrbit.DistanceMetric(
        mean_orbit, cluster_orbits
    ).ravel()
    variance = float(squared_distances.sum()) / (cluster_size - 1)
    return float(variance)


def density(df: pd.DataFrame, label_column: str = "label", verbose: bool = False) -> pd.DataFrame:
    """
    Compute a variance-style density score per label group.

    Smaller scores mean member orbits are closer to the cluster Fréchet mean
    (more tightly packed). Larger scores mean more dispersed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing orbit rows and a label column. TLEs are
        Schema-validated (``line1`` / ``line2``).
    label_column : str
        Name of the cluster label column. Defaults to "label".
    verbose : bool
        If True, print optimizer diagnostics while computing Fréchet means.
    """
    if label_column not in df.columns:
        raise ValueError(f"DataFrame is missing required column: '{label_column}'")

    Schema().validate(df)

    results = []
    groups = list(df.groupby(label_column, dropna=False))
    for label, group in tqdm(groups, desc="Calculating cluster densities"):
        cluster_density = _cluster_density(group, verbose=verbose)
        results.append({"label": label, "density": cluster_density})

    return pd.DataFrame(results)
