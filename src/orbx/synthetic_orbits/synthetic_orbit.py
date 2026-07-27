from collections.abc import Iterable
import warnings
import io
from contextlib import redirect_stdout
from orbx.clustering.Schema import Schema
import pandas as pd

def synthetic_orbit(
    df: pd.DataFrame,
    mode: str | Iterable[str] = "max_separation",
    n_samples: int = 5000,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generate synthetic orbit rows for each non-noise cluster.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``line1``, ``line2``, and ``label``.
    mode : str | Iterable[str], default "frechet"
        Synthetic orbit mode(s) to run per cluster. Supported modes:
        ``"frechet"`` and ``"max_separation"``.
        You can pass a single string or a list/tuple of modes.
    n_samples : int, default 5000
        Number of random samples for initial candidate solutions in "max_separation" mode.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only synthetic rows with columns:
        ``line1``, ``line2``, ``label``, and ``synthetic_type``.
    """
    allowed_modes = {"frechet", "max_separation"}
    output_columns = ["line1", "line2", "label", "synthetic_type"]

    if isinstance(mode, str):
        selected_modes = [mode]
    else:
        selected_modes = list(mode)

    if not selected_modes:
        raise ValueError("mode must include at least one mode")

    invalid_modes = set(selected_modes) - allowed_modes
    if invalid_modes:
        raise ValueError(
            f"Invalid mode(s): {sorted(invalid_modes)}. "
            f"Allowed modes: {sorted(allowed_modes)}"
        )

    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    Schema().validate(df)

    from orbx.synthetic_orbits.orbit_finder.frechet_orbit_finder import frechet_orbit
    from orbx.synthetic_orbits.orbit_finder.max_separation_orbit_finder import (
        get_maximally_separated_orbit,
    )

    synthetic_rows = []

    for label, df_cluster in df.groupby("label"):
        if label == -1:
            continue

        if "frechet" in selected_modes:
            try:
                if verbose:
                    df_aug = frechet_orbit(df_cluster.copy())
                else:
                    with redirect_stdout(io.StringIO()):
                        df_aug = frechet_orbit(df_cluster.copy())
                        
                if df_aug is None:
                    if verbose: 
                        warnings.warn(
                            f"Fréchet failed for label {label}: insufficient members ({len(df_cluster)})",
                            stacklevel=2,
                        )
                else:
                    synth_row = df_aug.iloc[-1].copy()
                    synth_row["label"] = label
                    synth_row["synthetic_type"] = "frechet"
                    synthetic_rows.append(synth_row[output_columns])
            except Exception as e:
                if verbose:
                    warnings.warn(f"Fréchet failed for label {label}: {e}", stacklevel=2)

        if "max_separation" in selected_modes:
            try:
                if verbose:
                    df_aug, _ = get_maximally_separated_orbit(
                        df_cluster.copy(), n_samples=n_samples, return_diagnostics=True
                    )
                else:
                    with redirect_stdout(io.StringIO()):
                        df_aug, _ = get_maximally_separated_orbit(
                            df_cluster.copy(), n_samples=n_samples, return_diagnostics=True
                        )
                
                if df_aug is None:
                    if verbose:
                        warnings.warn(
                            f"Max-separation failed for label {label}: insufficient members ({len(df_cluster)})",
                            stacklevel=2,
                        )
                else:
                    synth_row = df_aug.iloc[-1].copy()
                    synth_row["label"] = label
                    synth_row["synthetic_type"] = "max_separation"
                    synthetic_rows.append(synth_row[output_columns])
            except Exception as e:
                if verbose:
                    warnings.warn(
                        f"Max-separation failed for label {label}: {e}",
                        stacklevel=2,
                    )

    if not synthetic_rows:
        return pd.DataFrame(columns=output_columns)

    return pd.DataFrame(synthetic_rows)