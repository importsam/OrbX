from collections.abc import Iterable
import warnings
import io
from contextlib import redirect_stdout
from orbx.clustering.Schema import Schema
import pandas as pd


def _load_finders():
    """Lazy Orekit-backed imports (patchable in unit tests)."""
    from orbx.synthetic_orbits.orbit_finder.frechet_orbit_finder import frechet_orbit
    from orbx.synthetic_orbits.orbit_finder.max_separation_orbit_finder import (
        get_maximally_separated_orbit,
    )

    return frechet_orbit, get_maximally_separated_orbit


def synthetic_orbit(
    df: pd.DataFrame,
    mode: str | Iterable[str] = "max_separation",
    n_samples: int = 5000,
    verbose: bool = False,
    skip_errors: bool = False,
) -> pd.DataFrame:
    """
    Generate synthetic orbit rows for each non-noise cluster.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``line1``, ``line2``, and ``label``.
    mode : str | Iterable[str], default "max_separation"
        Synthetic orbit mode(s) to run per cluster. Supported modes:
        ``"frechet"`` and ``"max_separation"``.
        You can pass a single string or a list/tuple of modes.
    n_samples : int, default 5000
        Number of random samples for initial candidate solutions in "max_separation" mode.
    verbose : bool, default False
        If ``True``, show optimiser stdout. If ``False``, suppress stdout.
    skip_errors : bool, default False
        If ``False`` (default), raise on the first cluster/mode failure.
        If ``True``, skip failed clusters/modes and emit a warning instead.

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

    if "label" not in df.columns:
        raise ValueError("Input DataFrame is missing required column: 'label'")

    Schema().validate(df)

    frechet_orbit, get_maximally_separated_orbit = _load_finders()

    synthetic_rows = []

    def _handle_failure(message: str, cause: Exception | None = None) -> None:
        if skip_errors:
            warnings.warn(message, stacklevel=3)
            return
        if cause is not None:
            raise RuntimeError(message) from cause
        raise RuntimeError(message)

    def _run_mode(label, df_cluster, mode_name, synthetic_type, runner):
        try:
            if verbose:
                df_aug = runner()
            else:
                with redirect_stdout(io.StringIO()):
                    df_aug = runner()
        except Exception as e:
            _handle_failure(f"{mode_name} failed for label {label}: {e}", cause=e)
            return

        if df_aug is None:
            _handle_failure(
                f"{mode_name} failed for label {label}: "
                f"insufficient members ({len(df_cluster)})"
            )
            return

        synth_row = df_aug.iloc[-1].copy()
        synth_row["label"] = label
        synth_row["synthetic_type"] = synthetic_type
        synthetic_rows.append(synth_row[output_columns])

    for label, df_cluster in df.groupby("label"):
        if label == -1:
            continue

        if "frechet" in selected_modes:
            _run_mode(
                label,
                df_cluster,
                "Fréchet",
                "frechet",
                lambda: frechet_orbit(df_cluster.copy()),
            )

        if "max_separation" in selected_modes:
            def _max_sep():
                df_aug, _ = get_maximally_separated_orbit(
                    df_cluster.copy(),
                    n_samples=n_samples,
                    return_diagnostics=True,
                )
                return df_aug

            _run_mode(
                label,
                df_cluster,
                "Max-separation",
                "max_separation",
                _max_sep,
            )

    if not synthetic_rows:
        return pd.DataFrame(columns=output_columns)

    return pd.DataFrame(synthetic_rows)
