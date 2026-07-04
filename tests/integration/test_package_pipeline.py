from pathlib import Path

import pytest

from orbx import cluster, density, synthetic_orbit
from tests.helpers.loaders import spacetrack_parse_file

REPO_ROOT = Path(__file__).resolve().parents[2]
THREE_LE_PATH = REPO_ROOT / "3le_1126"


@pytest.mark.integration
@pytest.mark.slow
def test_package_pipeline():
    if not THREE_LE_PATH.is_file():
        pytest.skip(f"3-line element file not found: {THREE_LE_PATH}")

    df = spacetrack_parse_file(THREE_LE_PATH)

    df = df[
        (df["apogee"] >= 500)
        & (df["apogee"] <= 520)
    ]

    assert not df.empty

    df = df[["line1", "line2"]]

    labels = cluster(df)
    df = df.copy()
    df["label"] = labels

    assert len(df["label"].unique()) >= 2

    df = df[df["label"] != -1]

    assert not df.empty

    density_df = density(df, verbose=False)
    density_df = density_df.sort_values(by="density", ascending=False)

    assert not density_df.empty
    assert {"label", "density"}.issubset(density_df.columns)

    synthetic_df = synthetic_orbit(df, verbose=False)

    assert not synthetic_df.empty
    assert {"line1", "line2", "label", "synthetic_type"}.issubset(synthetic_df.columns)
