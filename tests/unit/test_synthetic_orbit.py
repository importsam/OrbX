from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from orbx.synthetic_orbits.synthetic_orbit import synthetic_orbit


def test_synthetic_orbit_rejects_invalid_mode(sample_tle_df):
    df = sample_tle_df.copy()
    df["label"] = 0

    with pytest.raises(ValueError, match="Invalid mode"):
        synthetic_orbit(df, mode="bogus")


def test_synthetic_orbit_rejects_empty_mode_list(sample_tle_df):
    df = sample_tle_df.copy()
    df["label"] = 0

    with pytest.raises(ValueError, match="mode must include at least one mode"):
        synthetic_orbit(df, mode=[])


def test_synthetic_orbit_rejects_invalid_n_samples(sample_tle_df):
    df = sample_tle_df.copy()
    df["label"] = 0

    with pytest.raises(ValueError, match="n_samples must be >= 1"):
        synthetic_orbit(df, n_samples=0)


def test_synthetic_orbit_validates_tle_input(sample_tle_df):
    df = sample_tle_df.copy()
    df["label"] = 0
    df.loc[0, "line1"] = "not a tle"

    with pytest.raises(ValueError, match="Invalid TLE format"):
        synthetic_orbit(df)


def test_synthetic_orbit_requires_label_column(sample_tle_df):
    with pytest.raises(ValueError, match="missing required column: 'label'"):
        synthetic_orbit(sample_tle_df)


def test_synthetic_orbit_raises_on_optimizer_failure(sample_tle_df):
    df = sample_tle_df.copy()
    df["label"] = 0

    max_sep = MagicMock(side_effect=RuntimeError("optimizer blew up"))
    with patch(
        "orbx.synthetic_orbits.synthetic_orbit._load_finders",
        return_value=(MagicMock(), max_sep),
    ):
        with pytest.raises(RuntimeError, match="Max-separation failed for label 0"):
            synthetic_orbit(df, mode="max_separation")


def test_synthetic_orbit_raises_when_finder_returns_none(sample_tle_df):
    df = sample_tle_df.copy()
    df["label"] = 0

    max_sep = MagicMock(return_value=(None, {}))
    with patch(
        "orbx.synthetic_orbits.synthetic_orbit._load_finders",
        return_value=(MagicMock(), max_sep),
    ):
        with pytest.raises(RuntimeError, match="insufficient members"):
            synthetic_orbit(df, mode="max_separation")


def test_synthetic_orbit_skip_errors_warns_and_continues(sample_tle_df):
    df = sample_tle_df.copy()
    df["label"] = 0

    max_sep = MagicMock(side_effect=RuntimeError("optimizer blew up"))
    with patch(
        "orbx.synthetic_orbits.synthetic_orbit._load_finders",
        return_value=(MagicMock(), max_sep),
    ):
        with pytest.warns(UserWarning, match="Max-separation failed for label 0"):
            result = synthetic_orbit(df, mode="max_separation", skip_errors=True)

    assert result.empty
    assert list(result.columns) == ["line1", "line2", "label", "synthetic_type"]


def test_synthetic_orbit_notices_when_processing_noise(sample_tle_df, capsys):
    df = sample_tle_df.copy()
    df["label"] = [-1, -1]

    synth_row = pd.Series(
        {
            "line1": df.iloc[0]["line1"],
            "line2": df.iloc[0]["line2"],
            "label": -1,
            "synthetic_type": "max_separation",
        }
    )
    aug = pd.concat([df, synth_row.to_frame().T], ignore_index=True)
    max_sep = MagicMock(return_value=(aug, {}))

    with patch(
        "orbx.synthetic_orbits.synthetic_orbit._load_finders",
        return_value=(MagicMock(), max_sep),
    ):
        result = synthetic_orbit(df, mode="max_separation")

    captured = capsys.readouterr()
    assert "Notice: processing noise label -1" in captured.out
    assert set(result["label"]) == {-1}
