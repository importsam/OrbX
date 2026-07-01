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
