import pandas as pd
import pytest

from orbx.density.Density import _cluster_density, density


def test_cluster_density_returns_zero_for_single_member_cluster(sample_tle_df):
    assert _cluster_density(sample_tle_df.iloc[:1]) == 0.0


def test_density_validates_tle_input(sample_tle_df):
    df = sample_tle_df.copy()
    df["label"] = 0
    df.loc[0, "line1"] = "not a tle"

    with pytest.raises(ValueError, match="Invalid TLE format"):
        density(df)


def test_density_returns_one_row_per_label(sample_tle_df):
    df = sample_tle_df.copy()
    df["label"] = [0, 1]

    result = density(df)

    assert set(result["label"]) == {0, 1}
    assert result.loc[result["label"] == 0, "density"].iloc[0] == 0.0
