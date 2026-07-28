import pandas as pd
import pytest

from orbx import cluster


def test_cluster_rejects_invalid_tle_before_clustering(sample_tle_df):
    df = sample_tle_df.copy()
    df.loc[0, "line1"] = "not a tle"

    with pytest.raises(ValueError, match="Invalid TLE format"):
        cluster(df)


def test_cluster_does_not_mutate_input_dataframe(two_satellite_tle_df):
    df = two_satellite_tle_df.copy()
    before = df.copy(deep=True)

    try:
        cluster(df)
    except Exception:
        # Clustering may fail on tiny inputs; enrichment must not have mutated df.
        pass

    pd.testing.assert_frame_equal(df, before)
    assert "satNo" not in df.columns
