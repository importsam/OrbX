import pandas as pd
import pytest

from orbx import cluster


def test_cluster_rejects_invalid_tle_before_clustering(sample_tle_df):
    df = sample_tle_df.copy()
    df.loc[0, "line1"] = "not a tle"

    with pytest.raises(ValueError, match="Invalid TLE format"):
        cluster(df)
