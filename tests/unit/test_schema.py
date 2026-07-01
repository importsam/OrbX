import pandas as pd
import pytest

from orbx.clustering.Schema import Schema


def test_validate_accepts_valid_tle_df(sample_tle_df):
    Schema().validate(sample_tle_df)


def test_validate_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        Schema().validate(pd.DataFrame({"line1": ["1 00000U"]}))


def test_validate_rejects_empty_dataframe(sample_tle_df):
    with pytest.raises(ValueError, match="empty"):
        Schema().validate(sample_tle_df.iloc[0:0])


def test_validate_rejects_null_tle_values(sample_tle_df):
    df = sample_tle_df.copy()
    df.loc[0, "line1"] = None
    with pytest.raises(ValueError, match="null values"):
        Schema().validate(df)


def test_validate_rejects_invalid_tle(sample_tle_df):
    df = sample_tle_df.copy()
    df.loc[0, "line1"] = "not a tle"
    with pytest.raises(ValueError, match="Invalid TLE format"):
        Schema().validate(df)
