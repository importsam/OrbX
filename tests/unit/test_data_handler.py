import pandas as pd
import pytest

from orbx.clustering.data_handling.DataHandler import DataHandler


def test_tle_to_keplerian_preserves_row_count(two_satellite_tle_df):
    enriched = DataHandler().tle_to_keplerian(two_satellite_tle_df.copy())

    assert len(enriched) == len(two_satellite_tle_df)
    assert enriched.iloc[0]["satNo"] == "25544"
    assert enriched.iloc[1]["satNo"] == "58979"


def test_tle_to_keplerian_adds_expected_columns(two_satellite_tle_df):
    enriched = DataHandler().tle_to_keplerian(two_satellite_tle_df.copy())

    for column in (
        "satNo",
        "inclination",
        "apogee",
        "raan",
        "argument_of_perigee",
        "eccentricity",
        "mean_motion",
    ):
        assert column in enriched.columns


def test_tle_to_keplerian_includes_row_context_on_parse_error(sample_tle_df):
    df = sample_tle_df.copy()
    df.loc[0, "line2"] = "2 25544  bad data here"

    with pytest.raises(ValueError, match="Error parsing TLE at row: 0"):
        DataHandler().tle_to_keplerian(df)


def test_get_points_returns_six_dimensional_embedding(two_satellite_tle_df):
    handler = DataHandler()
    enriched = handler.tle_to_keplerian(two_satellite_tle_df.copy())
    points = handler.get_points(enriched)

    assert points.shape == (len(enriched), 6)
