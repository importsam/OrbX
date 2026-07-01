import pandas as pd
import pytest

ISS_LINE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
ISS_LINE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391 56353"

NOAA_LINE1 = "1 43013U 17073A   25365.50000000  .00000093  00000-0  33518-4 0  9997"
NOAA_LINE2 = "2 43013  98.5742 152.9220 0011408  89.1978 271.0120 14.19996472  5789"


@pytest.fixture
def sample_tle_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "line1": [ISS_LINE1, ISS_LINE1],
            "line2": [ISS_LINE2, ISS_LINE2],
        }
    )


@pytest.fixture
def two_satellite_tle_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "line1": [ISS_LINE1, NOAA_LINE1],
            "line2": [ISS_LINE2, NOAA_LINE2],
        }
    )
