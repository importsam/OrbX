import pandas as pd
import pytest

ISS_LINE1 = "1 25544U 98067A   26182.50817465  .00006185  00000-0  11827-3 0  9997"
ISS_LINE2 = "2 25544  51.6311 229.1989 0004224 255.0896 104.9625 15.49503254573972"

NOAA_LINE1 = "1 58979U 24031Q   26183.83336806  .00061200  00000-0  18088-2 0  9991"
NOAA_LINE2 = "2 58979  53.1598 121.8985 0000471 135.4548 154.2508 15.34381792  5853"


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
