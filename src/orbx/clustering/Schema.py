import pandas as pd
from sgp4.io import twoline2rv
import sgp4.earth_gravity as earth_gravity

REQUIRED_COLUMNS = {"line1", "line2"}

class Schema:

    def validate(self, df: pd.DataFrame) -> None:
        """
        Validates input DataFrame and computes all derived orbital elements.
        
        Required columns:
            line1 (str): TLE line 1
            line2 (str): TLE line 2
        
        Optional columns:
            name (str): Satellite name. Defaults to sat_id if not provided.
        
        Returns a DataFrame with all orbital elements computed and ready for clustering.
        """
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Input DataFrame is missing required columns: {missing}\n"
                f"DataFrame must contain at minimum 'line1' and 'line2'."
            )
        if df.empty: 
            raise ValueError("Input DataFrame is empty")
        if df[["line1", "line2"]].isnull().any().any():
            raise ValueError("Input DataFrame contains null values in 'line1' or 'line2' columns")
        
        # Next we want to validate the two line elements.
        
        for _, row in df.iterrows():
            try:
                _ = twoline2rv(row['line1'], row['line2'], earth_gravity.wgs72)
            except ValueError as e:
                raise ValueError(f"Invalid TLE format for TLE: {row['line1']}, {row['line2']}: {e}")

        return None