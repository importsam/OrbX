import pandas as pd 
from src.satellite_clustering.data_handling.tle_parser import TLEParser

REQUIRED_COLUMNS = {"line1", "line2"}

class Schema: 
    
    def __init__(self):
        self.tle_parser = TLEParser(source="Space-Track")

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
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
        df = df.copy()
        df_kep = self.tle_parser.tle_to_keplerian(df)
        
        if "name" not in df_kep.columns:
            df_kep["name"] = df_kep["sat_id"]
        return df_kep