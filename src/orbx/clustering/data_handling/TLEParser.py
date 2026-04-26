from math import pi
import pandas as pd
from orbx.Models import Satellite
from orbx.Configs import OrbitalConstants

class TLEParser:
    def __init__(self):
        self.constants = OrbitalConstants()
                
    def tle_to_keplerian(self, input_df) -> pd.DataFrame:
        
        # for each row, compute the keplerian elements and add to df
        for index, row in input_df.iterrows():
            sat_obj = self._parse_tle_group(
                row['line1'],
                row['line2']
            )
            
            # add to df 
            input_df = pd.concat([input_df, pd.DataFrame([{
                'satNo': sat_obj.sat_no,
                'inclination': sat_obj.inclination,
                'apogee': sat_obj.apogee,
                'raan': sat_obj.raan,
                'argument_of_perigee': sat_obj.argument_of_perigee,
                'eccentricity': sat_obj.eccentricity,
                'mean_motion': sat_obj.mean_motion
            }])], ignore_index=True)
        
        return input_df
 
    def _parse_tle_group(self, line1: str, line2: str) -> Satellite:
        """Parse a single TLE group into a Satellite object"""
        if not (line1.startswith('1 ') and line2.startswith('2 ')):
            raise ValueError("Invalid TLE format")

        sat_no = line1[2:7].strip()

        inclination = float(line2[8:16].strip())
        mean_motion = float(line2[52:63].strip())
        eccentricity = float("0." + line2[26:33].strip())
        raan = float(line2[17:25].strip())
        argument_of_perigee = float(line2[34:42].strip())
        apogee = self._calculate_apogee(mean_motion, eccentricity)

        return Satellite(
            sat_no=sat_no,
            line1=line1,
            line2=line2,
            inclination=inclination,
            apogee=apogee,
            raan=raan,
            argument_of_perigee=argument_of_perigee,
            eccentricity=eccentricity,
            mean_motion=mean_motion
        )
    
    def _calculate_apogee(self, mean_motion: float, eccentricity: float) -> float:
        """Calculate apogee in kilometers from mean motion and eccentricity"""
        n = mean_motion * 2 * pi / self.constants.SECONDS_IN_DAY
        a = (self.constants.GM_EARTH / (n ** 2)) ** (1/3)
        apogee_m = a * (1 + eccentricity) - self.constants.EARTH_RADIUS_M
        return apogee_m / 1000  # Convert to km