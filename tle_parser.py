from pathlib import Path
from math import pi
import pandas as pd

from models import Satellite
from configs import OrbitalConstants, PathConfig

class TLEParser:
    
    def __init__(self, source: str):
        self.constants = OrbitalConstants()
        self.source = source # "Space-Track", "Celestrak", "UDL"
        # By default, should load in the data and hold a df containing it
        paths = PathConfig()
        self.df = pd.DataFrame()
        if source == "Space-Track":
            self.spacetrack_parse_file(paths.spacetrack_data)
        elif source == "Celestrak":
            self.celestrak_parse_file(paths.celestrak_data)
        elif source == "UDL":
            self.udl_parse_file(paths.udl_data)
        else:
            raise ValueError("Unsupported source. Valid Options: 'Space-Track', 'Celestrak', 'UDL'")
        
        # fix satNo to be string
        self.df['satNo'] = self.df['satNo'].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
        
    def spacetrack_parse_file(self, filepath: Path) -> None:
        """Takes tle data and converts to df"""
        
        print(f"Parsing TLE data from Space-Track...")
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                break
                
            sat_obj = self._parse_tle_group(
                lines[i].strip(),
                lines[i+1].strip(), 
                lines[i+2].strip()
            )
            
            # add to df 
            self.df = pd.concat([self.df, pd.DataFrame([{
                'satNo': sat_obj.sat_no,
                'line1': sat_obj.line1,
                'line2': sat_obj.line2,
                'inclination': sat_obj.inclination,
                'apogee': sat_obj.apogee,
                'raan': sat_obj.raan,
                'argument_of_perigee': sat_obj.argument_of_perigee,
                'eccentricity': sat_obj.eccentricity,
                'mean_motion': sat_obj.mean_motion
            }])], ignore_index=True)
    
    def celestrak_parse_file(self, filepath: Path) -> None:
        raise NotImplementedError()
    
    def udl_parse_file(self, filepath: Path) -> None:
        raise NotImplementedError()
    
    def _parse_tle_group(self, name: str, line1: str, line2: str) -> Satellite:
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