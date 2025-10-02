from typing import Iterator
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
        if source == "Space-Track":
            self.sat_df = self.spacetrack_parse_file(paths.spacetrack_data)
        elif source == "Celestrak":
            self.sat_df = self.celestrak_parse_file(paths.celestrak_data)
        elif source == "UDL":
            self.sat_df = self.udl_parse_file(paths.udl_data)
        else:
            raise ValueError("Unsupported source. Valid Options: 'Space-Track', 'Celestrak', 'UDL'")
        
    def spacetrack_parse_file(self, filepath: Path) -> Iterator[Satellite]:
        """Satellite object generator"""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                break
                
            sat = self._parse_tle_group(
                lines[i].strip(),
                lines[i+1].strip(), 
                lines[i+2].strip()
            )
            
            if sat:
                yield sat
                
    def celestrak_parse_file(self, filepath: Path) -> Iterator[Satellite]:
        raise NotImplementedError()
    
    def udl_parse_file(self, filepath: Path) -> Iterator[Satellite]:
        raise NotImplementedError()
    
    def _parse_tle_group(self, name: str, line1: str, line2: str) -> Satellite:
        """Parse a single TLE group into a Satellite object"""
        if not (line1.startswith('1 ') and line2.startswith('2 ')):
            raise ValueError("Invalid TLE format")

        sat_no = line1[2:7].strip()
        inclination = float(line2[8:16].strip())
        mean_motion = float(line2[52:63].strip())
        eccentricity = float("0." + line2[26:33].strip())
        
        apogee = self._calculate_apogee(mean_motion, eccentricity)
        
        return Satellite(
            sat_no=sat_no,
            line1=line1,
            line2=line2,
            inclination=inclination,
            apogee=apogee
        )

    def _get_tle_df(self) -> 'pd.DataFrame':
        """Convert TLE data to DataFrame"""

        satellites = list(self.parse_file)
        
        return pd.DataFrame([
            {
                'satNo': sat.sat_no,
                'line1': sat.line1,
                'line2': sat.line2,
                'inclination': sat.inclination,
                'apogee': sat.apogee
            }
            for sat in satellites
        ])

    def _calculate_apogee(self, mean_motion: float, eccentricity: float) -> float:
        """Calculate apogee in kilometers from mean motion and eccentricity"""
        n = mean_motion * 2 * pi / self.constants.SECONDS_IN_DAY
        a = (self.constants.GM_EARTH / (n ** 2)) ** (1/3)
        apogee_m = a * (1 + eccentricity) - self.constants.EARTH_RADIUS_M
        return apogee_m / 1000  # Convert to km