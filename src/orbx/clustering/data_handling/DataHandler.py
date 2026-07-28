import pandas as pd
import numpy as np
from orbx.Configs import OrbitalConstants
from orbx.Models import Satellite
from math import pi

class DataHandler:
    
    def __init__(self):
        self.orbital_constants = OrbitalConstants()

    def get_points(self, df: pd.DataFrame):
            """Takes in a dataframe of Satellite objects. Converts each to a point in the 5D manifold embedded in 6D.
            This is so the raw data can be passed into clustering algs, quality metrics, etc.

            Args:
                df (pd.DataFrame): _description_

            Returns:
                _type_: _description_
            """

            # Convert degrees -> radians
            i = np.deg2rad(df["inclination"].values)
            Omega = np.deg2rad(df["raan"].values)
            omega = np.deg2rad(df["argument_of_perigee"].values)
            e = df["eccentricity"].values
            n = df["mean_motion"].values  # rev/day

            # Constants
            MU = self.orbital_constants.GM_EARTH  # m^3/s^2

            # Semi-major axis from mean motion
            n_rad = 2 * np.pi * n / 86400.0
            a = (MU / n_rad**2) ** (1 / 3)

            # Semi-latus rectum
            p = a * (1 - e**2)
            sqrt_p = np.sqrt(p)

            # Angular momentum vector u
            u = np.column_stack(
                [
                    sqrt_p * np.sin(i) * np.sin(Omega),
                    -sqrt_p * np.sin(i) * np.cos(Omega),
                    sqrt_p * np.cos(i),
                ]
            )

            # LRL vector v
            v = np.column_stack(
                [
                    e
                    * sqrt_p
                    * (
                        np.cos(omega) * np.cos(Omega)
                        - np.cos(i) * np.sin(omega) * np.sin(Omega)
                    ),
                    e
                    * sqrt_p
                    * (
                        np.cos(omega) * np.sin(Omega)
                        + np.cos(i) * np.sin(omega) * np.cos(Omega)
                    ),
                    e * sqrt_p * (np.sin(i) * np.sin(omega)),
                ]
            )

            X = np.hstack([u, v])

            return X

    def tle_to_keplerian(self, input_df) -> pd.DataFrame:
        """Parse TLEs into Keplerian columns on a copy; does not change input_df (anymore lol)."""
        df = input_df.copy()

        # Pre-allocate lists for keplerian elements
        sat_nos = []
        inclinations = []
        apogees = []
        raans = []
        arguments_of_perigee = []
        eccentricities = []
        mean_motions = []

        for index, row in df.iterrows():
            try:
                sat_obj = self._parse_tle_group(
                    row['line1'],
                    row['line2']
                )
            except ValueError as e:
                raise ValueError(f"Error parsing TLE at row: {index}, TLE: {row['line1']}, {row['line2']}, Error: {e}")

            sat_nos.append(sat_obj.sat_no)
            inclinations.append(sat_obj.inclination)
            apogees.append(sat_obj.apogee)
            raans.append(sat_obj.raan)
            arguments_of_perigee.append(sat_obj.argument_of_perigee)
            eccentricities.append(sat_obj.eccentricity)
            mean_motions.append(sat_obj.mean_motion)

        df['satNo'] = sat_nos
        df['inclination'] = inclinations
        df['apogee'] = apogees
        df['raan'] = raans
        df['argument_of_perigee'] = arguments_of_perigee
        df['eccentricity'] = eccentricities
        df['mean_motion'] = mean_motions

        return df
    
    def _parse_tle_group(self, line1: str, line2: str) -> Satellite:

        sat_no = line1[2:7].strip()

        # should use sgp4 to extract keplerians
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
        n = mean_motion * 2 * pi / self.orbital_constants.SECONDS_IN_DAY
        a = (self.orbital_constants.GM_EARTH / (n ** 2)) ** (1/3)
        apogee_m = a * (1 + eccentricity) - self.orbital_constants.EARTH_RADIUS_M
        return apogee_m / 1000  # Convert to km


