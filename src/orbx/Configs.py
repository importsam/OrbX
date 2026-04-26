from dataclasses import dataclass

@dataclass
class OrbitalConstants:
    EARTH_RADIUS_M = 6371000  # meters
    GM_EARTH = 3.986004418e14  # m^3/s^2
    SECONDS_IN_DAY = 86400