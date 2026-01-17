from dataclasses import dataclass, field
from typing import Optional

@dataclass 
class OrbitalRegime:
    apogee_min: Optional[float] = None
    apogee_max: Optional[float] = None
    incl_min: Optional[float] = None
    incl_max: Optional[float] = None
    
@dataclass 
class OrbitalRegimes:
    
    @staticmethod 
    def all():
        return OrbitalRegime()
    
    @staticmethod 
    def leo():
        return OrbitalRegime(
            apogee_min = 0,
            apogee_max=2000
        )

@dataclass
class CZMLConfig:
    
    density_percentile: Optional[float] = 0.99
    radius: float = 50
    
    orbital_regime: OrbitalRegime = field(default_factory=OrbitalRegime)
    
    @property 
    def apogee_min(self):
        return self.orbital_regime.apogee_min
    
    @property
    def apogee_max(self):
        return self.orbital_regime.apogee_max
    
    @property
    def incl_min(self):
        return self.orbital_regime.incl_min
    
    @property
    def incl_max(self):
        return self.orbital_regime.incl_max
