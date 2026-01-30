import numpy as np
from orbdtools import (
    ArcObs,
    Body
)


def main():
    obs_data = np.loadtxt('T25872_KUN2_2.dat',dtype=str,skiprows=1)
    obs_data = obs_data[::10]
    t = obs_data[:,0] # Obsevation time in UTC
    radec = obs_data[:,1:3].astype(float) # Ra and Dec of the space object, in [hour,deg]
    xyz_site = obs_data[:,3:6].astype(float) # Cartesian coordinates of the site in GCRF, in [km]
    radec[:,0] *= 15 # Convert hour angle to degrees
    
    # Load the extracted data to ArcObs, and eliminate outliers using the method of LOWESS.
    arc_optical = ArcObs({'t':t,'radec':radec,'xyz_site':xyz_site})
    arc_optical.lowess_smooth() # Eliminate outliers using the method of LOWESS
    
    # Set the Earth as the central body of attraction.
    earth = Body.from_name('Earth')
    arc_iod = arc_optical.iod(earth)
    
    # Apply gauss method
    arc_iod.gauss(ellipse_only=False)
    print(arc_iod.df.to_string())
    
if __name__ == "__main__":
    main()