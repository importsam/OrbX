"""
Converts keplerian elements to TLE using orekit.
"""

import orekit
orekit.initVM()

from orekit.pyhelpers import setup_orekit_curdir
setup_orekit_curdir()
import os
if not os.path.isdir("orekit-data"):
    orekit.pyhelpers.download_orekit_data_curdir()

import numpy as np
from org.orekit.propagation.analytical.tle import TLE
from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from org.orekit.frames import FramesFactory
from org.orekit.utils import Constants
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.analytical.tle.generation import LeastSquaresTleGenerationAlgorithm
from unique_orbits.uct_fitting.orbit_finder.DMT import VectorizedKeplerianOrbit

def test_orbit(df):
    """
    Take the TLE generated from the optimum keplerian elements and verify that
    it still has the lowest average distance to all provided neighbouring orbits.
    """

    import numpy as np

    def get_distance_matrix(df):
        line1 = df['line1'].values
        line2 = df['line2'].values
        
        orbits = VectorizedKeplerianOrbit(line1, line2)
        
        distance_matrix = VectorizedKeplerianOrbit.DistanceMetric(orbits, orbits)
                    
        return distance_matrix

    # Compute the distance matrix for the provided TLE dataframe.
    distance_matrix = get_distance_matrix(df)
    
    # Compute the average distance for each orbit (row).
    avg_distance = np.mean(distance_matrix, axis=1)
    
    # Sort indexes by average distance in ascending order.
    sorted_indexes = np.argsort(avg_distance)
    
    # Create a sorted list of tuples: (TLE line1, average distance)
    sorted_candidates = []
    print("\nVerification: Sorted results by average distance:")
    for idx in sorted_indexes:
        tle_line1 = df.iloc[idx]['line1']
        candidate_distance = avg_distance[idx]
        sorted_candidates.append((tle_line1, candidate_distance))
        print(f"{idx}: line1: {tle_line1} | Average distance: {candidate_distance}")
    print("\n")
    
    return sorted_candidates

def convert_kep_to_tle(keplerian_elements, mean_anomaly, initialDate):

    # Define TLE meta-data
    satNo = 99999
    classification = 'U'
    launchYear = 2020
    launchNumber = 42
    launchPiece = 'A'
    bStar = 1e-5
    mean_motion_first_derivative = 0.0
    mean_motion_second_derivative = 0.0
    ephemeris_type = 0
    element_number = 999
    revolution_number = 100

    # Create the orbit
    try:
        # get python datetime object for right now in UTC
        a = float(keplerian_elements[0] * 1000) 
        e = float(abs(keplerian_elements[1]))
        i = float(keplerian_elements[2])
        omega = float(keplerian_elements[3])
        raan = float(keplerian_elements[4])
        M = float(mean_anomaly)
                                
        orbitToConvert = KeplerianOrbit(
            a,
            e,
            i,
            omega,
            raan,
            M,
            PositionAngleType.MEAN, # anomaly type
            FramesFactory.getEME2000(), # inertial frame
            initialDate, # epoch date
            Constants.IERS2010_EARTH_MU  # Earth's gravitational parameter
        )
        
    except Exception as ex:
        print(f"\nDetailed orbit creation error:")
        import traceback
        traceback.print_exc()
        raise
    
    print(orbitToConvert)

    # Create the initial spacecraft state
    initial_state = SpacecraftState(orbitToConvert)
    
    mean_motion = float(np.sqrt(Constants.EIGEN5C_EARTH_MU / np.power(a, 3)))

    # Build the template TLE
    try:
        templateTLE = TLE(
            satNo, 
            classification,
            launchYear,
            launchNumber,
            launchPiece,
            ephemeris_type,
            element_number,
            initialDate,
            mean_motion,
            mean_motion_first_derivative, 
            mean_motion_second_derivative,
            e,
            i,
            omega,
            raan,
            M,
            revolution_number,
            bStar
        )
        
    except Exception as ex:
        print(f"\nDetailed TLE creation error:")
        import traceback
        traceback.print_exc()
        raise

    print("\nTemplate TLE:")
    print(templateTLE.getLine1())
    print(templateTLE.getLine2())

    # Generate the fitted TLE
    fixedPoint = LeastSquaresTleGenerationAlgorithm()
    fittedTLE = TLE.stateToTLE(initial_state, templateTLE, fixedPoint)
    
    print("\nGenerated TLE:")
    print(fittedTLE.getLine1())
    print(fittedTLE.getLine2())

    # return (fittedTLE.getLine1(), fittedTLE.getLine2())
    return (templateTLE.getLine1(), templateTLE.getLine2())