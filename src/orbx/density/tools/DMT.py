from sgp4.api import Satrec
import numpy as np

"""
This file contains the code for the distance metric between two orbits. 
It takes the two-line elements, breaks it down into keplerian elements,
converts those to two different vectors:
1. The vector which  is normal to the orbital plane (3 components)
2. The LRL vector. (3 components)

This makes each orbit represented as a point in a 6D space.

The distance metric is then the Euclidean distance between these two points. If they are very similar,
the distances are going to be quite small, and that distance will grow as they differ.

Please refer to the following paper for details:
https://academic.oup.com/mnras/article/462/2/2275/2589463

Credit to Dr. Yasir Latif (of Space Protocol) for the translation of the concept from paper to the Python code in this file.
"""


class VectorizedKeplerianOrbit:
    def __init__(self, line1_array, line2_array=None):
        if line2_array is not None:
            satellites = np.vectorize(Satrec.twoline2rv)(line1_array, line2_array)
            radius_earth_km = satellites[0].radiusearthkm

            self.a = np.array([sat.a * radius_earth_km for sat in satellites])
            self.e = np.array([sat.ecco for sat in satellites])
            self.i = np.array([sat.inclo for sat in satellites])
            self.omega = np.array([sat.argpo for sat in satellites])
            self.raan = np.array([sat.nodeo for sat in satellites])
            self.q = np.array([sat.altp * radius_earth_km for sat in satellites])
            self.p = self.a * (1 - self.e ** 2)
        else:
            self.a, self.e, self.i, self.omega, self.raan, self.q, self.p = line1_array

    def __getitem__(self, key):
        return VectorizedKeplerianOrbit(
            np.array([self.a[key], self.e[key], self.i[key], self.omega[key], self.raan[key], self.q[key], self.p[key]])
        )

    @staticmethod
    def DistanceMetric(orbit1, orbit2):
        w1 = orbit1.omega[:, np.newaxis]
        w2 = orbit2.omega[np.newaxis, :]

        c1 = np.cos(orbit1.i)[:, np.newaxis]
        c2 = np.cos(orbit2.i)[np.newaxis, :]

        s1 = np.sin(orbit1.i)[:, np.newaxis]
        s2 = np.sin(orbit2.i)[np.newaxis, :]

        delta = orbit1.raan[:, np.newaxis] - orbit2.raan[np.newaxis, :]
        cosI = c1 * c2 + s1 * s2 * np.cos(delta)

        cosP = s1 * s2 * np.sin(w1) * np.sin(w2) + \
               (np.cos(w1) * np.cos(w2) + c1 * c2 * np.sin(w1) * np.sin(w2)) * np.cos(delta) + \
               (c2 * np.cos(w1) * np.sin(w2) - c1 * np.sin(w1) * np.cos(w2)) * np.sin(delta)

        q = (1 + orbit1.e[:, np.newaxis]**2) * orbit1.p[:, np.newaxis] + \
            (1 + orbit2.e[np.newaxis, :]**2) * orbit2.p[np.newaxis, :] - \
            2 * np.sqrt(orbit1.p[:, np.newaxis] * orbit2.p[np.newaxis, :]) * \
            (cosI + orbit1.e[:, np.newaxis] * orbit2.e[np.newaxis, :] * cosP)

        return q
        