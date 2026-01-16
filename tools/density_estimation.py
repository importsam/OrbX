import numpy as np

class DensityEstimator:
    def __init__(self):
        self.radius = 100
    
    def density(self, distance_matrix, r):
        N_i = (distance_matrix <= r).sum(axis=1)
        V = (np.pi**3 / 6) * r**6
        densities = N_i / V
        return densities

    def assign_density(self, elset_df, distance_matrix):
        densities = self.density(distance_matrix, self.radius)
        elset_df = elset_df.copy()
        elset_df['density'] = densities
        return elset_df