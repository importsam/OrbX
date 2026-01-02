from clustering_algs.affinity_propagation import AffinityPropagation
from clustering_algs.OPTICS import OPTICS
from clustering_algs.StarCluster import StarCluster
from clustering_algs.agglomerative_clustering import AgglomerativeClusterer


class ClusterWrapper:
    def __init__(self):
        self.affinity_propagation = AffinityPropagation()
        self.optics = OPTICS()
        self.star_cluster = StarCluster()
        self.agglomerative = AgglomerativeClusterer()
        
    def run_all(self, distance_matrix):
        affinity_labels = self.affinity_propagation.fit(distance_matrix)
        print(f"Affinity Propagation found {len(set(affinity_labels))} clusters")
        
        # optics_labels = self.optics.fit(distance_matrix)
        # print(f"OPTICS found {len(set(optics_labels))} clusters")
        
        # star_labels = self.star_cluster.fit(distance_matrix)
        # print(f"Star Clustering found {len(set(star_labels))} clusters")
        
        # agglomerative_labels = self.agglomerative.fit(distance_matrix)
        # print(f"Agglomerative Clustering found {len(set(agglomerative_labels))} clusters")
        
        