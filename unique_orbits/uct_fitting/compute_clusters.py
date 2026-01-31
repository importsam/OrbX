from sklearn.cluster import AffinityPropagation
import numpy as np

def compute_clusters(distance_matrix, damping):
    
    kmeans = AffinityPropagation(affinity='precomputed', damping=damping)
    kmeans.fit(np.exp(-distance_matrix/np.var(distance_matrix)))

    # Get the cluster labels
    labels = kmeans.labels_

    return labels

if __name__ == '__main__':
    import pickle
    distance_matrix= '/home/yasir/drive/datasets/distance_matrix.pkl'

    with open(distance_matrix, 'rb') as f:
        distance_matrix = pickle.load(f)
        
        
    distance_matrix = np.array(distance_matrix)
    distance_matrix = distance_matrix[::30,::30]
    labels = compute_clusters(distance_matrix)

    print(labels)