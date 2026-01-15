from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score
import numpy as np
from tqdm import tqdm


class KMeansWrapper:

    def __init__(
        self,
        k_max: int = 100,
        k_min: int = 2,
        patience: int = 100,
        random_state: int = 42
    ):
        self.k_max = k_max
        self.k_min = k_min
        self.patience = patience
        self.random_state = random_state

    def run(self, distance_matrix: np.ndarray, X: np.ndarray):
        # print size before
        print(f"Running KMeans on {len(X)} points")
        # --- 1. Remove outliers ---
        lof = LocalOutlierFactor()
        inliers = lof.fit_predict(X) == 1

        clean_X = X[inliers]
        clean_dist = distance_matrix[np.ix_(inliers, inliers)]
        print(f"Removed {len(X) - len(clean_X)} outliers, {len(clean_X)} points remain")

        best_k = None
        best_silhouette = -1.0
        best_labels = None
        history = []

        no_improve_count = 0

        # --- 2. Sweep k from large → small ---
        for k in tqdm(
            range(self.k_max, self.k_min - 1, -1),
            desc="KMeans k sweep"
        ):

            if k >= len(clean_X):
                continue

            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state
            )

            labels = kmeans.fit_predict(clean_X)

            # Skip degenerate cases
            if len(np.unique(labels)) < 2:
                continue

            try:
                sil = silhouette_score(
                    clean_dist,
                    labels,
                    metric="precomputed"
                )

                history.append({
                    "k": k,
                    "Silhouette": sil
                })

                print(f"k={k:2d} | Silhouette={sil:.4f}")

                # --- 3. Early stopping based on silhouette ---
                if sil > best_silhouette:
                    best_silhouette = sil
                    best_k = k
                    best_labels = labels
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count >= self.patience:
                    print(f"\nEarly stopping at k={k}")
                    break

            except Exception as e:
                print(f"k={k} failed: {e}")

        # --- Final report ---
        print("\n=== Best clustering result ===")
        print(f"Best k          : {best_k}")
        print(f"Best silhouette : {best_silhouette:.4f}")
        
        full_labels = np.full(len(X), -1)  # -1 = outlier
        full_labels[inliers] = best_labels

        return full_labels

