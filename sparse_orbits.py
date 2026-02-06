from data_handler import DataHandler
import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from configs import PathConfig
from tools.distance_matrix import get_distance_matrix

class SparseOrbits:
    
    def __init__(self): 
        self.data_handler = DataHandler(
            inclination_range=[0, 180],
            apogee_range=[300, 700]
        )
        self.path_config = PathConfig()  # to get output_plot dir
        
    def graph_tsne(self, name: str = "all_points"):
        """
        Load all satellites in the configured ranges and
        make an unsupervised t-SNE plot of all points (PNG only).
        """
        data = self.data_handler.load_data()
        df = data["orbit_df"]      # DataFrame with satNo, inclination, apogee, density, etc.
        X = data["X"]              # feature matrix for t-SNE

        if df.empty or X.size == 0:
            print("No data available for t-SNE.")
            return

        self._plot_tsne_png(X=X, df=df, name=name)

    def _plot_tsne_png(
        self,
        X: np.ndarray,
        df,
        name: str = "all_points",
        highlight_indices: np.ndarray | None = None,
    ):
        """
        Internal t-SNE plotter for SparseOrbits.
        Produces a single PNG, colouring points by inclination.
        Optionally highlights selected points in red.
        """
        print("\nRunning t-SNE and generating PNG...\n")

        tsne = TSNE(
            n_components=2,
            random_state=42,
            init="pca",
            perplexity=75,
        )
        X_2d = tsne.fit_transform(X)

        plt.figure(figsize=(9, 7), dpi=150)

        # base scatter: all points in blue
        plt.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            c="tab:blue",
            s=10,
            alpha=0.6,
            edgecolor="none",
        )

        # overlay: highlight selected indices in red
        if highlight_indices is not None and len(highlight_indices) > 0:
            hi = np.asarray(highlight_indices, dtype=int)
            plt.scatter(
                X_2d[hi, 0],
                X_2d[hi, 1],
                c="red",
                s=30,
                alpha=0.9,
                edgecolor="black",
                linewidths=0.5,
                label="Most isolated orbits",
            )

            satnos = df.iloc[hi]["satNo"].astype(str).values

            for (idx, x, y, s) in zip(hi, X_2d[hi, 0], X_2d[hi, 1], satnos):
                # default: bottom-left
                offset = (-6, -5)
                ha, va = "right", "top"

                if s == "54880":
                    # top-left for satNo 54880
                    offset = (-6, 5)
                    ha, va = "right", "bottom"

                plt.annotate(
                    s,
                    (x, y),
                    textcoords="offset points",
                    xytext=offset,
                    ha=ha,
                    va=va,
                    fontsize=8,
                    color="black",
                )


        plt.title("Isolated Orbits in LEO (300-700km, k=10)")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        if highlight_indices is not None:
            plt.legend(loc="best")
        plt.tight_layout()

        output_file_png = (
            self.path_config.output_plot
            / f"tsne_sparse_orbits_{name if name else 'None'}.png"
        )
        plt.savefig(output_file_png, dpi=150)
        plt.close()

        print(f"t-SNE PNG saved to {output_file_png}")



    @staticmethod
    def knn_isolation_scores(distance_matrix: np.ndarray, k: int = 10) -> np.ndarray:
        """
        For each point i, compute the mean distance to its k nearest neighbors (excluding itself).
        Returns a 1D array scores[i] = mean_kNN_distance(i).
        """
        n = distance_matrix.shape[0]
        if k >= n:
            raise ValueError("k must be smaller than number of points")

        # copy to avoid touching original
        dist = distance_matrix.copy()

        # set diagonal to +inf so self-distance is never picked as a neighbor
        np.fill_diagonal(dist, np.inf)

        # sort distances for each row, take k smallest, then average
        # argsort gives indices, but we can just sort the distances directly
        # axis=1: sort each row independently
        sorted_dists = np.sort(dist, axis=1)
        knn_dists = sorted_dists[:, :k]
        scores = knn_dists.mean(axis=1)

        return scores
    
    def graph_tsne_with_isolated(self, top_n: int = 3, name: str = "isolated"):
        """
        Compute k-NN isolation scores from the orbital distance matrix,
        pick the top-N most isolated orbits, and plot them in red on t-SNE.
        """
        data = self.data_handler.load_data()
        df = data["orbit_df"]
        X = data["X"]

        if df.empty or X.size == 0:
            print("No data available for t-SNE / isolation.")
            return

        # compute distance matrix on the same df used for X
        distance_matrix, key = get_distance_matrix(df)
        # key: {'satNo_idx_dict': {satNo: idx}, 'idx_satNo_dict': {idx: satNo}}

        scores = self.knn_isolation_scores(distance_matrix)

        # get indices of the top-N largest scores (most isolated)
        # argsort ascending, so take last N
        idx_sorted = np.argsort(scores)
        top_indices = idx_sorted[-top_n:]

        print("Top isolated orbits (index, satNo, score):")
        idx_to_satno = key["idx_satNo_dict"]
        for i in top_indices:
            satno = idx_to_satno[int(i)]
            print(f"  idx={i}, satNo={satno}, mean_kNN_dist={scores[i]:.3e}")

        # t-SNE + highlight
        self._plot_tsne_png(X=X, df=df, name=name, highlight_indices=top_indices)

        # histogram of k-NN isolation scores
        self.plot_knn_histogram(
            scores=scores,
            k=10,               # or pass k as a parameter to graph_tsne_with_isolated
            top_indices=top_indices,
            name=name,
        )


    def plot_knn_histogram(
        self,
        scores: np.ndarray,
        k: int = 10,
        top_indices: np.ndarray | None = None,
        name: str = "knn_hist",
    ):
        # keep only positive scores (log-scale requirement)
        scores = scores[scores > 0]

        vmin = scores.min()
        vmax = scores.max()

        # log-spaced bins
        bins = np.logspace(np.log10(vmin), np.log10(vmax), 40)

        plt.figure(figsize=(8, 5), dpi=150)

        plt.hist(scores, bins=bins, color="tab:blue", alpha=0.7, edgecolor="black")

        if top_indices is not None and len(top_indices) > 0:
            for idx in np.asarray(top_indices, dtype=int):
                if scores[idx] > 0:
                    plt.axvline(
                        scores[idx],
                        color="red",
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.9,
                    )

        plt.xscale("log")
        plt.xlabel(f"Mean distance to 10-NN (log)")
        plt.ylabel("Number of orbits")
        plt.title(r"Distribution of Mean 10-NN Distances ($\rho_{k}$)")
        plt.tight_layout()

        output_file_png = (
            self.path_config.output_plot
            / f"knn_isolation_hist_{name if name else 'None'}.png"
        )
        plt.savefig(output_file_png, dpi=150)
        plt.close()

        print(f"k-NN isolation histogram saved to {output_file_png}")
