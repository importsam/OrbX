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
    ):
        """
        Internal t-SNE plotter for SparseOrbits.
        Produces a single PNG, colouring points by inclination.
        """
        print("\nRunning t-SNE and generating PNG...\n")

        # t-SNE
        tsne = TSNE(
            n_components=2,
            random_state=42,
            init="pca",
            perplexity=75,
        )
        X_2d = tsne.fit_transform(X)

        # Matplotlib PNG
        plt.figure(figsize=(9, 7), dpi=150)
        sc = plt.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            c=df["inclination"],
            cmap="viridis",
            s=10,
            alpha=0.7,
        )
        cbar = plt.colorbar(sc)
        cbar.set_label("Inclination (deg)")

        title = f"t-SNE: Orbital Points ({name})"
        plt.title(title)
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.tight_layout()

        output_file_png = (
            self.path_config.output_plot
            / f"tsne_sparse_orbits_{name if name else 'None'}.png"
        )
        plt.savefig(output_file_png, dpi=150)
        plt.close()

        print(f"t-SNE PNG saved to {output_file_png}")


    def nearest_neighbors(self, df):
        """
        Create a dataframe containing the distance to the 
        kth neighbor. 
        This can be used later for ranking 
        """
        
        distance_matrix, key = get_distance_matrix(df)
        
        