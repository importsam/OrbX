from data_handler import DataHandler
from graph import Grapher  # assuming your Grapher is in grapher.py
import numpy as np

from sklearn.manifold import TSNE
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt

class SparseOrbits:
    
    def __init__(self): 
        self.data_handler = DataHandler(
            inclination_range=[0, 180],
            apogee_range=[300, 700]
        )
        self.grapher = Grapher()
        
    def graph_tsne(self):
        """
        Load all satellites in the configured ranges and
        make an unsupervised t-SNE plot of all points.
        """
        # Load data & features
        data = self.data_handler.load_data()
        df = data["orbit_df"]      # pandas DataFrame with columns like satNo, inclination, apogee, density
        X = data["X"]              # numpy array (features for t-SNE)

        # Sanity check (optional)
        if df.empty or X.size == 0:
            print("No data available for t-SNE.")
            return

        # No labels => color by inclination (like your unsupervised branch)
        self.grapher.plot_tsne(
            X=X,
            df=df,
            name="all_points",
            labels=None
        )

