import random
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from configs import PathConfig
import numpy as np

class Grapher:
    """Creates visualizations of satellite clusters"""
    
    def __init__(self):
        self.path_config = PathConfig()
    
    def plot_clusters(self, df: pd.DataFrame, output_path: Path):
        print("\nPlotting clusters...\n")
        if df.empty:
            print("No data available to plot.")
            return
        
        unique_labels = sorted(df['label'].unique())
        cluster_colors = self._generate_colors(unique_labels)
        
        fig = self._create_figure(df, unique_labels, cluster_colors)
        fig.write_html(str(output_path), include_plotlyjs="cdn")
        print(f"Plot saved to {output_path}")
    
    def _generate_colors(self, labels: list) -> dict:
        """Generate random colors for each cluster"""
        return {
            label: f"rgb({random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)})"
            for label in labels
        }
    
    def _create_figure(
        self, 
        df: pd.DataFrame, 
        labels: list, 
        colors: dict
    ) -> go.Figure:
        """Create Plotly figure with cluster traces"""
        fig = go.Figure()
        
        for label in labels:
            cluster_df = df[df['label'] == label]
            fig.add_trace(go.Scatter(
                x=cluster_df['inclination'],
                y=cluster_df['apogee'],
                mode='markers',
                marker=dict(size=5, opacity=0.6, color=colors[label]),
                name=f"Cluster {label}",
                text=cluster_df['satNo'],
                hovertemplate=(
                    "SatNo: %{text}<br>"
                    "Inclination: %{x}<br>"
                    "Apogee: %{y} km<br>"
                    f"Cluster: {label}<extra></extra>"
                )
            ))
        
        fig.update_layout(
            title="Satellite clustering",
            xaxis_title="Inclination (degrees)",
            yaxis_title="Apogee (km)",
            template="plotly_white",
            width=900,
            height=700,
            legend_title="Clusters"
        )
        
        return fig
    
    def plot_tsne(self, X: np.ndarray, df: pd.DataFrame):
        tsne = TSNE(n_components=2, random_state=42, init='pca')
        X_2d = tsne.fit_transform(X)

        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df['inclination'], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Inclination (degrees)')
        plt.title('t-SNE Visualization of Orbital Manifold')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.tight_layout()
        plt.savefig(self.path_config.output_plot / 'tsne_orbital_points.png', dpi=300, bbox_inches='tight')