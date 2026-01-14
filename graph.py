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
    
    def plot_tsne(
        self,
        X: np.ndarray,
        df: pd.DataFrame,
        name: str = "None",
        labels: np.ndarray = None
    ):
        print("\nRunning t-SNE and generating interactive plot...\n")

        tsne = TSNE(
            n_components=2,
            random_state=42,
            init='pca',
            perplexity=30
        )
        X_2d = tsne.fit_transform(X)

        fig = go.Figure()

        if labels is not None:
            # Colour by cluster labels
            unique_labels = np.unique(labels)

            for lbl in unique_labels:
                mask = labels == lbl
                fig.add_trace(go.Scatter(
                    x=X_2d[mask, 0],
                    y=X_2d[mask, 1],
                    mode='markers',
                    name=f'Cluster {lbl}',
                    marker=dict(
                        size=6,
                        opacity=0.7
                    ),
                    text=df.loc[mask, 'satNo'],
                    hovertemplate=(
                        "SatNo: %{text}<br>"
                        "t-SNE 1: %{x}<br>"
                        "t-SNE 2: %{y}<br>"
                        f"Cluster: {lbl}<br>"
                        "Inclination: %{customdata[0]}°<br>"
                        "Apogee: %{customdata[1]} km"
                        "<extra></extra>"
                    ),
                    customdata=df.loc[mask, ['inclination', 'apogee']].values
                ))

            title = f"t-SNE: Orbital Points by Clusters ({name})"

        else:
            # Colour by inclination (continuous)
            fig.add_trace(go.Scatter(
                x=X_2d[:, 0],
                y=X_2d[:, 1],
                mode='markers',
                marker=dict(
                    size=6,
                    opacity=0.7,
                    color=df['inclination'],
                    colorscale='Viridis',
                    colorbar=dict(title='Inclination (deg)')
                ),
                text=df['satNo'],
                hovertemplate=(
                    "SatNo: %{text}<br>"
                    "t-SNE 1: %{x}<br>"
                    "t-SNE 2: %{y}<br>"
                    "Inclination: %{marker.color}°<br>"
                    "Apogee: %{customdata} km"
                    "<extra></extra>"
                ),
                customdata=df['apogee']
            ))

            title = f"t-SNE: Orbital Points by Inclination ({name})"

        fig.update_layout(
            title=title,
            xaxis_title="t-SNE Component 1",
            yaxis_title="t-SNE Component 2",
            template="plotly_white",
            width=900,
            height=700,
            legend_title="Clusters"
        )

        output_file = self.path_config.output_plot / f"tsne_orbital_points_{name if name else 'None'}.html"
        fig.write_html(str(output_file), include_plotlyjs="cdn")

        print(f"t-SNE plot saved to {output_file}")

