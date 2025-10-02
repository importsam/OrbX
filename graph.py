import random
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

class Grapher:
    """Creates visualizations of satellite clusters"""
    
    def plot_clusters(self, df: pd.DataFrame, output_path: Path):
        """Create scatter plot of satellite clusters"""
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