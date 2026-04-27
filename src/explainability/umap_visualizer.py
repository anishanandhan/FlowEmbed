"""
UMAP Visualizer — Embedding space visualization with UMAP and t-SNE.

Creates the "killer demo visual" — 2D scatter plots showing traffic type
clusters separating in embedding space. YouTube and Netflix cluster together,
gaming clusters separately, malware is an isolated outlier.
"""

import numpy as np
from typing import Optional, Dict, List
import logging
from pathlib import Path

from src.config import UMAP_N_NEIGHBORS, UMAP_MIN_DIST, UMAP_N_COMPONENTS

logger = logging.getLogger(__name__)


def compute_umap(
    embeddings: np.ndarray,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    min_dist: float = UMAP_MIN_DIST,
    n_components: int = UMAP_N_COMPONENTS,
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduce embeddings to 2D using UMAP.

    Args:
        embeddings: High-dimensional embeddings [N, embedding_dim].
        n_neighbors: UMAP parameter controlling local vs global structure.
        min_dist: UMAP parameter controlling cluster tightness.

    Returns:
        2D coordinates [N, 2].
    """
    import umap

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric="cosine",
        random_state=random_state,
    )

    coords = reducer.fit_transform(embeddings)
    logger.info(f"UMAP reduction: {embeddings.shape} → {coords.shape}")
    return coords


def compute_tsne(
    embeddings: np.ndarray,
    perplexity: float = 30.0,
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduce embeddings to 2D using t-SNE (alternative visualization).
    """
    from sklearn.manifold import TSNE

    reducer = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        metric="cosine",
    )

    coords = reducer.fit_transform(embeddings)
    logger.info(f"t-SNE reduction: {embeddings.shape} → {coords.shape}")
    return coords


def plot_embeddings(
    coords: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[Dict[int, str]] = None,
    title: str = "Flow Embedding Space (UMAP)",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8),
    dark_theme: bool = True,
) -> None:
    """
    Create a scatter plot of 2D embeddings colored by traffic class.

    Args:
        coords: 2D coordinates [N, 2].
        labels: Integer class labels [N].
        label_names: Dict mapping label IDs to names.
        title: Plot title.
        save_path: Optional path to save the plot.
        dark_theme: Use dark background (matching dashboard).
    """
    import matplotlib.pyplot as plt
    import matplotlib

    if dark_theme:
        plt.style.use("dark_background")
        bg_color = "#0f0f1a"
        text_color = "#e0e0e0"
    else:
        bg_color = "white"
        text_color = "black"

    # Curated color palette for traffic classes
    colors = [
        "#FF6B6B",  # streaming - coral red
        "#4ECDC4",  # gaming - teal
        "#45B7D1",  # voip - sky blue
        "#F7DC6F",  # social_media - gold
        "#BB8FCE",  # browsing - purple
        "#82E0AA",  # file_transfer - green
        "#F0B27A",  # vpn - orange
        "#E74C3C",  # malware_c2 - bright red
        "#00D2FF",  # xr_ar - cyan
    ]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    unique_labels = np.unique(labels)

    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = label_names.get(label, f"Class {label}") if label_names else f"Class {label}"
        color = colors[i % len(colors)]

        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color,
            label=name,
            alpha=0.7,
            s=20,
            edgecolors="none",
        )

    ax.set_title(title, fontsize=16, fontweight="bold", color=text_color, pad=15)
    ax.set_xlabel("UMAP-1", fontsize=12, color=text_color)
    ax.set_ylabel("UMAP-2", fontsize=12, color=text_color)
    ax.tick_params(colors=text_color)

    legend = ax.legend(
        loc="upper right",
        fontsize=10,
        framealpha=0.3,
        facecolor=bg_color,
        edgecolor=text_color,
    )
    for text in legend.get_texts():
        text.set_color(text_color)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=bg_color)
        logger.info(f"Saved embedding plot to {save_path}")

    plt.close()


def get_plotly_data(
    coords: np.ndarray,
    labels: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    label_names: Optional[Dict[int, str]] = None,
) -> dict:
    """
    Generate Plotly-compatible JSON data for the React dashboard.

    Returns a dict that can be sent via WebSocket to the frontend
    for interactive visualization.
    """
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#F7DC6F", "#BB8FCE",
        "#82E0AA", "#F0B27A", "#E74C3C", "#00D2FF",
    ]

    traces = []
    unique_labels = np.unique(labels)

    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = label_names.get(int(label), f"Class {label}") if label_names else f"Class {label}"

        trace = {
            "x": coords[mask, 0].tolist(),
            "y": coords[mask, 1].tolist(),
            "mode": "markers",
            "type": "scatter",
            "name": name,
            "marker": {
                "color": colors[i % len(colors)],
                "size": 6,
                "opacity": 0.7,
            },
        }

        if confidences is not None:
            trace["text"] = [
                f"Confidence: {c:.2%}" for c in confidences[mask]
            ]
            trace["hoverinfo"] = "text+name"

        traces.append(trace)

    layout = {
        "title": "Flow Embedding Space (UMAP)",
        "paper_bgcolor": "#0f0f1a",
        "plot_bgcolor": "#0f0f1a",
        "font": {"color": "#e0e0e0"},
        "xaxis": {"title": "UMAP-1", "gridcolor": "#1a1a2e"},
        "yaxis": {"title": "UMAP-2", "gridcolor": "#1a1a2e"},
        "legend": {"bgcolor": "rgba(0,0,0,0.3)"},
    }

    return {"data": traces, "layout": layout}
