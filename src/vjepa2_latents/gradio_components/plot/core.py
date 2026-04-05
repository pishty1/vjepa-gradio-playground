from __future__ import annotations

from typing import Sequence

import numpy as np

from ..projection.core import (
    compute_projection_bundle,
    projection_component_labels,
    projection_method_display_name,
)

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - exercised only when optional dependency is missing
    go = None


def _validate_component_indices(component_indices: Sequence[int], component_count: int) -> tuple[int, ...]:
    indices = tuple(int(index) for index in component_indices)
    if len(indices) not in {2, 3}:
        raise ValueError("Select exactly 2 or 3 components")
    if len(set(indices)) != len(indices):
        raise ValueError("Selected components must be distinct")
    if min(indices) < 0 or max(indices) >= component_count:
        raise ValueError("Selected component index is out of bounds")
    return indices


def build_projection_figure_from_data(
    projection: np.ndarray,
    coordinates: np.ndarray,
    *,
    method: str = "pca",
    component_indices: Sequence[int] = (0, 1, 2),
    component_labels: Sequence[str] | None = None,
    max_points: int = 4000,
):
    if go is None:
        raise RuntimeError("plotly is required to build the interactive latent-space plot")
    if projection.ndim != 2:
        raise ValueError(f"Expected 2D projection array, got shape {projection.shape}")
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"Expected coordinates with shape [n, 3], got {coordinates.shape}")
    if projection.shape[0] != coordinates.shape[0]:
        raise ValueError("Projection rows and coordinates rows must match")

    indices = _validate_component_indices(component_indices, projection.shape[1])
    reduced_projection = projection[:, indices]
    labels = list(component_labels) if component_labels is not None else projection_component_labels(method, projection.shape[1])
    selected_labels = [labels[index] if index < len(labels) else f"C{index + 1}" for index in indices]

    if reduced_projection.shape[0] > max_points:
        selection = np.linspace(0, reduced_projection.shape[0] - 1, num=max_points, dtype=int)
        reduced_projection = reduced_projection[selection]
        coordinates = coordinates[selection]

    colors = coordinates[:, 0]
    hover_text = [
        f"t={time_index}, h={row_index}, w={column_index}"
        for time_index, row_index, column_index in coordinates.tolist()
    ]

    if len(indices) == 3:
        hover_template = (
            f"%{{text}}<br>{selected_labels[0]}=%{{x:.3f}}<br>{selected_labels[1]}=%{{y:.3f}}<br>{selected_labels[2]}=%{{z:.3f}}<extra></extra>"
        )
        trace = go.Scatter3d(
            x=reduced_projection[:, 0],
            y=reduced_projection[:, 1],
            z=reduced_projection[:, 2],
            mode="markers",
            marker={
                "size": 4,
                "color": colors,
                "colorscale": "Viridis",
                "opacity": 0.85,
                "colorbar": {"title": "Latent t"},
            },
            text=hover_text,
            hovertemplate=hover_template,
        )
        figure = go.Figure(data=[trace])
        figure.update_layout(
            title=f"Latent-space {projection_method_display_name(method)} projection",
            scene={
                "xaxis_title": selected_labels[0],
                "yaxis_title": selected_labels[1],
                "zaxis_title": selected_labels[2],
            },
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
            template="plotly_white",
        )
        return figure

    hover_template = f"%{{text}}<br>{selected_labels[0]}=%{{x:.3f}}<br>{selected_labels[1]}=%{{y:.3f}}<extra></extra>"
    trace = go.Scatter(
        x=reduced_projection[:, 0],
        y=reduced_projection[:, 1],
        mode="markers",
        marker={
            "size": 7,
            "color": colors,
            "colorscale": "Viridis",
            "opacity": 0.8,
            "colorbar": {"title": "Latent t"},
        },
        text=hover_text,
        hovertemplate=hover_template,
    )
    figure = go.Figure(data=[trace])
    figure.update_layout(
        title=f"Latent-space {projection_method_display_name(method)} projection",
        xaxis_title=selected_labels[0],
        yaxis_title=selected_labels[1],
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        template="plotly_white",
    )
    return figure


def build_projection_figure(
    latent_grid: np.ndarray,
    *,
    method: str = "pca",
    n_components: int = 3,
    max_points: int = 4000,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_metric: str = "euclidean",
    umap_random_state: int | None = 42,
):
    projection_bundle = compute_projection_bundle(
        latent_grid,
        method=method,
        n_components=max(2, int(n_components)),
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        umap_metric=umap_metric,
        umap_random_state=umap_random_state,
    )
    component_count = min(3, int(projection_bundle["projection"].shape[1]))
    return build_projection_figure_from_data(
        projection_bundle["projection"],
        projection_bundle["coordinates"],
        method=str(projection_bundle["method"]),
        component_indices=tuple(range(component_count)),
        component_labels=projection_bundle["component_labels"],
        max_points=max_points,
    )


def build_pca_figure(latent_grid: np.ndarray, max_points: int = 4000):
    return build_projection_figure(latent_grid, method="pca", n_components=3, max_points=max_points)
