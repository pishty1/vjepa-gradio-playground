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


def _limit_projection_rows(
    projection: np.ndarray,
    coordinates: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if projection.shape[0] <= max_points:
        return projection, coordinates
    selection = np.linspace(0, projection.shape[0] - 1, num=max_points, dtype=int)
    return projection[selection], coordinates[selection]


def _limit_projection_rows_by_time(
    projection: np.ndarray,
    coordinates: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if projection.shape[0] <= max_points:
        return projection, coordinates

    time_values = np.unique(coordinates[:, 0])
    if time_values.size == 0:
        return projection, coordinates

    per_time_limit = max(1, max_points // int(time_values.size))
    selected_indices: list[np.ndarray] = []
    for time_value in time_values:
        time_indices = np.flatnonzero(coordinates[:, 0] == time_value)
        if time_indices.size > per_time_limit:
            sampled_indices = time_indices[
                np.linspace(0, time_indices.size - 1, num=per_time_limit, dtype=int)
            ]
        else:
            sampled_indices = time_indices
        selected_indices.append(sampled_indices)

    if not selected_indices:
        return projection, coordinates

    selection = np.concatenate(selected_indices)
    return projection[selection], coordinates[selection]


def _make_scatter_trace(
    reduced_projection: np.ndarray,
    coordinates: np.ndarray,
    selected_labels: Sequence[str],
    *,
    animated: bool,
):
    hover_text = [
        f"t={time_index}, h={row_index}, w={column_index}"
        for time_index, row_index, column_index in coordinates.tolist()
    ]

    if len(selected_labels) == 3:
        trace_kwargs = {
            "x": reduced_projection[:, 0],
            "y": reduced_projection[:, 1],
            "z": reduced_projection[:, 2],
            "mode": "markers",
            "marker": {
                "size": 4,
                **(
                    {"color": "#2563eb", "opacity": 0.8}
                    if animated
                    else {"color": coordinates[:, 0], "colorscale": "Viridis", "opacity": 0.85, "colorbar": {"title": "Latent t"}}
                ),
            },
            "text": hover_text,
            "hovertemplate": (
                f"%{{text}}<br>{selected_labels[0]}=%{{x:.3f}}<br>{selected_labels[1]}=%{{y:.3f}}<br>{selected_labels[2]}=%{{z:.3f}}<extra></extra>"
            ),
        }
        return go.Scatter3d(**trace_kwargs)

    trace_kwargs = {
        "x": reduced_projection[:, 0],
        "y": reduced_projection[:, 1],
        "mode": "markers",
        "marker": {
            "size": 7,
            **(
                {"color": "#2563eb", "opacity": 0.85}
                if animated
                else {"color": coordinates[:, 0], "colorscale": "Viridis", "opacity": 0.8, "colorbar": {"title": "Latent t"}}
            ),
        },
        "text": hover_text,
        "hovertemplate": f"%{{text}}<br>{selected_labels[0]}=%{{x:.3f}}<br>{selected_labels[1]}=%{{y:.3f}}<extra></extra>",
    }
    return go.Scatter(**trace_kwargs)


def _add_animation_controls(figure, frame_names: Sequence[str], *, base_title: str) -> None:
    if len(frame_names) <= 1:
        return

    first_frame = frame_names[0]
    figure.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 1.12,
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            list(frame_names),
                            {
                                "frame": {"duration": 500, "redraw": True},
                                "mode": "immediate",
                                "fromcurrent": True,
                                "transition": {"duration": 150, "easing": "linear"},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Time step: "},
                "pad": {"t": 35},
                "steps": [
                    {
                        "label": frame_name,
                        "method": "animate",
                        "args": [
                            [frame_name],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for frame_name in frame_names
                ],
            }
        ],
    )
    figure.update_layout(title=f"{base_title} · animation starting at {first_frame}")


def build_projection_figure_from_data(
    projection: np.ndarray,
    coordinates: np.ndarray,
    *,
    method: str = "pca",
    component_indices: Sequence[int] = (0, 1, 2),
    component_labels: Sequence[str] | None = None,
    max_points: int = 4000,
    animate_over_time: bool = False,
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

    if animate_over_time:
        reduced_projection, coordinates = _limit_projection_rows_by_time(reduced_projection, coordinates, max_points)
    else:
        reduced_projection, coordinates = _limit_projection_rows(reduced_projection, coordinates, max_points)

    frame_names: list[str] = []
    frames = []
    if animate_over_time:
        time_values = np.unique(coordinates[:, 0])
        for time_value in time_values:
            frame_mask = coordinates[:, 0] == time_value
            frame_projection = reduced_projection[frame_mask]
            frame_coordinates = coordinates[frame_mask]
            if frame_projection.size == 0:
                continue
            frame_name = f"t={int(time_value)}"
            frame_names.append(frame_name)
            frames.append(
                go.Frame(
                    data=[_make_scatter_trace(frame_projection, frame_coordinates, selected_labels, animated=True)],
                    traces=[0],
                    name=frame_name,
                )
            )

    if animate_over_time and frames:
        first_frame = frames[0]
        initial_trace = first_frame.data[0]
    else:
        initial_trace = _make_scatter_trace(reduced_projection, coordinates, selected_labels, animated=animate_over_time)

    if len(indices) == 3:
        figure = go.Figure(data=[initial_trace], frames=frames)
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
        if animate_over_time:
            _add_animation_controls(figure, frame_names, base_title=f"Latent-space {projection_method_display_name(method)} projection")
        return figure

    figure = go.Figure(data=[initial_trace], frames=frames)
    figure.update_layout(
        title=f"Latent-space {projection_method_display_name(method)} projection",
        xaxis_title=selected_labels[0],
        yaxis_title=selected_labels[1],
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        template="plotly_white",
    )
    if animate_over_time:
        _add_animation_controls(figure, frame_names, base_title=f"Latent-space {projection_method_display_name(method)} projection")
    return figure


def build_projection_figure(
    latent_grid: np.ndarray,
    *,
    method: str = "pca",
    n_components: int = 3,
    max_points: int = 4000,
    animate_over_time: bool = False,
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
        animate_over_time=animate_over_time,
    )


def build_pca_figure(latent_grid: np.ndarray, max_points: int = 4000):
    return build_projection_figure(latent_grid, method="pca", n_components=3, max_points=max_points)
