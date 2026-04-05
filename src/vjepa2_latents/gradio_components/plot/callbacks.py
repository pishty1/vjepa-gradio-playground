from __future__ import annotations

from pathlib import Path
from typing import Any

from ...gradio_utils import _format_hint_status, _log_gradio_step
from ..projection import load_saved_projection
from .core import build_projection_figure_from_data
from .helpers import _format_plot_status


def build_plot_step(
    projection_state: dict[str, Any] | None,
    plot_dimensions: int,
    plot_max_points: int,
    plot_x_component: int,
    plot_y_component: int,
    plot_z_component: int | None,
):
    _log_gradio_step("build_plot", f"dimensions={plot_dimensions} components=({plot_x_component}, {plot_y_component}, {plot_z_component})")
    if not projection_state or not projection_state.get("output_prefix"):
        return None, _format_hint_status("Plot not ready", "Compute or load a projection before building a plot.")

    projection = projection_state.get("projection")
    coordinates = projection_state.get("coordinates")
    metadata = projection_state.get("metadata")
    if projection is None or coordinates is None or metadata is None:
        projection, coordinates, metadata = load_saved_projection(Path(projection_state["output_prefix"]))
    component_indices = [int(plot_x_component) - 1, int(plot_y_component) - 1]
    if int(plot_dimensions) == 3:
        if plot_z_component is None:
            return None, _format_hint_status("Plot not ready", "Choose a Z component for a 3D plot.")
        component_indices.append(int(plot_z_component) - 1)

    figure = build_projection_figure_from_data(
        projection,
        coordinates,
        method=metadata["method"],
        component_indices=tuple(component_indices),
        component_labels=metadata.get("component_labels"),
        max_points=max(100, int(plot_max_points)),
    )
    _log_gradio_step("build_plot", f"plotted_components={component_indices}")
    return figure, _format_plot_status(metadata["method"], component_indices, metadata.get("method_label"))
