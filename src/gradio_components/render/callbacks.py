from __future__ import annotations

from pathlib import Path
from typing import Any

from gradio_utils import _format_hint_status, _load_latent_metadata, _log_gradio_step, _serialize_json
from ..projection import load_saved_projection
from .video import create_visualizations_from_projection
from .helpers import _format_render_status


def create_rgb_videos_step(
    latent_state: dict[str, Any] | None,
    projection_state: dict[str, Any] | None,
    rgb_r_component: int,
    rgb_g_component: int,
    rgb_b_component: int | None,
    upscale_factor: int,
):
    _log_gradio_step("render_rgb", f"components=({rgb_r_component}, {rgb_g_component}, {rgb_b_component}) upscale={upscale_factor}")
    if not projection_state or not projection_state.get("output_prefix"):
        return (
            _format_hint_status("RGB videos not ready", "Compute or load a projection before generating RGB videos."),
            None,
            "{}",
        )
    if rgb_b_component is None:
        return (
            _format_hint_status("RGB videos not ready", "RGB rendering needs at least 3 projected components."),
            None,
            "{}",
        )

    projection_output_prefix = Path(projection_state["output_prefix"])
    projection = projection_state.get("projection")
    projection_metadata = projection_state.get("metadata")
    if projection is None or projection_metadata is None:
        projection, _, projection_metadata = load_saved_projection(projection_output_prefix)
    latent_output_prefix_text = None
    if latent_state and latent_state.get("output_prefix"):
        latent_output_prefix_text = latent_state["output_prefix"]
    elif projection_metadata.get("latent_output_prefix"):
        latent_output_prefix_text = projection_metadata["latent_output_prefix"]
    if not latent_output_prefix_text:
        return (
            _format_hint_status("RGB videos not ready", "Could not locate the latent metadata required for side-by-side rendering."),
            None,
            "{}",
        )

    latent_output_prefix = Path(latent_output_prefix_text)
    latent_metadata = None
    if latent_state and latent_state.get("metadata") is not None:
        latent_metadata = latent_state["metadata"]
    if latent_metadata is None:
        latent_metadata = _load_latent_metadata(latent_output_prefix)
    rgb_components = (int(rgb_r_component) - 1, int(rgb_g_component) - 1, int(rgb_b_component) - 1)
    output_dir = projection_output_prefix.parent / "renders"
    artifacts = create_visualizations_from_projection(
        projection=projection,
        latent_grid_shape=projection_metadata["latent_grid_shape"],
        metadata=latent_metadata,
        output_dir=output_dir,
        rgb_components=rgb_components,
        method=projection_metadata["method"],
        component_labels=projection_metadata.get("component_labels"),
        upscale_factor=max(1, int(upscale_factor)),
    )
    payload = {
        "projection_output_prefix": str(projection_output_prefix),
        "latent_output_prefix": str(latent_output_prefix),
        "rgb_components": [component + 1 for component in rgb_components],
        "artifacts": {
            "latent_video_path": str(artifacts.latent_video_path),
            "side_by_side_video_path": str(artifacts.side_by_side_video_path),
            "latent_video_shape": list(artifacts.latent_video_shape),
            "side_by_side_video_shape": list(artifacts.side_by_side_video_shape),
            "display_fps": artifacts.display_fps,
        },
    }
    status = _format_render_status(
        projection_metadata["method"],
        rgb_components,
        artifacts.latent_video_path,
        artifacts.side_by_side_video_path,
        projection_metadata.get("method_label"),
    )
    _log_gradio_step("render_rgb", f"side_by_side={artifacts.side_by_side_video_path}")
    return status, str(artifacts.side_by_side_video_path), _serialize_json(payload)
