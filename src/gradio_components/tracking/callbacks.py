from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from gradio_utils import _format_hint_status, _load_latent_metadata, _log_gradio_step, _serialize_json
from ..projection import load_saved_latents
from ..render.video import load_aligned_source_frames
from .core import (
    annotate_selected_patch,
    create_patch_similarity_video,
    map_click_to_latent_token,
)
from .helpers import _format_tracking_ready_status, _format_tracking_result_status, _tracking_frame_choices


def prepare_tracking_step(latent_state: dict[str, Any] | None, tracking_frame_index: int | str | None = 0):
    _log_gradio_step("tracking_prepare", f"frame_index={tracking_frame_index}")
    if not latent_state or not latent_state.get("output_prefix"):
        return (
            gr.update(choices=[], value=None, interactive=False),
            None,
            _format_hint_status("Patch similarity not ready", "Load latents first, then show the first frame for tracking."),
            "{}",
            None,
            None,
        )

    latent_output_prefix = Path(latent_state["output_prefix"])
    latent_grid = latent_state.get("latent_grid")
    if latent_grid is None:
        latent_grid, _ = load_saved_latents(latent_output_prefix)

    metadata = latent_state.get("metadata")
    if metadata is None:
        metadata = _load_latent_metadata(latent_output_prefix)

    try:
        source_frames, display_fps, source_frame_indices = load_aligned_source_frames(metadata, latent_grid.shape)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise gr.Error(str(error)) from error

    frame_choices = _tracking_frame_choices(source_frame_indices)
    selected_frame_index = int(tracking_frame_index) if tracking_frame_index not in (None, "") else 0
    selected_frame_index = max(0, min(selected_frame_index, len(source_frames) - 1))
    selected_video_frame = source_frame_indices[selected_frame_index] if source_frame_indices else selected_frame_index

    payload = {
        "latent_output_prefix": str(latent_output_prefix),
        "video_path": metadata.get("video_path"),
        "latent_grid_shape": list(latent_grid.shape),
        "source_frame_shape": list(source_frames.shape),
        "source_frame_indices": list(source_frame_indices),
        "selected_frame_index": selected_frame_index,
        "selected_video_frame_index": selected_video_frame,
        "display_fps": display_fps,
    }
    tracking_state = {
        "latent_output_prefix": str(latent_output_prefix),
        "source_frames": source_frames,
        "display_fps": display_fps,
        "latent_grid_shape": list(latent_grid.shape),
        "source_frame_indices": list(source_frame_indices),
        "selected_frame_index": selected_frame_index,
    }
    return (
        gr.update(choices=frame_choices, value=selected_frame_index, interactive=True),
        source_frames[selected_frame_index],
        _format_tracking_ready_status(latent_grid.shape, display_fps),
        _serialize_json(payload),
        tracking_state,
        None,
    )


def select_patch_similarity_step(
    latent_state: dict[str, Any] | None,
    tracking_state: dict[str, Any] | None,
    evt: gr.SelectData,
):
    _log_gradio_step("tracking_select", f"click_index={getattr(evt, 'index', None)}")
    if evt is None or getattr(evt, "index", None) is None:
        return (
            None,
            _format_hint_status("Patch similarity not ready", "Click inside the first frame to select a latent patch."),
            None,
            "{}",
            tracking_state,
        )
    if not latent_state or not latent_state.get("output_prefix"):
        return (
            None,
            _format_hint_status("Patch similarity not ready", "Load latents first, then show the first frame for tracking."),
            None,
            "{}",
            tracking_state,
        )

    latent_output_prefix = Path(latent_state["output_prefix"])
    latent_grid = latent_state.get("latent_grid")
    if latent_grid is None:
        latent_grid, _ = load_saved_latents(latent_output_prefix)

    metadata = latent_state.get("metadata")
    if metadata is None:
        metadata = _load_latent_metadata(latent_output_prefix)

    source_frames = None
    if tracking_state and tracking_state.get("latent_output_prefix") == str(latent_output_prefix):
        source_frames = tracking_state.get("source_frames")
    if source_frames is None:
        source_frames, display_fps, source_frame_indices = load_aligned_source_frames(metadata, latent_grid.shape)
    else:
        display_fps = float(tracking_state.get("display_fps", 0.0) or 0.0)
        source_frame_indices = list(tracking_state.get("source_frame_indices", []))

    selected_frame_index = int(tracking_state.get("selected_frame_index", 0) if tracking_state else 0)
    selected_frame_index = max(0, min(selected_frame_index, len(source_frames) - 1))
    selected_video_frame = source_frame_indices[selected_frame_index] if source_frame_indices else selected_frame_index

    click_index = evt.index
    if not isinstance(click_index, (tuple, list)) or len(click_index) < 2:
        raise gr.Error("Image selection did not provide click coordinates.")
    click_xy = (int(click_index[0]), int(click_index[1]))
    token_index = map_click_to_latent_token(click_xy, source_frames[selected_frame_index].shape, latent_grid.shape, time_index=selected_frame_index)
    preview_frame = annotate_selected_patch(source_frames[selected_frame_index], token_index, latent_grid.shape)

    output_dir = latent_output_prefix.parent / "tracking"
    artifacts = create_patch_similarity_video(
        latent_grid=latent_grid,
        metadata=metadata,
        output_dir=output_dir,
        token_index=token_index,
        source_frames=source_frames,
    )
    payload = {
        "latent_output_prefix": str(latent_output_prefix),
        "click_xy": [int(click_xy[0]), int(click_xy[1])],
        "selected_token": {
            "t": int(token_index[0]),
            "h": int(token_index[1]),
            "w": int(token_index[2]),
        },
        "selected_frame_index": int(selected_frame_index),
        "selected_video_frame_index": int(selected_video_frame),
        "display_fps": artifacts.display_fps,
        "similarity_video_path": str(artifacts.similarity_video_path),
        "similarity_video_shape": list(artifacts.similarity_video_shape),
        "similarity_range": [artifacts.similarity_min, artifacts.similarity_max],
    }
    next_tracking_state = {
        "latent_output_prefix": str(latent_output_prefix),
        "source_frames": source_frames,
        "display_fps": display_fps if display_fps > 0 else artifacts.display_fps,
        "latent_grid_shape": list(latent_grid.shape),
        "selected_token": list(token_index),
        "click_xy": list(click_xy),
        "selected_frame_index": int(selected_frame_index),
        "selected_video_frame_index": int(selected_video_frame),
    }
    return (
        preview_frame,
        _format_tracking_result_status(
            token_index,
            click_xy,
            artifacts.similarity_video_path,
            frame_index=selected_frame_index,
            video_frame_index=selected_video_frame,
        ),
        str(artifacts.similarity_video_path),
        _serialize_json(payload),
        next_tracking_state,
    )
