from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from ..gradio_utils import (
    _format_hint_status,
    _format_segmentation_prompt_status,
    _format_segmentation_ready_status,
    _format_segmentation_result_status,
    _load_latent_metadata,
    _log_gradio_step,
    _serialize_json,
    _tracking_frame_choices,
)
from ..visualization import load_aligned_source_frames, load_saved_latents, map_click_to_latent_token
from .core import annotate_prompt_points, create_segmentation_video



def prepare_segmentation_step(latent_state: dict[str, Any] | None, segmentation_frame_index: int | str | None = 0):
    _log_gradio_step("vos_prepare", f"frame_index={segmentation_frame_index}")
    if not latent_state or not latent_state.get("output_prefix"):
        return (
            gr.update(choices=[], value=None, interactive=False),
            None,
            _format_hint_status("VOS segmentation not ready", "Load latents first, then show a frame for segmentation prompts."),
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
    selected_frame_index = 0
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
        "paper_aligned_defaults": {
            "top_k": 5,
            "temperature": 0.2,
            "context_frames": 15,
            "spatial_radius": 12,
            "prompt_frame_only": 0,
        },
        "prompt_points": {},
    }
    segmentation_state = {
        "latent_output_prefix": str(latent_output_prefix),
        "source_frames": source_frames,
        "display_fps": display_fps,
        "latent_grid_shape": list(latent_grid.shape),
        "source_frame_indices": list(source_frame_indices),
        "selected_frame_index": selected_frame_index,
        "prompt_points": {},
        "prompt_tokens": {},
    }
    return (
        gr.update(choices=frame_choices[:1], value=selected_frame_index, interactive=False),
        source_frames[selected_frame_index],
        _format_segmentation_ready_status(latent_grid.shape, display_fps),
        _serialize_json(payload),
        segmentation_state,
        None,
    )



def select_segmentation_prompt_step(
    latent_state: dict[str, Any] | None,
    segmentation_state: dict[str, Any] | None,
    prompt_label: str,
    evt: gr.SelectData,
):
    _log_gradio_step("vos_prompt", f"label={prompt_label} click_index={getattr(evt, 'index', None)}")
    if evt is None or getattr(evt, "index", None) is None:
        return (
            None,
            _format_hint_status("VOS segmentation not ready", "Click inside the selected frame to add a foreground/background prompt."),
            "{}",
            segmentation_state,
        )
    if not latent_state or not latent_state.get("output_prefix"):
        return (
            None,
            _format_hint_status("VOS segmentation not ready", "Load latents first, then show a frame for segmentation prompts."),
            "{}",
            segmentation_state,
        )

    normalized_label = str(prompt_label or "foreground").strip().lower()
    if normalized_label not in {"foreground", "background"}:
        normalized_label = "foreground"

    latent_output_prefix = Path(latent_state["output_prefix"])
    latent_grid = latent_state.get("latent_grid")
    if latent_grid is None:
        latent_grid, _ = load_saved_latents(latent_output_prefix)

    metadata = latent_state.get("metadata")
    if metadata is None:
        metadata = _load_latent_metadata(latent_output_prefix)

    source_frames = None
    if segmentation_state and segmentation_state.get("latent_output_prefix") == str(latent_output_prefix):
        source_frames = segmentation_state.get("source_frames")
    if source_frames is None:
        source_frames, display_fps, source_frame_indices = load_aligned_source_frames(metadata, latent_grid.shape)
    else:
        display_fps = float(segmentation_state.get("display_fps", 0.0) or 0.0)
        source_frame_indices = list(segmentation_state.get("source_frame_indices", []))

    selected_frame_index = int(segmentation_state.get("selected_frame_index", 0) if segmentation_state else 0)
    selected_frame_index = max(0, min(selected_frame_index, len(source_frames) - 1))
    selected_video_frame = source_frame_indices[selected_frame_index] if source_frame_indices else selected_frame_index

    click_index = evt.index
    if not isinstance(click_index, (tuple, list)) or len(click_index) < 2:
        raise gr.Error("Image selection did not provide click coordinates.")
    click_xy = [int(click_index[0]), int(click_index[1])]
    token_index = map_click_to_latent_token(click_xy, source_frames[selected_frame_index].shape, latent_grid.shape, time_index=selected_frame_index)

    prompt_points = dict(segmentation_state.get("prompt_points", {}) if segmentation_state else {})
    prompt_tokens = dict(segmentation_state.get("prompt_tokens", {}) if segmentation_state else {})
    prompt_points[normalized_label] = click_xy
    prompt_tokens[normalized_label] = list(token_index)
    preview_frame = annotate_prompt_points(source_frames[selected_frame_index], prompt_points)

    payload = {
        "latent_output_prefix": str(latent_output_prefix),
        "selected_frame_index": int(selected_frame_index),
        "selected_video_frame_index": int(selected_video_frame),
        "display_fps": display_fps,
        "prompt_points": prompt_points,
        "prompt_tokens": prompt_tokens,
    }
    next_segmentation_state = {
        "latent_output_prefix": str(latent_output_prefix),
        "source_frames": source_frames,
        "display_fps": display_fps,
        "latent_grid_shape": list(latent_grid.shape),
        "source_frame_indices": list(source_frame_indices),
        "selected_frame_index": int(selected_frame_index),
        "selected_video_frame_index": int(selected_video_frame),
        "prompt_points": prompt_points,
        "prompt_tokens": prompt_tokens,
    }
    return (
        preview_frame,
        _format_segmentation_prompt_status(
            prompt_points,
            frame_index=selected_frame_index,
            video_frame_index=selected_video_frame,
        ),
        _serialize_json(payload),
        next_segmentation_state,
    )



def run_segmentation_step(
    latent_state: dict[str, Any] | None,
    segmentation_state: dict[str, Any] | None,
    vos_knn_neighbors: int | float,
):
    _log_gradio_step("vos_run", f"k={vos_knn_neighbors}")
    if not latent_state or not latent_state.get("output_prefix"):
        return (
            _format_hint_status("VOS segmentation not ready", "Load latents first, then choose prompt points."),
            None,
            "{}",
            segmentation_state,
        )
    if not segmentation_state:
        return (
            _format_hint_status("VOS segmentation not ready", "Show a frame and select foreground/background prompts first."),
            None,
            "{}",
            segmentation_state,
        )

    prompt_tokens = dict(segmentation_state.get("prompt_tokens", {}))
    prompt_points = dict(segmentation_state.get("prompt_points", {}))
    if "foreground" not in prompt_tokens or "background" not in prompt_tokens:
        return (
            _format_hint_status("VOS segmentation not ready", "Select both a foreground point and a background point before running segmentation."),
            None,
            "{}",
            segmentation_state,
        )

    latent_output_prefix = Path(latent_state["output_prefix"])
    latent_grid = latent_state.get("latent_grid")
    if latent_grid is None:
        latent_grid, _ = load_saved_latents(latent_output_prefix)

    metadata = latent_state.get("metadata")
    if metadata is None:
        metadata = _load_latent_metadata(latent_output_prefix)

    source_frames = None
    if segmentation_state.get("latent_output_prefix") == str(latent_output_prefix):
        source_frames = segmentation_state.get("source_frames")
    if source_frames is None:
        source_frames, display_fps, source_frame_indices = load_aligned_source_frames(metadata, latent_grid.shape)
    else:
        display_fps = float(segmentation_state.get("display_fps", 0.0) or 0.0)
        source_frame_indices = list(segmentation_state.get("source_frame_indices", []))

    selected_frame_index = int(segmentation_state.get("selected_frame_index", 0))
    selected_frame_index = max(0, min(selected_frame_index, len(source_frames) - 1))
    selected_video_frame = source_frame_indices[selected_frame_index] if source_frame_indices else selected_frame_index

    output_dir = latent_output_prefix.parent / "tracking"
    artifacts = create_segmentation_video(
        latent_grid=latent_grid,
        metadata=metadata,
        output_dir=output_dir,
        foreground_token=prompt_tokens["foreground"],
        background_token=prompt_tokens["background"],
        source_frames=source_frames,
        foreground_click_xy=prompt_points.get("foreground"),
        background_click_xy=prompt_points.get("background"),
        k_neighbors=max(1, int(vos_knn_neighbors)),
        temperature=0.2,
        context_frames=15,
        spatial_radius=12,
    )
    payload = {
        "latent_output_prefix": str(latent_output_prefix),
        "selected_frame_index": int(selected_frame_index),
        "selected_video_frame_index": int(selected_video_frame),
        "display_fps": artifacts.display_fps,
        "foreground_prompt": {
            "click_xy": prompt_points["foreground"],
            "token": {
                "t": int(prompt_tokens["foreground"][0]),
                "h": int(prompt_tokens["foreground"][1]),
                "w": int(prompt_tokens["foreground"][2]),
            },
        },
        "background_prompt": {
            "click_xy": prompt_points["background"],
            "token": {
                "t": int(prompt_tokens["background"][0]),
                "h": int(prompt_tokens["background"][1]),
                "w": int(prompt_tokens["background"][2]),
            },
        },
        "segmentation_video_path": str(artifacts.segmentation_video_path),
        "segmentation_video_shape": list(artifacts.segmentation_video_shape),
        "knn_neighbors": int(artifacts.knn_neighbors),
        "temperature": artifacts.temperature,
        "context_frames": artifacts.context_frames,
        "spatial_radius": artifacts.spatial_radius,
        "foreground_ratio_per_frame": [round(value, 4) for value in artifacts.foreground_ratio_per_frame],
    }
    next_segmentation_state = {
        **segmentation_state,
        "display_fps": display_fps if display_fps > 0 else artifacts.display_fps,
        "selected_video_frame_index": int(selected_video_frame),
        "segmentation_video_path": str(artifacts.segmentation_video_path),
    }
    return (
        _format_segmentation_result_status(
            prompt_tokens["foreground"],
            prompt_tokens["background"],
            artifacts.segmentation_video_path,
            frame_index=selected_frame_index,
            video_frame_index=selected_video_frame,
            knn_neighbors=artifacts.knn_neighbors,
        ),
        str(artifacts.segmentation_video_path),
        _serialize_json(payload),
        next_segmentation_state,
    )
