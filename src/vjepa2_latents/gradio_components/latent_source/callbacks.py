from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import gradio as gr

from .extractor import (
    estimate_extraction_requirements,
    extract_latents,
    probe_video,
    select_frame_indices,
)
from ...gradio_utils import _format_hint_status, _log_gradio_step, _serialize_json
from ..projection import load_saved_latents, summarize_latents
from .catalog import saved_latent_choices
from .config import APP_OUTPUT_DIR, CHECKPOINT_DIR, VENDOR_REPO
from .helpers import (
    _clean_latent_metadata_for_ui,
    _create_session_dir,
    _format_extraction_status,
    _format_latent_status,
    _format_preflight_status,
    _latent_state,
    _normalize_prefix,
    _resolve_video_path,
    _summarize_timings_for_ui,
)
def estimate_limits_step(
    video_file: str | None,
    model_name: str,
    crop_height: int,
    crop_width: int,
    num_frames: int,
    sample_fps: float | None,
    start_second: float,
    device_name: str,
):
    _log_gradio_step("estimate", f"video={video_file or 'default'} model={model_name} device={device_name}")
    video_path = _resolve_video_path(video_file)
    video_meta = probe_video(video_path)
    frame_indices = select_frame_indices(
        video_fps=video_meta["fps"],
        frame_count=video_meta["frame_count"],
        num_frames=int(num_frames),
        start_frame=0,
        start_second=start_second if start_second > 0 else None,
        sample_fps=sample_fps if sample_fps and sample_fps > 0 else None,
    )
    preflight = estimate_extraction_requirements(
        model_name=model_name,
        num_frames=int(num_frames),
        crop_size=(int(crop_height), int(crop_width)),
        device_name=device_name,
    )
    payload = {
        **preflight,
        "video_path": str(video_path),
        "video_metadata": video_meta,
        "selected_frame_count": len(frame_indices),
        "selected_frame_range": [frame_indices[0], frame_indices[-1]],
    }
    _log_gradio_step("estimate", f"selected_frames={len(frame_indices)} token_count={preflight['token_count']}")
    return _format_preflight_status(preflight, video_meta, frame_indices), _serialize_json(payload)


def refresh_saved_latent_choices(current_value: str | None = None):
    choices = saved_latent_choices(APP_OUTPUT_DIR)
    valid_values = {value for _, value in choices}
    value = current_value if current_value in valid_values else (choices[0][1] if choices else None)
    return gr.update(choices=choices, value=value)


def toggle_latent_source_mode(latent_source_mode: str, current_saved_latent_prefix: str | None = None):
    show_extract = latent_source_mode == "extract"
    selector_update = refresh_saved_latent_choices(current_saved_latent_prefix) if not show_extract else gr.update()
    return gr.update(visible=show_extract), gr.update(visible=not show_extract), selector_update


def extract_latents_step(
    video_file: str | None,
    model_name: str,
    crop_height: int,
    crop_width: int,
    num_frames: int,
    sample_fps: float | None,
    start_second: float,
    device_name: str,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    _log_gradio_step("extract", f"starting model={model_name} device={device_name}")
    video_path = _resolve_video_path(video_file)
    session_dir = _create_session_dir("vjepa2-extract-")
    output_prefix = session_dir / "latents"
    progress(0.1, desc="Running V-JEPA encoder")
    try:
        result = extract_latents(
            video_path=video_path,
            output_prefix=output_prefix,
            vendor_repo=VENDOR_REPO,
            model_name=model_name,
            checkpoint_path=None,
            checkpoint_dir=CHECKPOINT_DIR,
            num_frames=num_frames,
            crop_size=(int(crop_height), int(crop_width)),
            sample_fps=sample_fps if sample_fps and sample_fps > 0 else None,
            start_frame=0,
            start_second=start_second if start_second > 0 else None,
            device_name=device_name,
            dry_run=False,
            save_pt=False,
        )
    except RuntimeError as error:
        if "MPS backend out of memory" in str(error):
            preflight = estimate_extraction_requirements(
                model_name=model_name,
                num_frames=int(num_frames),
                crop_size=(int(crop_height), int(crop_width)),
                device_name=device_name,
            )
            raise gr.Error(
                "Apple MPS ran out of memory for this configuration. "
                f"Estimated risk is `{preflight['risk_level']}` with `{preflight['token_count']}` patch tokens. "
                "Try fewer frames, a smaller crop, or switch the device to `cpu`."
            ) from error
        raise

    payload = {
        **result,
        "model": model_name,
        "video_path": str(video_path),
        "video_name": video_path.name,
        "latent_output_prefix": str(output_prefix),
        "latent_npy_path": str(output_prefix.with_suffix(".npy")),
        "latent_metadata_path": str(output_prefix.with_suffix(".metadata.json")),
    }
    payload["timings"] = _summarize_timings_for_ui(result.get("timings"))
    latent_grid, metadata = load_saved_latents(output_prefix)
    status = _format_extraction_status(result, output_prefix, model_name=model_name, video_path=video_path)
    latent_state = _latent_state(output_prefix, latent_grid, metadata)
    _log_gradio_step("extract", f"saved_prefix={output_prefix} latent_grid_shape={latent_grid.shape}")
    return (
        status,
        str(output_prefix),
        _serialize_json(payload),
        latent_state,
        None,
        "",
        "{}",
        None,
        None,
        "",
        "{}",
        refresh_saved_latent_choices(str(output_prefix)),
    )


def load_latents_step(
    saved_latent_prefix: str | None,
    latent_state: dict[str, Any] | None,
):
    active_prefix = saved_latent_prefix or (latent_state.get("output_prefix") if latent_state else None)
    _log_gradio_step("load_latents", f"prefix={active_prefix or 'none'}")
    output_prefix = _normalize_prefix(saved_latent_prefix, (".npy", ".metadata.json"))
    if output_prefix is None and latent_state and latent_state.get("output_prefix"):
        output_prefix = Path(latent_state["output_prefix"])
    if output_prefix is None:
        return (
            _format_hint_status(
                "Latents not loaded",
                "Choose a saved latent run from the list before loading latents.",
            ),
            "",
            "{}",
            latent_state,
            None,
            "",
            "{}",
            None,
            None,
            _format_hint_status(
                "RGB videos not ready",
                "Load latents first, then compute or load a projection before rendering videos.",
            ),
            "{}",
            refresh_saved_latent_choices(),
        )

    latent_grid, metadata = load_saved_latents(output_prefix)
    summary = summarize_latents(latent_grid)
    payload = {
        **_clean_latent_metadata_for_ui(metadata),
        "summary": summary,
        "latent_output_prefix": str(output_prefix),
    }
    status = _format_latent_status(output_prefix, metadata, summary)
    next_state = _latent_state(output_prefix, latent_grid, metadata)
    _log_gradio_step("load_latents", f"loaded prefix={output_prefix} latent_grid_shape={latent_grid.shape}")
    return (
        status,
        str(output_prefix),
        _serialize_json(payload),
        next_state,
        None,
        "",
        "{}",
        None,
        None,
        "",
        "{}",
        refresh_saved_latent_choices(str(output_prefix)),
    )
