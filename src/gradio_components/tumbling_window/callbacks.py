from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from gradio_utils import _log_gradio_step, _serialize_json
from ..latent_source import CHECKPOINT_DIR, _resolve_video_path
from ..latent_source.extractor import (
    auto_device,
    download_checkpoint_if_needed,
    load_encoder,
    preprocess_video,
    probe_video,
    read_video_frames,
    reshape_patch_tokens,
    run_encoder_synchronously,
)
from .core import (
    build_tumbling_window_heatmap_figure,
    compare_overlapping_latent_windows,
    derive_tumbling_window_ranges,
)
from .helpers import format_tumbling_window_status


def _html_to_iframe(content: str, *, height: int) -> str:
    return (
        "<iframe "
        f"style=\"width:100%;height:{height}px;border:0;\" "
        'sandbox="allow-scripts" '
        f"srcdoc=\"{escape(content, quote=True)}\" "
        'loading="lazy"></iframe>'
    )


def _figure_to_html(figure) -> str:
    figure_html = figure.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True, "displaylogo": False},
    )
    return _html_to_iframe(figure_html, height=820)


def _run_window(
    *,
    video_path: Path,
    encoder: Any,
    device: Any,
    crop_size: tuple[int, int],
    window_frames: int,
    window_start: int,
    tubelet_size: int,
) -> dict[str, Any]:
    window_frame_indices = list(range(int(window_start), int(window_start) + int(window_frames)))
    window_frames_rgb = read_video_frames(video_path, window_frame_indices)
    window_tensor = preprocess_video(window_frames_rgb, crop_size)
    raw_tokens, encoder_seconds = run_encoder_synchronously(encoder, window_tensor, device)
    time_patches = int(window_frames) // int(tubelet_size)
    height_patches = int(crop_size[0]) // 16
    width_patches = int(crop_size[1]) // 16
    latent_grid, tokens_stripped = reshape_patch_tokens(
        raw_tokens,
        time_patches=time_patches,
        height_patches=height_patches,
        width_patches=width_patches,
    )
    return {
        "frame_indices": window_frame_indices,
        "raw_tokens_shape": list(raw_tokens.shape),
        "latent_grid": latent_grid.detach().cpu().numpy().astype("float32", copy=False),
        "encoder_seconds": float(encoder_seconds),
        "tokens_stripped": int(tokens_stripped),
    }


def compare_tumbling_windows_step(
    video_file: str | None,
    model_name: str,
    crop_height: int | float,
    crop_width: int | float,
    device_name: str | None,
    start_frame: int | float,
    overlap_time_slices: int | float,
    window_frames: int | float,
):
    _log_gradio_step(
        "tumbling_window_compare",
        f"model={model_name}, start_frame={start_frame}, overlap_time_slices={overlap_time_slices}, window_frames={window_frames}",
    )
    tubelet_size = 2
    video_path = _resolve_video_path(video_file)
    crop_size = (int(crop_height), int(crop_width))
    window_frames = int(window_frames)
    start_frame = int(start_frame)
    overlap_time_slices = int(overlap_time_slices)
    device = auto_device(device_name)

    video_meta = probe_video(video_path)
    ranges = derive_tumbling_window_ranges(
        start_frame=start_frame,
        window_frames=window_frames,
        overlap_latent_steps=overlap_time_slices,
        tubelet_size=tubelet_size,
        available_frames=int(video_meta["frame_count"]),
    )

    checkpoint_path = download_checkpoint_if_needed(model_name, None, CHECKPOINT_DIR)
    encoder = load_encoder(
        model_name=model_name,
        num_frames=window_frames,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    left_run = _run_window(
        video_path=video_path,
        encoder=encoder,
        device=device,
        crop_size=crop_size,
        window_frames=window_frames,
        window_start=ranges.left_start,
        tubelet_size=tubelet_size,
    )
    right_run = _run_window(
        video_path=video_path,
        encoder=encoder,
        device=device,
        crop_size=crop_size,
        window_frames=window_frames,
        window_start=ranges.right_start,
        tubelet_size=tubelet_size,
    )

    analysis = compare_overlapping_latent_windows(
        left_run["latent_grid"],
        right_run["latent_grid"],
        left_start=ranges.left_start,
        right_start=ranges.right_start,
        tubelet_size=tubelet_size,
    )
    figure = build_tumbling_window_heatmap_figure(analysis["difference_overlap"])

    comparison = dict(analysis["comparison"])
    comparison["requested_overlap_time_slices"] = overlap_time_slices
    payload = {
        "video_path": str(video_path),
        "model": model_name,
        "device": str(device),
        "crop_size": list(crop_size),
        "video_metadata": video_meta,
        "window_ranges": {
            "left_start": ranges.left_start,
            "left_end": ranges.left_end,
            "right_start": ranges.right_start,
            "right_end": ranges.right_end,
            "overlap_start_frame": ranges.overlap_start_frame,
            "overlap_end_frame": ranges.overlap_end_frame,
            "overlap_frames": ranges.overlap_frames,
            "requested_overlap_latent_steps": ranges.overlap_latent_steps,
        },
        "left_run": {
            "frame_indices": left_run["frame_indices"],
            "raw_tokens_shape": left_run["raw_tokens_shape"],
            "latent_grid_shape": list(left_run["latent_grid"].shape),
            "encoder_seconds": left_run["encoder_seconds"],
            "tokens_stripped": left_run["tokens_stripped"],
        },
        "right_run": {
            "frame_indices": right_run["frame_indices"],
            "raw_tokens_shape": right_run["raw_tokens_shape"],
            "latent_grid_shape": list(right_run["latent_grid"].shape),
            "encoder_seconds": right_run["encoder_seconds"],
            "tokens_stripped": right_run["tokens_stripped"],
        },
        "comparison": comparison,
    }
    status = format_tumbling_window_status(
        video_path=video_path,
        model_name=model_name,
        device=device,
        comparison=comparison,
        available_frames=int(video_meta["frame_count"]),
        crop_size=crop_size,
        left_encoder_seconds=left_run["encoder_seconds"],
        right_encoder_seconds=right_run["encoder_seconds"],
        left_tokens_stripped=left_run["tokens_stripped"],
        right_tokens_stripped=right_run["tokens_stripped"],
    )
    return status, _figure_to_html(figure), _serialize_json(payload)