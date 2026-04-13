from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from gradio_utils import _format_hint_status


def sync_overlap_time_slice_control(window_frames: int | float, overlap_time_slices: int | float | None = None):
    max_steps = max(1, int(window_frames) // 2)
    if overlap_time_slices in (None, ""):
        value = min(1, max_steps)
    else:
        value = max(1, min(int(overlap_time_slices), max_steps))
    return gr.update(maximum=max_steps, value=value)


def format_tumbling_window_status(
    *,
    video_path: Path,
    model_name: str,
    device: Any,
    comparison: dict[str, Any],
    available_frames: int,
    crop_size: tuple[int, int],
    left_encoder_seconds: float,
    right_encoder_seconds: float,
    left_tokens_stripped: int,
    right_tokens_stripped: int,
) -> str:
    overlap_range = comparison.get("overlap_frame_range", [None, None])
    return "\n".join(
        [
            "## Tumbling-window comparison ready",
            f"- video: `{video_path}`",
            f"- model/device: `{model_name}` on `{device}`",
            f"- crop/window: `{crop_size[0]}x{crop_size[1]}` and `{comparison['window_frames']}` frames",
            f"- windows: `{comparison['left_window']}` vs `{comparison['right_window']}`",
            f"- overlap: frames `{overlap_range[0]}-{overlap_range[1]}` across `{comparison['overlap_latent_steps']}` latent time slices",
            f"- cosine similarity: mean `{comparison['mean_token_cosine_similarity_overlapping']:.6f}`, min `{comparison['min_token_cosine_similarity_overlapping']:.6f}`, max `{comparison['max_token_cosine_similarity_overlapping']:.6f}`",
            f"- absolute diff: max `{comparison['max_abs_diff_overlapping']:.6f}`, mean `{comparison['mean_abs_diff_overlapping']:.6f}`",
            f"- encoder seconds: `{left_encoder_seconds:.3f}` / `{right_encoder_seconds:.3f}` with stripped tokens `{left_tokens_stripped}` / `{right_tokens_stripped}`",
            f"- source frames available: `{available_frames}`",
        ]
    )


def initial_tumbling_window_status() -> str:
    return _format_hint_status(
        "Tumbling-window comparison not ready",
        "Choose a video in the latent-source section, set window controls here, and run the comparison.",
    )