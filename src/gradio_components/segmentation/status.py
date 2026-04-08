from __future__ import annotations

from pathlib import Path
from typing import Sequence


def _format_segmentation_ready_status(latent_grid_shape: Sequence[int], display_fps: float) -> str:
    _, time_steps, grid_h, grid_w, _ = [int(value) for value in latent_grid_shape]
    return "\n".join(
        [
            "## VOS segmentation ready",
            "- paper-style propagation uses the first frame only: click one foreground point and one background point, then run weighted top-k cosine propagation.",
            f"- latent token grid: `{time_steps} x {grid_h} x {grid_w}`",
            "- propagation defaults: `top-k=5`, `temperature=0.2`, `context_frames=15`, `spatial_radius=12`",
            f"- playback fps: `{display_fps:.3f}`",
        ]
    )


def _format_segmentation_prompt_status(
    prompt_points: dict[str, Sequence[int]] | None,
    *,
    frame_index: int | None = None,
    video_frame_index: int | None = None,
) -> str:
    prompt_points = prompt_points or {}
    lines = ["## Segmentation prompts updated"]
    for label in ("foreground", "background"):
        point = prompt_points.get(label)
        if point is None:
            lines.append(f"- {label}: `not selected`")
        else:
            lines.append(f"- {label}: `x={int(point[0])}, y={int(point[1])}`")
    if frame_index is not None:
        frame_line = f"- prompt frame: `{frame_index + 1}`"
        if video_frame_index is not None:
            frame_line = f"- prompt frame: `{frame_index + 1}` (video frame `{video_frame_index}`)"
        lines.append(frame_line)
    if "foreground" not in prompt_points or "background" not in prompt_points:
        lines.append("- next: add the missing prompt point, then click **Run VOS segmentation**.")
    else:
        lines.append("- next: click **Run VOS segmentation** to classify all latent tokens as foreground/background.")
    return "\n".join(lines)


def _format_segmentation_result_status(
    foreground_token: Sequence[int],
    background_token: Sequence[int],
    video_path: Path,
    *,
    frame_index: int | None = None,
    video_frame_index: int | None = None,
    knn_neighbors: int = 1,
) -> str:
    fg_t, fg_h, fg_w = [int(value) for value in foreground_token]
    bg_t, bg_h, bg_w = [int(value) for value in background_token]
    lines = [
        "## VOS segmentation video ready",
        f"- foreground token: `t={fg_t}, h={fg_h}, w={fg_w}`",
        f"- background token: `t={bg_t}, h={bg_h}, w={bg_w}`",
        f"- top-k neighbors: `{int(knn_neighbors)}`",
        "- propagation defaults: `temperature=0.2`, `context_frames=15`, `spatial_radius=12`",
    ]
    if frame_index is not None:
        frame_line = f"- prompt frame: `{frame_index + 1}`"
        if video_frame_index is not None:
            frame_line = f"- prompt frame: `{frame_index + 1}` (video frame `{video_frame_index}`)"
        lines.append(frame_line)
    lines.append(f"- output video: `{video_path}`")
    return "\n".join(lines)