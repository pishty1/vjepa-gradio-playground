from __future__ import annotations

from pathlib import Path
from typing import Sequence


def _format_tracking_ready_status(latent_grid_shape: Sequence[int], display_fps: float) -> str:
    _, time_steps, grid_h, grid_w, _ = [int(value) for value in latent_grid_shape]
    return "\n".join(
        [
            "## Patch similarity ready",
            "- choose one of the available source frames, then click any object or patch in the image to select a latent token.",
            f"- latent token grid: `{time_steps} x {grid_h} x {grid_w}`",
            f"- playback fps: `{display_fps:.3f}`",
        ]
    )


def _format_tracking_result_status(
    token_index: Sequence[int],
    click_xy: Sequence[int],
    video_path: Path,
    *,
    frame_index: int | None = None,
    video_frame_index: int | None = None,
) -> str:
    time_index, row_index, column_index = [int(value) for value in token_index]
    click_x, click_y = [int(value) for value in click_xy[:2]]
    lines = [
        "## Patch similarity video ready",
        f"- selected click: `x={click_x}, y={click_y}`",
        f"- selected latent token: `t={time_index}, h={row_index}, w={column_index}`",
    ]
    if frame_index is not None:
        frame_line = f"- tracked frame: `{frame_index + 1}`"
        if video_frame_index is not None:
            frame_line = f"- tracked frame: `{frame_index + 1}` (video frame `{video_frame_index}`)"
        lines.append(frame_line)
    lines.append(f"- output video: `{video_path}`")
    return "\n".join(lines)


def _tracking_frame_choices(source_frame_indices: Sequence[int]) -> list[tuple[str, int]]:
    return [(f"Frame {position + 1} (video frame {frame_index})", position) for position, frame_index in enumerate(source_frame_indices)]
