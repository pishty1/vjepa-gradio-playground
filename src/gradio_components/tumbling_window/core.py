from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - exercised only when optional dependency is missing
    go = None


@dataclass(frozen=True)
class TumblingWindowRanges:
    left_start: int
    left_end: int
    right_start: int
    right_end: int
    overlap_start_frame: int
    overlap_end_frame: int
    overlap_frames: int
    overlap_latent_steps: int


def derive_tumbling_window_ranges(
    *,
    start_frame: int,
    window_frames: int,
    overlap_latent_steps: int,
    tubelet_size: int = 2,
    available_frames: int | None = None,
) -> TumblingWindowRanges:
    left_start = int(start_frame)
    window_frames = int(window_frames)
    overlap_latent_steps = int(overlap_latent_steps)
    tubelet_size = int(tubelet_size)

    if left_start < 0:
        raise ValueError("Start frame must be zero or greater.")
    if tubelet_size <= 0:
        raise ValueError("Tubelet size must be positive.")
    if window_frames <= 0 or window_frames % tubelet_size != 0:
        raise ValueError(f"Window length must be a positive multiple of the tubelet size {tubelet_size}.")

    max_overlap_steps = window_frames // tubelet_size
    if overlap_latent_steps <= 0 or overlap_latent_steps > max_overlap_steps:
        raise ValueError(
            f"Overlap time slices must be between 1 and {max_overlap_steps} for a {window_frames}-frame window."
        )

    overlap_frames = overlap_latent_steps * tubelet_size
    right_start = left_start + window_frames - overlap_frames
    left_end = left_start + window_frames
    right_end = right_start + window_frames
    overlap_start_frame = max(left_start, right_start)
    overlap_end_frame = min(left_end, right_end) - 1
    if overlap_end_frame < overlap_start_frame:
        raise ValueError("The requested windows do not overlap.")
    if available_frames is not None and right_end > int(available_frames):
        raise ValueError(
            "Requested tumbling windows exceed the available video frames: "
            f"need frames {left_start}-{left_end - 1} and {right_start}-{right_end - 1}, "
            f"but the video has only {int(available_frames)} frames."
        )

    return TumblingWindowRanges(
        left_start=left_start,
        left_end=left_end,
        right_start=right_start,
        right_end=right_end,
        overlap_start_frame=overlap_start_frame,
        overlap_end_frame=overlap_end_frame,
        overlap_frames=overlap_end_frame - overlap_start_frame + 1,
        overlap_latent_steps=overlap_latent_steps,
    )


def overlap_time_slice(
    *,
    window_start: int,
    overlap_start_frame: int,
    overlap_end_frame: int,
    latent_time_steps: int,
    tubelet_size: int = 2,
) -> slice:
    overlap_start_offset = int(overlap_start_frame) - int(window_start)
    overlap_end_offset = int(overlap_end_frame) - int(window_start)
    start_index = max(0, (overlap_start_offset + tubelet_size - 1) // tubelet_size)
    end_index = min(int(latent_time_steps), (overlap_end_offset - tubelet_size + 1) // tubelet_size + 1)
    if end_index <= start_index:
        raise ValueError(f"No overlapping latent timesteps exist for window starting at {window_start}.")
    return slice(start_index, end_index)


def compare_overlapping_latent_windows(
    left_latent: np.ndarray,
    right_latent: np.ndarray,
    *,
    left_start: int,
    right_start: int,
    tubelet_size: int = 2,
) -> dict[str, Any]:
    left_latent = np.asarray(left_latent, dtype=np.float32)
    right_latent = np.asarray(right_latent, dtype=np.float32)
    if left_latent.ndim != 5 or right_latent.ndim != 5:
        raise ValueError(
            f"Expected latent windows with shape [batch, time, height, width, dim], got {left_latent.shape} and {right_latent.shape}."
        )
    if left_latent.shape[0] != right_latent.shape[0]:
        raise ValueError("Both tumbling windows must have the same batch size.")

    left_window_frames = int(left_latent.shape[1]) * int(tubelet_size)
    right_window_frames = int(right_latent.shape[1]) * int(tubelet_size)
    left_end = int(left_start) + left_window_frames
    right_end = int(right_start) + right_window_frames
    overlap_start_frame = max(int(left_start), int(right_start))
    overlap_end_frame = min(left_end, right_end) - 1
    if overlap_end_frame < overlap_start_frame:
        raise ValueError("The selected windows do not overlap in source-frame space.")

    left_time_slice = overlap_time_slice(
        window_start=int(left_start),
        overlap_start_frame=overlap_start_frame,
        overlap_end_frame=overlap_end_frame,
        latent_time_steps=int(left_latent.shape[1]),
        tubelet_size=tubelet_size,
    )
    right_time_slice = overlap_time_slice(
        window_start=int(right_start),
        overlap_start_frame=overlap_start_frame,
        overlap_end_frame=overlap_end_frame,
        latent_time_steps=int(right_latent.shape[1]),
        tubelet_size=tubelet_size,
    )

    left_overlap = left_latent[:, left_time_slice]
    right_overlap = right_latent[:, right_time_slice]
    if left_overlap.shape != right_overlap.shape:
        raise ValueError(
            f"Overlapping latent slices must have matching shapes, got {left_overlap.shape} and {right_overlap.shape}."
        )

    difference_overlap = left_overlap - right_overlap
    left_flat = left_overlap.reshape(-1, left_overlap.shape[-1])
    right_flat = right_overlap.reshape(-1, right_overlap.shape[-1])
    left_norm = left_flat / np.clip(np.linalg.norm(left_flat, axis=1, keepdims=True), 1e-12, None)
    right_norm = right_flat / np.clip(np.linalg.norm(right_flat, axis=1, keepdims=True), 1e-12, None)
    per_token_cosine = np.sum(left_norm * right_norm, axis=1).reshape(left_overlap.shape[1:4])

    comparison = {
        "left_window": f"frames {int(left_start)}-{left_end - 1}",
        "right_window": f"frames {int(right_start)}-{right_end - 1}",
        "window_frames": left_window_frames,
        "window_offset": int(right_start) - int(left_start),
        "overlap_frames": overlap_end_frame - overlap_start_frame + 1,
        "overlap_frame_range": [overlap_start_frame, overlap_end_frame],
        "overlap_latent_steps": int(left_overlap.shape[1]),
        "overlap_left_time_indices": [left_time_slice.start, left_time_slice.stop - 1],
        "overlap_right_time_indices": [right_time_slice.start, right_time_slice.stop - 1],
        "allclose_overlapping": bool(np.allclose(left_overlap, right_overlap, rtol=1e-5, atol=1e-6)),
        "max_abs_diff_overlapping": float(np.max(np.abs(difference_overlap))),
        "mean_abs_diff_overlapping": float(np.mean(np.abs(difference_overlap))),
        "mean_token_cosine_similarity_overlapping": float(np.mean(per_token_cosine)),
        "min_token_cosine_similarity_overlapping": float(np.min(per_token_cosine)),
        "max_token_cosine_similarity_overlapping": float(np.max(per_token_cosine)),
    }

    return {
        "comparison": comparison,
        "left_overlap": left_overlap,
        "right_overlap": right_overlap,
        "difference_overlap": difference_overlap,
        "per_token_cosine": per_token_cosine,
    }


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
                                "frame": {"duration": 800, "redraw": True},
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
                "currentvalue": {"prefix": "Overlap time slice: "},
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


def build_tumbling_window_heatmap_figure(difference_overlap: np.ndarray):
    if go is None:
        raise RuntimeError("plotly is required to build the tumbling-window heatmap")

    difference_overlap = np.asarray(difference_overlap, dtype=np.float32)
    if difference_overlap.ndim == 5:
        if difference_overlap.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for animated heatmap rendering, got {difference_overlap.shape[0]}.")
        difference_frames = difference_overlap[0]
    elif difference_overlap.ndim == 4:
        difference_frames = difference_overlap
    else:
        raise ValueError(
            f"Expected overlap difference with shape [batch, time, height, width, dim] or [time, height, width, dim], got {difference_overlap.shape}."
        )

    if difference_frames.shape[0] <= 0:
        raise ValueError("No overlapping latent time slices are available to visualize.")

    shared_abs_max = max(float(np.max(np.abs(difference_frames))), 1e-12)

    def _reshape_matrix(frame_3d: np.ndarray) -> np.ndarray:
        return frame_3d.reshape(frame_3d.shape[0] * frame_3d.shape[1], frame_3d.shape[2])

    initial_matrix = _reshape_matrix(difference_frames[0])
    heatmap = go.Heatmap(
        z=initial_matrix,
        colorscale=[
            [0.0, "#ff0000"],
            [0.5, "#ffffff"],
            [1.0, "#0000ff"],
        ],
        zmin=-shared_abs_max,
        zmax=shared_abs_max,
        colorbar={"title": "First run minus second run"},
        hovertemplate="token=%{y}<br>latent dim=%{x}<br>diff=%{z:.5f}<extra></extra>",
    )
    frame_names = [f"{time_index + 1}/{difference_frames.shape[0]}" for time_index in range(difference_frames.shape[0])]
    frames = [
        go.Frame(
            data=[go.Heatmap(z=_reshape_matrix(difference_frames[time_index]))],
            traces=[0],
            name=frame_names[time_index],
        )
        for time_index in range(difference_frames.shape[0])
    ]

    figure = go.Figure(data=[heatmap], frames=frames)
    figure.update_layout(
        title="Overlap latent difference heatmap",
        xaxis_title=f"Latent dimension ({difference_frames.shape[-1]})",
        yaxis_title=(
            f"Spatial token index ({difference_frames.shape[1]} × {difference_frames.shape[2]} = {difference_frames.shape[1] * difference_frames.shape[2]})"
        ),
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
        template="plotly_white",
        height=760,
    )
    _add_animation_controls(figure, frame_names, base_title="Overlap latent difference heatmap")
    return figure