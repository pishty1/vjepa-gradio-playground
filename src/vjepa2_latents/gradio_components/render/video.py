from __future__ import annotations

import importlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from ..latent_source.extractor import normalize_crop_size, prepare_display_frames, read_video_frames
from ..projection.core import compute_projection_bundle, projection_component_labels, projection_method_display_name


@dataclass(frozen=True)
class VisualizationArtifacts:
    latent_video_path: Path
    side_by_side_video_path: Path
    latent_video_shape: tuple[int, int, int, int]
    side_by_side_video_shape: tuple[int, int, int, int]
    display_fps: float


def projection_rgb_frames(
    projection: np.ndarray,
    latent_grid_shape: Sequence[int],
    *,
    rgb_components: Sequence[int] = (0, 1, 2),
    upscale_factor: int = 24,
) -> np.ndarray:
    if projection.ndim != 2:
        raise ValueError(f"Expected 2D projection array, got shape {projection.shape}")
    if len(latent_grid_shape) != 5:
        raise ValueError(f"Expected latent grid shape with 5 dimensions, got {latent_grid_shape}")

    rgb_indices = tuple(int(index) for index in rgb_components)
    if len(rgb_indices) != 3:
        raise ValueError("Exactly 3 projection components are required for RGB rendering")
    if len(set(rgb_indices)) != 3:
        raise ValueError("RGB components must be distinct")
    if min(rgb_indices) < 0 or max(rgb_indices) >= projection.shape[1]:
        raise ValueError("RGB component index is out of bounds")

    _, time_steps, grid_h, grid_w, _ = [int(value) for value in latent_grid_shape]
    expected_rows = time_steps * grid_h * grid_w
    if projection.shape[0] != expected_rows:
        raise ValueError("Projection row count does not match the latent grid shape")

    selected_projection = projection[:, rgb_indices]
    rgb = minmax_scale(selected_projection)
    intensity = minmax_scale(np.linalg.norm(selected_projection, axis=1, keepdims=True))
    rgb = np.clip(rgb * (0.35 + 0.65 * intensity), 0.0, 1.0)
    frames = (rgb.reshape(time_steps, grid_h, grid_w, 3) * 255.0).astype(np.uint8)
    if upscale_factor > 1:
        frames = np.repeat(np.repeat(frames, upscale_factor, axis=1), upscale_factor, axis=2)
    return frames


def latent_rgb_frames(latent_grid: np.ndarray, upscale_factor: int = 24) -> np.ndarray:
    projection_bundle = compute_projection_bundle(latent_grid, method="pca", n_components=3)
    return projection_rgb_frames(
        projection_bundle["projection"],
        latent_grid.shape,
        rgb_components=(0, 1, 2),
        upscale_factor=upscale_factor,
    )


def minmax_scale(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    min_value = array.min(axis=0, keepdims=True)
    max_value = array.max(axis=0, keepdims=True)
    scale = max_value - min_value
    scale[scale == 0] = 1.0
    return (array - min_value) / scale


def infer_latent_fps(frame_indices: list[int], source_fps: float, tubelet_size: int = 2) -> float:
    if source_fps <= 0:
        return 4.0
    if len(frame_indices) < 2:
        return max(1.0, source_fps / tubelet_size)
    diffs = np.diff(frame_indices)
    mean_stride = float(np.mean(diffs)) if len(diffs) else 1.0
    effective_clip_fps = source_fps / max(mean_stride, 1e-6)
    return max(1.0, effective_clip_fps / max(tubelet_size, 1))


def load_aligned_source_frames(
    metadata: dict[str, Any],
    latent_grid_shape: Sequence[int],
) -> tuple[np.ndarray, float, list[int]]:
    latent_grid_shape_tuple = tuple(int(value) for value in latent_grid_shape)
    if len(latent_grid_shape_tuple) != 5:
        raise ValueError(f"Expected latent grid shape with 5 dimensions, got {latent_grid_shape}")

    frame_indices = [int(index) for index in metadata["frame_indices"]]
    source_fps = float(metadata.get("video_metadata", {}).get("fps", 0.0) or 0.0)
    tubelet_size = int(metadata.get("tubelet_size", 2) or 2)
    latent_fps = infer_latent_fps(frame_indices, source_fps, tubelet_size=tubelet_size)
    source_frame_indices = frame_indices[:: max(tubelet_size, 1)][: latent_grid_shape_tuple[1]]
    source_frames = read_video_frames(Path(metadata["video_path"]), source_frame_indices)
    crop_size = normalize_crop_size(metadata.get("crop_size", [latent_grid_shape_tuple[2] * 16, latent_grid_shape_tuple[3] * 16]))
    source_frames = prepare_display_frames(source_frames, crop_size)
    return source_frames, latent_fps, source_frame_indices


def _fit_frame(frame: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    target_height, target_width = target_size
    if frame.shape[0] == target_height and frame.shape[1] == target_width:
        return frame

    source_height, source_width = frame.shape[:2]
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)

    canvas = np.full((target_height, target_width, 3), 245, dtype=np.uint8)
    top = (target_height - resized_height) // 2
    left = (target_width - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def side_by_side_frames(
    source_frames: np.ndarray,
    latent_frames: np.ndarray,
    *,
    left_label: str = "Source",
    right_label: str = "Latent PCA RGB",
    gap: int = 16,
    title_height: int = 36,
    panel_size: tuple[int, int] | None = None,
) -> np.ndarray:
    if len(source_frames) != len(latent_frames):
        raise ValueError("source_frames and latent_frames must have the same length")

    if panel_size is None:
        panel_height = max(source_frames.shape[1], latent_frames.shape[1])
        panel_width = max(source_frames.shape[2], latent_frames.shape[2])
    else:
        panel_height, panel_width = (int(panel_size[0]), int(panel_size[1]))
    canvas_height = panel_height + title_height + gap
    canvas_width = panel_width * 2 + gap * 3

    combined_frames: list[np.ndarray] = []
    for source_frame, latent_frame in zip(source_frames, latent_frames, strict=True):
        source_panel = _fit_frame(source_frame, (panel_height, panel_width))
        latent_panel = _fit_frame(latent_frame, (panel_height, panel_width))
        canvas = np.full((canvas_height, canvas_width, 3), 245, dtype=np.uint8)

        y_offset = title_height
        canvas[y_offset : y_offset + panel_height, gap : gap + panel_width] = source_panel
        canvas[y_offset : y_offset + panel_height, gap * 2 + panel_width : gap * 2 + panel_width * 2] = latent_panel

        cv2.putText(canvas, left_label, (gap, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (32, 32, 32), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            right_label,
            (gap * 2 + panel_width, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (32, 32, 32),
            2,
            cv2.LINE_AA,
        )
        combined_frames.append(canvas)

    return np.stack(combined_frames, axis=0)


def _ensure_even_frame_size(frames: np.ndarray) -> np.ndarray:
    frame_height, frame_width = frames.shape[1:3]
    pad_height = frame_height % 2
    pad_width = frame_width % 2
    if pad_height == 0 and pad_width == 0:
        return frames
    return np.pad(
        frames,
        ((0, 0), (0, pad_height), (0, pad_width), (0, 0)),
        mode="constant",
        constant_values=245,
    )


def _resolve_ffmpeg_executable() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    try:
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
    except ImportError as error:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "ffmpeg is required to write browser-compatible MP4 files. Install `ffmpeg` or `imageio-ffmpeg`."
        ) from error

    return imageio_ffmpeg.get_ffmpeg_exe()


def write_video(video_path: Path, frames: np.ndarray, fps: float) -> Path:
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    frames = _ensure_even_frame_size(frames)
    frame_height, frame_width = frames.shape[1:3]
    ffmpeg_path = _resolve_ffmpeg_executable()
    command = [
        ffmpeg_path,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{frame_width}x{frame_height}",
        "-r",
        str(float(fps)),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    completed = subprocess.run(
        command,
        input=np.ascontiguousarray(frames, dtype=np.uint8).tobytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to write {video_path}: {completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return video_path


def create_visualizations(
    *,
    latent_grid: np.ndarray,
    metadata: dict[str, Any],
    output_dir: Path,
) -> VisualizationArtifacts:
    projection_bundle = compute_projection_bundle(latent_grid, method="pca", n_components=3)
    return create_visualizations_from_projection(
        projection=projection_bundle["projection"],
        latent_grid_shape=latent_grid.shape,
        metadata=metadata,
        output_dir=output_dir,
        rgb_components=(0, 1, 2),
        method="pca",
        component_labels=projection_bundle["component_labels"],
    )


def create_visualizations_from_projection(
    *,
    projection: np.ndarray,
    latent_grid_shape: Sequence[int],
    metadata: dict[str, Any],
    output_dir: Path,
    rgb_components: Sequence[int] = (0, 1, 2),
    method: str = "pca",
    component_labels: Sequence[str] | None = None,
    upscale_factor: int = 24,
) -> VisualizationArtifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latent_grid_shape_tuple = tuple(int(value) for value in latent_grid_shape)
    source_frames, latent_fps, _ = load_aligned_source_frames(metadata, latent_grid_shape_tuple)
    latent_frames = projection_rgb_frames(
        projection,
        latent_grid_shape_tuple,
        rgb_components=rgb_components,
        upscale_factor=upscale_factor,
    )

    labels = list(component_labels) if component_labels is not None else projection_component_labels(method, projection.shape[1])
    rgb_label = ", ".join(labels[index] if index < len(labels) else f"C{index + 1}" for index in rgb_components)
    combined_frames = side_by_side_frames(
        source_frames,
        latent_frames,
        right_label=f"Latent {projection_method_display_name(method)} RGB ({rgb_label})",
        panel_size=latent_frames.shape[1:3],
    )

    component_suffix = "-".join(str(index + 1) for index in rgb_components)
    method_name = method.strip().lower()
    latent_video_path = write_video(output_dir / f"latent_{method_name}_rgb_c{component_suffix}.mp4", latent_frames, fps=latent_fps)
    side_by_side_video_path = write_video(
        output_dir / f"latent_{method_name}_side_by_side_c{component_suffix}.mp4",
        combined_frames,
        fps=latent_fps,
    )

    return VisualizationArtifacts(
        latent_video_path=latent_video_path,
        side_by_side_video_path=side_by_side_video_path,
        latent_video_shape=tuple(latent_frames.shape),
        side_by_side_video_shape=tuple(combined_frames.shape),
        display_fps=latent_fps,
    )
