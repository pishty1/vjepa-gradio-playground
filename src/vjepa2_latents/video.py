from __future__ import annotations

import importlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .extractor import normalize_crop_size, prepare_display_frames, read_video_frames
from .projection import (
    compute_projection_bundle,
    flatten_latent_grid,
    minmax_scale,
    projection_component_labels,
    projection_method_display_name,
)

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - exercised only when optional dependency is missing
    go = None


@dataclass(frozen=True)
class VisualizationArtifacts:
    latent_video_path: Path
    side_by_side_video_path: Path
    latent_video_shape: tuple[int, int, int, int]
    side_by_side_video_shape: tuple[int, int, int, int]
    display_fps: float


@dataclass(frozen=True)
class PatchSimilarityArtifacts:
    similarity_video_path: Path
    similarity_video_shape: tuple[int, int, int, int]
    display_fps: float
    token_index: tuple[int, int, int]
    similarity_min: float
    similarity_max: float


def _validate_component_indices(component_indices: Sequence[int], component_count: int) -> tuple[int, ...]:
    indices = tuple(int(index) for index in component_indices)
    if len(indices) not in {2, 3}:
        raise ValueError("Select exactly 2 or 3 components")
    if len(set(indices)) != len(indices):
        raise ValueError("Selected components must be distinct")
    if min(indices) < 0 or max(indices) >= component_count:
        raise ValueError("Selected component index is out of bounds")
    return indices


def build_projection_figure_from_data(
    projection: np.ndarray,
    coordinates: np.ndarray,
    *,
    method: str = "pca",
    component_indices: Sequence[int] = (0, 1, 2),
    component_labels: Sequence[str] | None = None,
    max_points: int = 4000,
):
    if go is None:
        raise RuntimeError("plotly is required to build the interactive latent-space plot")
    if projection.ndim != 2:
        raise ValueError(f"Expected 2D projection array, got shape {projection.shape}")
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"Expected coordinates with shape [n, 3], got {coordinates.shape}")
    if projection.shape[0] != coordinates.shape[0]:
        raise ValueError("Projection rows and coordinates rows must match")

    indices = _validate_component_indices(component_indices, projection.shape[1])
    reduced_projection = projection[:, indices]
    labels = list(component_labels) if component_labels is not None else projection_component_labels(method, projection.shape[1])
    selected_labels = [labels[index] if index < len(labels) else f"C{index + 1}" for index in indices]

    if reduced_projection.shape[0] > max_points:
        selection = np.linspace(0, reduced_projection.shape[0] - 1, num=max_points, dtype=int)
        reduced_projection = reduced_projection[selection]
        coordinates = coordinates[selection]

    colors = coordinates[:, 0]
    hover_text = [
        f"t={time_index}, h={row_index}, w={column_index}"
        for time_index, row_index, column_index in coordinates.tolist()
    ]

    if len(indices) == 3:
        hover_template = (
            f"%{{text}}<br>{selected_labels[0]}=%{{x:.3f}}<br>{selected_labels[1]}=%{{y:.3f}}<br>{selected_labels[2]}=%{{z:.3f}}<extra></extra>"
        )
        trace = go.Scatter3d(
            x=reduced_projection[:, 0],
            y=reduced_projection[:, 1],
            z=reduced_projection[:, 2],
            mode="markers",
            marker={
                "size": 4,
                "color": colors,
                "colorscale": "Viridis",
                "opacity": 0.85,
                "colorbar": {"title": "Latent t"},
            },
            text=hover_text,
            hovertemplate=hover_template,
        )
        figure = go.Figure(data=[trace])
        figure.update_layout(
            title=f"Latent-space {projection_method_display_name(method)} projection",
            scene={
                "xaxis_title": selected_labels[0],
                "yaxis_title": selected_labels[1],
                "zaxis_title": selected_labels[2],
            },
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
            template="plotly_white",
        )
        return figure

    hover_template = f"%{{text}}<br>{selected_labels[0]}=%{{x:.3f}}<br>{selected_labels[1]}=%{{y:.3f}}<extra></extra>"
    trace = go.Scatter(
        x=reduced_projection[:, 0],
        y=reduced_projection[:, 1],
        mode="markers",
        marker={
            "size": 7,
            "color": colors,
            "colorscale": "Viridis",
            "opacity": 0.8,
            "colorbar": {"title": "Latent t"},
        },
        text=hover_text,
        hovertemplate=hover_template,
    )
    figure = go.Figure(data=[trace])
    figure.update_layout(
        title=f"Latent-space {projection_method_display_name(method)} projection",
        xaxis_title=selected_labels[0],
        yaxis_title=selected_labels[1],
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        template="plotly_white",
    )
    return figure


def build_projection_figure(
    latent_grid: np.ndarray,
    *,
    method: str = "pca",
    n_components: int = 3,
    max_points: int = 4000,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_metric: str = "euclidean",
    umap_random_state: int | None = 42,
):
    projection_bundle = compute_projection_bundle(
        latent_grid,
        method=method,
        n_components=max(2, int(n_components)),
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        umap_metric=umap_metric,
        umap_random_state=umap_random_state,
    )
    component_count = min(3, int(projection_bundle["projection"].shape[1]))
    return build_projection_figure_from_data(
        projection_bundle["projection"],
        projection_bundle["coordinates"],
        method=str(projection_bundle["method"]),
        component_indices=tuple(range(component_count)),
        component_labels=projection_bundle["component_labels"],
        max_points=max_points,
    )


def build_pca_figure(latent_grid: np.ndarray, max_points: int = 4000):
    return build_projection_figure(latent_grid, method="pca", n_components=3, max_points=max_points)


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


def map_click_to_latent_token(
    click_xy: Sequence[int | float],
    image_shape: Sequence[int],
    latent_grid_shape: Sequence[int],
    *,
    time_index: int = 0,
) -> tuple[int, int, int]:
    if len(click_xy) < 2:
        raise ValueError("click_xy must contain x and y coordinates")
    if len(image_shape) < 2:
        raise ValueError("image_shape must contain height and width")
    if len(latent_grid_shape) != 5:
        raise ValueError(f"Expected latent grid shape with 5 dimensions, got {latent_grid_shape}")

    image_height = int(image_shape[0])
    image_width = int(image_shape[1])
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image dimensions must be positive")

    _, time_steps, grid_h, grid_w, _ = [int(value) for value in latent_grid_shape]
    clamped_time = min(max(int(time_index), 0), max(time_steps - 1, 0))
    click_x = min(max(float(click_xy[0]), 0.0), image_width - 1)
    click_y = min(max(float(click_xy[1]), 0.0), image_height - 1)
    token_h = min(int((click_y / image_height) * grid_h), max(grid_h - 1, 0))
    token_w = min(int((click_x / image_width) * grid_w), max(grid_w - 1, 0))
    return clamped_time, token_h, token_w


def cosine_similarity_volume(
    latent_grid: np.ndarray,
    token_index: Sequence[int],
) -> np.ndarray:
    if latent_grid.ndim != 5:
        raise ValueError(f"Expected latent grid with 5 dims [b, t, h, w, d], got {latent_grid.shape}")
    if len(token_index) != 3:
        raise ValueError("token_index must contain exactly (t, h, w)")

    features, coordinates = flatten_latent_grid(latent_grid)
    time_index, row_index, column_index = [int(value) for value in token_index]
    matches = np.where(
        (coordinates[:, 0] == time_index)
        & (coordinates[:, 1] == row_index)
        & (coordinates[:, 2] == column_index)
    )[0]
    if len(matches) == 0:
        raise ValueError(f"Token index {tuple(token_index)} is out of bounds for latent grid {latent_grid.shape}")

    normalized_features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    query = normalized_features[matches[0]]
    similarity = normalized_features @ query
    _, time_steps, grid_h, grid_w, _ = latent_grid.shape
    return similarity.reshape(time_steps, grid_h, grid_w).astype(np.float32, copy=False)


def annotate_selected_patch(
    frame: np.ndarray,
    token_index: Sequence[int],
    latent_grid_shape: Sequence[int],
    *,
    color: tuple[int, int, int] = (255, 96, 64),
    thickness: int = 2,
) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected RGB frame with shape [h, w, 3], got {frame.shape}")
    if len(token_index) != 3:
        raise ValueError("token_index must contain exactly (t, h, w)")
    if len(latent_grid_shape) != 5:
        raise ValueError(f"Expected latent grid shape with 5 dimensions, got {latent_grid_shape}")

    annotated = np.array(frame, copy=True)
    _, _, grid_h, grid_w, _ = [int(value) for value in latent_grid_shape]
    _, row_index, column_index = [int(value) for value in token_index]
    cell_height = frame.shape[0] / max(grid_h, 1)
    cell_width = frame.shape[1] / max(grid_w, 1)
    top = int(round(row_index * cell_height))
    bottom = max(top + 1, int(round((row_index + 1) * cell_height)) - 1)
    left = int(round(column_index * cell_width))
    right = max(left + 1, int(round((column_index + 1) * cell_width)) - 1)
    cv2.rectangle(annotated, (left, top), (right, bottom), color, thickness)
    return annotated


def similarity_heatmap_frames(
    source_frames: np.ndarray,
    similarity_volume: np.ndarray,
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    if source_frames.ndim != 4 or source_frames.shape[-1] != 3:
        raise ValueError(f"Expected source_frames with shape [t, h, w, 3], got {source_frames.shape}")
    if similarity_volume.ndim != 3:
        raise ValueError(f"Expected similarity_volume with shape [t, h, w], got {similarity_volume.shape}")
    if source_frames.shape[0] != similarity_volume.shape[0]:
        raise ValueError("source_frames and similarity_volume must have the same number of time steps")

    blended_frames: list[np.ndarray] = []
    blend_alpha = float(np.clip(alpha, 0.0, 1.0))
    positive_similarity = np.clip(np.asarray(similarity_volume, dtype=np.float32), 0.0, None)
    positive_values = positive_similarity[positive_similarity > 0.0]
    if positive_values.size == 0:
        normalized = np.zeros_like(positive_similarity, dtype=np.float32)
    else:
        lower = float(np.percentile(positive_values, 70.0))
        upper = float(np.percentile(positive_values, 99.5))
        if upper <= lower:
            upper = float(positive_values.max())
        scale = max(upper - lower, 1e-6)
        normalized = np.clip((positive_similarity - lower) / scale, 0.0, 1.0)
        normalized = np.power(normalized, 0.65)

    for frame, similarity_map in zip(source_frames, normalized, strict=True):
        heatmap = cv2.resize(similarity_map, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        heatmap_uint8 = np.clip(heatmap * 255.0, 0.0, 255.0).astype(np.uint8)
        colorized = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_TURBO)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB).astype(np.float32)
        frame_float = frame.astype(np.float32)
        alpha_map = (blend_alpha * np.power(heatmap, 0.9))[..., None]
        blended = frame_float * (1.0 - alpha_map) + colorized * alpha_map
        blended_frames.append(np.clip(blended, 0.0, 255.0).astype(np.uint8))
    return np.stack(blended_frames, axis=0)


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


def create_patch_similarity_video(
    *,
    latent_grid: np.ndarray,
    metadata: dict[str, Any],
    output_dir: Path,
    token_index: Sequence[int],
    source_frames: np.ndarray | None = None,
    alpha: float = 0.45,
) -> PatchSimilarityArtifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if source_frames is None:
        source_frames, latent_fps, _ = load_aligned_source_frames(metadata, latent_grid.shape)
    else:
        latent_fps = infer_latent_fps(
            [int(index) for index in metadata["frame_indices"]],
            float(metadata.get("video_metadata", {}).get("fps", 0.0) or 0.0),
            tubelet_size=int(metadata.get("tubelet_size", 2) or 2),
        )

    normalized_token = tuple(int(value) for value in token_index)
    similarity_volume = cosine_similarity_volume(latent_grid, normalized_token)
    overlay_frames = similarity_heatmap_frames(source_frames, similarity_volume, alpha=alpha)
    _, time_steps, grid_h, grid_w, _ = latent_grid.shape
    flattened_similarity = similarity_volume.reshape(time_steps, grid_h * grid_w)
    peak_patch_indices = np.argmax(flattened_similarity, axis=1)
    peak_rows, peak_columns = np.unravel_index(peak_patch_indices, (grid_h, grid_w))

    for frame_index in range(time_steps):
        overlay_frames[frame_index] = annotate_selected_patch(
            overlay_frames[frame_index],
            (frame_index, int(peak_rows[frame_index]), int(peak_columns[frame_index])),
            latent_grid.shape,
            color=(48, 255, 255),
            thickness=3,
        )
    overlay_frames[0] = annotate_selected_patch(
        overlay_frames[0],
        normalized_token,
        latent_grid.shape,
        color=(255, 96, 64),
        thickness=4,
    )

    component_suffix = f"t{normalized_token[0] + 1}_h{normalized_token[1] + 1}_w{normalized_token[2] + 1}"
    similarity_video_path = write_video(
        output_dir / f"patch_similarity_{component_suffix}.mp4",
        overlay_frames,
        fps=latent_fps,
    )
    return PatchSimilarityArtifacts(
        similarity_video_path=similarity_video_path,
        similarity_video_shape=tuple(overlay_frames.shape),
        display_fps=latent_fps,
        token_index=normalized_token,
        similarity_min=float(similarity_volume.min()),
        similarity_max=float(similarity_volume.max()),
    )
