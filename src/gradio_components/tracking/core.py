from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from ..projection.core import flatten_latent_grid
from ..render.video import infer_latent_fps, load_aligned_source_frames, write_video


@dataclass(frozen=True)
class PatchSimilarityArtifacts:
    similarity_video_path: Path
    similarity_video_shape: tuple[int, int, int, int]
    display_fps: float
    token_index: tuple[int, int, int]
    similarity_min: float
    similarity_max: float


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

    def select_token(click_x: float, click_y: float) -> tuple[int, int, int]:
        bounded_x = min(max(click_x, 0.0), image_width - 1)
        bounded_y = min(max(click_y, 0.0), image_height - 1)
        token_h = min(int((bounded_y / image_height) * grid_h), max(grid_h - 1, 0))
        token_w = min(int((bounded_x / image_width) * grid_w), max(grid_w - 1, 0))
        return clamped_time, token_h, token_w

    def token_center_distance(token_index: tuple[int, int, int], click_x: float, click_y: float) -> float:
        _, token_h, token_w = token_index
        cell_width = image_width / max(grid_w, 1)
        cell_height = image_height / max(grid_h, 1)
        center_x = (token_w + 0.5) * cell_width
        center_y = (token_h + 0.5) * cell_height
        return float((center_x - click_x) ** 2 + (center_y - click_y) ** 2)

    click_x = float(click_xy[0])
    click_y = float(click_xy[1])
    token_xy = select_token(click_x, click_y)
    token_yx = select_token(click_y, click_x)
    if token_center_distance(token_yx, click_x, click_y) < token_center_distance(token_xy, click_x, click_y):
        return token_yx
    return token_xy


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
