from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from ..projection import flatten_latent_grid


@dataclass(frozen=True)
class SegmentationArtifacts:
    segmentation_video_path: Path
    segmentation_video_shape: tuple[int, int, int, int]
    display_fps: float
    foreground_token: tuple[int, int, int]
    background_token: tuple[int, int, int]
    knn_neighbors: int
    temperature: float
    context_frames: int
    spatial_radius: int
    foreground_ratio_per_frame: tuple[float, ...]



def annotate_prompt_points(
    frame: np.ndarray,
    click_points: dict[str, Sequence[int | float] | None] | None,
    *,
    radius: int = 8,
) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected RGB frame with shape [h, w, 3], got {frame.shape}")

    annotated = np.array(frame, copy=True)
    if not click_points:
        return annotated

    point_styles = {
        "foreground": ((64, 224, 96), "FG"),
        "background": ((255, 96, 96), "BG"),
    }
    for label, point in click_points.items():
        if label not in point_styles or point is None or len(point) < 2:
            continue
        color, text = point_styles[label]
        center = (int(round(float(point[0]))), int(round(float(point[1]))))
        cv2.circle(annotated, center, int(radius), color, thickness=-1)
        cv2.circle(annotated, center, int(radius) + 2, (255, 255, 255), thickness=2)
        cv2.putText(
            annotated,
            text,
            (center[0] + int(radius) + 6, max(18, center[1] - int(radius) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated



def knn_binary_segmentation_volume(
    latent_grid: np.ndarray,
    foreground_token: Sequence[int],
    background_token: Sequence[int],
    *,
    k_neighbors: int = 5,
    temperature: float = 0.2,
    context_frames: int = 15,
    spatial_radius: int = 12,
) -> np.ndarray:
    if latent_grid.ndim != 5:
        raise ValueError(f"Expected latent grid with 5 dims [b, t, h, w, d], got {latent_grid.shape}")

    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")

    _, time_steps, grid_h, grid_w, _ = latent_grid.shape
    query_time_index = int(foreground_token[0])
    if any(int(token[0]) != query_time_index for token in (foreground_token, background_token)):
        raise ValueError("foreground_token and background_token must come from the same prompt frame")
    if query_time_index != 0:
        raise ValueError("paper-style VOS propagation expects prompts from the first frame (t=0)")

    def _token_feature(token_index: Sequence[int], features: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        if len(token_index) != 3:
            raise ValueError("Prompt token must contain exactly (t, h, w)")
        time_index, row_index, column_index = [int(value) for value in token_index]
        matches = np.where(
            (coordinates[:, 0] == time_index)
            & (coordinates[:, 1] == row_index)
            & (coordinates[:, 2] == column_index)
        )[0]
        if len(matches) == 0:
            raise ValueError(f"Token index {tuple(token_index)} is out of bounds for latent grid {latent_grid.shape}")
        return features[matches[0]]

    features, coordinates = flatten_latent_grid(latent_grid)
    normalized_features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    feature_grid = normalized_features.reshape(time_steps, grid_h, grid_w, -1)

    prompt_vectors = np.stack(
        [
            _token_feature(foreground_token, normalized_features, coordinates),
            _token_feature(background_token, normalized_features, coordinates),
        ],
        axis=0,
    )
    prompt_labels = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    effective_k = max(1, int(k_neighbors))
    radius = max(0, int(spatial_radius))
    context = max(0, int(context_frames))

    probability_volume = np.zeros((time_steps, grid_h, grid_w, 2), dtype=np.float32)
    hard_label_volume = np.zeros((time_steps, grid_h, grid_w), dtype=np.int32)

    prompt_positions = {
        (int(foreground_token[1]), int(foreground_token[2])): np.array([1.0, 0.0], dtype=np.float32),
        (int(background_token[1]), int(background_token[2])): np.array([0.0, 1.0], dtype=np.float32),
    }

    def _build_memory_for_frame(frame_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if frame_index == 0:
            memory_coords = np.array(
                [
                    [0, int(foreground_token[1]), int(foreground_token[2])],
                    [0, int(background_token[1]), int(background_token[2])],
                ],
                dtype=np.int32,
            )
            return prompt_vectors, prompt_labels, memory_coords

        memory_features_list: list[np.ndarray] = []
        memory_label_list: list[np.ndarray] = []
        memory_coord_list: list[np.ndarray] = []

        first_frame_coords = np.stack(
            np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij"),
            axis=-1,
        ).reshape(-1, 2)
        memory_features_list.append(feature_grid[0].reshape(-1, feature_grid.shape[-1]))
        memory_label_list.append(probability_volume[0].reshape(-1, 2))
        memory_coord_list.append(
            np.column_stack(
                [
                    np.zeros(len(first_frame_coords), dtype=np.int32),
                    first_frame_coords[:, 0],
                    first_frame_coords[:, 1],
                ]
            )
        )

        start_frame = max(1, frame_index - context)
        for previous_frame in range(start_frame, frame_index):
            frame_coords = np.stack(
                np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij"),
                axis=-1,
            ).reshape(-1, 2)
            memory_features_list.append(feature_grid[previous_frame].reshape(-1, feature_grid.shape[-1]))
            memory_label_list.append(probability_volume[previous_frame].reshape(-1, 2))
            memory_coord_list.append(
                np.column_stack(
                    [
                        np.full(len(frame_coords), previous_frame, dtype=np.int32),
                        frame_coords[:, 0],
                        frame_coords[:, 1],
                    ]
                )
            )

        return (
            np.concatenate(memory_features_list, axis=0),
            np.concatenate(memory_label_list, axis=0),
            np.concatenate(memory_coord_list, axis=0),
        )

    for frame_index in range(time_steps):
        memory_features, memory_labels, memory_coords = _build_memory_for_frame(frame_index)
        for row_index in range(grid_h):
            for column_index in range(grid_w):
                prompt_label = prompt_positions.get((row_index, column_index)) if frame_index == 0 else None
                if prompt_label is not None:
                    probability_volume[frame_index, row_index, column_index] = prompt_label
                    hard_label_volume[frame_index, row_index, column_index] = int(np.argmax(prompt_label))
                    continue

                valid_memory = np.ones(len(memory_coords), dtype=bool)
                if frame_index > 0 and radius > 0:
                    delta_rows = memory_coords[:, 1] - row_index
                    delta_columns = memory_coords[:, 2] - column_index
                    local_mask = (delta_rows * delta_rows + delta_columns * delta_columns) <= (radius * radius)
                    valid_memory = (memory_coords[:, 0] == 0) | local_mask
                elif frame_index > 0 and radius == 0:
                    valid_memory = memory_coords[:, 0] == 0

                filtered_features = memory_features[valid_memory]
                filtered_labels = memory_labels[valid_memory]
                if len(filtered_features) == 0:
                    filtered_features = prompt_vectors
                    filtered_labels = prompt_labels

                similarity = filtered_features @ feature_grid[frame_index, row_index, column_index]
                top_k = max(1, min(effective_k, similarity.shape[0]))
                nearest_indices = np.argpartition(-similarity, top_k - 1)[:top_k]
                nearest_similarity = similarity[nearest_indices]
                nearest_labels = filtered_labels[nearest_indices]
                weights = np.exp((nearest_similarity - float(np.max(nearest_similarity))) / float(temperature))
                weights = weights / np.maximum(np.sum(weights), 1e-12)
                probabilities = np.sum(nearest_labels * weights[:, None], axis=0)
                probability_volume[frame_index, row_index, column_index] = probabilities
                hard_label_volume[frame_index, row_index, column_index] = int(np.argmax(probabilities))

    return hard_label_volume == 0



def segmentation_mask_frames(
    source_frames: np.ndarray,
    segmentation_volume: np.ndarray,
    *,
    alpha: float = 0.45,
    mask_color: tuple[int, int, int] = (64, 224, 96),
) -> np.ndarray:
    if source_frames.ndim != 4 or source_frames.shape[-1] != 3:
        raise ValueError(f"Expected source_frames with shape [t, h, w, 3], got {source_frames.shape}")
    if segmentation_volume.ndim != 3:
        raise ValueError(f"Expected segmentation_volume with shape [t, h, w], got {segmentation_volume.shape}")
    if source_frames.shape[0] != segmentation_volume.shape[0]:
        raise ValueError("source_frames and segmentation_volume must have the same number of time steps")

    blended_frames: list[np.ndarray] = []
    blend_alpha = float(np.clip(alpha, 0.0, 1.0))
    overlay_color = np.asarray(mask_color, dtype=np.float32)

    for frame, mask in zip(source_frames, segmentation_volume, strict=True):
        resized_mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        binary_mask = resized_mask.astype(bool)
        frame_float = frame.astype(np.float32)
        blended = frame_float.copy()
        if np.any(binary_mask):
            alpha_map = (binary_mask.astype(np.float32) * blend_alpha)[..., None]
            blended = frame_float * (1.0 - alpha_map) + overlay_color * alpha_map

            contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            outlined = np.clip(blended, 0.0, 255.0).astype(np.uint8)
            cv2.drawContours(outlined, contours, -1, tuple(int(v) for v in mask_color), 2)
            blended_frames.append(outlined)
        else:
            blended_frames.append(np.clip(blended, 0.0, 255.0).astype(np.uint8))

    return np.stack(blended_frames, axis=0)



def create_segmentation_video(
    *,
    latent_grid: np.ndarray,
    metadata: dict[str, Any],
    output_dir: Path,
    foreground_token: Sequence[int],
    background_token: Sequence[int],
    source_frames: np.ndarray | None = None,
    foreground_click_xy: Sequence[int | float] | None = None,
    background_click_xy: Sequence[int | float] | None = None,
    k_neighbors: int = 5,
    temperature: float = 0.2,
    context_frames: int = 15,
    spatial_radius: int = 12,
    alpha: float = 0.45,
) -> SegmentationArtifacts:
    from ..video import infer_latent_fps, load_aligned_source_frames, write_video

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

    normalized_foreground = tuple(int(value) for value in foreground_token)
    normalized_background = tuple(int(value) for value in background_token)
    segmentation_volume = knn_binary_segmentation_volume(
        latent_grid,
        normalized_foreground,
        normalized_background,
        k_neighbors=k_neighbors,
        temperature=temperature,
        context_frames=context_frames,
        spatial_radius=spatial_radius,
    )
    overlay_frames = segmentation_mask_frames(source_frames, segmentation_volume, alpha=alpha)
    overlay_frames[0] = annotate_prompt_points(
        overlay_frames[0],
        {
            "foreground": foreground_click_xy,
            "background": background_click_xy,
        },
    )

    output_path = write_video(
        output_dir
        / (
            f"vos_segmentation_fg_t{normalized_foreground[0] + 1}_h{normalized_foreground[1] + 1}_w{normalized_foreground[2] + 1}"
            f"_bg_t{normalized_background[0] + 1}_h{normalized_background[1] + 1}_w{normalized_background[2] + 1}.mp4"
        ),
        overlay_frames,
        fps=latent_fps,
    )
    foreground_ratio_per_frame = tuple(float(frame_mask.mean()) for frame_mask in segmentation_volume)
    return SegmentationArtifacts(
        segmentation_video_path=output_path,
        segmentation_video_shape=tuple(overlay_frames.shape),
        display_fps=latent_fps,
        foreground_token=normalized_foreground,
        background_token=normalized_background,
        knn_neighbors=max(1, int(k_neighbors)),
        temperature=float(temperature),
        context_frames=max(0, int(context_frames)),
        spatial_radius=max(0, int(spatial_radius)),
        foreground_ratio_per_frame=foreground_ratio_per_frame,
    )
