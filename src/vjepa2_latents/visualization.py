from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import numpy as np

from .projection import (
    MLX_VIS_METHOD_SPECS,
    ProjectionArtifacts,
    compute_pca_projection,
    compute_projection_bundle,
    compute_umap_projection,
    flatten_latent_grid,
    has_mlx_vis_support as _has_mlx_vis_support,
    _filtered_constructor_kwargs,
    has_umap_support,
    load_saved_latents,
    load_saved_projection,
    minmax_scale,
    normalize_projection_method,
    projection_component_labels,
    projection_method_display_name,
    projection_mode_display_name,
    save_projection_artifacts,
    summarize_latents,
)
from .vos import (
    SegmentationArtifacts,
    annotate_prompt_points,
    create_segmentation_video,
    knn_binary_segmentation_volume,
    segmentation_mask_frames,
)
from .video import (
    PatchSimilarityArtifacts,
    VisualizationArtifacts,
    _ensure_even_frame_size,
    _resolve_ffmpeg_executable,
    annotate_selected_patch,
    build_pca_figure,
    build_projection_figure,
    build_projection_figure_from_data,
    cosine_similarity_volume,
    create_patch_similarity_video,
    create_visualizations,
    create_visualizations_from_projection,
    infer_latent_fps,
    latent_rgb_frames,
    load_aligned_source_frames,
    map_click_to_latent_token,
    projection_rgb_frames,
    side_by_side_frames,
    similarity_heatmap_frames,
)

__all__ = [
    "MLX_VIS_METHOD_SPECS",
    "PatchSimilarityArtifacts",
    "SegmentationArtifacts",
    "ProjectionArtifacts",
    "VisualizationArtifacts",
    "_ensure_even_frame_size",
    "_resolve_ffmpeg_executable",
    "annotate_prompt_points",
    "annotate_selected_patch",
    "build_pca_figure",
    "build_projection_figure",
    "build_projection_figure_from_data",
    "compute_mlx_projection",
    "compute_pca_projection",
    "compute_projection_bundle",
    "compute_umap_projection",
    "cosine_similarity_volume",
    "create_patch_similarity_video",
    "create_segmentation_video",
    "create_visualizations",
    "create_visualizations_from_projection",
    "flatten_latent_grid",
    "has_mlx_vis_support",
    "has_umap_support",
    "infer_latent_fps",
    "knn_binary_segmentation_volume",
    "latent_rgb_frames",
    "load_aligned_source_frames",
    "load_saved_latents",
    "load_saved_projection",
    "map_click_to_latent_token",
    "minmax_scale",
    "normalize_projection_method",
    "projection_component_labels",
    "projection_method_display_name",
    "projection_mode_display_name",
    "projection_rgb_frames",
    "save_projection_artifacts",
    "segmentation_mask_frames",
    "side_by_side_frames",
    "similarity_heatmap_frames",
    "summarize_latents",
    "write_video",
]


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


def compute_mlx_projection(
    features: np.ndarray,
    *,
    method: str,
    n_components: int = 3,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int | None = 42,
) -> np.ndarray:
    if not has_mlx_vis_support():
        raise RuntimeError(
            "This projection method requires the optional `mlx-vis` dependency. "
            "Install it on Apple Silicon with `pip install mlx-vis` to enable MLX-accelerated reducers."
        )

    if features.ndim != 2:
        raise ValueError(f"Expected 2D features array, got shape {features.shape}")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if features.shape[0] < 2:
        raise ValueError("MLX projections require at least 2 samples")

    reducer_name = normalize_projection_method(method)
    spec = MLX_VIS_METHOD_SPECS.get(reducer_name)
    if spec is None:
        raise ValueError(f"Unknown MLX projection method: {method}")

    mlx_vis = importlib.import_module("mlx_vis")
    reducer_class = getattr(mlx_vis, spec["class_name"])
    effective_neighbors = max(2, min(int(n_neighbors), features.shape[0] - 1))
    constructor_kwargs = _filtered_constructor_kwargs(
        reducer_class,
        {
            "n_components": int(n_components),
            "n_neighbors": effective_neighbors,
            "min_dist": float(min_dist),
            "metric": metric,
            "random_state": random_state,
        },
    )
    reducer = reducer_class(**constructor_kwargs)
    projection = reducer.fit_transform(features)
    return np.asarray(projection, dtype=np.float32)


def has_mlx_vis_support() -> bool:
    return _has_mlx_vis_support()
