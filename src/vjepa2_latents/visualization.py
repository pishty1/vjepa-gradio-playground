from __future__ import annotations

import importlib
import json
import importlib.util
import inspect
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .extractor import normalize_crop_size, prepare_display_frames, read_video_frames

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
class ProjectionArtifacts:
    projection_path: Path
    metadata_path: Path
    projection_shape: tuple[int, int]
    method: str
    component_labels: tuple[str, ...]


@dataclass(frozen=True)
class PatchSimilarityArtifacts:
    similarity_video_path: Path
    similarity_video_shape: tuple[int, int, int, int]
    display_fps: float
    token_index: tuple[int, int, int]
    similarity_min: float
    similarity_max: float


MLX_VIS_METHOD_SPECS: dict[str, dict[str, str]] = {
    "umap_mlx": {"class_name": "UMAP", "display_name": "UMAP-MLX"},
    "tsne_mlx": {"class_name": "TSNE", "display_name": "t-SNE-MLX"},
    "pacmap_mlx": {"class_name": "PaCMAP", "display_name": "PaCMAP-MLX"},
    "localmap_mlx": {"class_name": "LocalMAP", "display_name": "LocalMAP-MLX"},
    "trimap_mlx": {"class_name": "TriMap", "display_name": "TriMap-MLX"},
    "dreams_mlx": {"class_name": "DREAMS", "display_name": "DREAMS-MLX"},
    "cne_mlx": {"class_name": "CNE", "display_name": "CNE-MLX"},
    "mmae_mlx": {"class_name": "MMAE", "display_name": "MMAE-MLX"},
}


def has_umap_support() -> bool:
    return importlib.util.find_spec("umap.umap_") is not None


def has_mlx_vis_support() -> bool:
    return importlib.util.find_spec("mlx_vis") is not None


def normalize_projection_method(method: str) -> str:
    normalized = method.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "pca": "pca",
        "umap": "umap",
        "mlx_umap": "umap_mlx",
        "umap_mlx": "umap_mlx",
        "mlx_tsne": "tsne_mlx",
        "tsne": "tsne_mlx",
        "tsne_mlx": "tsne_mlx",
        "mlx_pacmap": "pacmap_mlx",
        "pacmap": "pacmap_mlx",
        "pacmap_mlx": "pacmap_mlx",
        "mlx_localmap": "localmap_mlx",
        "localmap": "localmap_mlx",
        "localmap_mlx": "localmap_mlx",
        "mlx_trimap": "trimap_mlx",
        "trimap": "trimap_mlx",
        "trimap_mlx": "trimap_mlx",
        "mlx_dreams": "dreams_mlx",
        "dreams": "dreams_mlx",
        "dreams_mlx": "dreams_mlx",
        "mlx_cne": "cne_mlx",
        "cne": "cne_mlx",
        "cne_mlx": "cne_mlx",
        "mlx_mmae": "mmae_mlx",
        "mmae": "mmae_mlx",
        "mmae_mlx": "mmae_mlx",
    }
    return aliases.get(normalized, normalized)


def projection_method_display_name(method: str) -> str:
    reducer_name = normalize_projection_method(method)
    if reducer_name == "pca":
        return "PCA"
    if reducer_name == "umap":
        return "UMAP"
    spec = MLX_VIS_METHOD_SPECS.get(reducer_name)
    if spec is not None:
        return spec["display_name"]
    return method.strip() or "Projection"


def _normalize_pca_mode(pca_mode: str | None) -> str:
    if pca_mode is None:
        return "global"
    normalized = pca_mode.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "global": "global",
        "global_only": "global",
        "all": "global",
        "spatial": "spatial",
        "spatial_only": "spatial",
        "time_averaged_spatial": "spatial",
        "temporal": "temporal",
        "temporal_only": "temporal",
        "space_averaged_temporal": "temporal",
    }
    if normalized not in aliases:
        raise ValueError(f"Unknown PCA mode: {pca_mode}")
    return aliases[normalized]


def projection_mode_display_name(method: str, pca_mode: str | None = None) -> str:
    reducer_name = normalize_projection_method(method)
    if reducer_name == "pca":
        normalized_mode = _normalize_pca_mode(pca_mode)
        if normalized_mode == "spatial":
            return "Spatial-only PCA"
        if normalized_mode == "temporal":
            return "Temporal-only PCA"
        return "PCA"
    return projection_method_display_name(reducer_name)


def _projection_inputs_for_mode(latent_grid: np.ndarray, pca_mode: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    if latent_grid.ndim != 5:
        raise ValueError(f"Expected latent grid with 5 dims [b, t, h, w, d], got {latent_grid.shape}")
    if latent_grid.shape[0] != 1:
        raise ValueError(f"Expected batch size 1 for visualization, got {latent_grid.shape[0]}")

    normalized_mode = _normalize_pca_mode(pca_mode)
    if normalized_mode == "global":
        return flatten_latent_grid(latent_grid)

    _, time_steps, grid_h, grid_w, embed_dim = latent_grid.shape
    if normalized_mode == "spatial":
        reduced_grid = latent_grid.mean(axis=1, keepdims=True)
        features = reduced_grid.reshape(grid_h * grid_w, embed_dim)
        coordinates = np.stack(np.unravel_index(np.arange(features.shape[0]), (1, grid_h, grid_w)), axis=1)
        return features.astype(np.float32, copy=False), coordinates.astype(np.int32, copy=False)

    reduced_grid = latent_grid.mean(axis=(2, 3), keepdims=True)
    features = reduced_grid.reshape(time_steps, embed_dim)
    coordinates = np.stack(np.unravel_index(np.arange(features.shape[0]), (time_steps, 1, 1)), axis=1)
    return features.astype(np.float32, copy=False), coordinates.astype(np.int32, copy=False)


def _filtered_constructor_kwargs(constructor: Any, candidate_kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(constructor)
    accepts_var_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    if accepts_var_kwargs:
        return {key: value for key, value in candidate_kwargs.items() if value is not None}

    accepted_names = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    return {
        key: value
        for key, value in candidate_kwargs.items()
        if key in accepted_names and value is not None
    }


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


def load_saved_latents(output_prefix: Path) -> tuple[np.ndarray, dict[str, Any]]:
    output_prefix = Path(output_prefix)
    latent_grid = np.load(output_prefix.with_suffix(".npy"))
    metadata = json.loads(output_prefix.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    return latent_grid, metadata


def flatten_latent_grid(latent_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if latent_grid.ndim != 5:
        raise ValueError(f"Expected latent grid with 5 dims [b, t, h, w, d], got {latent_grid.shape}")
    if latent_grid.shape[0] != 1:
        raise ValueError(f"Expected batch size 1 for visualization, got {latent_grid.shape[0]}")

    _, time_steps, grid_h, grid_w, embed_dim = latent_grid.shape
    features = latent_grid.reshape(time_steps * grid_h * grid_w, embed_dim)
    coordinates = np.stack(np.unravel_index(np.arange(features.shape[0]), (time_steps, grid_h, grid_w)), axis=1)
    return features.astype(np.float32, copy=False), coordinates.astype(np.int32, copy=False)


def minmax_scale(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    min_value = array.min(axis=0, keepdims=True)
    max_value = array.max(axis=0, keepdims=True)
    scale = max_value - min_value
    scale[scale == 0] = 1.0
    return (array - min_value) / scale


def compute_pca_projection(features: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features array, got shape {features.shape}")
    if n_components <= 0:
        raise ValueError("n_components must be positive")

    max_components = min(features.shape[0], features.shape[1])
    component_count = min(n_components, max_components)
    centered = features - features.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vh[:component_count].T
    variance = singular_values**2
    explained = variance / variance.sum() if variance.sum() > 0 else np.zeros_like(variance)
    return projection.astype(np.float32, copy=False), explained[:component_count].astype(np.float32, copy=False)


def compute_umap_projection(
    features: np.ndarray,
    n_components: int = 3,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int | None = 42,
) -> np.ndarray:
    if not has_umap_support():
        raise RuntimeError("UMAP requires the optional `umap-learn` dependency. Install it with `pip install umap-learn`.")
    import umap.umap_ as umap

    if features.ndim != 2:
        raise ValueError(f"Expected 2D features array, got shape {features.shape}")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if features.shape[0] < 2:
        raise ValueError("UMAP requires at least 2 samples")

    effective_neighbors = max(2, min(int(n_neighbors), features.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=int(n_components),
        n_neighbors=effective_neighbors,
        min_dist=float(min_dist),
        metric=metric,
        random_state=random_state,
    )
    projection = reducer.fit_transform(features)
    return projection.astype(np.float32, copy=False)


def projection_component_labels(
    method: str,
    component_count: int,
    explained_variance: np.ndarray | None = None,
) -> list[str]:
    reducer_name = normalize_projection_method(method)
    if reducer_name == "pca":
        labels: list[str] = []
        for index in range(component_count):
            if explained_variance is not None and index < len(explained_variance):
                labels.append(f"PC{index + 1} ({explained_variance[index] * 100:.1f}%)")
            else:
                labels.append(f"PC{index + 1}")
        return labels
    prefix = projection_method_display_name(reducer_name).replace(" ", "")
    return [f"{prefix}{index + 1}" for index in range(component_count)]


def compute_projection_bundle(
    latent_grid: np.ndarray,
    *,
    method: str = "pca",
    n_components: int = 3,
    pca_mode: str | None = None,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_metric: str = "euclidean",
    umap_random_state: int | None = 42,
) -> dict[str, Any]:
    reducer_name = normalize_projection_method(method)
    requested_components = max(2, int(n_components))
    if reducer_name == "pca":
        features, coordinates = _projection_inputs_for_mode(latent_grid, pca_mode)
    else:
        features, coordinates = flatten_latent_grid(latent_grid)

    if reducer_name == "pca":
        projection, explained = compute_pca_projection(features, n_components=requested_components)
        explained_list: list[float] | None = [float(value) for value in explained]
        projection_backend = "numpy-svd"
    elif reducer_name == "umap":
        projection = compute_umap_projection(
            features,
            n_components=requested_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=umap_random_state,
        )
        explained = None
        explained_list = None
        projection_backend = "umap-learn"
    elif reducer_name in MLX_VIS_METHOD_SPECS:
        projection = compute_mlx_projection(
            features,
            method=reducer_name,
            n_components=requested_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=umap_random_state,
        )
        explained = None
        explained_list = None
        projection_backend = "mlx-vis"
    else:
        raise ValueError(f"Unknown projection method: {method}")

    method_label = projection_mode_display_name(reducer_name, pca_mode)
    component_labels = projection_component_labels(reducer_name, projection.shape[1], explained)
    return {
        "projection": projection,
        "coordinates": coordinates,
        "method": reducer_name,
        "method_label": method_label,
        "pca_mode": _normalize_pca_mode(pca_mode) if reducer_name == "pca" else None,
        "latent_grid_shape": list(latent_grid.shape),
        "component_labels": component_labels,
        "explained_variance": explained_list,
        "settings": {
            "method": reducer_name,
            "method_label": method_label,
            "pca_mode": _normalize_pca_mode(pca_mode) if reducer_name == "pca" else None,
            "n_components": int(projection.shape[1]),
            "umap_n_neighbors": int(umap_n_neighbors),
            "umap_min_dist": float(umap_min_dist),
            "umap_metric": umap_metric,
            "umap_random_state": umap_random_state,
            "projection_backend": projection_backend,
        },
    }


def save_projection_artifacts(
    output_prefix: Path,
    projection_bundle: dict[str, Any],
    *,
    latent_output_prefix: Path | None = None,
) -> ProjectionArtifacts:
    output_prefix = Path(output_prefix)
    projection_path = output_prefix.with_suffix(".projection.npz")
    metadata_path = output_prefix.with_suffix(".projection.metadata.json")

    np.savez_compressed(
        projection_path,
        projection=projection_bundle["projection"],
        coordinates=projection_bundle["coordinates"],
    )

    metadata_payload = {
        key: value
        for key, value in projection_bundle.items()
        if key not in {"projection", "coordinates"}
    }
    if latent_output_prefix is not None:
        metadata_payload["latent_output_prefix"] = str(latent_output_prefix)
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    return ProjectionArtifacts(
        projection_path=projection_path,
        metadata_path=metadata_path,
        projection_shape=tuple(projection_bundle["projection"].shape),
        method=str(projection_bundle["method"]),
        component_labels=tuple(str(label) for label in projection_bundle["component_labels"]),
    )


def load_saved_projection(output_prefix: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    output_prefix = Path(output_prefix)
    stored = np.load(output_prefix.with_suffix(".projection.npz"))
    projection = np.asarray(stored["projection"], dtype=np.float32)
    coordinates = np.asarray(stored["coordinates"], dtype=np.int32)
    metadata = json.loads(output_prefix.with_suffix(".projection.metadata.json").read_text(encoding="utf-8"))
    return projection, coordinates, metadata


def summarize_latents(latent_grid: np.ndarray) -> dict[str, Any]:
    features, _ = flatten_latent_grid(latent_grid)
    norms = np.linalg.norm(features, axis=1)
    return {
        "latent_shape": list(latent_grid.shape),
        "latent_dtype": str(latent_grid.dtype),
        "patch_norm_mean": float(norms.mean()),
        "patch_norm_std": float(norms.std()),
        "patch_norm_min": float(norms.min()),
        "patch_norm_max": float(norms.max()),
    }


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
