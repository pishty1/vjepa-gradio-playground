from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProjectionArtifacts:
    projection_path: Path
    metadata_path: Path
    projection_shape: tuple[int, int]
    method: str
    component_labels: tuple[str, ...]


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
        and parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
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
