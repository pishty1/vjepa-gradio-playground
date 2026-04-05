from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import gradio as gr

from .gradio_components.latent_source.extractor import MODEL_SPECS, log_step
from .gradio_components.segmentation.status import (
    _format_segmentation_prompt_status,
    _format_segmentation_ready_status,
    _format_segmentation_result_status,
)
from .gradio_components.tracking.helpers import (
    _format_tracking_ready_status,
    _format_tracking_result_status,
    _tracking_frame_choices,
)
from .gradio_components.projection import projection_method_display_name

MODEL_LABELS = {
    "vit_base_384": "ViT-B/16 · 80M · 384",
    "vit_large_384": "ViT-L/16 · 300M · 384",
    "vit_giant_384": "ViT-g/16 · 1B · 384",
    "vit_gigantic_384": "ViT-G/16 · 2B · 384",
}

MODEL_CHOICES = [
    (MODEL_LABELS.get(model_name, model_name), model_name)
    for model_name in sorted(MODEL_SPECS.keys())
]

PROJECTION_METHOD_CHOICES = [
    ("PCA", "PCA"),
    ("UMAP (`umap-learn`)", "UMAP"),
    ("UMAP (`mlx-vis`)", "UMAP-MLX"),
    ("t-SNE (`mlx-vis`)", "TSNE-MLX"),
    ("PaCMAP (`mlx-vis`)", "PaCMAP-MLX"),
    ("LocalMAP (`mlx-vis`)", "LocalMAP-MLX"),
    ("TriMap (`mlx-vis`)", "TriMap-MLX"),
    ("DREAMS (`mlx-vis`)", "DREAMS-MLX"),
    ("CNE (`mlx-vis`)", "CNE-MLX"),
    ("MMAE (`mlx-vis`)", "MMAE-MLX"),
]

NEIGHBOR_TUNED_METHODS = {"umap", "umap_mlx", "pacmap_mlx", "localmap_mlx"}
PCA_MODE_CHOICES = [
    ("Global PCA", "global"),
    ("Spatial-only PCA", "spatial"),
    ("Temporal-only PCA", "temporal"),
]


def _normalize_prefix(prefix_text: str | None, suffixes: Sequence[str]) -> Path | None:
    if not prefix_text or not prefix_text.strip():
        return None
    candidate = Path(prefix_text.strip()).expanduser().resolve()
    candidate_text = str(candidate)
    for suffix in suffixes:
        if candidate_text.endswith(suffix):
            return Path(candidate_text[: -len(suffix)])
    return candidate


def _serialize_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2)


def _load_latent_metadata(output_prefix: Path) -> dict[str, Any]:
    return json.loads(output_prefix.with_suffix(".metadata.json").read_text(encoding="utf-8"))


def _format_projection_status(output_prefix: Path, metadata: dict[str, Any]) -> str:
    labels = metadata.get("component_labels", [])
    method_text = metadata.get("method_label") or projection_method_display_name(metadata["method"])
    backend = metadata.get("settings", {}).get("projection_backend", "unknown")
    return "\n".join(
        [
            "## Projection ready",
            f"- prefix: `{output_prefix}`",
            f"- method: `{method_text}`",
            f"- backend: `{backend}`",
            f"- components: `{metadata['settings']['n_components']}`",
            f"- labels: `{', '.join(labels)}`",
            "- next: build a plot and/or generate RGB videos from any 3 projected components.",
        ]
    )


def _format_plot_status(method: str, component_indices: Sequence[int], method_label: str | None = None) -> str:
    selection = ", ".join(f"C{index + 1}" for index in component_indices)
    return "\n".join(
        [
            "## Plot updated",
            f"- method: `{method_label or projection_method_display_name(method)}`",
            f"- plotted components: `{selection}`",
        ]
    )


def _format_render_status(
    method: str,
    rgb_components: Sequence[int],
    latent_video_path: Path,
    side_by_side_video_path: Path,
    method_label: str | None = None,
) -> str:
    component_text = ", ".join(f"C{index + 1}" for index in rgb_components)
    return "\n".join(
        [
            "## RGB videos created",
            f"- method: `{method_label or projection_method_display_name(method)}`",
            f"- RGB components: `{component_text}`",
            f"- latent video: `{latent_video_path}`",
            f"- side-by-side video: `{side_by_side_video_path}`",
        ]
    )


def _format_hint_status(title: str, message: str) -> str:
    return "\n".join([f"## {title}", f"- {message}"])


def _normalize_projection_settings(
    projection_method: str,
    projection_pca_mode: str,
    projection_components: int | float,
    umap_n_neighbors: int | float,
    umap_min_dist: float,
    umap_metric: str,
    umap_random_state: float | None,
) -> dict[str, Any]:
    pca_mode = (
        projection_pca_mode.strip().lower().replace("-", "_").replace(" ", "_")
        if projection_method.strip().lower() == "pca"
        else "global"
    )
    return {
        "method": projection_method.strip().lower(),
        "pca_mode": pca_mode,
        "n_components": max(2, int(projection_components)),
        "umap_n_neighbors": max(2, int(umap_n_neighbors)),
        "umap_min_dist": float(umap_min_dist),
        "umap_metric": umap_metric,
        "umap_random_state": None if umap_random_state in (None, "") else int(umap_random_state),
    }


def _component_selector_updates(component_count: int):
    choices = list(range(1, component_count + 1))
    plot_dimensions_value = 3 if component_count >= 3 else 2
    z_value = 3 if component_count >= 3 else None
    return (
        gr.update(choices=[2, 3] if component_count >= 3 else [2], value=plot_dimensions_value),
        gr.update(choices=choices, value=1),
        gr.update(choices=choices, value=2 if component_count >= 2 else 1),
        gr.update(choices=choices, value=z_value),
        gr.update(choices=choices, value=1),
        gr.update(choices=choices, value=2 if component_count >= 2 else 1),
        gr.update(choices=choices, value=z_value),
    )


def _projection_state(output_prefix: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "output_prefix": str(output_prefix),
        "method": metadata["method"],
        "component_count": int(metadata["settings"]["n_components"]),
        "latent_output_prefix": metadata.get("latent_output_prefix"),
        "projection": metadata.get("projection"),
        "coordinates": metadata.get("coordinates"),
        "metadata": metadata,
    }


def _log_gradio_step(step_name: str, message: str) -> None:
    log_step(f"Gradio · {step_name}: {message}")
