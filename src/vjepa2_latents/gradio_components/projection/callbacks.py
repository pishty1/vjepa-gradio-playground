from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from ...gradio_utils import (
    _component_selector_updates,
    _format_hint_status,
    _log_gradio_step,
    _normalize_prefix,
    _normalize_projection_settings,
    _projection_state,
    _serialize_json,
)
from .core import (
    compute_projection_bundle,
    load_saved_latents,
    load_saved_projection,
    save_projection_artifacts,
)
from .helpers import _format_projection_status


def compute_projection_step(
    latent_state: dict[str, Any] | None,
    projection_method: str,
    projection_pca_mode: str,
    projection_components: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    umap_random_state: float | None,
):
    _log_gradio_step("compute_projection", f"method={projection_method} components={projection_components}")
    if not latent_state or not latent_state.get("output_prefix"):
        selector_updates = _component_selector_updates(3)
        return (
            _format_hint_status("Projection not computed", "Load latents before computing a projection."),
            "",
            "{}",
            None,
            *selector_updates,
            gr.update(),
            None,
            "",
            _format_hint_status("Plot not ready", "Compute or load a projection before building a plot."),
            _format_hint_status("RGB videos not ready", "Compute or load a projection before generating RGB videos."),
            "{}",
        )

    latent_output_prefix = Path(latent_state["output_prefix"])
    latent_grid = latent_state.get("latent_grid")
    if latent_grid is None:
        latent_grid, _ = load_saved_latents(latent_output_prefix)
    settings = _normalize_projection_settings(
        projection_method,
        projection_pca_mode,
        projection_components,
        umap_n_neighbors,
        umap_min_dist,
        umap_metric,
        umap_random_state,
    )
    try:
        projection_bundle = compute_projection_bundle(latent_grid, **settings)
    except (RuntimeError, ValueError) as error:
        raise gr.Error(str(error)) from error
    projection_name = settings["method"]
    if projection_name == "pca" and settings.get("pca_mode") and settings["pca_mode"] != "global":
        projection_name = f"{settings['pca_mode']}_pca"
    projection_output_prefix = latent_output_prefix.parent / f"projection_{projection_name}_{projection_bundle['projection'].shape[1]}"
    artifacts = save_projection_artifacts(
        projection_output_prefix,
        projection_bundle,
        latent_output_prefix=latent_output_prefix,
    )
    projection_metadata = {
        key: value
        for key, value in projection_bundle.items()
        if key not in {"projection", "coordinates"}
    }
    projection_metadata.update(
        {
            "latent_output_prefix": str(latent_output_prefix),
            "projection_output_prefix": str(projection_output_prefix),
            "projection_npz_path": str(artifacts.projection_path),
            "projection_metadata_path": str(artifacts.metadata_path),
        }
    )
    payload = {**projection_metadata}
    status = _format_projection_status(projection_output_prefix, projection_metadata)
    selector_updates = _component_selector_updates(artifacts.projection_shape[1])
    _log_gradio_step("compute_projection", f"saved_prefix={projection_output_prefix} projection_shape={projection_bundle['projection'].shape}")
    return (
        status,
        str(projection_output_prefix),
        _serialize_json(payload),
        {
            **_projection_state(projection_output_prefix, projection_metadata),
            "projection": projection_bundle["projection"],
            "coordinates": projection_bundle["coordinates"],
        },
        *selector_updates,
        gr.update(minimum=1, maximum=int(projection_bundle["projection"].shape[0]), value=int(projection_bundle["projection"].shape[0])),
        None,
        "",
        _format_hint_status("Plot not ready", "Build the plot to preview the selected projection components."),
        _format_hint_status("RGB videos not ready", "Create RGB videos after choosing projection components."),
        "{}",
    )


def load_projection_step(
    projection_prefix_text: str,
    projection_state: dict[str, Any] | None,
):
    _log_gradio_step("load_projection", f"prefix={projection_prefix_text or projection_state.get('output_prefix') if projection_state else 'none'}")
    output_prefix = _normalize_prefix(projection_prefix_text, (".projection.npz", ".projection.metadata.json"))
    if output_prefix is None and projection_state and projection_state.get("output_prefix"):
        output_prefix = Path(projection_state["output_prefix"])
    if output_prefix is None:
        selector_updates = _component_selector_updates(3)
        return (
            _format_hint_status("Projection not loaded", "Provide a projection prefix before loading a saved projection."),
            projection_prefix_text or "",
            "{}",
            projection_state,
            *selector_updates,
            gr.update(),
            None,
            "",
            _format_hint_status("Plot not ready", "Build the plot to preview the selected projection components."),
            _format_hint_status("RGB videos not ready", "Create RGB videos after choosing projection components."),
            "{}",
        )

    projection, coordinates, metadata = load_saved_projection(output_prefix)
    payload = {
        **metadata,
        "latent_output_prefix": metadata.get("latent_output_prefix") or (
            str(Path(output_prefix).with_name("latents")) if output_prefix is not None else None
        ),
        "projection_output_prefix": str(output_prefix),
        "projection_shape": list(projection.shape),
    }
    status = _format_projection_status(output_prefix, metadata)
    selector_updates = _component_selector_updates(projection.shape[1])
    _log_gradio_step("load_projection", f"loaded prefix={output_prefix} projection_shape={projection.shape}")
    return (
        status,
        str(output_prefix),
        _serialize_json(payload),
        {
            **_projection_state(output_prefix, metadata),
            "projection": projection,
            "coordinates": coordinates,
            "metadata": metadata,
        },
        *selector_updates,
        gr.update(minimum=1, maximum=int(projection.shape[0]), value=int(projection.shape[0])),
        None,
        "",
        _format_hint_status("Plot not ready", "Build the plot to preview the selected projection components."),
        _format_hint_status("RGB videos not ready", "Create RGB videos after choosing projection components."),
        "{}",
    )
