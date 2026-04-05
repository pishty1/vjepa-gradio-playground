from __future__ import annotations

from importlib import import_module

__all__ = [
    "MLX_VIS_METHOD_SPECS",
    "ProjectionArtifacts",
    "_filtered_constructor_kwargs",
    "build_projection_section",
    "compute_mlx_projection",
    "compute_pca_projection",
    "compute_projection_bundle",
    "compute_projection_step",
    "compute_umap_projection",
    "flatten_latent_grid",
    "has_mlx_vis_support",
    "has_umap_support",
    "load_saved_latents",
    "load_saved_projection",
    "minmax_scale",
    "normalize_projection_method",
    "projection_component_labels",
    "projection_method_display_name",
    "projection_mode_display_name",
    "save_projection_artifacts",
    "summarize_latents",
    "toggle_projection_controls",
    "load_projection_step",
]

_EXPORT_MODULES = {
    "MLX_VIS_METHOD_SPECS": ".core",
    "ProjectionArtifacts": ".core",
    "_filtered_constructor_kwargs": ".core",
    "build_projection_section": ".ui",
    "compute_mlx_projection": ".core",
    "compute_pca_projection": ".core",
    "compute_projection_bundle": ".core",
    "compute_projection_step": ".callbacks",
    "compute_umap_projection": ".core",
    "flatten_latent_grid": ".core",
    "has_mlx_vis_support": ".core",
    "has_umap_support": ".core",
    "load_saved_latents": ".core",
    "load_saved_projection": ".core",
    "minmax_scale": ".core",
    "normalize_projection_method": ".core",
    "projection_component_labels": ".core",
    "projection_method_display_name": ".core",
    "projection_mode_display_name": ".core",
    "save_projection_artifacts": ".core",
    "summarize_latents": ".core",
    "toggle_projection_controls": ".helpers",
    "load_projection_step": ".callbacks",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
