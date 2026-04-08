from __future__ import annotations

from importlib import import_module

__all__ = [
    "APP_OUTPUT_DIR",
    "CHECKPOINT_DIR",
    "DEFAULT_CROP_HEIGHT",
    "DEFAULT_CROP_WIDTH",
    "DEFAULT_DEVICE",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_VIDEO",
    "_clean_latent_metadata_for_ui",
    "_create_session_dir",
    "_format_extraction_status",
    "_format_latent_status",
    "_format_preflight_status",
    "_latent_state",
    "_normalize_prefix",
    "_resolve_video_path",
    "_summarize_timings_for_ui",
    "build_latent_source_section",
    "estimate_limits_step",
    "extract_latents_step",
    "load_latents_step",
    "refresh_saved_latent_choices",
    "saved_latent_choices",
    "toggle_latent_source_mode",
]

_EXPORT_MODULES = {
    "APP_OUTPUT_DIR": ".config",
    "CHECKPOINT_DIR": ".config",
    "DEFAULT_CROP_HEIGHT": ".config",
    "DEFAULT_CROP_WIDTH": ".config",
    "DEFAULT_DEVICE": ".config",
    "DEFAULT_MODEL_NAME": ".config",
    "DEFAULT_VIDEO": ".config",
    "_clean_latent_metadata_for_ui": ".helpers",
    "_create_session_dir": ".helpers",
    "_format_extraction_status": ".helpers",
    "_format_latent_status": ".helpers",
    "_format_preflight_status": ".helpers",
    "_latent_state": ".helpers",
    "_normalize_prefix": ".helpers",
    "_resolve_video_path": ".helpers",
    "_summarize_timings_for_ui": ".helpers",
    "build_latent_source_section": ".ui",
    "estimate_limits_step": ".callbacks",
    "extract_latents_step": ".callbacks",
    "load_latents_step": ".callbacks",
    "refresh_saved_latent_choices": ".callbacks",
    "saved_latent_choices": ".catalog",
    "toggle_latent_source_mode": ".callbacks",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
