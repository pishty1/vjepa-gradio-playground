from __future__ import annotations

from importlib import import_module

__all__ = [
    "TumblingWindowRanges",
    "build_tumbling_window_tab",
    "build_tumbling_window_heatmap_figure",
    "compare_overlapping_latent_windows",
    "compare_tumbling_windows_step",
    "derive_tumbling_window_ranges",
    "sync_overlap_time_slice_control",
]

_EXPORT_MODULES = {
    "TumblingWindowRanges": ".core",
    "build_tumbling_window_tab": ".ui",
    "build_tumbling_window_heatmap_figure": ".core",
    "compare_overlapping_latent_windows": ".core",
    "compare_tumbling_windows_step": ".callbacks",
    "derive_tumbling_window_ranges": ".core",
    "sync_overlap_time_slice_control": ".helpers",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value