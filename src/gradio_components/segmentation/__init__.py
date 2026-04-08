from __future__ import annotations

from importlib import import_module

__all__ = [
    "SegmentationArtifacts",
    "annotate_prompt_points",
    "build_segmentation_tab",
    "create_segmentation_video",
    "knn_binary_segmentation_volume",
    "prepare_segmentation_step",
    "run_segmentation_step",
    "segmentation_mask_frames",
    "select_segmentation_prompt_step",
    "_format_segmentation_prompt_status",
    "_format_segmentation_ready_status",
    "_format_segmentation_result_status",
]

_EXPORT_MODULES = {
    "SegmentationArtifacts": ".core",
    "annotate_prompt_points": ".core",
    "build_segmentation_tab": ".ui",
    "create_segmentation_video": ".core",
    "knn_binary_segmentation_volume": ".core",
    "prepare_segmentation_step": ".callbacks",
    "run_segmentation_step": ".callbacks",
    "segmentation_mask_frames": ".core",
    "select_segmentation_prompt_step": ".callbacks",
    "_format_segmentation_prompt_status": ".status",
    "_format_segmentation_ready_status": ".status",
    "_format_segmentation_result_status": ".status",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
