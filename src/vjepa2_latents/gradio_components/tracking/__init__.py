from __future__ import annotations

from importlib import import_module

__all__ = [
	"PatchSimilarityArtifacts",
	"annotate_selected_patch",
	"build_tracking_tab",
	"cosine_similarity_volume",
	"create_patch_similarity_video",
	"map_click_to_latent_token",
	"prepare_tracking_step",
	"select_patch_similarity_step",
	"similarity_heatmap_frames",
]

_EXPORT_MODULES = {
	"PatchSimilarityArtifacts": ".core",
	"annotate_selected_patch": ".core",
	"build_tracking_tab": ".ui",
	"cosine_similarity_volume": ".core",
	"create_patch_similarity_video": ".core",
	"map_click_to_latent_token": ".core",
	"prepare_tracking_step": ".callbacks",
	"select_patch_similarity_step": ".callbacks",
	"similarity_heatmap_frames": ".core",
}


def __getattr__(name: str):
	module_name = _EXPORT_MODULES.get(name)
	if module_name is None:
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
	module = import_module(module_name, __name__)
	value = getattr(module, name)
	globals()[name] = value
	return value
