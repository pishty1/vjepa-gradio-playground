from __future__ import annotations

from importlib import import_module

__all__ = [
	"VisualizationArtifacts",
	"_ensure_even_frame_size",
	"_resolve_ffmpeg_executable",
	"build_render_section",
	"create_rgb_videos_step",
	"create_visualizations",
	"create_visualizations_from_projection",
	"infer_latent_fps",
	"latent_rgb_frames",
	"load_aligned_source_frames",
	"projection_rgb_frames",
	"side_by_side_frames",
	"write_video",
]

_EXPORT_MODULES = {
	"VisualizationArtifacts": ".video",
	"_ensure_even_frame_size": ".video",
	"_resolve_ffmpeg_executable": ".video",
	"build_render_section": ".ui",
	"create_rgb_videos_step": ".callbacks",
	"create_visualizations": ".video",
	"create_visualizations_from_projection": ".video",
	"infer_latent_fps": ".video",
	"latent_rgb_frames": ".video",
	"load_aligned_source_frames": ".video",
	"projection_rgb_frames": ".video",
	"side_by_side_frames": ".video",
	"write_video": ".video",
}


def __getattr__(name: str):
	module_name = _EXPORT_MODULES.get(name)
	if module_name is None:
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
	module = import_module(module_name, __name__)
	value = getattr(module, name)
	globals()[name] = value
	return value
