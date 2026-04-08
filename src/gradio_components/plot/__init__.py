from __future__ import annotations

from importlib import import_module

__all__ = [
	"build_pca_figure",
	"build_plot_section",
	"build_plot_step",
	"build_projection_figure",
	"build_projection_figure_from_data",
	"toggle_plot_dimensions",
]

_EXPORT_MODULES = {
	"build_pca_figure": ".core",
	"build_plot_section": ".ui",
	"build_plot_step": ".callbacks",
	"build_projection_figure": ".core",
	"build_projection_figure_from_data": ".core",
	"toggle_plot_dimensions": ".helpers",
}


def __getattr__(name: str):
	module_name = _EXPORT_MODULES.get(name)
	if module_name is None:
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
	module = import_module(module_name, __name__)
	value = getattr(module, name)
	globals()[name] = value
	return value
