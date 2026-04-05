from __future__ import annotations

from typing import Sequence

import gradio as gr

from ..projection import projection_method_display_name


def toggle_plot_dimensions(plot_dimensions: int):
    return gr.update(visible=int(plot_dimensions) == 3)


def _format_plot_status(method: str, component_indices: Sequence[int], method_label: str | None = None) -> str:
    selection = ", ".join(f"C{index + 1}" for index in component_indices)
    return "\n".join(
        [
            "## Plot updated",
            f"- method: `{method_label or projection_method_display_name(method)}`",
            f"- plotted components: `{selection}`",
        ]
    )
