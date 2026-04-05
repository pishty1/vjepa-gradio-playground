from __future__ import annotations

from pathlib import Path
from typing import Sequence

import gradio as gr

from ...gradio_utils import NEIGHBOR_TUNED_METHODS
from .core import projection_method_display_name


def toggle_projection_controls(projection_method: str):
    normalized = projection_method.strip().lower().replace("-", "_")
    return gr.update(visible=normalized in NEIGHBOR_TUNED_METHODS), gr.update(visible=normalized == "pca")


def _format_projection_status(output_prefix: Path, metadata: dict[str, object]) -> str:
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
