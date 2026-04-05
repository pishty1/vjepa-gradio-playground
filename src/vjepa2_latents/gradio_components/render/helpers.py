from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..projection import projection_method_display_name


def _format_render_status(
    method: str,
    rgb_components: Sequence[int],
    latent_video_path: Path,
    side_by_side_video_path: Path,
    method_label: str | None = None,
) -> str:
    component_text = ", ".join(f"C{index + 1}" for index in rgb_components)
    return "\n".join(
        [
            "## RGB videos created",
            f"- method: `{method_label or projection_method_display_name(method)}`",
            f"- RGB components: `{component_text}`",
            f"- latent video: `{latent_video_path}`",
            f"- side-by-side video: `{side_by_side_video_path}`",
        ]
    )
