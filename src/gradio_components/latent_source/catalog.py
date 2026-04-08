from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_metadata(metadata_path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _frame_count(metadata: dict[str, Any]) -> int | None:
    frame_indices = metadata.get("frame_indices")
    if isinstance(frame_indices, list) and frame_indices:
        return len(frame_indices)

    latent_grid_shape = metadata.get("latent_grid_shape")
    tubelet_size = metadata.get("tubelet_size")
    if (
        isinstance(latent_grid_shape, list)
        and len(latent_grid_shape) >= 2
        and isinstance(tubelet_size, (int, float))
    ):
        return int(latent_grid_shape[1]) * int(tubelet_size)
    return None


def _crop_text(metadata: dict[str, Any]) -> str | None:
    crop_size = metadata.get("crop_size")
    if isinstance(crop_size, list) and len(crop_size) >= 2:
        return f"{int(crop_size[0])}x{int(crop_size[1])}"
    return None


def _latent_grid_text(metadata: dict[str, Any]) -> str | None:
    latent_grid_shape = metadata.get("latent_grid_shape")
    if isinstance(latent_grid_shape, list) and len(latent_grid_shape) >= 4:
        return f"latent={int(latent_grid_shape[1])}x{int(latent_grid_shape[2])}x{int(latent_grid_shape[3])}"
    return None


def _video_name(metadata: dict[str, Any], output_prefix: Path) -> str:
    video_path = metadata.get("video_path")
    if isinstance(video_path, str) and video_path:
        return Path(video_path).name
    return output_prefix.parent.name


def format_saved_latent_label(output_prefix: Path, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    modified_text = datetime.fromtimestamp(output_prefix.with_suffix(".metadata.json").stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    parts = [modified_text, _video_name(metadata, output_prefix)]

    model_name = metadata.get("model")
    if isinstance(model_name, str) and model_name:
        parts.append(model_name)

    frame_count = _frame_count(metadata)
    if frame_count is not None:
        parts.append(f"{frame_count}f")

    crop_text = _crop_text(metadata)
    if crop_text is not None:
        parts.append(crop_text)

    latent_grid_text = _latent_grid_text(metadata)
    if latent_grid_text is not None:
        parts.append(latent_grid_text)

    parts.append(output_prefix.parent.name)
    return " · ".join(parts)


def iter_saved_latent_prefixes(app_output_dir: Path) -> list[Path]:
    if not app_output_dir.exists():
        return []

    prefixes: list[tuple[float, Path]] = []
    for metadata_path in app_output_dir.glob("**/*.metadata.json"):
        output_prefix = metadata_path.with_suffix("")
        output_prefix = output_prefix.with_suffix("")
        if not output_prefix.with_suffix(".npy").exists():
            continue
        try:
            modified_time = metadata_path.stat().st_mtime
        except OSError:
            continue
        prefixes.append((modified_time, output_prefix))

    prefixes.sort(key=lambda item: item[0], reverse=True)
    return [output_prefix for _, output_prefix in prefixes]


def saved_latent_choices(app_output_dir: Path) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for output_prefix in iter_saved_latent_prefixes(app_output_dir):
        metadata = _load_metadata(output_prefix.with_suffix(".metadata.json"))
        choices.append((format_saved_latent_label(output_prefix, metadata), str(output_prefix)))
    return choices
