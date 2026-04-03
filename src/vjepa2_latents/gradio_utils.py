from __future__ import annotations

import json
import platform
import tempfile
from pathlib import Path
from typing import Any, Sequence

import gradio as gr

from .extractor import MODEL_SPECS, log_step
from .visualization import projection_method_display_name
from .visualization import summarize_latents

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = ROOT / "testvideo.mp4"
VENDOR_REPO = ROOT / "vendor" / "vjepa2"
CHECKPOINT_DIR = ROOT / "checkpoints"
APP_OUTPUT_DIR = ROOT / ".gradio_outputs"
DEFAULT_DEVICE = "mps" if platform.system() == "Darwin" else "auto"
DEFAULT_MODEL_NAME = "vit_base_384"
DEFAULT_CROP_HEIGHT = 384
DEFAULT_CROP_WIDTH = 384

MODEL_LABELS = {
    "vit_base_384": "ViT-B/16 · 80M · 384",
    "vit_large_384": "ViT-L/16 · 300M · 384",
    "vit_giant_384": "ViT-g/16 · 1B · 384",
    "vit_gigantic_384": "ViT-G/16 · 2B · 384",
}

MODEL_CHOICES = [
    (MODEL_LABELS.get(model_name, model_name), model_name)
    for model_name in sorted(MODEL_SPECS.keys())
]

PROJECTION_METHOD_CHOICES = [
    ("PCA", "PCA"),
    ("UMAP (`umap-learn`)", "UMAP"),
    ("UMAP (`mlx-vis`)", "UMAP-MLX"),
    ("t-SNE (`mlx-vis`)", "TSNE-MLX"),
    ("PaCMAP (`mlx-vis`)", "PaCMAP-MLX"),
    ("LocalMAP (`mlx-vis`)", "LocalMAP-MLX"),
    ("TriMap (`mlx-vis`)", "TriMap-MLX"),
    ("DREAMS (`mlx-vis`)", "DREAMS-MLX"),
    ("CNE (`mlx-vis`)", "CNE-MLX"),
    ("MMAE (`mlx-vis`)", "MMAE-MLX"),
]

NEIGHBOR_TUNED_METHODS = {"umap", "umap_mlx", "pacmap_mlx", "localmap_mlx"}
PCA_MODE_CHOICES = [
    ("Global PCA", "global"),
    ("Spatial-only PCA", "spatial"),
    ("Temporal-only PCA", "temporal"),
]


def _resolve_video_path(video_file: str | None) -> Path:
    if video_file:
        return Path(video_file).resolve()
    if DEFAULT_VIDEO.exists():
        return DEFAULT_VIDEO.resolve()
    raise gr.Error("Upload a video or provide an example video first.")


def _create_session_dir(prefix: str) -> Path:
    APP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=APP_OUTPUT_DIR))


def _normalize_prefix(prefix_text: str | None, suffixes: Sequence[str]) -> Path | None:
    if not prefix_text or not prefix_text.strip():
        return None
    candidate = Path(prefix_text.strip()).expanduser().resolve()
    candidate_text = str(candidate)
    for suffix in suffixes:
        if candidate_text.endswith(suffix):
            return Path(candidate_text[: -len(suffix)])
    return candidate


def _serialize_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2)


def _round_ui_number(value: Any, digits: int = 3) -> Any:
    if isinstance(value, float):
        if value == 0.0:
            return 0.0
        if abs(value) < 0.01:
            return round(value, 6)
        return round(value, digits)
    return value


def _summarize_timings_for_ui(timings: dict[str, Any] | None) -> dict[str, Any]:
    if not timings:
        return {}

    summary: dict[str, Any] = {}

    encoder_forward = timings.get("encoder_forward_pass")
    if isinstance(encoder_forward, dict):
        summary["encoder_forward_pass"] = {
            key: _round_ui_number(encoder_forward[key])
            for key in (
                "device_executes_asynchronously",
                "measured_synchronously",
                "forward_run_seconds",
                "total_wall_seconds",
            )
            if key in encoder_forward
        }

    reshape_timings = timings.get("reshape_patch_tokens")
    if isinstance(reshape_timings, dict) and "total_seconds" in reshape_timings:
        summary["reshape_patch_tokens"] = {"total_seconds": _round_ui_number(reshape_timings["total_seconds"])}

    output_timings = timings.get("output_serialization")
    if isinstance(output_timings, dict) and "total_seconds" in output_timings:
        summary["output_serialization"] = {"total_seconds": _round_ui_number(output_timings["total_seconds"])}

    encoder_setup = timings.get("encoder_setup")
    if isinstance(encoder_setup, dict) and "total_seconds" in encoder_setup:
        summary["encoder_setup"] = {"total_seconds": _round_ui_number(encoder_setup["total_seconds"])}

    major_phases = timings.get("major_phases")
    if isinstance(major_phases, dict):
        phase_summary = {
            label: _round_ui_number(seconds)
            for label, seconds in major_phases.items()
            if isinstance(seconds, (int, float)) and float(seconds) >= 0.01
        }
        if phase_summary:
            summary["major_phases"] = phase_summary

    if "total_extraction_seconds" in timings:
        summary["total_extraction_seconds"] = _round_ui_number(timings["total_extraction_seconds"])

    return summary


def _clean_latent_metadata_for_ui(metadata: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(metadata)
    cleaned_timings = _summarize_timings_for_ui(cleaned.get("timings"))
    if cleaned_timings:
        cleaned["timings"] = cleaned_timings
    else:
        cleaned.pop("timings", None)
    return cleaned


def _load_latent_metadata(output_prefix: Path) -> dict[str, Any]:
    return json.loads(output_prefix.with_suffix(".metadata.json").read_text(encoding="utf-8"))


def _format_extraction_status(
    result: dict[str, Any],
    output_prefix: Path,
    *,
    model_name: str | None = None,
    video_path: Path | None = None,
) -> str:
    outputs = result.get("outputs", {})
    model_value = result.get("model") or model_name or "unknown"
    video_value = result.get("video_path") or (str(video_path) if video_path is not None else "unknown")
    return "\n".join(
        [
            "## Latents extracted",
            f"- output prefix: `{output_prefix}`",
            f"- device: `{result.get('device', 'unknown')}`",
            f"- model: `{model_value}`",
            f"- video: `{video_value}`",
            f"- latent tensor: `{outputs.get('npy', output_prefix.with_suffix('.npy'))}`",
            "- next: use **Load latents** to inspect the latent grid before projecting it.",
        ]
    )


def _format_preflight_status(preflight: dict[str, Any], video_meta: dict[str, Any], frame_indices: Sequence[int]) -> str:
    risk_level = str(preflight["risk_level"]).upper()
    range_text = f"{frame_indices[0]}..{frame_indices[-1]}" if frame_indices else "n/a"
    lines = [
        "## System fit estimate",
        f"- risk: `{risk_level}`",
        f"- device: `{preflight['device']}`",
        f"- selected frames: `{len(frame_indices)}` (`{range_text}`)",
        f"- source video: `{video_meta['frame_count']} frames @ {video_meta['fps']:.3f} fps, {video_meta['width']}x{video_meta['height']}`",
        f"- patch tokens: `{preflight['token_count']}` = `{preflight['time_patches']} x {preflight['height_patches']} x {preflight['width_patches']}`",
        f"- latent tensor size: `{preflight['latent_tensor_mib']:.1f} MiB`",
        f"- attention pressure vs 16f/384 baseline: `{preflight['quadratic_token_factor_vs_16f_384']:.2f}x`",
    ]

    mps_limit = preflight.get("mps_recommended_max_memory_gib")
    if mps_limit is not None:
        lines.append(f"- MPS recommended memory cap: `{mps_limit:.2f} GiB`")

    if preflight["device"].startswith("mps"):
        if preflight["risk_level"] == "high":
            lines.append("- warning: this configuration is likely to hit Apple MPS memory limits; reduce frames or crop size, or switch to CPU.")
        elif preflight["risk_level"] == "medium":
            lines.append("- warning: this configuration may be unstable on Apple MPS; watch for OOM during forward pass or CPU copy.")
        else:
            lines.append("- note: the saved `.npy` file is not the main memory cost; peak memory happens during attention and MPS synchronization.")

    return "\n".join(lines)


def _format_latent_status(output_prefix: Path, metadata: dict[str, Any], summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "## Latents loaded",
            f"- prefix: `{output_prefix}`",
            f"- latent grid: `{metadata['latent_grid_shape']}`",
            f"- input tensor: `{metadata['input_tensor_shape']}`",
            f"- raw tokens: `{metadata['raw_token_shape']}`",
            f"- patch norm mean/std: `{summary['patch_norm_mean']:.4f}` / `{summary['patch_norm_std']:.4f}`",
            "- next: compute PCA or UMAP with the projection controls.",
        ]
    )


def _format_projection_status(output_prefix: Path, metadata: dict[str, Any]) -> str:
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


def _format_plot_status(method: str, component_indices: Sequence[int], method_label: str | None = None) -> str:
    selection = ", ".join(f"C{index + 1}" for index in component_indices)
    return "\n".join(
        [
            "## Plot updated",
            f"- method: `{method_label or projection_method_display_name(method)}`",
            f"- plotted components: `{selection}`",
        ]
    )


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


def _format_tracking_ready_status(latent_grid_shape: Sequence[int], display_fps: float) -> str:
    _, time_steps, grid_h, grid_w, _ = [int(value) for value in latent_grid_shape]
    return "\n".join(
        [
            "## Patch similarity ready",
            "- choose one of the available source frames, then click any object or patch in the image to select a latent token.",
            f"- latent token grid: `{time_steps} x {grid_h} x {grid_w}`",
            f"- playback fps: `{display_fps:.3f}`",
        ]
    )


def _format_tracking_result_status(
    token_index: Sequence[int],
    click_xy: Sequence[int],
    video_path: Path,
    *,
    frame_index: int | None = None,
    video_frame_index: int | None = None,
) -> str:
    time_index, row_index, column_index = [int(value) for value in token_index]
    click_x, click_y = [int(value) for value in click_xy[:2]]
    lines = [
        "## Patch similarity video ready",
        f"- selected click: `x={click_x}, y={click_y}`",
        f"- selected latent token: `t={time_index}, h={row_index}, w={column_index}`",
    ]
    if frame_index is not None:
        frame_line = f"- tracked frame: `{frame_index + 1}`"
        if video_frame_index is not None:
            frame_line = f"- tracked frame: `{frame_index + 1}` (video frame `{video_frame_index}`)"
        lines.append(frame_line)
    lines.append(f"- output video: `{video_path}`")
    return "\n".join(lines)


def _tracking_frame_choices(source_frame_indices: Sequence[int]) -> list[tuple[str, int]]:
    return [(f"Frame {position + 1} (video frame {frame_index})", position) for position, frame_index in enumerate(source_frame_indices)]


def _format_hint_status(title: str, message: str) -> str:
    return "\n".join([f"## {title}", f"- {message}"])


def _normalize_projection_settings(
    projection_method: str,
    projection_pca_mode: str,
    projection_components: int | float,
    umap_n_neighbors: int | float,
    umap_min_dist: float,
    umap_metric: str,
    umap_random_state: float | None,
) -> dict[str, Any]:
    pca_mode = (
        projection_pca_mode.strip().lower().replace("-", "_").replace(" ", "_")
        if projection_method.strip().lower() == "pca"
        else "global"
    )
    return {
        "method": projection_method.strip().lower(),
        "pca_mode": pca_mode,
        "n_components": max(2, int(projection_components)),
        "umap_n_neighbors": max(2, int(umap_n_neighbors)),
        "umap_min_dist": float(umap_min_dist),
        "umap_metric": umap_metric,
        "umap_random_state": None if umap_random_state in (None, "") else int(umap_random_state),
    }


def _component_selector_updates(component_count: int):
    choices = list(range(1, component_count + 1))
    plot_dimensions_value = 3 if component_count >= 3 else 2
    z_value = 3 if component_count >= 3 else None
    return (
        gr.update(choices=[2, 3] if component_count >= 3 else [2], value=plot_dimensions_value),
        gr.update(choices=choices, value=1),
        gr.update(choices=choices, value=2 if component_count >= 2 else 1),
        gr.update(choices=choices, value=z_value),
        gr.update(choices=choices, value=1),
        gr.update(choices=choices, value=2 if component_count >= 2 else 1),
        gr.update(choices=choices, value=z_value),
    )


def _projection_state(output_prefix: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "output_prefix": str(output_prefix),
        "method": metadata["method"],
        "component_count": int(metadata["settings"]["n_components"]),
        "latent_output_prefix": metadata.get("latent_output_prefix"),
        "projection": metadata.get("projection"),
        "coordinates": metadata.get("coordinates"),
        "metadata": metadata,
    }


def _latent_state(output_prefix: Path, latent_grid: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "output_prefix": str(output_prefix),
        "session_dir": str(output_prefix.parent),
        "latent_grid": latent_grid,
        "metadata": metadata,
        "summary": summarize_latents(latent_grid),
    }


def _log_gradio_step(step_name: str, message: str) -> None:
    log_step(f"Gradio · {step_name}: {message}")
