from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Sequence

import gradio as gr

from ..projection import summarize_latents
from .config import APP_OUTPUT_DIR, DEFAULT_VIDEO


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
    latent_grid_shape = metadata.get("latent_grid_shape", "unknown")
    input_tensor_shape = metadata.get("input_tensor_shape", "unknown")
    raw_token_shape = metadata.get("raw_token_shape")
    if raw_token_shape is None:
        raw_token_shape = metadata.get("token_shape", input_tensor_shape)
    return "\n".join(
        [
            "## Latents loaded",
            f"- prefix: `{output_prefix}`",
            f"- latent grid: `{latent_grid_shape}`",
            f"- input tensor: `{input_tensor_shape}`",
            f"- raw tokens: `{raw_token_shape}`",
            f"- patch norm mean/std: `{summary['patch_norm_mean']:.4f}` / `{summary['patch_norm_std']:.4f}`",
            "- next: compute PCA or UMAP with the projection controls.",
        ]
    )


def _latent_state(output_prefix: Path, latent_grid: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "output_prefix": str(output_prefix),
        "session_dir": str(output_prefix.parent),
        "latent_grid": latent_grid,
        "metadata": metadata,
        "summary": summarize_latents(latent_grid),
    }
