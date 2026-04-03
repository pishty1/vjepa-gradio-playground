from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from .extractor_checkpoint import (
    clean_state_dict,
    download_checkpoint,
    download_checkpoint_if_needed,
    ensure_vendor_imports,
    load_checkpoint_file,
    load_encoder,
    resolve_checkpoint_key,
    validate_checkpoint_file,
)
from .extractor_config import (
    MODEL_SPECS,
    auto_device,
    estimate_attention_scores_bytes,
    get_mps_memory_info,
    get_system_memory_bytes,
    normalize_crop_size,
    parse_crop_size,
)
from .extractor_logging import (
    bytes_to_mib,
    device_executes_asynchronously,
    log_step,
    log_timing,
    log_timing_summary,
)
from .extractor_tensor import (
    reshape_patch_tokens,
    reshape_patch_tokens_with_timings,
    run_encoder_synchronously,
    save_outputs,
)
from .extractor_video import (
    prepare_display_frames,
    preprocess_video,
    probe_video,
    read_video_frames,
    select_frame_indices,
)


def estimate_extraction_requirements(
    *,
    model_name: str,
    num_frames: int,
    crop_size: int | Sequence[int],
    device_name: str | None,
) -> dict[str, Any]:
    crop_height, crop_width = normalize_crop_size(crop_size)
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unsupported model '{model_name}'. Choices: {sorted(MODEL_SPECS)}")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    spec = MODEL_SPECS[model_name]
    device = auto_device(device_name)
    time_patches = int(num_frames) // 2
    height_patches = int(crop_height) // 16
    width_patches = int(crop_width) // 16
    token_count = time_patches * height_patches * width_patches
    input_tensor_bytes = int(1 * 3 * int(num_frames) * crop_height * crop_width * 4)
    latent_tensor_bytes = int(1 * token_count * spec.embed_dim * 4)
    baseline_tokens = 8 * 24 * 24
    quadratic_token_factor = (token_count / baseline_tokens) ** 2 if baseline_tokens > 0 else 0.0
    model_pressure_factor = quadratic_token_factor * (spec.embed_dim / 768.0)

    if model_pressure_factor >= 7.5:
        risk_level = "high"
    elif model_pressure_factor >= 4.0:
        risk_level = "medium"
    else:
        risk_level = "low"

    system_memory_bytes = get_system_memory_bytes()
    mps_memory = get_mps_memory_info() if device.type == "mps" else {
        "recommended_max_memory": None,
        "current_allocated_memory": None,
        "driver_allocated_memory": None,
    }
    recommended_max_memory = mps_memory.get("recommended_max_memory")
    latent_fraction_of_mps_limit = None
    if recommended_max_memory:
        latent_fraction_of_mps_limit = float(latent_tensor_bytes) / float(recommended_max_memory)

    return {
        "device": str(device),
        "model": model_name,
        "crop_size": [crop_height, crop_width],
        "num_frames": int(num_frames),
        "time_patches": time_patches,
        "height_patches": height_patches,
        "width_patches": width_patches,
        "token_count": token_count,
        "input_tensor_shape": [1, 3, int(num_frames), crop_height, crop_width],
        "latent_shape": [1, time_patches, height_patches, width_patches, spec.embed_dim],
        "input_tensor_bytes": input_tensor_bytes,
        "latent_tensor_bytes": latent_tensor_bytes,
        "input_tensor_mib": bytes_to_mib(input_tensor_bytes),
        "latent_tensor_mib": bytes_to_mib(latent_tensor_bytes),
        "quadratic_token_factor_vs_16f_384": quadratic_token_factor,
        "model_pressure_factor": model_pressure_factor,
        "risk_level": risk_level,
        "system_memory_bytes": system_memory_bytes,
        "system_memory_gib": None if system_memory_bytes is None else float(system_memory_bytes) / (1024.0**3),
        "mps_memory": mps_memory,
        "mps_recommended_max_memory_gib": None
        if recommended_max_memory is None
        else float(recommended_max_memory) / (1024.0**3),
        "latent_fraction_of_mps_limit": latent_fraction_of_mps_limit,
        "notes": [
            "The saved latent tensor is much smaller than the model's peak activation memory.",
            "Transformer attention memory grows roughly with the square of the token count.",
            "On Apple MPS, out-of-memory often surfaces when pending GPU work is synchronized or copied back to CPU.",
        ],
    }


def extract_latents(
    *,
    video_path: Path,
    output_prefix: Path,
    vendor_repo: Path,
    model_name: str,
    checkpoint_path: Path | None,
    checkpoint_dir: Path,
    num_frames: int,
    crop_size: int | Sequence[int],
    sample_fps: float | None,
    start_frame: int,
    start_second: float | None,
    device_name: str | None,
    dry_run: bool,
    save_pt: bool = True,
) -> dict[str, Any]:
    extraction_start = time.perf_counter()
    major_phase_timings: dict[str, float] = {}

    video_path = video_path.resolve()
    output_prefix = output_prefix.resolve()
    vendor_repo = vendor_repo.resolve()
    checkpoint_dir = checkpoint_dir.resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unsupported model '{model_name}'. Choices: {sorted(MODEL_SPECS)}")
    if num_frames % 2 != 0:
        raise ValueError("num_frames must be divisible by tubelet size 2")

    device = auto_device(device_name)
    log_step(f"Selected device: {device}")

    probe_start = time.perf_counter()
    video_meta = probe_video(video_path)
    major_phase_timings["probe video metadata"] = time.perf_counter() - probe_start
    log_step(
        f"Video metadata: {video_meta['frame_count']} frames at {video_meta['fps']:.3f} fps, "
        f"resolution {video_meta['width']}x{video_meta['height']}"
    )

    frame_selection_start = time.perf_counter()
    frame_indices = select_frame_indices(
        video_fps=video_meta["fps"],
        frame_count=video_meta["frame_count"],
        num_frames=num_frames,
        start_frame=start_frame,
        start_second=start_second,
        sample_fps=sample_fps,
    )
    major_phase_timings["select frame indices"] = time.perf_counter() - frame_selection_start
    log_step(f"Reading {len(frame_indices)} frames: {frame_indices[0]}..{frame_indices[-1]}")

    decode_start = time.perf_counter()
    frames = read_video_frames(video_path, frame_indices)
    major_phase_timings["decode video frames"] = time.perf_counter() - decode_start
    crop_height, crop_width = normalize_crop_size(crop_size)
    log_step(f"Preprocessing clip to {crop_height}x{crop_width}")

    preprocess_start = time.perf_counter()
    video_tensor = preprocess_video(frames, crop_size=crop_size)
    major_phase_timings["preprocess video clip"] = time.perf_counter() - preprocess_start
    input_shape = list(video_tensor.shape)

    _, _, input_time, input_height, input_width = video_tensor.shape
    time_patches = input_time // 2
    height_patches = input_height // 16
    width_patches = input_width // 16
    expected_patches = time_patches * height_patches * width_patches
    spec = MODEL_SPECS[model_name]
    estimated_attention_scores_bytes = estimate_attention_scores_bytes(
        num_heads=spec.num_heads,
        token_count=expected_patches,
        dtype=video_tensor.dtype,
    )
    if device.type == "mps":
        log_step(
            "MPS sequence summary: "
            f"tokens={expected_patches} ({time_patches}x{height_patches}x{width_patches}), "
            f"estimated attention scores={bytes_to_mib(estimated_attention_scores_bytes):.1f} MiB across {spec.num_heads} heads"
        )

    if dry_run:
        log_step("Dry run complete")
        dummy_tokens = torch.zeros((1, expected_patches, spec.embed_dim), dtype=torch.float32)
        latent_grid, stripped = reshape_patch_tokens(
            dummy_tokens,
            time_patches=time_patches,
            height_patches=height_patches,
            width_patches=width_patches,
        )
        return {
            "mode": "dry-run",
            "device": str(device),
            "input_shape": input_shape,
            "latent_shape": list(latent_grid.shape),
            "frame_indices": frame_indices,
            "tokens_stripped": stripped,
        }

    checkpoint_resolution_start = time.perf_counter()
    resolved_checkpoint = download_checkpoint_if_needed(model_name, checkpoint_path, checkpoint_dir)
    major_phase_timings["resolve checkpoint"] = time.perf_counter() - checkpoint_resolution_start

    encoder_setup_timings: dict[str, float] = {}
    encoder = load_encoder(
        model_name=model_name,
        num_frames=num_frames,
        checkpoint_path=resolved_checkpoint,
        vendor_repo=vendor_repo,
        device=device,
        timings_out=encoder_setup_timings,
    )
    major_phase_timings["load encoder"] = encoder_setup_timings["total_seconds"]
    log_timing("load encoder", encoder_setup_timings["total_seconds"])

    log_step("Running encoder synchronously")
    raw_tokens, encoder_forward_seconds = run_encoder_synchronously(encoder, video_tensor, device)
    major_phase_timings["encoder run"] = encoder_forward_seconds
    log_timing("encoder run", encoder_forward_seconds)

    log_step("Reshaping patch tokens into latent grid")
    latent_grid, stripped, reshape_timings = reshape_patch_tokens_with_timings(
        raw_tokens,
        time_patches=time_patches,
        height_patches=height_patches,
        width_patches=width_patches,
    )
    major_phase_timings["reshape patch tokens"] = reshape_timings["total_seconds"]
    log_timing("reshape patch tokens into latent grid", reshape_timings["total_seconds"])
    raw_token_shape = list(raw_tokens.shape)
    latent_grid_shape = list(latent_grid.shape)

    metadata = {
        "video_path": str(video_path),
        "video_name": video_path.name,
        "model": model_name,
        "device": str(device),
        "checkpoint_path": str(resolved_checkpoint),
        "video_metadata": video_meta,
        "frame_indices": frame_indices,
        "input_tensor_shape": input_shape,
        "raw_token_shape": raw_token_shape,
        "tokens_stripped": stripped,
        "latent_grid_shape": latent_grid_shape,
        "patch_size": 16,
        "tubelet_size": 2,
        "crop_size": [crop_height, crop_width],
        "native_checkpoint_resolution": spec.native_resolution,
        "timings": {
            "encoder_forward_pass": {
                "device_executes_asynchronously": device_executes_asynchronously(device),
                "measured_synchronously": True,
                "forward_run_seconds": encoder_forward_seconds,
                "total_wall_seconds": encoder_forward_seconds,
            },
            "reshape_patch_tokens": reshape_timings,
        },
        "notes": [
            "V-JEPA 2.1 checkpoints are trained at 384 resolution.",
            "This extractor supports smaller inference crops such as 256 via RoPE-enabled variable-size inference.",
            "The PyTorch encoder in the official repo does not usually emit an extra modality token; stripping is automatic only if the token count is off by one.",
            "On asynchronous devices such as Apple MPS, encoder enqueue time can be much smaller than the actual forward wall time because GPU work completes later at synchronization points.",
        ],
    }
    del encoder
    del video_tensor
    del frames
    del raw_tokens
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    log_step(f"Starting output serialization with prefix {output_prefix}")
    output_serialization_timings: dict[str, float] = {}
    output_paths = save_outputs(
        latent_grid=latent_grid,
        output_prefix=output_prefix,
        metadata=metadata,
        save_pt=save_pt,
        timings_out=output_serialization_timings,
    )
    metadata["timings"]["output_serialization"] = output_serialization_timings
    major_phase_timings["serialize outputs"] = output_serialization_timings["total_seconds"]

    finalize_metadata_start = time.perf_counter()
    metadata["timings"]["total_extraction_seconds"] = time.perf_counter() - extraction_start
    metadata["timings"]["major_phases"] = major_phase_timings
    metadata["timings"]["encoder_setup"] = encoder_setup_timings
    metadata["timings"]["major_phase_subtotal_seconds"] = sum(major_phase_timings.values())
    metadata["timings"]["major_phase_unaccounted_overhead_seconds"] = max(
        0.0,
        metadata["timings"]["total_extraction_seconds"] - metadata["timings"]["major_phase_subtotal_seconds"],
    )
    output_paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    finalize_metadata_seconds = time.perf_counter() - finalize_metadata_start
    major_phase_timings["finalize metadata write"] = finalize_metadata_seconds
    metadata["timings"]["major_phases"] = major_phase_timings
    metadata["timings"]["major_phase_subtotal_seconds"] = sum(major_phase_timings.values())
    metadata["timings"]["major_phase_unaccounted_overhead_seconds"] = max(
        0.0,
        metadata["timings"]["total_extraction_seconds"] - metadata["timings"]["major_phase_subtotal_seconds"],
    )
    output_paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    extraction_summary_timings = {
        "encoder run": encoder_forward_seconds,
        "decode video frames": major_phase_timings["decode video frames"],
        "load encoder": major_phase_timings["load encoder"],
        "preprocess video clip": major_phase_timings["preprocess video clip"],
        "resolve checkpoint": major_phase_timings["resolve checkpoint"],
        "serialize outputs": major_phase_timings["serialize outputs"],
        "reshape patch tokens": major_phase_timings["reshape patch tokens"],
    }
    log_timing_summary(
        "Extraction timing summary",
        extraction_summary_timings,
        metadata["timings"]["total_extraction_seconds"],
        min_seconds=0.05,
    )
    log_timing("total extraction", metadata["timings"]["total_extraction_seconds"])
    log_step("Extraction complete")

    return {
        "mode": "extract",
        "device": str(device),
        "input_shape": input_shape,
        "raw_token_shape": raw_token_shape,
        "latent_shape": latent_grid_shape,
        "frame_indices": frame_indices,
        "tokens_stripped": stripped,
        "timings": metadata["timings"],
        "outputs": {key: str(value) for key, value in output_paths.items()},
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract patch-level V-JEPA 2.1 latents from a video clip.")
    parser.add_argument("--video", type=Path, default=Path("testvideo.mp4"), help="Path to the input video.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("skate_latents"),
        help="Output file prefix; .pt, .npy, and .metadata.json are written.",
    )
    parser.add_argument(
        "--vendor-repo",
        type=Path,
        default=Path("vendor/vjepa2"),
        help="Path to the cloned official facebookresearch/vjepa2 repository.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_SPECS),
        default="vit_large_384",
        help="Checkpoint/model variant to load.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional local checkpoint path. If omitted, the official checkpoint is downloaded into --checkpoint-dir.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory for downloaded checkpoints.",
    )
    parser.add_argument("--num-frames", type=int, default=16, help="Number of frames in the extracted clip.")
    parser.add_argument(
        "--crop-size",
        type=parse_crop_size,
        default=256,
        help="Center crop size. Accepts `N` for square crops or `HxW` for rectangular crops, e.g. `384` or `384x640`.",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=None,
        help="Optional resampling FPS. Omit to use consecutive frames.",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="First source frame to use.")
    parser.add_argument(
        "--start-second",
        type=float,
        default=None,
        help="Alternative to --start-frame; overrides it when provided.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run on: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preprocessing and shape math only, without downloading weights or executing the encoder.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = extract_latents(
        video_path=args.video,
        output_prefix=args.output_prefix,
        vendor_repo=args.vendor_repo,
        model_name=args.model,
        checkpoint_path=args.checkpoint_path,
        checkpoint_dir=args.checkpoint_dir,
        num_frames=args.num_frames,
        crop_size=args.crop_size,
        sample_fps=args.sample_fps,
        start_frame=args.start_frame,
        start_second=args.start_second,
        device_name=args.device,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))
    return 0


__all__ = [
    "MODEL_SPECS",
    "ModelSpec",
    "auto_device",
    "bytes_to_mib",
    "build_arg_parser",
    "center_crop",
    "clean_state_dict",
    "device_executes_asynchronously",
    "download_checkpoint",
    "download_checkpoint_if_needed",
    "ensure_vendor_imports",
    "estimate_attention_scores_bytes",
    "estimate_extraction_requirements",
    "extract_latents",
    "get_mps_memory_info",
    "get_system_memory_bytes",
    "load_checkpoint_file",
    "load_encoder",
    "log_step",
    "log_timing",
    "log_timing_summary",
    "main",
    "normalize_crop_size",
    "parse_crop_size",
    "prepare_display_frames",
    "prepare_video_frames",
    "preprocess_video",
    "probe_video",
    "read_video_frames",
    "reshape_patch_tokens",
    "reshape_patch_tokens_with_timings",
    "resolve_checkpoint_key",
    "resize_to_cover",
    "run_encoder_synchronously",
    "save_outputs",
    "select_frame_indices",
    "validate_checkpoint_file",
]
