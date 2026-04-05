from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from .checkpoint import (
    clean_state_dict,
    download_checkpoint,
    download_checkpoint_if_needed,
    ensure_vendor_imports,
    load_checkpoint_file,
    load_encoder,
    resolve_checkpoint_key,
    validate_checkpoint_file,
)
from .config import (
    MODEL_SPECS,
    ModelSpec,
    auto_device,
    estimate_attention_scores_bytes,
    get_mps_memory_info,
    get_system_memory_bytes,
    normalize_crop_size,
    parse_crop_size,
)
from .tensor import (
    reshape_patch_tokens,
    reshape_patch_tokens_with_timings,
    run_encoder_synchronously,
    save_outputs,
)
from .utils.logging import bytes_to_mib, log_step, log_timing, log_timing_summary
from .video import (
    center_crop,
    prepare_display_frames,
    prepare_video_frames,
    preprocess_video,
    probe_video,
    read_video_frames,
    resize_to_cover,
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
        "video_metadata": video_meta,
        "model_name": model_name,
        "model_spec": {
            "arch_name": spec.arch_name,
            "embed_dim": spec.embed_dim,
            "num_heads": spec.num_heads,
            "native_resolution": spec.native_resolution,
        },
        "checkpoint_path": str(resolved_checkpoint),
        "device": str(device),
        "crop_size": [crop_height, crop_width],
        "num_frames": int(num_frames),
        "sample_fps": None if sample_fps is None else float(sample_fps),
        "start_frame": int(start_frame),
        "start_second": None if start_second is None else float(start_second),
        "frame_indices": frame_indices,
        "tubelet_size": 2,
        "input_tensor_shape": input_shape,
        "raw_patch_tokens_shape": raw_token_shape,
        "latent_grid_shape": latent_grid_shape,
        "tokens_stripped": int(stripped),
        "estimated_attention_scores_bytes": int(estimated_attention_scores_bytes),
        "timings": {
            "major_phases": major_phase_timings,
            "encoder_setup": encoder_setup_timings,
            "reshape_patch_tokens": reshape_timings,
        },
    }

    save_output_timings: dict[str, float] = {}
    outputs = save_outputs(
        latent_grid=latent_grid,
        output_prefix=output_prefix,
        metadata=metadata,
        save_pt=save_pt,
        timings_out=save_output_timings,
    )
    major_phase_timings["save outputs"] = save_output_timings.get("total_seconds", 0.0)
    total_seconds = time.perf_counter() - extraction_start
    metadata["timings"]["total_seconds"] = total_seconds
    log_timing_summary("Extraction timing summary", major_phase_timings, total_seconds)

    return {
        "video_path": str(video_path),
        "output_prefix": str(output_prefix),
        "device": str(device),
        "model_name": model_name,
        "checkpoint_path": str(resolved_checkpoint),
        "frame_indices": frame_indices,
        "video_metadata": video_meta,
        "input_shape": input_shape,
        "raw_patch_tokens_shape": raw_token_shape,
        "latent_shape": latent_grid_shape,
        "tokens_stripped": int(stripped),
        "estimated_attention_scores_bytes": int(estimated_attention_scores_bytes),
        "outputs": {key: str(path) for key, path in outputs.items()},
        "timings": metadata["timings"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract V-JEPA 2.1 latent grids from a video clip")
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("output_prefix", type=Path, help="Output path prefix, e.g. outputs/clip")
    parser.add_argument("--vendor-repo", type=Path, default=Path("vendor/vjepa2"), help="Path to facebookresearch/vjepa2 checkout")
    parser.add_argument("--model", dest="model_name", default="vit_base_384", choices=sorted(MODEL_SPECS), help="Encoder variant")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional explicit checkpoint path")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Where downloaded checkpoints are cached")
    parser.add_argument("--num-frames", type=int, default=16, help="Number of frames to encode (must be divisible by 2)")
    parser.add_argument("--crop-size", type=parse_crop_size, default=384, help="Center crop size as INT or HxW, e.g. 384 or 320x512")
    parser.add_argument("--sample-fps", type=float, default=None, help="Optional temporal sampling FPS")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index")
    parser.add_argument("--start-second", type=float, default=None, help="Optional start time in seconds")
    parser.add_argument("--device", dest="device_name", default=None, help="torch device override, e.g. cpu, cuda, mps")
    parser.add_argument("--dry-run", action="store_true", help="Only validate shapes and selection logic without loading the model")
    parser.add_argument("--no-save-pt", dest="save_pt", action="store_false", help="Skip writing the .pt latent tensor")
    parser.set_defaults(save_pt=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = extract_latents(
        video_path=args.video,
        output_prefix=args.output_prefix,
        vendor_repo=args.vendor_repo,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        num_frames=args.num_frames,
        crop_size=args.crop_size,
        sample_fps=args.sample_fps,
        start_frame=args.start_frame,
        start_second=args.start_second,
        device_name=args.device_name,
        dry_run=args.dry_run,
        save_pt=args.save_pt,
    )
    print(json.dumps(result, indent=2))
    return 0