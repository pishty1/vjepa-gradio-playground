from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import einops
import numpy as np
import torch

from .extractor_logging import log_step, log_timing


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def run_encoder_synchronously(
    encoder: torch.nn.Module,
    video_tensor: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    input_tensor = video_tensor.to(device)
    _synchronize_device(device)
    encoder_start = time.perf_counter()
    with torch.inference_mode():
        raw_tokens = encoder(input_tensor)
    _synchronize_device(device)
    encoder_seconds = time.perf_counter() - encoder_start
    return raw_tokens, encoder_seconds


def reshape_patch_tokens(
    patch_tokens: torch.Tensor,
    *,
    time_patches: int,
    height_patches: int,
    width_patches: int,
    strip_leading_tokens: int | None = None,
) -> tuple[torch.Tensor, int]:
    latent_grid, stripped, _ = reshape_patch_tokens_with_timings(
        patch_tokens,
        time_patches=time_patches,
        height_patches=height_patches,
        width_patches=width_patches,
        strip_leading_tokens=strip_leading_tokens,
    )
    return latent_grid, stripped


def reshape_patch_tokens_with_timings(
    patch_tokens: torch.Tensor,
    *,
    time_patches: int,
    height_patches: int,
    width_patches: int,
    strip_leading_tokens: int | None = None,
) -> tuple[torch.Tensor, int, dict[str, float]]:
    timings: dict[str, float] = {}
    reshape_start = time.perf_counter()

    expected_patches = time_patches * height_patches * width_patches
    token_count = patch_tokens.shape[1]

    auto_detect_start = time.perf_counter()
    if strip_leading_tokens is None:
        if token_count == expected_patches:
            strip_leading_tokens = 0
        elif token_count == expected_patches + 1:
            strip_leading_tokens = 1
        else:
            raise ValueError(
                f"Token count {token_count} does not match expected patch count {expected_patches} "
                "and is not off by a single leading token."
            )
    timings["auto_detect_leading_tokens_seconds"] = time.perf_counter() - auto_detect_start

    validate_start = time.perf_counter()
    if strip_leading_tokens < 0:
        raise ValueError("strip_leading_tokens must be >= 0")
    if token_count - strip_leading_tokens != expected_patches:
        raise ValueError(
            f"After stripping {strip_leading_tokens} tokens, got {token_count - strip_leading_tokens} patches "
            f"but expected {expected_patches}."
        )
    timings["validate_patch_token_count_seconds"] = time.perf_counter() - validate_start

    strip_start = time.perf_counter()
    patch_tokens = patch_tokens[:, strip_leading_tokens:, :]
    timings["strip_leading_tokens_seconds"] = time.perf_counter() - strip_start

    rearrange_start = time.perf_counter()
    latent_grid = einops.rearrange(
        patch_tokens,
        "b (t h w) d -> b t h w d",
        t=time_patches,
        h=height_patches,
        w=width_patches,
    )
    timings["rearrange_to_latent_grid_seconds"] = time.perf_counter() - rearrange_start
    timings["total_seconds"] = time.perf_counter() - reshape_start
    return latent_grid, strip_leading_tokens, timings


def save_outputs(
    *,
    latent_grid: torch.Tensor,
    output_prefix: Path,
    metadata: dict[str, Any],
    save_pt: bool = True,
    timings_out: dict[str, float] | None = None,
) -> dict[str, Path]:
    serialization_start = time.perf_counter()

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    npy_path = output_prefix.with_suffix(".npy")
    metadata_path = output_prefix.with_suffix(".metadata.json")

    latent_shape = tuple(int(value) for value in latent_grid.shape)
    latent_dtype = latent_grid.dtype
    latent_device = latent_grid.device
    element_size = torch.tensor([], dtype=latent_dtype).element_size()
    total_bytes = int(latent_grid.numel()) * int(element_size)
    log_step(
        "Preparing latent grid for serialization: "
        f"shape={latent_shape}, dtype={latent_dtype}, device={latent_device}, size={total_bytes / (1024 * 1024):.1f} MiB"
    )
    timings = metadata.setdefault("timings", {})
    output_timings = timings.setdefault("output_serialization", {})

    log_step("Copying latent grid to CPU memory")
    copy_start = time.perf_counter()
    latent_grid_cpu = latent_grid.detach().to("cpu")
    output_timings["copy_latent_grid_to_cpu_seconds"] = time.perf_counter() - copy_start

    log_step(f"Writing NumPy latent grid to {npy_path}")
    numpy_start = time.perf_counter()
    np.save(npy_path, latent_grid_cpu.numpy())
    output_timings["write_numpy_latent_grid_seconds"] = time.perf_counter() - numpy_start

    log_step(f"Writing metadata to {metadata_path}")
    metadata_json = json.dumps(metadata, indent=2)
    metadata_start = time.perf_counter()
    metadata_path.write_text(metadata_json, encoding="utf-8")
    output_timings["write_metadata_seconds"] = time.perf_counter() - metadata_start

    outputs = {
        "npy": npy_path,
        "metadata": metadata_path,
    }
    if save_pt:
        pt_path = output_prefix.with_suffix(".pt")
        log_step(f"Writing PyTorch tensor to {pt_path}")
        torch_save_start = time.perf_counter()
        torch.save(latent_grid_cpu, pt_path)
        output_timings["write_pytorch_tensor_seconds"] = time.perf_counter() - torch_save_start
        outputs["pt"] = pt_path

    output_timings["total_seconds"] = time.perf_counter() - serialization_start
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if timings_out is not None:
        timings_out.clear()
        timings_out.update(output_timings)

    log_timing("output serialization total", output_timings["total_seconds"])
    log_step("Output serialization complete")
    return outputs
