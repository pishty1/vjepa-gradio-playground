from __future__ import annotations

import sys

import torch


def log_step(message: str) -> None:
    print(f"[vjepa2] {message}", file=sys.stderr, flush=True)


def format_seconds(seconds: float) -> str:
    return f"{float(seconds):.3f}s"


def log_timing(label: str, seconds: float) -> None:
    log_step(f"Timing · {label}: {format_seconds(seconds)}")


def log_timing_summary(
    title: str,
    timings: dict[str, float],
    total_seconds: float,
    *,
    min_seconds: float = 0.0,
    max_entries: int | None = None,
) -> None:
    log_step(f"{title}:")
    ranked = sorted(timings.items(), key=lambda item: item[1], reverse=True)
    displayed = 0
    for label, seconds in ranked:
        if seconds < min_seconds:
            continue
        share = 0.0 if total_seconds <= 0.0 else (seconds / total_seconds) * 100.0
        log_step(f"  - {label}: {format_seconds(seconds)} ({share:.1f}%)")
        displayed += 1
        if max_entries is not None and displayed >= max_entries:
            break
    subtotal_seconds = sum(timings.values())
    overhead_seconds = max(0.0, total_seconds - subtotal_seconds)
    if overhead_seconds >= min_seconds:
        share = 0.0 if total_seconds <= 0.0 else (overhead_seconds / total_seconds) * 100.0
        log_step(f"  - unaccounted overhead: {format_seconds(overhead_seconds)} ({share:.1f}%)")


def bytes_to_mib(num_bytes: int) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


def device_executes_asynchronously(device: torch.device | None) -> bool:
    if device is None:
        return False
    if device.type == "cuda":
        return torch.cuda.is_available()
    if device.type == "mps":
        return hasattr(torch, "mps") and hasattr(torch.mps, "synchronize")
    return False
