from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Sequence

import torch


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


def normalize_crop_size(crop_size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(crop_size, int):
        if crop_size <= 0:
            raise ValueError("crop_size must be positive")
        return crop_size, crop_size

    values = tuple(int(value) for value in crop_size)
    if len(values) != 2:
        raise ValueError("crop_size sequence must contain exactly 2 values: height and width")
    crop_height, crop_width = values
    if crop_height <= 0 or crop_width <= 0:
        raise ValueError("crop dimensions must be positive")
    return crop_height, crop_width


def parse_crop_size(value: str) -> int | tuple[int, int]:
    text = value.strip().lower()
    if "x" in text:
        height_text, width_text = text.split("x", maxsplit=1)
        return normalize_crop_size((int(height_text), int(width_text)))
    size = int(text)
    if size <= 0:
        raise ValueError("crop size must be positive")
    return size


@dataclass(frozen=True)
class ModelSpec:
    arch_name: str
    checkpoint_keys: tuple[str, ...]
    checkpoint_url: str
    embed_dim: int
    num_heads: int
    native_resolution: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "vit_base_384": ModelSpec(
        arch_name="vit_base",
        checkpoint_keys=("ema_encoder", "target_encoder", "encoder"),
        checkpoint_url="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt",
        embed_dim=768,
        num_heads=12,
        native_resolution=384,
    ),
    "vit_large_384": ModelSpec(
        arch_name="vit_large",
        checkpoint_keys=("ema_encoder", "target_encoder", "encoder"),
        checkpoint_url="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt",
        embed_dim=1024,
        num_heads=16,
        native_resolution=384,
    ),
    "vit_giant_384": ModelSpec(
        arch_name="vit_giant_xformers",
        checkpoint_keys=("target_encoder", "ema_encoder", "encoder"),
        checkpoint_url="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitg_384.pt",
        embed_dim=1408,
        num_heads=22,
        native_resolution=384,
    ),
    "vit_gigantic_384": ModelSpec(
        arch_name="vit_gigantic_xformers",
        checkpoint_keys=("target_encoder", "ema_encoder", "encoder"),
        checkpoint_url="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt",
        embed_dim=1664,
        num_heads=26,
        native_resolution=384,
    ),
}


def auto_device(explicit: str | None = None) -> torch.device:
    if explicit and explicit != "auto":
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and platform.system() == "Darwin":
        return torch.device("mps")
    return torch.device("cpu")


def estimate_attention_scores_bytes(*, num_heads: int, token_count: int, dtype: torch.dtype) -> int:
    element_size = torch.tensor([], dtype=dtype).element_size()
    return int(num_heads) * int(token_count) * int(token_count) * int(element_size)


def get_system_memory_bytes() -> int | None:
    page_size_names = ("SC_PAGE_SIZE", "SC_PAGESIZE")
    page_size: int | None = None
    for name in page_size_names:
        try:
            page_size = int(os.sysconf(name))
            break
        except (AttributeError, OSError, ValueError):
            continue
    try:
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        page_count = 0
    if not page_size or page_count <= 0:
        return None
    return int(page_size) * int(page_count)


def get_mps_memory_info() -> dict[str, int | None]:
    if not hasattr(torch, "mps"):
        return {
            "recommended_max_memory": None,
            "current_allocated_memory": None,
            "driver_allocated_memory": None,
        }

    def _safe_int(name: str) -> int | None:
        value = getattr(torch.mps, name, None)
        if value is None:
            return None
        if callable(value):
            try:
                return int(value())
            except Exception:
                return None
        try:
            return int(value)
        except Exception:
            return None

    return {
        "recommended_max_memory": _safe_int("recommended_max_memory"),
        "current_allocated_memory": _safe_int("current_allocated_memory"),
        "driver_allocated_memory": _safe_int("driver_allocated_memory"),
    }