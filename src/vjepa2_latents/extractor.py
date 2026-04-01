from __future__ import annotations

import argparse
import importlib.machinery
import json
import os
import platform
import sys
import time
import types
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import einops
import numpy as np
import torch
import torch.nn.functional as torch_f

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


def log_step(message: str) -> None:
    print(f"[vjepa2] {message}", file=sys.stderr, flush=True)


def format_seconds(seconds: float) -> str:
    return f"{float(seconds):.3f}s"


def log_timing(label: str, seconds: float) -> None:
    log_step(f"Timing · {label}: {format_seconds(seconds)}")


def synchronize_device(device: torch.device | None) -> None:
    if device is None:
        return
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


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


@dataclass(frozen=True)
class ModelSpec:
    """Static information needed to construct and load a specific checkpoint."""

    arch_name: str
    checkpoint_keys: tuple[str, ...]
    checkpoint_url: str
    embed_dim: int
    num_heads: int
    native_resolution: int


MODEL_SPECS: dict[str, ModelSpec] = {
    # These names are the user-facing CLI options. Each entry maps to the
    # architecture constructor and checkpoint layout used by the official repo.
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


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize checkpoint keys to the names expected by the bare encoder.

    The official checkpoints often prefix weights with wrappers such as
    ``module.`` or ``backbone.`` depending on how training was launched.
    """

    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        cleaned[key] = value
    return cleaned


def ensure_vendor_imports(vendor_repo: Path) -> None:
    """Expose the cloned upstream repo on ``sys.path`` for local imports."""

    vendor_repo = vendor_repo.resolve()
    if not vendor_repo.exists():
        raise FileNotFoundError(
            f"Expected the official repo at {vendor_repo}. Clone facebookresearch/vjepa2 there first."
        )
    vendor_path = str(vendor_repo)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)

    vendor_app_dir = vendor_repo / "app"
    if vendor_app_dir.exists():
        current_app = sys.modules.get("app")
        current_app_paths = [str(path) for path in getattr(current_app, "__path__", [])]
        if str(vendor_app_dir) not in current_app_paths:
            vendor_app_package = types.ModuleType("app")
            vendor_app_package.__path__ = [str(vendor_app_dir)]
            vendor_app_package.__package__ = "app"
            vendor_app_package.__spec__ = importlib.machinery.ModuleSpec(
                name="app",
                loader=None,
                is_package=True,
            )
            vendor_app_package.__spec__.submodule_search_locations = [str(vendor_app_dir)]
            sys.modules["app"] = vendor_app_package


def auto_device(explicit: str | None = None) -> torch.device:
    """Choose a runtime device, preferring CUDA, then Apple MPS, then CPU."""

    if explicit and explicit != "auto":
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        if platform.system() == "Darwin":
            return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint_file(checkpoint_path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RuntimeError(f"Failed to read checkpoint {checkpoint_path}: {error}") from error
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint {checkpoint_path} did not contain a state-dict dictionary")
    return checkpoint


def resolve_checkpoint_key(checkpoint: dict[str, Any], checkpoint_keys: Sequence[str]) -> str:
    for checkpoint_key in checkpoint_keys:
        if checkpoint_key in checkpoint:
            return checkpoint_key
    raise RuntimeError(
        f"Checkpoint is missing expected keys {list(checkpoint_keys)}. "
        f"Available keys: {sorted(checkpoint.keys())}"
    )


def validate_checkpoint_file(checkpoint_path: Path, checkpoint_keys: Sequence[str]) -> str:
    checkpoint = load_checkpoint_file(checkpoint_path)
    try:
        return resolve_checkpoint_key(checkpoint, checkpoint_keys)
    except RuntimeError as error:
        raise RuntimeError(f"Checkpoint {checkpoint_path} is invalid: {error}") from error


def download_checkpoint(url: str, target_path: Path) -> None:
    temp_path = target_path.with_suffix(target_path.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()
    try:
        urllib.request.urlretrieve(url, temp_path)
        os.replace(temp_path, target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def download_checkpoint_if_needed(model_name: str, checkpoint_path: Path | None, checkpoint_dir: Path) -> Path:
    """Return a local checkpoint path, downloading the official weights if needed."""

    if checkpoint_path is not None:
        checkpoint_path = checkpoint_path.resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        validate_checkpoint_file(checkpoint_path, MODEL_SPECS[model_name].checkpoint_keys)
        log_step(f"Using local checkpoint: {checkpoint_path}")
        return checkpoint_path

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    spec = MODEL_SPECS[model_name]
    target_path = checkpoint_dir / Path(spec.checkpoint_url).name
    if target_path.exists():
        try:
            validate_checkpoint_file(target_path, spec.checkpoint_keys)
            log_step(f"Found cached checkpoint: {target_path}")
            return target_path
        except RuntimeError as error:
            log_step(f"Cached checkpoint is invalid, removing and re-downloading: {error}")
            target_path.unlink()

    log_step(f"Downloading checkpoint to {target_path}")
    download_checkpoint(spec.checkpoint_url, target_path)
    validate_checkpoint_file(target_path, spec.checkpoint_keys)
    return target_path


def load_encoder(
    *,
    model_name: str,
    num_frames: int,
    checkpoint_path: Path,
    vendor_repo: Path,
    device: torch.device,
    timings_out: dict[str, float] | None = None,
) -> torch.nn.Module:
    """Construct the upstream encoder and load the selected pretrained weights."""

    load_encoder_start = time.perf_counter()

    vendor_import_start = time.perf_counter()
    ensure_vendor_imports(vendor_repo)
    vendor_import_seconds = time.perf_counter() - vendor_import_start

    import_model_start = time.perf_counter()
    from app.vjepa_2_1.models import vision_transformer as vit_encoder
    import_model_seconds = time.perf_counter() - import_model_start

    spec = MODEL_SPECS[model_name]
    log_step(f"Building encoder {model_name} for {num_frames} frames on {device}")
    # We instantiate the encoder at the checkpoint's native spatial resolution,
    # but RoPE interpolation in the upstream implementation lets us run on other
    # crop sizes such as 256 or 384 at inference time.
    build_encoder_start = time.perf_counter()
    encoder = vit_encoder.__dict__[spec.arch_name](
        img_size=(spec.native_resolution, spec.native_resolution),
        patch_size=16,
        num_frames=num_frames,
        tubelet_size=2,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
        modality_embedding=True,
    )
    build_encoder_seconds = time.perf_counter() - build_encoder_start

    log_step(f"Loading checkpoint weights from {checkpoint_path}")
    load_checkpoint_start = time.perf_counter()
    checkpoint = load_checkpoint_file(checkpoint_path)
    load_checkpoint_seconds = time.perf_counter() - load_checkpoint_start

    resolve_key_start = time.perf_counter()
    checkpoint_key = resolve_checkpoint_key(checkpoint, spec.checkpoint_keys)
    resolve_key_seconds = time.perf_counter() - resolve_key_start
    log_step(f"Using checkpoint key '{checkpoint_key}'")

    clean_state_start = time.perf_counter()
    pretrained = clean_state_dict(checkpoint[checkpoint_key])
    clean_state_seconds = time.perf_counter() - clean_state_start

    load_state_dict_start = time.perf_counter()
    load_result = encoder.load_state_dict(pretrained, strict=True)
    load_state_dict_seconds = time.perf_counter() - load_state_dict_start
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint load mismatch. Missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )

    move_encoder_start = time.perf_counter()
    encoder.eval().to(device)
    synchronize_device(device)
    move_encoder_seconds = time.perf_counter() - move_encoder_start

    if timings_out is not None:
        timings_out.clear()
        timings_out.update(
            {
                "ensure_vendor_imports_seconds": vendor_import_seconds,
                "import_model_module_seconds": import_model_seconds,
                "build_encoder_module_seconds": build_encoder_seconds,
                "load_checkpoint_file_seconds": load_checkpoint_seconds,
                "resolve_checkpoint_key_seconds": resolve_key_seconds,
                "clean_state_dict_seconds": clean_state_seconds,
                "load_state_dict_seconds": load_state_dict_seconds,
                "move_encoder_to_device_seconds": move_encoder_seconds,
                "total_seconds": time.perf_counter() - load_encoder_start,
            }
        )
    return encoder


def run_encoder_synchronously(
    encoder: torch.nn.Module,
    video_tensor: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Run the encoder and measure end-to-end wall time, including async device completion."""

    encoder_start = time.perf_counter()
    with torch.inference_mode():
        raw_tokens = encoder(video_tensor.to(device))
        synchronize_device(device)
    encoder_seconds = time.perf_counter() - encoder_start
    return raw_tokens, encoder_seconds


def probe_video(video_path: Path) -> dict[str, Any]:
    """Read cheap video metadata without decoding the full clip."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


def select_frame_indices(
    *,
    video_fps: float,
    frame_count: int,
    num_frames: int,
    start_frame: int = 0,
    start_second: float | None = None,
    sample_fps: float | None = None,
) -> list[int]:
    """Choose the exact source-frame indices used to build the model clip."""

    if start_second is not None:
        start_frame = int(round(start_second * video_fps))

    if sample_fps is not None:
        # Convert the desired sampling rate into source-video frame positions.
        if video_fps <= 0:
            raise ValueError("Video FPS metadata is required when using --sample-fps")
        if sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        stride = video_fps / sample_fps
        indices = [int(round(start_frame + i * stride)) for i in range(num_frames)]
        if len(set(indices)) != len(indices):
            raise ValueError(
                "Sampling produced duplicate frame indices. Lower --sample-fps or omit it for consecutive frames."
            )
    else:
        indices = list(range(start_frame, start_frame + num_frames))

    if not indices:
        raise ValueError("No frame indices were selected")
    if indices[0] < 0:
        raise ValueError("Selected frames start before the beginning of the video")
    if indices[-1] >= frame_count:
        raise ValueError(
            f"Video has {frame_count} frames, but the requested window ends at frame {indices[-1]}"
        )
    return indices


def read_video_frames(video_path: Path, frame_indices: list[int]) -> np.ndarray:
    """Decode the selected frames as an array shaped ``[time, height, width, channel]``."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames: list[np.ndarray] = []
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            cap.release()
            raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    return np.stack(frames, axis=0)


def resize_to_cover(video_tchw: torch.Tensor, crop_size: int | Sequence[int]) -> torch.Tensor:
    """Resize each frame so both output dimensions can cover the requested crop."""

    _, _, height, width = video_tchw.shape
    crop_height, crop_width = normalize_crop_size(crop_size)
    scale = max(crop_height / height, crop_width / width)
    new_height = int(round(height * scale))
    new_width = int(round(width * scale))
    return torch_f.interpolate(
        video_tchw,
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    )


def center_crop(video_tchw: torch.Tensor, crop_size: int | Sequence[int]) -> torch.Tensor:
    """Center-crop the resized clip to the requested height and width."""

    _, _, height, width = video_tchw.shape
    crop_height, crop_width = normalize_crop_size(crop_size)
    if height < crop_height or width < crop_width:
        raise ValueError(
            f"Center crop {crop_height}x{crop_width} is larger than resized frames {height}x{width}"
        )
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return video_tchw[:, :, top : top + crop_height, left : left + crop_width]


def prepare_video_frames(frames_thwc: np.ndarray, crop_size: int | Sequence[int]) -> torch.Tensor:
    """Resize and crop raw RGB frames into `[time, channel, height, width]` float tensors."""

    video_tchw = torch.from_numpy(frames_thwc).float().permute(0, 3, 1, 2) / 255.0
    video_tchw = resize_to_cover(video_tchw, crop_size)
    return center_crop(video_tchw, crop_size)


def prepare_display_frames(frames_thwc: np.ndarray, crop_size: int | Sequence[int]) -> np.ndarray:
    """Apply the model's resize/crop pipeline without normalization for side-by-side display."""

    video_tchw = prepare_video_frames(frames_thwc, crop_size)
    frames = video_tchw.permute(0, 2, 3, 1).mul(255.0).clamp(0.0, 255.0)
    return frames.to(dtype=torch.uint8).cpu().numpy()


def preprocess_video(frames_thwc: np.ndarray, crop_size: int | Sequence[int]) -> torch.Tensor:
    """Convert raw RGB frames into a normalized model input tensor.

    Input is decoded as ``[time, height, width, channel]`` and returned as the
    V-JEPA layout ``[batch, channel, time, height, width]``.
    """

    video_tchw = prepare_video_frames(frames_thwc, crop_size)
    mean = IMAGENET_MEAN.view(1, 3, 1, 1)
    std = IMAGENET_STD.view(1, 3, 1, 1)
    video_tchw = (video_tchw - mean) / std
    return video_tchw.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


def reshape_patch_tokens(
    patch_tokens: torch.Tensor,
    *,
    time_patches: int,
    height_patches: int,
    width_patches: int,
    strip_leading_tokens: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Rebuild the spatiotemporal patch grid from the flat token sequence."""

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
    """Rebuild the latent grid and report timings for the main reshape sub-steps."""

    timings: dict[str, float] = {}
    reshape_start = time.perf_counter()

    expected_patches = time_patches * height_patches * width_patches
    token_count = patch_tokens.shape[1]

    auto_detect_start = time.perf_counter()
    if strip_leading_tokens is None:
        # Some ViT variants prepend a special token. The V-JEPA 2.1 encoder used
        # here usually does not, but we detect the off-by-one case automatically.
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
    # The flat token order from the encoder is time-major patch order, so we can
    # reconstruct the latent video grid directly with einops.
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
    """Persist the latent grid in both PyTorch and NumPy-friendly formats."""

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
    """Run the full latent-extraction pipeline for one video clip."""

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
    # V-JEPA 2.1 tokenizes with 16x16 spatial patches and a temporal tubelet of 2.
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
        # Dry-run mode validates preprocessing and shape math without paying the
        # cost of loading a large checkpoint or running the encoder.
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
        # Keep the key run facts next to the saved tensors so downstream notebook
        # analysis can recover how the clip was produced.
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
    cleanup_start = time.perf_counter()
    if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
        synchronize_device(device)
    major_phase_timings["cleanup and cache clear"] = time.perf_counter() - cleanup_start

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
    """Define the CLI for extraction and quick preprocessing checks."""

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
    """CLI entrypoint used by both direct execution and the thin wrapper script."""

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
