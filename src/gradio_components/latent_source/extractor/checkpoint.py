from __future__ import annotations

import contextlib
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import torch

from .config import MODEL_SPECS
from .utils.logging import log_step


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = WORKSPACE_ROOT / "src"


def _resolve_sys_path_entry(entry: str) -> Path | None:
    if not entry:
        return Path.cwd().resolve()
    try:
        return Path(entry).resolve()
    except OSError:
        return None


@contextlib.contextmanager
def isolate_torch_hub_imports() -> Any:
    original_sys_path = list(sys.path)
    blocked_entries = {WORKSPACE_ROOT.resolve(), SOURCE_ROOT.resolve(), Path.cwd().resolve()}
    sys.path[:] = [entry for entry in original_sys_path if _resolve_sys_path_entry(entry) not in blocked_entries]

    previous_app_module = sys.modules.pop("app", None)
    try:
        yield
    finally:
        sys.path[:] = original_sys_path
        if previous_app_module is not None:
            sys.modules["app"] = previous_app_module
        else:
            sys.modules.pop("app", None)


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        cleaned[key] = value
    return cleaned


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


def load_hub_encoder(*, model_name: str, num_frames: int) -> torch.nn.Module:
    spec = MODEL_SPECS[model_name]
    log_step(f"Loading {spec.hub_name} from torch.hub")
    with isolate_torch_hub_imports():
        loaded = torch.hub.load(
            "facebookresearch/vjepa2",
            spec.hub_name,
            pretrained=False,
            num_frames=num_frames,
            trust_repo=True,
        )
    encoder = loaded[0] if isinstance(loaded, tuple) else loaded
    if not isinstance(encoder, torch.nn.Module):
        raise RuntimeError(f"torch.hub returned an unexpected object for {spec.hub_name}: {type(encoder)!r}")
    return encoder


def load_encoder(
    *,
    model_name: str,
    num_frames: int,
    checkpoint_path: Path,
    device: torch.device,
    timings_out: dict[str, float] | None = None,
) -> torch.nn.Module:
    load_encoder_start = time.perf_counter()

    spec = MODEL_SPECS[model_name]

    build_encoder_start = time.perf_counter()
    encoder = load_hub_encoder(model_name=model_name, num_frames=num_frames)
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
    move_encoder_seconds = time.perf_counter() - move_encoder_start

    if timings_out is not None:
        timings_out.clear()
        timings_out.update(
            {
                "load_torch_hub_encoder_seconds": build_encoder_seconds,
                "load_checkpoint_file_seconds": load_checkpoint_seconds,
                "resolve_checkpoint_key_seconds": resolve_key_seconds,
                "clean_state_dict_seconds": clean_state_seconds,
                "load_state_dict_seconds": load_state_dict_seconds,
                "move_encoder_to_device_seconds": move_encoder_seconds,
                "total_seconds": time.perf_counter() - load_encoder_start,
            }
        )
    return encoder