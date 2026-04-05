from __future__ import annotations

import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_VIDEO = ROOT / "testvideo.mp4"
VENDOR_REPO = ROOT / "vendor" / "vjepa2"
CHECKPOINT_DIR = ROOT / "checkpoints"
APP_OUTPUT_DIR = ROOT / ".gradio_outputs"
DEFAULT_DEVICE = "mps" if platform.system() == "Darwin" else "auto"
DEFAULT_MODEL_NAME = "vit_base_384"
DEFAULT_CROP_HEIGHT = 384
DEFAULT_CROP_WIDTH = 384
