#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from vjepa2_latents.gradio_app import build_demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch()
