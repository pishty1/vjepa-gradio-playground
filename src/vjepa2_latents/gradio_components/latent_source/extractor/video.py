from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_f

from .config import IMAGENET_MEAN, IMAGENET_STD, normalize_crop_size


def probe_video(video_path: Path) -> dict[str, Any]:
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
    if start_second is not None:
        start_frame = int(round(start_second * video_fps))

    if sample_fps is not None:
        if video_fps <= 0:
            raise ValueError("Video FPS metadata is required when using --sample-fps")
        if sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        stride = video_fps / sample_fps
        indices = [int(round(start_frame + index * stride)) for index in range(num_frames)]
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
    video_tchw = torch.from_numpy(frames_thwc).float().permute(0, 3, 1, 2) / 255.0
    video_tchw = resize_to_cover(video_tchw, crop_size)
    return center_crop(video_tchw, crop_size)


def prepare_display_frames(frames_thwc: np.ndarray, crop_size: int | Sequence[int]) -> np.ndarray:
    video_tchw = prepare_video_frames(frames_thwc, crop_size)
    frames = video_tchw.permute(0, 2, 3, 1).mul(255.0).clamp(0.0, 255.0)
    return frames.to(dtype=torch.uint8).cpu().numpy()


def preprocess_video(frames_thwc: np.ndarray, crop_size: int | Sequence[int]) -> torch.Tensor:
    video_tchw = prepare_video_frames(frames_thwc, crop_size)
    mean = IMAGENET_MEAN.view(1, 3, 1, 1)
    std = IMAGENET_STD.view(1, 3, 1, 1)
    video_tchw = (video_tchw - mean) / std
    return video_tchw.permute(1, 0, 2, 3).unsqueeze(0).contiguous()