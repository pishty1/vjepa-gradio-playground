from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vjepa2_latents.extractor import reshape_patch_tokens, select_frame_indices


class ReshapePatchTokensTests(unittest.TestCase):
    def test_keeps_plain_patch_tokens(self) -> None:
        tokens = torch.randn(1, 8 * 16 * 16, 1024)
        grid, stripped = reshape_patch_tokens(
            tokens,
            time_patches=8,
            height_patches=16,
            width_patches=16,
        )
        self.assertEqual(grid.shape, (1, 8, 16, 16, 1024))
        self.assertEqual(stripped, 0)

    def test_auto_strips_single_leading_token(self) -> None:
        tokens = torch.randn(1, 1 + 8 * 16 * 16, 1024)
        grid, stripped = reshape_patch_tokens(
            tokens,
            time_patches=8,
            height_patches=16,
            width_patches=16,
        )
        self.assertEqual(grid.shape, (1, 8, 16, 16, 1024))
        self.assertEqual(stripped, 1)


class SelectFrameIndicesTests(unittest.TestCase):
    def test_consecutive_frames(self) -> None:
        indices = select_frame_indices(
            video_fps=30.0,
            frame_count=200,
            num_frames=4,
            start_frame=10,
        )
        self.assertEqual(indices, [10, 11, 12, 13])

    def test_sample_fps_stride(self) -> None:
        indices = select_frame_indices(
            video_fps=30.0,
            frame_count=200,
            num_frames=4,
            start_frame=0,
            sample_fps=10.0,
        )
        self.assertEqual(indices, [0, 3, 6, 9])


if __name__ == "__main__":
    unittest.main()
