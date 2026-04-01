from pathlib import Path
import sys
import unittest

import numpy as np
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vjepa2_latents.visualization import (
    _ensure_even_frame_size,
    build_projection_figure,
    build_projection_figure_from_data,
    compute_projection_bundle,
    compute_pca_projection,
    compute_umap_projection,
    has_umap_support,
    infer_latent_fps,
    latent_rgb_frames,
    load_saved_projection,
    projection_rgb_frames,
    save_projection_artifacts,
    side_by_side_frames,
)


class PcaProjectionTests(unittest.TestCase):
    def test_projection_shape_and_variance(self) -> None:
        features = np.arange(60, dtype=np.float32).reshape(12, 5)
        projection, explained = compute_pca_projection(features, n_components=3)
        self.assertEqual(projection.shape, (12, 3))
        self.assertEqual(explained.shape, (3,))
        self.assertTrue(np.all(explained >= 0))


class ProjectionFigureTests(unittest.TestCase):
    def test_builds_2d_pca_figure(self) -> None:
        latent_grid = np.random.default_rng(1).normal(size=(1, 3, 4, 5, 6)).astype(np.float32)
        figure = build_projection_figure(latent_grid, method="pca", n_components=2, max_points=50)
        self.assertEqual(len(figure.data), 1)
        self.assertEqual(figure.data[0].type, "scatter")

    def test_builds_3d_figure_from_saved_projection_data(self) -> None:
        latent_grid = np.random.default_rng(3).normal(size=(1, 2, 3, 4, 7)).astype(np.float32)
        bundle = compute_projection_bundle(latent_grid, method="pca", n_components=5)
        figure = build_projection_figure_from_data(
            bundle["projection"],
            bundle["coordinates"],
            method="pca",
            component_indices=(0, 2, 4),
            component_labels=bundle["component_labels"],
            max_points=50,
        )
        self.assertEqual(len(figure.data), 1)
        self.assertEqual(figure.data[0].type, "scatter3d")


class ProjectionArtifactTests(unittest.TestCase):
    def test_round_trips_projection_artifacts(self) -> None:
        latent_grid = np.random.default_rng(4).normal(size=(1, 2, 3, 4, 6)).astype(np.float32)
        bundle = compute_projection_bundle(latent_grid, method="pca", n_components=4)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_prefix = Path(temp_dir) / "projection"
            artifacts = save_projection_artifacts(output_prefix, bundle)
            projection, coordinates, metadata = load_saved_projection(output_prefix)

        self.assertEqual(tuple(projection.shape), artifacts.projection_shape)
        self.assertEqual(coordinates.shape[1], 3)
        self.assertEqual(metadata["method"], "pca")
        self.assertEqual(len(metadata["component_labels"]), projection.shape[1])


@unittest.skipIf(not has_umap_support(), "umap-learn not installed")
class UmapProjectionTests(unittest.TestCase):
    def test_projection_shape(self) -> None:
        features = np.random.default_rng(2).normal(size=(32, 8)).astype(np.float32)
        projection = compute_umap_projection(features, n_components=2, n_neighbors=10, min_dist=0.05)
        self.assertEqual(projection.shape, (32, 2))


class LatentRgbFramesTests(unittest.TestCase):
    def test_latent_rgb_frames_shape(self) -> None:
        latent_grid = np.random.default_rng(0).normal(size=(1, 2, 3, 4, 6)).astype(np.float32)
        frames = latent_rgb_frames(latent_grid, upscale_factor=2)
        self.assertEqual(frames.shape, (2, 6, 8, 3))
        self.assertEqual(frames.dtype, np.uint8)

    def test_projection_rgb_frames_use_selected_components(self) -> None:
        latent_grid = np.random.default_rng(5).normal(size=(1, 2, 3, 4, 6)).astype(np.float32)
        bundle = compute_projection_bundle(latent_grid, method="pca", n_components=5)
        frames = projection_rgb_frames(bundle["projection"], latent_grid.shape, rgb_components=(1, 2, 4), upscale_factor=2)
        self.assertEqual(frames.shape, (2, 6, 8, 3))
        self.assertEqual(frames.dtype, np.uint8)


class SideBySideFramesTests(unittest.TestCase):
    def test_side_by_side_layout(self) -> None:
        source_frames = np.zeros((2, 32, 48, 3), dtype=np.uint8)
        latent_frames = np.zeros((2, 16, 24, 3), dtype=np.uint8)
        combined = side_by_side_frames(source_frames, latent_frames)
        self.assertEqual(combined.shape[0], 2)
        self.assertEqual(combined.shape[3], 3)
        self.assertGreater(combined.shape[1], 32)
        self.assertGreater(combined.shape[2], 96)

    def test_side_by_side_can_use_bounded_panel_size(self) -> None:
        source_frames = np.zeros((1, 2160, 4096, 3), dtype=np.uint8)
        latent_frames = np.zeros((1, 1152, 1152, 3), dtype=np.uint8)
        combined = side_by_side_frames(source_frames, latent_frames, panel_size=(1152, 1152))
        self.assertEqual(combined.shape, (1, 1204, 2352, 3))
        
    def test_preserves_source_aspect_ratio_with_padding(self) -> None:
        source_frames = np.full((1, 20, 60, 3), 100, dtype=np.uint8)
        latent_frames = np.full((1, 60, 60, 3), 200, dtype=np.uint8)
        combined = side_by_side_frames(source_frames, latent_frames)
        
        source_panel = combined[0, 36:96, 16:76]
        self.assertTrue(np.all(source_panel[:15, :, :] == 245))
        self.assertTrue(np.all(source_panel[-15:, :, :] == 245))
        self.assertTrue(np.any(source_panel[20:40, :, :] == 100))


class InferLatentFpsTests(unittest.TestCase):
    def test_uses_frame_stride(self) -> None:
        fps = infer_latent_fps([0, 2, 4, 6], source_fps=24.0, tubelet_size=2)
        self.assertAlmostEqual(fps, 6.0)


class VideoWriteHelpersTests(unittest.TestCase):
    def test_ensure_even_frame_size_pads_odd_dimensions(self) -> None:
        frames = np.zeros((2, 11, 13, 3), dtype=np.uint8)
        padded = _ensure_even_frame_size(frames)
        self.assertEqual(padded.shape, (2, 12, 14, 3))


if __name__ == "__main__":
    unittest.main()
