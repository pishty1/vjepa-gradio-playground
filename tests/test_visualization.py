from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gradio_components.plot import (
    build_projection_figure,
    build_projection_figure_from_data,
)
from gradio_components.projection import (
    compute_mlx_projection,
    compute_pca_projection,
    compute_projection_bundle,
    compute_umap_projection,
    has_umap_support,
    load_saved_projection,
    projection_method_display_name,
    save_projection_artifacts,
)
from gradio_components.render import (
    _ensure_even_frame_size,
    infer_latent_fps,
    latent_rgb_frames,
    projection_rgb_frames,
    side_by_side_frames,
)
from gradio_components.segmentation import (
    annotate_prompt_points,
    knn_binary_segmentation_volume,
    segmentation_mask_frames,
)
from gradio_components.tracking import (
    annotate_selected_patch,
    cosine_similarity_volume,
    map_click_to_latent_token,
    similarity_heatmap_frames,
)
from gradio_components.tumbling_window import (
    build_tumbling_window_heatmap_figure,
    compare_overlapping_latent_windows,
    derive_tumbling_window_ranges,
)


class PcaProjectionTests(unittest.TestCase):
    def test_projection_shape_and_variance(self) -> None:
        features = np.arange(60, dtype=np.float32).reshape(12, 5)
        projection, explained = compute_pca_projection(features, n_components=3)
        self.assertEqual(projection.shape, (12, 3))
        self.assertEqual(explained.shape, (3,))
        self.assertTrue(np.all(explained >= 0))

    def test_spatial_and_temporal_pca_modes_reduce_different_axes(self) -> None:
        latent_grid = np.arange(1 * 3 * 2 * 4 * 6, dtype=np.float32).reshape(1, 3, 2, 4, 6)

        spatial_bundle = compute_projection_bundle(latent_grid, method="pca", pca_mode="spatial", n_components=2)
        temporal_bundle = compute_projection_bundle(latent_grid, method="pca", pca_mode="temporal", n_components=2)

        self.assertEqual(spatial_bundle["projection"].shape, (8, 2))
        self.assertEqual(spatial_bundle["coordinates"].shape, (8, 3))
        self.assertTrue(np.all(spatial_bundle["coordinates"][:, 0] == 0))
        self.assertEqual(spatial_bundle["method_label"], "Spatial-only PCA")

        self.assertEqual(temporal_bundle["projection"].shape, (3, 2))
        self.assertEqual(temporal_bundle["coordinates"].shape, (3, 3))
        self.assertTrue(np.all(temporal_bundle["coordinates"][:, 1:] == 0))
        self.assertEqual(temporal_bundle["method_label"], "Temporal-only PCA")


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

    def test_builds_animated_projection_figure_with_time_frames(self) -> None:
        latent_grid = np.random.default_rng(9).normal(size=(1, 3, 2, 2, 5)).astype(np.float32)
        bundle = compute_projection_bundle(latent_grid, method="pca", n_components=3)
        figure = build_projection_figure_from_data(
            bundle["projection"],
            bundle["coordinates"],
            method="pca",
            component_indices=(0, 1),
            component_labels=bundle["component_labels"],
            max_points=50,
            animate_over_time=True,
        )
        self.assertEqual(len(figure.data), 1)
        self.assertEqual(figure.data[0].type, "scatter")
        self.assertEqual(len(figure.frames), 3)
        self.assertTrue(figure.layout.sliders)
        self.assertEqual(tuple(figure.frames[0].traces), (0,))
        self.assertEqual(figure.frames[0].name, "t=0")
        self.assertEqual(len(figure.data[0].x), len(figure.frames[0].data[0].x))


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


class MlxProjectionTests(unittest.TestCase):
    def test_missing_mlx_vis_dependency_raises_clear_error(self) -> None:
        features = np.random.default_rng(7).normal(size=(16, 8)).astype(np.float32)
        with mock.patch("gradio_components.projection.core.has_mlx_vis_support", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "mlx-vis"):
                compute_mlx_projection(features, method="umap_mlx", n_components=2)

    def test_projection_bundle_supports_mocked_mlx_reducers(self) -> None:
        latent_grid = np.random.default_rng(8).normal(size=(1, 2, 3, 4, 6)).astype(np.float32)

        class FakeTSNE:
            def __init__(self, n_components: int):
                self.n_components = n_components

            def fit_transform(self, features: np.ndarray) -> np.ndarray:
                rows = features.shape[0]
                return np.arange(rows * self.n_components, dtype=np.float32).reshape(rows, self.n_components)

        fake_module = type("FakeMlxVis", (), {"TSNE": FakeTSNE})

        with (
            mock.patch("gradio_components.projection.core.has_mlx_vis_support", return_value=True),
            mock.patch("gradio_components.projection.core.importlib.import_module", return_value=fake_module),
        ):
            bundle = compute_projection_bundle(
                latent_grid,
                method="TSNE-MLX",
                n_components=2,
                umap_n_neighbors=99,
                umap_min_dist=0.25,
                umap_metric="cosine",
                umap_random_state=17,
            )

        self.assertEqual(bundle["method"], "tsne_mlx")
        self.assertEqual(bundle["method_label"], "t-SNE-MLX")
        self.assertEqual(bundle["projection"].shape, (24, 2))
        self.assertEqual(bundle["settings"]["projection_backend"], "mlx-vis")
        self.assertEqual(bundle["component_labels"], ["t-SNE-MLX1", "t-SNE-MLX2"])


class ProjectionMethodNamingTests(unittest.TestCase):
    def test_display_names_are_human_friendly(self) -> None:
        self.assertEqual(projection_method_display_name("pca"), "PCA")
        self.assertEqual(projection_method_display_name("UMAP-MLX"), "UMAP-MLX")
        self.assertEqual(projection_method_display_name("TSNE-MLX"), "t-SNE-MLX")


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


class PatchSimilarityTests(unittest.TestCase):
    def test_maps_click_to_latent_token_coordinates(self) -> None:
        token_index = map_click_to_latent_token((95, 63), (64, 96, 3), (1, 2, 4, 6, 8))
        self.assertEqual(token_index, (0, 3, 5))

    def test_maps_swapped_click_coordinates_to_the_visible_patch(self) -> None:
        token_index = map_click_to_latent_token((20, 50), (64, 96, 3), (1, 2, 4, 6, 8))
        self.assertEqual(token_index, (0, 3, 1))

    def test_computes_cosine_similarity_volume(self) -> None:
        latent_grid = np.array(
            [[[[[1.0, 0.0], [0.0, 1.0]]], [[[1.0, 0.0], [-1.0, 0.0]]]]],
            dtype=np.float32,
        )

        similarity = cosine_similarity_volume(latent_grid, (0, 0, 0))

        self.assertEqual(similarity.shape, (2, 1, 2))
        self.assertAlmostEqual(float(similarity[0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(similarity[0, 0, 1]), 0.0)
        self.assertAlmostEqual(float(similarity[1, 0, 0]), 1.0)
        self.assertAlmostEqual(float(similarity[1, 0, 1]), -1.0)

    def test_builds_similarity_overlay_and_annotation(self) -> None:
        source_frames = np.full((2, 32, 48, 3), 120, dtype=np.uint8)
        similarity = np.array(
            [
                [[1.0, 0.0], [0.0, -1.0]],
                [[0.5, 0.25], [-0.25, -0.5]],
            ],
            dtype=np.float32,
        )

        overlay = similarity_heatmap_frames(source_frames, similarity, alpha=0.5)
        annotated = annotate_selected_patch(overlay[0], (0, 1, 1), (1, 2, 2, 2, 4))
        hotspot_delta = np.abs(overlay[0, 4, 4].astype(np.int16) - source_frames[0, 4, 4].astype(np.int16)).sum()
        background_delta = np.abs(overlay[0, 28, 44].astype(np.int16) - source_frames[0, 28, 44].astype(np.int16)).sum()

        self.assertEqual(overlay.shape, source_frames.shape)
        self.assertEqual(overlay.dtype, np.uint8)
        self.assertGreater(hotspot_delta, background_delta)
        self.assertEqual(annotated.shape, overlay[0].shape)
        self.assertFalse(np.array_equal(annotated, overlay[0]))


class TumblingWindowTests(unittest.TestCase):
    def test_derives_tumbling_window_ranges_from_overlap_steps(self) -> None:
        ranges = derive_tumbling_window_ranges(
            start_frame=50,
            window_frames=20,
            overlap_latent_steps=1,
            tubelet_size=2,
            available_frames=200,
        )

        self.assertEqual(ranges.left_start, 50)
        self.assertEqual(ranges.right_start, 68)
        self.assertEqual(ranges.overlap_start_frame, 68)
        self.assertEqual(ranges.overlap_end_frame, 69)
        self.assertEqual(ranges.overlap_frames, 2)

    def test_compare_overlapping_latent_windows_reports_similarity_metrics(self) -> None:
        left_latent = np.zeros((1, 4, 1, 1, 2), dtype=np.float32)
        right_latent = np.zeros((1, 4, 1, 1, 2), dtype=np.float32)
        left_latent[0, 2, 0, 0] = np.array([1.0, 0.0], dtype=np.float32)
        left_latent[0, 3, 0, 0] = np.array([0.0, 1.0], dtype=np.float32)
        right_latent[0, 0, 0, 0] = np.array([1.0, 0.0], dtype=np.float32)
        right_latent[0, 1, 0, 0] = np.array([0.0, 1.0], dtype=np.float32)

        result = compare_overlapping_latent_windows(
            left_latent,
            right_latent,
            left_start=0,
            right_start=4,
            tubelet_size=2,
        )

        self.assertTrue(result["comparison"]["allclose_overlapping"])
        self.assertEqual(result["comparison"]["overlap_latent_steps"], 2)
        self.assertAlmostEqual(result["comparison"]["mean_token_cosine_similarity_overlapping"], 1.0)
        self.assertEqual(result["difference_overlap"].shape, (1, 2, 1, 1, 2))

    def test_build_tumbling_window_heatmap_figure_animates_time_slices(self) -> None:
        difference_overlap = np.arange(1 * 2 * 2 * 3 * 4, dtype=np.float32).reshape(1, 2, 2, 3, 4)

        figure = build_tumbling_window_heatmap_figure(difference_overlap)

        self.assertEqual(figure.data[0].type, "heatmap")
        self.assertEqual(len(figure.frames), 2)
        self.assertTrue(figure.layout.sliders)

    def test_knn_binary_segmentation_volume_separates_foreground_and_background(self) -> None:
        latent_grid = np.array(
            [
                [
                    [
                        [[1.0, 0.0], [0.9, 0.1]],
                        [[0.0, 1.0], [0.1, 0.9]],
                    ],
                    [
                        [[0.95, 0.05], [0.85, 0.15]],
                        [[0.05, 0.95], [0.15, 0.85]],
                    ],
                ]
            ],
            dtype=np.float32,
        )

        segmentation = knn_binary_segmentation_volume(latent_grid, (0, 0, 0), (0, 1, 0), k_neighbors=1)

        self.assertEqual(segmentation.shape, (2, 2, 2))
        self.assertTrue(bool(segmentation[0, 0, 0]))
        self.assertTrue(bool(segmentation[1, 0, 1]))
        self.assertFalse(bool(segmentation[0, 1, 0]))
        self.assertFalse(bool(segmentation[1, 1, 1]))

    def test_segmentation_mask_frames_and_prompt_annotation(self) -> None:
        source_frames = np.full((2, 32, 48, 3), 120, dtype=np.uint8)
        segmentation = np.array(
            [
                [[1, 1], [0, 0]],
                [[0, 1], [0, 1]],
            ],
            dtype=bool,
        )

        overlay = segmentation_mask_frames(source_frames, segmentation, alpha=0.5)
        annotated = annotate_prompt_points(overlay[0], {"foreground": (8, 8), "background": (36, 24)})

        self.assertEqual(overlay.shape, source_frames.shape)
        self.assertEqual(overlay.dtype, np.uint8)
        self.assertFalse(np.array_equal(overlay[0, 4, 4], source_frames[0, 4, 4]))
        self.assertFalse(np.array_equal(annotated, overlay[0]))


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

    def test_write_video_uses_ffmpeg_h264_yuv420p(self) -> None:
        frames = np.zeros((2, 10, 12, 3), dtype=np.uint8)

        with mock.patch("gradio_components.render.video._resolve_ffmpeg_executable", return_value="ffmpeg"):
            with mock.patch("gradio_components.render.video.subprocess.run") as run_mock:
                run_mock.return_value = mock.Mock(returncode=0, stderr=b"")
                output = _ensure_even_frame_size(frames)
                self.assertEqual(output.shape, (2, 10, 12, 3))
                from gradio_components.render import write_video

                with tempfile.TemporaryDirectory() as temp_dir:
                    video_path = Path(temp_dir) / "test.mp4"
                    write_video(video_path, frames, fps=24)

        args, kwargs = run_mock.call_args
        command = args[0]
        self.assertIn("ffmpeg", command[0])
        self.assertIn("-c:v", command)
        self.assertIn("libx264", command)
        self.assertIn("-pix_fmt", command)
        self.assertIn("yuv420p", command)
        self.assertEqual(kwargs["check"], False)
        self.assertEqual(kwargs["input"], frames.tobytes())


if __name__ == "__main__":
    unittest.main()
