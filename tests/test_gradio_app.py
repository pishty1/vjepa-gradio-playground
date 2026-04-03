from pathlib import Path
import sys
import unittest
import numpy as np
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from vjepa2_latents.gradio_app import (
    DEFAULT_CROP_HEIGHT,
    DEFAULT_CROP_WIDTH,
    DEFAULT_MODEL_NAME,
    _clean_latent_metadata_for_ui,
    build_plot_step,
    create_rgb_videos_step,
    compute_projection_step,
    prepare_tracking_step,
    select_patch_similarity_step,
    _summarize_timings_for_ui,
    extract_latents_step,
    toggle_projection_controls,
)


class GradioMetadataCleanupTests(unittest.TestCase):
    def test_gradio_defaults_prefer_base_model_and_384_crop(self) -> None:
        self.assertEqual(DEFAULT_MODEL_NAME, "vit_base_384")
        self.assertEqual(DEFAULT_CROP_HEIGHT, 384)
        self.assertEqual(DEFAULT_CROP_WIDTH, 384)

    def test_summarize_timings_for_ui_keeps_only_high_value_fields(self) -> None:
        timings = {
            "encoder_forward_pass": {
                "device_executes_asynchronously": True,
                "measured_synchronously": True,
                "forward_run_seconds": 1.7513040419726167,
                "total_wall_seconds": 1.7513040419726167,
                "sync_before_seconds": 0.001,
            },
            "reshape_patch_tokens": {
                "auto_detect_leading_tokens_seconds": 0.0000002,
                "total_seconds": 0.000311291,
            },
            "output_serialization": {
                "copy_latent_grid_to_cpu_seconds": 0.01019,
                "total_seconds": 0.027461707999464124,
            },
            "encoder_setup": {
                "ensure_vendor_imports_seconds": 0.00019,
                "total_seconds": 4.331428250006866,
            },
            "major_phases": {
                "probe video metadata": 0.1185,
                "select frame indices": 0.000002,
                "decode video frames": 0.9773,
                "finalize metadata write": 0.000145,
            },
            "total_extraction_seconds": 7.9271589999843854,
        }

        summarized = _summarize_timings_for_ui(timings)

        self.assertEqual(
            summarized["encoder_forward_pass"],
            {
                "device_executes_asynchronously": True,
                "measured_synchronously": True,
                "forward_run_seconds": 1.751,
                "total_wall_seconds": 1.751,
            },
        )
        self.assertEqual(summarized["reshape_patch_tokens"], {"total_seconds": 0.000311})
        self.assertEqual(summarized["output_serialization"], {"total_seconds": 0.027})
        self.assertEqual(summarized["encoder_setup"], {"total_seconds": 4.331})
        self.assertEqual(
            summarized["major_phases"],
            {
                "probe video metadata": 0.118,
                "decode video frames": 0.977,
            },
        )
        self.assertEqual(summarized["total_extraction_seconds"], 7.927)

    def test_clean_latent_metadata_for_ui_replaces_verbose_timings(self) -> None:
        metadata = {
            "latent_grid_shape": [1, 3, 48, 48, 768],
            "timings": {
                "reshape_patch_tokens": {
                    "auto_detect_leading_tokens_seconds": 0.0000002,
                    "total_seconds": 0.000311291,
                }
            },
            "notes": ["kept"],
        }

        cleaned = _clean_latent_metadata_for_ui(metadata)

        self.assertEqual(cleaned["latent_grid_shape"], [1, 3, 48, 48, 768])
        self.assertEqual(cleaned["notes"], ["kept"])
        self.assertEqual(cleaned["timings"], {"reshape_patch_tokens": {"total_seconds": 0.000311}})

    def test_extract_latents_step_keeps_video_outside_session_dir(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            video_path = temp_path / "example.mp4"
            video_path.write_bytes(b"not a real video, this is only for path handling")
            session_dir = temp_path / "session"
            session_dir.mkdir()

            fake_result = {
                "video_path": str(video_path),
                "timings": {},
                "outputs": {},
            }

            with (
                patch("vjepa2_latents.gradio_app._resolve_video_path", return_value=video_path),
                patch("vjepa2_latents.gradio_app._create_session_dir", return_value=session_dir),
                patch("vjepa2_latents.gradio_app.extract_latents", return_value=fake_result),
                patch("vjepa2_latents.gradio_app.load_saved_latents", return_value=(np.zeros((1, 1, 1, 1, 3), dtype=np.float32), {"latent_grid_shape": [1, 1, 1, 1, 3], "frame_indices": [0], "video_path": str(video_path), "video_metadata": {"fps": 24.0}, "tubelet_size": 2, "crop_size": [256, 256]})),
            ):
                status, latent_prefix, metadata_json, latent_state, *_ = extract_latents_step(
                    video_file=str(video_path),
                    model_name="vit_large_384",
                    crop_height=256,
                    crop_width=256,
                    num_frames=16,
                    sample_fps=None,
                    start_second=0,
                    device_name="cpu",
                )

            self.assertIn("Latents extracted", status)
            self.assertEqual(latent_prefix, str(session_dir / "latents"))
            self.assertEqual(latent_state["output_prefix"], str(session_dir / "latents"))
            self.assertFalse((session_dir / video_path.name).exists())
            self.assertIn(f'"video_name": "{video_path.name}"', metadata_json)

    def test_plot_and_render_steps_use_cached_state(self) -> None:
        projection = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        coordinates = np.array([[0, 0, 0]], dtype=np.int32)
        latent_state = {
            "output_prefix": "/tmp/latents",
            "latent_grid": np.zeros((1, 1, 1, 1, 3), dtype=np.float32),
            "metadata": {
                "video_path": "/tmp/example.mp4",
                "video_metadata": {"fps": 24.0},
                "frame_indices": [0, 2],
                "tubelet_size": 2,
                "crop_size": [384, 384],
                "latent_grid_shape": [1, 1, 1, 1, 3],
            },
        }
        projection_state = {
            "output_prefix": "/tmp/projection",
            "projection": projection,
            "coordinates": coordinates,
            "metadata": {
                "method": "pca",
                "latent_grid_shape": [1, 1, 1, 1, 3],
                "component_labels": ["PC1", "PC2", "PC3"],
                "latent_output_prefix": "/tmp/latents",
                "settings": {"n_components": 3},
            },
        }

        dummy_artifacts = SimpleNamespace(
            latent_video_path=Path("/tmp/latent.mp4"),
            side_by_side_video_path=Path("/tmp/side-by-side.mp4"),
            latent_video_shape=(1, 1, 1, 3),
            side_by_side_video_shape=(1, 1, 1, 3),
            display_fps=24.0,
        )

        with patch("vjepa2_latents.gradio_app.load_saved_projection", side_effect=AssertionError("disk should not be hit")):
            figure, plot_status = build_plot_step(projection_state, 2, 500, 1, 2, None)
            self.assertIsNotNone(figure)
            self.assertIn("Plot updated", plot_status)

        with (
            patch("vjepa2_latents.gradio_app.load_saved_projection", side_effect=AssertionError("disk should not be hit")),
            patch("vjepa2_latents.gradio_app._load_latent_metadata", side_effect=AssertionError("disk should not be hit")),
            patch("vjepa2_latents.gradio_app.create_visualizations_from_projection", return_value=dummy_artifacts),
        ):
            status, video_path, payload_json = create_rgb_videos_step(latent_state, projection_state, 1, 2, 3, 2)

        self.assertIn("RGB videos created", status)
        self.assertEqual(video_path, str(dummy_artifacts.side_by_side_video_path))
        self.assertIn('"display_fps": 24.0', payload_json)

    def test_toggle_projection_controls_shows_pca_and_umap_groups(self) -> None:
        umap_update, pca_update = toggle_projection_controls("PCA")

        self.assertFalse(umap_update["visible"])
        self.assertTrue(pca_update["visible"])

        umap_update, pca_update = toggle_projection_controls("UMAP")

        self.assertTrue(umap_update["visible"])
        self.assertFalse(pca_update["visible"])

    def test_compute_projection_step_passes_selected_pca_mode(self) -> None:
        latent_state = {
            "output_prefix": "/tmp/latents",
            "latent_grid": np.zeros((1, 2, 2, 2, 4), dtype=np.float32),
        }
        artifacts = SimpleNamespace(
            projection_path=Path("/tmp/projection.projection.npz"),
            metadata_path=Path("/tmp/projection.projection.metadata.json"),
            projection_shape=(8, 2),
            method="pca",
            component_labels=("PC1", "PC2"),
        )

        with patch("vjepa2_latents.gradio_app.compute_projection_bundle") as bundle_mock:
            bundle_mock.return_value = {
                "projection": np.zeros((8, 2), dtype=np.float32),
                "coordinates": np.zeros((8, 3), dtype=np.int32),
                "method": "pca",
                "method_label": "Spatial-only PCA",
                "latent_grid_shape": [1, 2, 2, 2, 4],
                "component_labels": ["PC1", "PC2"],
                "explained_variance": [1.0, 0.0],
                "settings": {"method": "pca", "method_label": "Spatial-only PCA", "pca_mode": "spatial", "n_components": 2, "umap_n_neighbors": 15, "umap_min_dist": 0.1, "umap_metric": "euclidean", "umap_random_state": 42, "projection_backend": "numpy-svd"},
            }
            with patch("vjepa2_latents.gradio_app.save_projection_artifacts", return_value=artifacts):
                compute_projection_step(latent_state, "PCA", "spatial", 2, 15, 0.1, "euclidean", 42)

        self.assertEqual(bundle_mock.call_args.kwargs["pca_mode"], "spatial")

    def test_prepare_tracking_step_returns_first_frame_and_state(self) -> None:
        latent_state = {
            "output_prefix": "/tmp/latents",
            "latent_grid": np.zeros((1, 2, 4, 4, 3), dtype=np.float32),
            "metadata": {
                "video_path": "/tmp/example.mp4",
                "frame_indices": [0, 2, 4, 6],
                "video_metadata": {"fps": 24.0},
                "tubelet_size": 2,
                "crop_size": [384, 384],
            },
        }
        source_frames = np.zeros((2, 48, 48, 3), dtype=np.uint8)

        with patch("vjepa2_latents.gradio_app.load_aligned_source_frames", return_value=(source_frames, 6.0)):
            preview, status, metadata_json, tracking_state, video_path = prepare_tracking_step(latent_state)

        self.assertEqual(preview.shape, (48, 48, 3))
        self.assertIn("Patch similarity ready", status)
        self.assertIn('"display_fps": 6.0', metadata_json)
        self.assertEqual(tracking_state["latent_output_prefix"], "/tmp/latents")
        self.assertIsNone(video_path)

    def test_select_patch_similarity_step_uses_cached_tracking_frames(self) -> None:
        latent_state = {
            "output_prefix": "/tmp/latents",
            "latent_grid": np.zeros((1, 2, 4, 4, 3), dtype=np.float32),
            "metadata": {
                "video_path": "/tmp/example.mp4",
                "frame_indices": [0, 2, 4, 6],
                "video_metadata": {"fps": 24.0},
                "tubelet_size": 2,
                "crop_size": [384, 384],
            },
        }
        tracking_state = {
            "latent_output_prefix": "/tmp/latents",
            "source_frames": np.zeros((2, 64, 64, 3), dtype=np.uint8),
            "display_fps": 6.0,
            "latent_grid_shape": [1, 2, 4, 4, 3],
        }
        artifacts = SimpleNamespace(
            similarity_video_path=Path("/tmp/patch_similarity.mp4"),
            similarity_video_shape=(2, 64, 64, 3),
            display_fps=6.0,
            token_index=(0, 1, 2),
            similarity_min=-0.25,
            similarity_max=1.0,
        )

        with (
            patch("vjepa2_latents.gradio_app.load_aligned_source_frames", side_effect=AssertionError("cached frames should be reused")),
            patch("vjepa2_latents.gradio_app.create_patch_similarity_video", return_value=artifacts) as create_mock,
        ):
            preview, status, video_path, metadata_json, next_state = select_patch_similarity_step(
                latent_state,
                tracking_state,
                SimpleNamespace(index=(33, 17)),
            )

        self.assertEqual(create_mock.call_args.kwargs["token_index"], (0, 1, 2))
        self.assertEqual(preview.shape, (64, 64, 3))
        self.assertIn("Patch similarity video ready", status)
        self.assertEqual(video_path, "/tmp/patch_similarity.mp4")
        self.assertIn('"w": 2', metadata_json)
        self.assertEqual(next_state["selected_token"], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()