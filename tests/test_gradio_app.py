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
    _summarize_timings_for_ui,
    extract_latents_step,
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


if __name__ == "__main__":
    unittest.main()