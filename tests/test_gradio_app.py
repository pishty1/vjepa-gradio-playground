from pathlib import Path
import sys
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from vjepa2_latents.gradio_app import _clean_latent_metadata_for_ui, _summarize_timings_for_ui, extract_latents_step


class GradioMetadataCleanupTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()