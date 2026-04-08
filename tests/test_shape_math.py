import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vjepa2_latents.gradio_components.latent_source.extractor import (
    MODEL_SPECS,
    auto_device,
    download_checkpoint_if_needed,
    estimate_extraction_requirements,
    isolate_torch_hub_imports,
    load_encoder,
    log_timing_summary,
    normalize_crop_size,
    parse_crop_size,
    prepare_display_frames,
    resolve_checkpoint_key,
    reshape_patch_tokens,
    run_encoder_synchronously,
    save_outputs,
    select_frame_indices,
)


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

    def test_timed_reshape_reports_substep_durations(self) -> None:
        from vjepa2_latents.gradio_components.latent_source.extractor import reshape_patch_tokens_with_timings

        tokens = torch.randn(1, 8 * 16 * 16, 1024)
        grid, stripped, timings = reshape_patch_tokens_with_timings(
            tokens,
            time_patches=8,
            height_patches=16,
            width_patches=16,
        )

        self.assertEqual(grid.shape, (1, 8, 16, 16, 1024))
        self.assertEqual(stripped, 0)
        self.assertIn("auto_detect_leading_tokens_seconds", timings)
        self.assertIn("validate_patch_token_count_seconds", timings)
        self.assertIn("strip_leading_tokens_seconds", timings)
        self.assertIn("rearrange_to_latent_grid_seconds", timings)
        self.assertIn("total_seconds", timings)


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


class CropSizeHelpersTests(unittest.TestCase):
    def test_model_specs_include_all_vjepa2_1_backbones(self) -> None:
        self.assertEqual(
            sorted(MODEL_SPECS),
            ["vit_base_384", "vit_giant_384", "vit_gigantic_384", "vit_large_384"],
        )
        self.assertEqual(MODEL_SPECS["vit_base_384"].arch_name, "vit_base")
        self.assertEqual(MODEL_SPECS["vit_base_384"].checkpoint_keys[0], "ema_encoder")
        self.assertEqual(MODEL_SPECS["vit_base_384"].embed_dim, 768)
        self.assertEqual(MODEL_SPECS["vit_gigantic_384"].arch_name, "vit_gigantic_xformers")
        self.assertEqual(MODEL_SPECS["vit_gigantic_384"].embed_dim, 1664)

    def test_checkpoint_key_resolution_accepts_fallback_keys(self) -> None:
        selected = resolve_checkpoint_key({"ema_encoder": {}, "predictor": {}}, ("target_encoder", "ema_encoder"))
        self.assertEqual(selected, "ema_encoder")

    def test_normalize_crop_size_accepts_rectangles(self) -> None:
        self.assertEqual(normalize_crop_size((384, 640)), (384, 640))

    def test_parse_crop_size_accepts_square_and_rectangle(self) -> None:
        self.assertEqual(parse_crop_size("384"), 384)
        self.assertEqual(parse_crop_size("384x640"), (384, 640))

    def test_prepare_display_frames_produces_requested_rectangle(self) -> None:
        frames = torch.randint(0, 255, (2, 300, 500, 3), dtype=torch.uint8).numpy()
        prepared = prepare_display_frames(frames, (256, 384))
        self.assertEqual(prepared.shape, (2, 256, 384, 3))


class AutoDeviceTests(unittest.TestCase):
    def test_auto_device_prefers_mps_on_macos(self) -> None:
        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.backends.mps.is_available", return_value=True),
            mock.patch("platform.system", return_value="Darwin"),
        ):
            self.assertEqual(auto_device("auto").type, "mps")

    def test_auto_device_falls_back_to_cpu_off_macos(self) -> None:
        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.backends.mps.is_available", return_value=True),
            mock.patch("platform.system", return_value="Linux"),
        ):
            self.assertEqual(auto_device("auto").type, "cpu")


class ExtractionEstimateTests(unittest.TestCase):
    def test_estimate_extraction_requirements_reports_high_pressure_for_50_frames(self) -> None:
        with mock.patch(
            "vjepa2_latents.gradio_components.latent_source.extractor.config.get_mps_memory_info",
            return_value={
                "recommended_max_memory": 26 * 1024**3,
                "current_allocated_memory": 0,
                "driver_allocated_memory": 0,
            },
        ):
            estimate = estimate_extraction_requirements(
                model_name="vit_base_384",
                num_frames=50,
                crop_size=(384, 384),
                device_name="mps",
            )

        self.assertEqual(estimate["token_count"], 25 * 24 * 24)
        self.assertEqual(estimate["latent_shape"], [1, 25, 24, 24, 768])
        self.assertEqual(estimate["risk_level"], "high")
        self.assertAlmostEqual(estimate["latent_tensor_mib"], 42.1875, places=3)

    def test_estimate_extraction_requirements_reports_low_pressure_for_small_clip(self) -> None:
        estimate = estimate_extraction_requirements(
            model_name="vit_base_384",
            num_frames=16,
            crop_size=(256, 256),
            device_name="cpu",
        )

        self.assertEqual(estimate["token_count"], 8 * 16 * 16)
        self.assertEqual(estimate["risk_level"], "low")


class CheckpointDownloadTests(unittest.TestCase):
    def test_invalid_cached_checkpoint_is_redownloaded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            target_path = checkpoint_dir / "vjepa2_1_vitb_dist_vitG_384.pt"
            target_path.write_bytes(b"not a real checkpoint")

            def fake_download(_url: str, destination: Path) -> None:
                torch.save({"ema_encoder": {}}, destination)

            with mock.patch(
                "vjepa2_latents.gradio_components.latent_source.extractor.checkpoint.download_checkpoint",
                side_effect=fake_download,
            ) as patched:
                resolved = download_checkpoint_if_needed("vit_base_384", None, checkpoint_dir)

            self.assertEqual(resolved, target_path)
            self.assertEqual(patched.call_count, 1)
            loaded = torch.load(resolved, map_location="cpu", weights_only=False)
            self.assertIn("ema_encoder", loaded)


class EncoderLoadTests(unittest.TestCase):
    def test_isolate_torch_hub_imports_hides_local_app_module_and_restores_state(self) -> None:
        original_sys_path = list(sys.path)
        fake_local_app = object()
        sys.modules["app"] = fake_local_app

        try:
            with isolate_torch_hub_imports():
                self.assertNotIn(Path.cwd().resolve(), [Path(path or ".").resolve() for path in sys.path])
                self.assertNotIn(ROOT.resolve(), [Path(path or ".").resolve() for path in sys.path])
                self.assertNotIn((ROOT / "src").resolve(), [Path(path or ".").resolve() for path in sys.path])
                self.assertNotIn("app", sys.modules)
            self.assertIs(sys.modules["app"], fake_local_app)
            self.assertEqual(sys.path, original_sys_path)
        finally:
            sys.modules.pop("app", None)
            sys.path[:] = original_sys_path

    def test_load_encoder_uses_torch_hub_loader(self) -> None:
        fake_encoder = torch.nn.Linear(4, 4, bias=False)
        checkpoint = {"ema_encoder": fake_encoder.state_dict()}

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)

            with mock.patch(
                "torch.hub.load",
                return_value=(fake_encoder, object()),
            ) as patched_hub_load:
                loaded_encoder = load_encoder(
                    model_name="vit_base_384",
                    num_frames=16,
                    checkpoint_path=checkpoint_path,
                    device=torch.device("cpu"),
                )

        self.assertIs(loaded_encoder, fake_encoder)
        self.assertIsInstance(loaded_encoder, torch.nn.Module)
        patched_hub_load.assert_called_once()
        self.assertEqual(patched_hub_load.call_args.args[:2], ("facebookresearch/vjepa2", "vjepa2_1_vit_base_384"))
        self.assertEqual(patched_hub_load.call_args.kwargs["pretrained"], False)
        self.assertEqual(patched_hub_load.call_args.kwargs["num_frames"], 16)


class OutputSaveTests(unittest.TestCase):
    def test_save_outputs_writes_expected_npy_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_prefix = Path(temp_dir) / "latents"
            latent_grid = torch.arange(1 * 5 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 5, 2, 3, 4)

            outputs = save_outputs(
                latent_grid=latent_grid,
                output_prefix=output_prefix,
                metadata={"ok": True},
                save_pt=False,
            )

            loaded = np.load(outputs["npy"])
            self.assertEqual(loaded.shape, (1, 5, 2, 3, 4))
            np.testing.assert_allclose(loaded, latent_grid.numpy())
            metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
            self.assertIn("timings", metadata)
            self.assertIn("output_serialization", metadata["timings"])
            self.assertIn("copy_latent_grid_to_cpu_seconds", metadata["timings"]["output_serialization"])
            self.assertIn("write_numpy_latent_grid_seconds", metadata["timings"]["output_serialization"])
            self.assertIn("write_metadata_seconds", metadata["timings"]["output_serialization"])

    def test_save_outputs_can_skip_pt_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_prefix = Path(temp_dir) / "latents"
            latent_grid = torch.zeros((1, 2, 3, 4, 5), dtype=torch.float32)
            outputs = save_outputs(
                latent_grid=latent_grid,
                output_prefix=output_prefix,
                metadata={"ok": True},
                save_pt=False,
            )

            self.assertEqual(set(outputs.keys()), {"npy", "metadata"})
            self.assertTrue(output_prefix.with_suffix(".npy").exists())
            self.assertTrue(output_prefix.with_suffix(".metadata.json").exists())
            self.assertFalse(output_prefix.with_suffix(".pt").exists())
            metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
            self.assertNotIn("write_pytorch_tensor_seconds", metadata["timings"]["output_serialization"])

    def test_save_outputs_logs_serialization_timings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_prefix = Path(temp_dir) / "latents"
            latent_grid = torch.ones((1, 2, 3, 4, 5), dtype=torch.float32)
            stderr_buffer = io.StringIO()

            with contextlib.redirect_stderr(stderr_buffer):
                save_outputs(
                    latent_grid=latent_grid,
                    output_prefix=output_prefix,
                    metadata={"ok": True},
                    save_pt=False,
                )

            stderr_text = stderr_buffer.getvalue()
            self.assertIn("Timing · output serialization total:", stderr_text)


class TimingSummaryTests(unittest.TestCase):
    def test_log_timing_summary_sorts_and_reports_overhead(self) -> None:
        stderr_buffer = io.StringIO()

        with contextlib.redirect_stderr(stderr_buffer):
            log_timing_summary(
                "Extraction timing summary",
                {
                    "decode video frames": 2.0,
                    "load encoder": 5.0,
                    "serialize outputs": 1.0,
                },
                total_seconds=10.0,
            )

        stderr_text = stderr_buffer.getvalue()
        self.assertIn("Extraction timing summary:", stderr_text)
        load_index = stderr_text.index("load encoder: 5.000s")
        decode_index = stderr_text.index("decode video frames: 2.000s")
        serialize_index = stderr_text.index("serialize outputs: 1.000s")
        self.assertLess(load_index, decode_index)
        self.assertLess(decode_index, serialize_index)
        self.assertIn("unaccounted overhead: 2.000s", stderr_text)


class EncoderRunTimingTests(unittest.TestCase):
    def test_run_encoder_synchronously_returns_tensor_and_duration(self) -> None:
        encoder = mock.Mock(return_value=torch.ones((1, 4, 8), dtype=torch.float32))
        video_tensor = torch.zeros((1, 3, 4, 16, 16), dtype=torch.float32)

        tokens, encoder_seconds = run_encoder_synchronously(
            encoder,
            video_tensor,
            torch.device("cpu"),
        )

        self.assertEqual(tokens.shape, (1, 4, 8))
        self.assertGreaterEqual(encoder_seconds, 0.0)
        encoder.assert_called_once()


if __name__ == "__main__":
    unittest.main()
