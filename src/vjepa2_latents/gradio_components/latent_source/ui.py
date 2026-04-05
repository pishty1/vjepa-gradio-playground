from __future__ import annotations

from typing import Any

import gradio as gr

from .catalog import saved_latent_choices
from .config import (
    APP_OUTPUT_DIR,
    DEFAULT_CROP_HEIGHT,
    DEFAULT_CROP_WIDTH,
    DEFAULT_DEVICE,
    DEFAULT_MODEL_NAME,
    DEFAULT_VIDEO,
)


def build_latent_source_section(*, model_choices: list[tuple[str, str]]) -> dict[str, Any]:
    saved_latent_options = saved_latent_choices(APP_OUTPUT_DIR)

    gr.Markdown("## 1. Choose latent source")
    latent_source_mode_input = gr.Radio(
        choices=[("Extract new latents", "extract"), ("Load saved latents", "load")],
        value="extract",
        label="Latent source",
    )

    with gr.Group(visible=True) as extract_source_group:
        with gr.Row():
            video_input = gr.Video(label="Input video", sources=["upload"])
            with gr.Column():
                model_input = gr.Dropdown(choices=model_choices, value=DEFAULT_MODEL_NAME, label="Model")
                crop_height_input = gr.Slider(minimum=256, maximum=768, step=128, value=DEFAULT_CROP_HEIGHT, label="Crop height")
                crop_width_input = gr.Slider(minimum=256, maximum=768, step=128, value=DEFAULT_CROP_WIDTH, label="Crop width")
                frames_input = gr.Slider(minimum=4, maximum=64, step=2, value=16, label="Frames")
                sample_fps_input = gr.Number(value=0, label="Sample FPS (0 = consecutive frames)")
                start_second_input = gr.Number(value=0, label="Start second")
                device_input = gr.Dropdown(choices=["auto", "cpu", "mps", "cuda"], value=DEFAULT_DEVICE, label="Device")
                gr.Markdown(
                    "First use of a model downloads its checkpoint into `checkpoints/`; later runs reuse the cached file. "
                    "On macOS with Apple Silicon, the default device is `mps`."
                )
                estimate_button = gr.Button("Estimate system fit")
                extract_button = gr.Button("Extract latents", variant="primary")
        if DEFAULT_VIDEO.exists():
            gr.Examples(examples=[[str(DEFAULT_VIDEO)]], inputs=[video_input], label="Example videos")

    extraction_status_output = gr.Markdown()
    preflight_status_output = gr.Markdown()
    with gr.Group(visible=False) as load_source_group:
        with gr.Row():
            saved_latent_input = gr.Dropdown(
                choices=saved_latent_options,
                value=saved_latent_options[0][1] if saved_latent_options else None,
                label="Saved latent runs",
                info="Recent runs include timestamp, video, frame count, crop, and latent grid info.",
            )
            refresh_saved_latents_button = gr.Button("Refresh saved runs")
        with gr.Accordion("Import latent files instead", open=False):
            with gr.Row():
                latent_npy_input = gr.File(label="Latent grid (.npy)", file_types=[".npy"], type="filepath")
                latent_metadata_input = gr.File(label="Latent metadata (.json)", file_types=[".json"], type="filepath")
        load_latents_button = gr.Button("Load latents", variant="primary")

    latent_prefix_input = gr.Textbox(label="Active latent prefix", interactive=False)
    with gr.Accordion("Extraction metadata", open=False):
        extraction_metadata_output = gr.Code(label="Extraction metadata", language="json", value="{}")
    with gr.Accordion("System fit estimate", open=False):
        preflight_metadata_output = gr.Code(label="System fit estimate", language="json", value="{}")
    latent_status_output = gr.Markdown()
    with gr.Accordion("Latent summary", open=False):
        latent_metadata_output = gr.Code(label="Latent summary", language="json", value="{}")

    return {
        "latent_source_mode_input": latent_source_mode_input,
        "extract_source_group": extract_source_group,
        "load_source_group": load_source_group,
        "video_input": video_input,
        "model_input": model_input,
        "crop_height_input": crop_height_input,
        "crop_width_input": crop_width_input,
        "frames_input": frames_input,
        "sample_fps_input": sample_fps_input,
        "start_second_input": start_second_input,
        "device_input": device_input,
        "estimate_button": estimate_button,
        "extract_button": extract_button,
        "saved_latent_input": saved_latent_input,
        "refresh_saved_latents_button": refresh_saved_latents_button,
        "latent_npy_input": latent_npy_input,
        "latent_metadata_input": latent_metadata_input,
        "load_latents_button": load_latents_button,
        "latent_prefix_input": latent_prefix_input,
        "extraction_status_output": extraction_status_output,
        "preflight_status_output": preflight_status_output,
        "extraction_metadata_output": extraction_metadata_output,
        "preflight_metadata_output": preflight_metadata_output,
        "latent_status_output": latent_status_output,
        "latent_metadata_output": latent_metadata_output,
    }
