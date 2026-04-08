from __future__ import annotations

from typing import Any

import gradio as gr

from gradio_utils import _format_hint_status


def build_segmentation_tab() -> dict[str, Any]:
    with gr.Tab("Foreground / Background VOS"):
        gr.Markdown(
            "Click one foreground point and one background point on the selected frame, then classify every latent token with a tiny KNN in latent space to render a binary segmentation mask video."
        )
        with gr.Row():
            segmentation_frame_input = gr.Dropdown(choices=[], value=None, label="Prompt frame (paper uses the first frame)")
            segmentation_prompt_label_input = gr.Radio(
                choices=[("Foreground (green)", "foreground"), ("Background (red)", "background")],
                value="foreground",
                label="Next click assigns",
            )
            vos_knn_neighbors_input = gr.Slider(minimum=1, maximum=15, step=1, value=5, label="Top-k neighbors")
        gr.Markdown(
            "Paper-aligned defaults are used for propagation: cosine similarity, temperature `0.2`, up to `15` context frames, local radius `12`, and first-frame-only prompts. "
            "This UI still uses sparse foreground/background clicks instead of the paper's dense first-frame object masks."
        )
        prepare_segmentation_button = gr.Button("Show frame for VOS prompts")
        run_segmentation_button = gr.Button("Run VOS segmentation", variant="primary")
        segmentation_status_output = gr.Markdown(
            value=_format_hint_status(
                "VOS segmentation not ready",
                "Load latents first, then choose a frame and add foreground/background prompts.",
            )
        )
        with gr.Row():
            with gr.Column(scale=1, min_width=0):
                segmentation_preview_output = gr.Image(
                    label="Prompt frame (click to place foreground/background points)",
                    type="numpy",
                )
            with gr.Column(scale=1, min_width=0):
                segmentation_video_output = gr.Video(
                    label="Binary segmentation mask video",
                    height=520,
                )
        with gr.Accordion("VOS segmentation metadata", open=False):
            segmentation_metadata_output = gr.Code(label="VOS segmentation metadata", language="json", value="{}")

    return {
        "segmentation_frame_input": segmentation_frame_input,
        "segmentation_prompt_label_input": segmentation_prompt_label_input,
        "vos_knn_neighbors_input": vos_knn_neighbors_input,
        "prepare_segmentation_button": prepare_segmentation_button,
        "run_segmentation_button": run_segmentation_button,
        "segmentation_status_output": segmentation_status_output,
        "segmentation_preview_output": segmentation_preview_output,
        "segmentation_video_output": segmentation_video_output,
        "segmentation_metadata_output": segmentation_metadata_output,
    }
