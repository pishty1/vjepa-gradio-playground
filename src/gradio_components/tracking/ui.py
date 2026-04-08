from __future__ import annotations

from typing import Any

import gradio as gr

from gradio_utils import _format_hint_status


def build_tracking_tab() -> dict[str, Any]:
    with gr.Tab("Patch Similarity / Dense Tracking"):
        gr.Markdown(
            "Load latents first, choose one of the available frames, and click a patch to compute cosine similarity against every latent token in the video."
        )
        tracking_frame_input = gr.Dropdown(choices=[], value=None, label="Frame to track")
        prepare_tracking_button = gr.Button("Show first frame")
        tracking_status_output = gr.Markdown(
            value=_format_hint_status(
                "Patch similarity not ready",
                "Load latents first, then choose a frame for tracking.",
            )
        )
        with gr.Row():
            with gr.Column(scale=1, min_width=0):
                tracking_preview_output = gr.Image(
                    label="First frame (click to choose a patch)",
                    type="numpy",
                )
            with gr.Column(scale=1, min_width=0):
                tracking_video_output = gr.Video(
                    label="Patch similarity heatmap video",
                    height=520,
                )
        with gr.Accordion("Patch similarity metadata", open=False):
            tracking_metadata_output = gr.Code(label="Patch similarity metadata", language="json", value="{}")

    return {
        "tracking_frame_input": tracking_frame_input,
        "prepare_tracking_button": prepare_tracking_button,
        "tracking_status_output": tracking_status_output,
        "tracking_preview_output": tracking_preview_output,
        "tracking_video_output": tracking_video_output,
        "tracking_metadata_output": tracking_metadata_output,
    }
