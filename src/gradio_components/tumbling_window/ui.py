from __future__ import annotations

from typing import Any

import gradio as gr

from .helpers import initial_tumbling_window_status


def build_tumbling_window_tab() -> dict[str, Any]:
    with gr.Tab("Tumbling Window Comparison"):
        gr.Markdown(
            "Use the video, model, crop, and device from **Choose latent source**, then compare two overlapping encoder windows. "
            "Pick the first window start frame, how many latent time slices overlap, and the window length in frames."
        )
        with gr.Row():
            tumbling_start_frame_input = gr.Number(value=50, precision=0, label="First window start frame")
            tumbling_overlap_time_slices_input = gr.Slider(
                minimum=1,
                maximum=32,
                step=1,
                value=1,
                label="Overlapping latent time slices",
            )
            tumbling_window_frames_input = gr.Slider(
                minimum=4,
                maximum=64,
                step=2,
                value=20,
                label="Window length (frames)",
            )
        compare_tumbling_windows_button = gr.Button("Compare tumbling windows")
        tumbling_status_output = gr.Markdown(value=initial_tumbling_window_status())
        tumbling_heatmap_output = gr.HTML(label="Overlap difference heatmap")
        with gr.Accordion("Tumbling-window metadata", open=False):
            tumbling_metadata_output = gr.Code(label="Tumbling-window metadata", language="json", value="{}")

    return {
        "tumbling_start_frame_input": tumbling_start_frame_input,
        "tumbling_overlap_time_slices_input": tumbling_overlap_time_slices_input,
        "tumbling_window_frames_input": tumbling_window_frames_input,
        "compare_tumbling_windows_button": compare_tumbling_windows_button,
        "tumbling_status_output": tumbling_status_output,
        "tumbling_heatmap_output": tumbling_heatmap_output,
        "tumbling_metadata_output": tumbling_metadata_output,
    }