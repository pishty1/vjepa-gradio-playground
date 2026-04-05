from __future__ import annotations

from typing import Any

import gradio as gr


def build_render_section() -> dict[str, Any]:
    gr.Markdown("## 4. Create RGB latent videos from any 3 projection components")
    with gr.Row():
        rgb_r_component_input = gr.Dropdown(choices=[1, 2, 3], value=1, label="R component")
        rgb_g_component_input = gr.Dropdown(choices=[1, 2, 3], value=2, label="G component")
        rgb_b_component_input = gr.Dropdown(choices=[1, 2, 3], value=3, label="B component")
        upscale_factor_input = gr.Slider(minimum=1, maximum=32, step=1, value=24, label="Upscale factor")
    create_rgb_button = gr.Button("Create RGB videos")
    render_status_output = gr.Markdown()
    side_by_side_output = gr.Video(label="Source vs latent side-by-side")
    with gr.Accordion("Render metadata", open=False):
        render_metadata_output = gr.Code(label="Render metadata", language="json", value="{}")

    return {
        "rgb_r_component_input": rgb_r_component_input,
        "rgb_g_component_input": rgb_g_component_input,
        "rgb_b_component_input": rgb_b_component_input,
        "upscale_factor_input": upscale_factor_input,
        "create_rgb_button": create_rgb_button,
        "render_status_output": render_status_output,
        "side_by_side_output": side_by_side_output,
        "render_metadata_output": render_metadata_output,
    }
