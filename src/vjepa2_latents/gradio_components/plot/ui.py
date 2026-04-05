from __future__ import annotations

from typing import Any

import gradio as gr


def build_plot_section() -> dict[str, Any]:
    gr.Markdown("## 3. Build a plot from chosen projection components")
    with gr.Row():
        plot_dimensions_input = gr.Radio([2, 3], value=3, label="Plot dimensions")
        plot_max_points_input = gr.Slider(minimum=500, maximum=12000, step=500, value=4000, label="Max plotted points")
        plot_x_component_input = gr.Dropdown(choices=[1, 2, 3], value=1, label="X component")
        plot_y_component_input = gr.Dropdown(choices=[1, 2, 3], value=2, label="Y component")
        plot_z_component_input = gr.Dropdown(choices=[1, 2, 3], value=3, label="Z component")
    build_plot_button = gr.Button("Build plot")
    plot_status_output = gr.Markdown()
    plot_output = gr.Plot(label="Latent space plot")

    return {
        "plot_dimensions_input": plot_dimensions_input,
        "plot_max_points_input": plot_max_points_input,
        "plot_x_component_input": plot_x_component_input,
        "plot_y_component_input": plot_y_component_input,
        "plot_z_component_input": plot_z_component_input,
        "build_plot_button": build_plot_button,
        "plot_status_output": plot_status_output,
        "plot_output": plot_output,
    }
