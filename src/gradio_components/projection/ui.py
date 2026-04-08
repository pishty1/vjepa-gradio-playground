from __future__ import annotations

from typing import Any

import gradio as gr

from gradio_utils import PCA_MODE_CHOICES, PROJECTION_METHOD_CHOICES


def build_projection_section(*, mlx_note: str) -> dict[str, Any]:
    gr.Markdown("## 2. Compute or load a projection")
    with gr.Row():
        with gr.Column():
            projection_method_input = gr.Dropdown(choices=PROJECTION_METHOD_CHOICES, value="PCA", label="Projection method")
            with gr.Group(visible=True) as pca_controls:
                projection_pca_mode_input = gr.Radio(
                    choices=PCA_MODE_CHOICES,
                    value="global",
                    label="PCA mode",
                )
                gr.Markdown("Use spatial-only PCA to average across time first, or temporal-only PCA to average across space first.")
            projection_components_input = gr.Slider(minimum=2, maximum=5, step=1, value=5, label="Projected components")
            with gr.Group(visible=False) as umap_controls:
                umap_n_neighbors_input = gr.Slider(minimum=2, maximum=200, step=1, value=15, label="Neighbor count")
                umap_min_dist_input = gr.Slider(minimum=0.0, maximum=0.99, step=0.01, value=0.1, label="UMAP min_dist")
                umap_metric_input = gr.Dropdown(
                    choices=["euclidean", "cosine", "manhattan", "chebyshev", "correlation"],
                    value="euclidean",
                    label="Distance metric",
                )
                umap_random_state_input = gr.Number(value=42, precision=0, label="Random state")
                gr.Markdown(
                    "These settings are used by `UMAP`, `UMAP-MLX`, `PaCMAP-MLX`, and `LocalMAP-MLX`. "
                    "Other `mlx-vis` reducers currently use their library defaults plus the selected component count."
                )
            gr.Markdown(mlx_note)
            compute_projection_button = gr.Button("Compute projection")
        with gr.Column():
            projection_prefix_input = gr.Textbox(label="Projection prefix (.projection.npz stem)")
            load_projection_button = gr.Button("Load saved projection")

    projection_status_output = gr.Markdown()
    with gr.Accordion("Projection metadata", open=False):
        projection_metadata_output = gr.Code(label="Projection metadata", language="json", value="{}")

    return {
        "projection_method_input": projection_method_input,
        "pca_controls": pca_controls,
        "projection_pca_mode_input": projection_pca_mode_input,
        "projection_components_input": projection_components_input,
        "umap_controls": umap_controls,
        "umap_n_neighbors_input": umap_n_neighbors_input,
        "umap_min_dist_input": umap_min_dist_input,
        "umap_metric_input": umap_metric_input,
        "umap_random_state_input": umap_random_state_input,
        "compute_projection_button": compute_projection_button,
        "projection_prefix_input": projection_prefix_input,
        "load_projection_button": load_projection_button,
        "projection_status_output": projection_status_output,
        "projection_metadata_output": projection_metadata_output,
    }
