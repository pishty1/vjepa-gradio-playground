from __future__ import annotations

import gradio as gr

from gradio_components.latent_source import (
    DEFAULT_CROP_HEIGHT,
    DEFAULT_CROP_WIDTH,
    DEFAULT_DEVICE,
    DEFAULT_MODEL_NAME,
    _clean_latent_metadata_for_ui,
    _format_extraction_status,
    _format_latent_status,
    _latent_state,
    _summarize_timings_for_ui,
    build_latent_source_section,
    estimate_limits_step,
    extract_latents_step,
    load_latents_step,
    refresh_saved_latent_choices,
    toggle_latent_source_mode,
)
from gradio_components.plot import build_plot_section, build_plot_step, toggle_plot_dimensions
from gradio_components.projection import (
    build_projection_section,
    compute_projection_step,
    load_projection_step,
    toggle_projection_controls,
)
from gradio_components.render import build_render_section, create_rgb_videos_step
from gradio_components.segmentation import (
    build_segmentation_tab,
    prepare_segmentation_step,
    run_segmentation_step,
    select_segmentation_prompt_step,
)
from gradio_components.tracking import (
    build_tracking_tab,
    prepare_tracking_step,
    select_patch_similarity_step,
)
from gradio_utils import MODEL_CHOICES
from gradio_components.projection import has_mlx_vis_support


def build_demo() -> gr.Blocks:
    description = (
        "Run the V-JEPA 2.1 pipeline in independent stages: choose or extract latents, "
        "compute PCA, `umap-learn`, or Apple-Silicon `mlx-vis` projections, build plots, and generate RGB videos from any 3 projected components."
    )
    mlx_note = (
        "`mlx-vis` is available in this environment for Apple-Silicon-accelerated reducers."
        if has_mlx_vis_support()
        else "Install optional `mlx-vis` on Apple Silicon to enable the MLX-backed projection methods."
    )

    with gr.Blocks(title="V-JEPA 2.1 Latent Explorer") as demo:
        gr.Markdown("# V-JEPA 2.1 Latent Explorer")
        gr.Markdown(description)

        latent_state = gr.State(value=None)
        projection_state = gr.State(value=None)
        tracking_state = gr.State(value=None)
        segmentation_state = gr.State(value=None)

        latent_source = build_latent_source_section(model_choices=MODEL_CHOICES)
        projection_section = build_projection_section(mlx_note=mlx_note)
        plot_section = build_plot_section()
        render_section = build_render_section()

        with gr.Tabs():
            tracking_tab = build_tracking_tab()
            segmentation_tab = build_segmentation_tab()

        latent_source["latent_source_mode_input"].change(
            fn=toggle_latent_source_mode,
            inputs=[latent_source["latent_source_mode_input"], latent_source["saved_latent_input"]],
            outputs=[
                latent_source["extract_source_group"],
                latent_source["load_source_group"],
                latent_source["saved_latent_input"],
            ],
        )

        latent_source["refresh_saved_latents_button"].click(
            fn=refresh_saved_latent_choices,
            inputs=[latent_source["saved_latent_input"]],
            outputs=[latent_source["saved_latent_input"]],
        )

        projection_section["projection_method_input"].change(
            fn=toggle_projection_controls,
            inputs=[projection_section["projection_method_input"]],
            outputs=[projection_section["umap_controls"], projection_section["pca_controls"]],
        )
        plot_section["plot_dimensions_input"].change(
            fn=toggle_plot_dimensions,
            inputs=[plot_section["plot_dimensions_input"]],
            outputs=[plot_section["plot_z_component_input"]],
        )

        latent_source["estimate_button"].click(
            fn=estimate_limits_step,
            inputs=[
                latent_source["video_input"],
                latent_source["model_input"],
                latent_source["crop_height_input"],
                latent_source["crop_width_input"],
                latent_source["frames_input"],
                latent_source["sample_fps_input"],
                latent_source["start_second_input"],
                latent_source["device_input"],
            ],
            outputs=[latent_source["preflight_status_output"], latent_source["preflight_metadata_output"]],
        )

        latent_source["extract_button"].click(
            fn=extract_latents_step,
            inputs=[
                latent_source["video_input"],
                latent_source["model_input"],
                latent_source["crop_height_input"],
                latent_source["crop_width_input"],
                latent_source["frames_input"],
                latent_source["sample_fps_input"],
                latent_source["start_second_input"],
                latent_source["device_input"],
            ],
            outputs=[
                latent_source["extraction_status_output"],
                latent_source["latent_prefix_input"],
                latent_source["extraction_metadata_output"],
                latent_state,
                projection_state,
                projection_section["projection_prefix_input"],
                projection_section["projection_metadata_output"],
                plot_section["plot_output"],
                render_section["side_by_side_output"],
                render_section["render_status_output"],
                render_section["render_metadata_output"],
                latent_source["saved_latent_input"],
            ],
            trigger_mode="once",
            concurrency_limit=1,
        )

        latent_source["load_latents_button"].click(
            fn=load_latents_step,
            inputs=[
                latent_source["saved_latent_input"],
                latent_state,
            ],
            outputs=[
                latent_source["latent_status_output"],
                latent_source["latent_prefix_input"],
                latent_source["latent_metadata_output"],
                latent_state,
                projection_state,
                projection_section["projection_prefix_input"],
                projection_section["projection_metadata_output"],
                plot_section["plot_output"],
                render_section["side_by_side_output"],
                render_section["render_status_output"],
                render_section["render_metadata_output"],
                latent_source["saved_latent_input"],
            ],
        )

        projection_section["compute_projection_button"].click(
            fn=compute_projection_step,
            inputs=[
                latent_state,
                projection_section["projection_method_input"],
                projection_section["projection_pca_mode_input"],
                projection_section["projection_components_input"],
                projection_section["umap_n_neighbors_input"],
                projection_section["umap_min_dist_input"],
                projection_section["umap_metric_input"],
                projection_section["umap_random_state_input"],
            ],
            outputs=[
                projection_section["projection_status_output"],
                projection_section["projection_prefix_input"],
                projection_section["projection_metadata_output"],
                projection_state,
                plot_section["plot_dimensions_input"],
                plot_section["plot_x_component_input"],
                plot_section["plot_y_component_input"],
                plot_section["plot_z_component_input"],
                render_section["rgb_r_component_input"],
                render_section["rgb_g_component_input"],
                render_section["rgb_b_component_input"],
                plot_section["plot_max_points_input"],
                plot_section["plot_output"],
                plot_section["plot_status_output"],
                render_section["render_status_output"],
                render_section["render_metadata_output"],
            ],
        )

        projection_section["load_projection_button"].click(
            fn=load_projection_step,
            inputs=[projection_section["projection_prefix_input"], projection_state],
            outputs=[
                projection_section["projection_status_output"],
                projection_section["projection_prefix_input"],
                projection_section["projection_metadata_output"],
                projection_state,
                plot_section["plot_dimensions_input"],
                plot_section["plot_x_component_input"],
                plot_section["plot_y_component_input"],
                plot_section["plot_z_component_input"],
                render_section["rgb_r_component_input"],
                render_section["rgb_g_component_input"],
                render_section["rgb_b_component_input"],
                plot_section["plot_max_points_input"],
                plot_section["plot_output"],
                plot_section["plot_status_output"],
                render_section["render_status_output"],
                render_section["render_metadata_output"],
            ],
        )

        plot_section["build_plot_button"].click(
            fn=build_plot_step,
            inputs=[
                projection_state,
                plot_section["plot_dimensions_input"],
                plot_section["plot_max_points_input"],
                plot_section["plot_animate_input"],
                plot_section["plot_x_component_input"],
                plot_section["plot_y_component_input"],
                plot_section["plot_z_component_input"],
            ],
            outputs=[plot_section["plot_output"], plot_section["plot_status_output"]],
        )

        render_section["create_rgb_button"].click(
            fn=create_rgb_videos_step,
            inputs=[
                latent_state,
                projection_state,
                render_section["rgb_r_component_input"],
                render_section["rgb_g_component_input"],
                render_section["rgb_b_component_input"],
                render_section["upscale_factor_input"],
            ],
            outputs=[
                render_section["render_status_output"],
                render_section["side_by_side_output"],
                render_section["render_metadata_output"],
            ],
        )

        tracking_tab["prepare_tracking_button"].click(
            fn=prepare_tracking_step,
            inputs=[latent_state, tracking_tab["tracking_frame_input"]],
            outputs=[
                tracking_tab["tracking_frame_input"],
                tracking_tab["tracking_preview_output"],
                tracking_tab["tracking_status_output"],
                tracking_tab["tracking_metadata_output"],
                tracking_state,
                tracking_tab["tracking_video_output"],
            ],
        )

        tracking_tab["tracking_frame_input"].change(
            fn=prepare_tracking_step,
            inputs=[latent_state, tracking_tab["tracking_frame_input"]],
            outputs=[
                tracking_tab["tracking_frame_input"],
                tracking_tab["tracking_preview_output"],
                tracking_tab["tracking_status_output"],
                tracking_tab["tracking_metadata_output"],
                tracking_state,
                tracking_tab["tracking_video_output"],
            ],
        )

        tracking_tab["tracking_preview_output"].select(
            fn=select_patch_similarity_step,
            inputs=[latent_state, tracking_state],
            outputs=[
                tracking_tab["tracking_preview_output"],
                tracking_tab["tracking_status_output"],
                tracking_tab["tracking_video_output"],
                tracking_tab["tracking_metadata_output"],
                tracking_state,
            ],
        )

        segmentation_tab["prepare_segmentation_button"].click(
            fn=prepare_segmentation_step,
            inputs=[latent_state, segmentation_tab["segmentation_frame_input"]],
            outputs=[
                segmentation_tab["segmentation_frame_input"],
                segmentation_tab["segmentation_preview_output"],
                segmentation_tab["segmentation_status_output"],
                segmentation_tab["segmentation_metadata_output"],
                segmentation_state,
                segmentation_tab["segmentation_video_output"],
            ],
        )

        segmentation_tab["segmentation_frame_input"].change(
            fn=prepare_segmentation_step,
            inputs=[latent_state, segmentation_tab["segmentation_frame_input"]],
            outputs=[
                segmentation_tab["segmentation_frame_input"],
                segmentation_tab["segmentation_preview_output"],
                segmentation_tab["segmentation_status_output"],
                segmentation_tab["segmentation_metadata_output"],
                segmentation_state,
                segmentation_tab["segmentation_video_output"],
            ],
        )

        segmentation_tab["segmentation_preview_output"].select(
            fn=select_segmentation_prompt_step,
            inputs=[latent_state, segmentation_state, segmentation_tab["segmentation_prompt_label_input"]],
            outputs=[
                segmentation_tab["segmentation_preview_output"],
                segmentation_tab["segmentation_status_output"],
                segmentation_tab["segmentation_metadata_output"],
                segmentation_state,
            ],
        )

        segmentation_tab["run_segmentation_button"].click(
            fn=run_segmentation_step,
            inputs=[latent_state, segmentation_state, segmentation_tab["vos_knn_neighbors_input"]],
            outputs=[
                segmentation_tab["segmentation_status_output"],
                segmentation_tab["segmentation_video_output"],
                segmentation_tab["segmentation_metadata_output"],
                segmentation_state,
            ],
        )

    return demo
