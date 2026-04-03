from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Sequence

import gradio as gr

from .extractor import estimate_extraction_requirements, extract_latents, probe_video, select_frame_indices
from .gradio_utils import (
    DEFAULT_CROP_HEIGHT,
    DEFAULT_CROP_WIDTH,
    DEFAULT_DEVICE,
    DEFAULT_MODEL_NAME,
    CHECKPOINT_DIR,
    DEFAULT_VIDEO,
    MODEL_CHOICES,
    NEIGHBOR_TUNED_METHODS,
    PCA_MODE_CHOICES,
    PROJECTION_METHOD_CHOICES,
    VENDOR_REPO,
    _clean_latent_metadata_for_ui,
    _component_selector_updates,
    _create_session_dir,
    _format_extraction_status,
    _format_hint_status,
    _format_latent_status,
    _format_plot_status,
    _format_preflight_status,
    _format_projection_status,
    _format_render_status,
    _format_tracking_ready_status,
    _format_tracking_result_status,
    _latent_state,
    _load_latent_metadata,
    _log_gradio_step,
    _normalize_prefix,
    _normalize_projection_settings,
    _projection_state,
    _resolve_video_path,
    _serialize_json,
    _summarize_timings_for_ui,
    _tracking_frame_choices,
    summarize_latents,
)
from .visualization import (
    annotate_selected_patch,
    build_projection_figure_from_data,
    compute_projection_bundle,
    create_patch_similarity_video,
    create_visualizations_from_projection,
    has_mlx_vis_support,
    load_aligned_source_frames,
    load_saved_latents,
    load_saved_projection,
    map_click_to_latent_token,
    save_projection_artifacts,
)
def estimate_limits_step(
    video_file: str | None,
    model_name: str,
    crop_height: int,
    crop_width: int,
    num_frames: int,
    sample_fps: float | None,
    start_second: float,
    device_name: str,
):
    _log_gradio_step("estimate", f"video={video_file or 'default'} model={model_name} device={device_name}")
    video_path = _resolve_video_path(video_file)
    video_meta = probe_video(video_path)
    frame_indices = select_frame_indices(
        video_fps=video_meta["fps"],
        frame_count=video_meta["frame_count"],
        num_frames=int(num_frames),
        start_frame=0,
        start_second=start_second if start_second > 0 else None,
        sample_fps=sample_fps if sample_fps and sample_fps > 0 else None,
    )
    preflight = estimate_extraction_requirements(
        model_name=model_name,
        num_frames=int(num_frames),
        crop_size=(int(crop_height), int(crop_width)),
        device_name=device_name,
    )
    payload = {
        **preflight,
        "video_path": str(video_path),
        "video_metadata": video_meta,
        "selected_frame_count": len(frame_indices),
        "selected_frame_range": [frame_indices[0], frame_indices[-1]],
    }
    _log_gradio_step("estimate", f"selected_frames={len(frame_indices)} token_count={preflight['token_count']}")
    return _format_preflight_status(preflight, video_meta, frame_indices), _serialize_json(payload)


def _latent_state(output_prefix: Path, latent_grid: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "output_prefix": str(output_prefix),
        "session_dir": str(output_prefix.parent),
        "latent_grid": latent_grid,
        "metadata": metadata,
        "summary": summarize_latents(latent_grid),
    }


def extract_latents_step(
    video_file: str | None,
    model_name: str,
    crop_height: int,
    crop_width: int,
    num_frames: int,
    sample_fps: float | None,
    start_second: float,
    device_name: str,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    _log_gradio_step("extract", f"starting model={model_name} device={device_name}")
    video_path = _resolve_video_path(video_file)
    session_dir = _create_session_dir("vjepa2-extract-")
    output_prefix = session_dir / "latents"
    progress(0.1, desc="Running V-JEPA encoder")
    try:
        result = extract_latents(
            video_path=video_path,
            output_prefix=output_prefix,
            vendor_repo=VENDOR_REPO,
            model_name=model_name,
            checkpoint_path=None,
            checkpoint_dir=CHECKPOINT_DIR,
            num_frames=num_frames,
            crop_size=(int(crop_height), int(crop_width)),
            sample_fps=sample_fps if sample_fps and sample_fps > 0 else None,
            start_frame=0,
            start_second=start_second if start_second > 0 else None,
            device_name=device_name,
            dry_run=False,
            save_pt=False,
        )
    except RuntimeError as error:
        if "MPS backend out of memory" in str(error):
            preflight = estimate_extraction_requirements(
                model_name=model_name,
                num_frames=int(num_frames),
                crop_size=(int(crop_height), int(crop_width)),
                device_name=device_name,
            )
            raise gr.Error(
                "Apple MPS ran out of memory for this configuration. "
                f"Estimated risk is `{preflight['risk_level']}` with `{preflight['token_count']}` patch tokens. "
                "Try fewer frames, a smaller crop, or switch the device to `cpu`."
            ) from error
        raise

    payload = {
        **result,
        "model": model_name,
        "video_path": str(video_path),
        "video_name": video_path.name,
        "latent_output_prefix": str(output_prefix),
        "latent_npy_path": str(output_prefix.with_suffix(".npy")),
        "latent_metadata_path": str(output_prefix.with_suffix(".metadata.json")),
    }
    payload["timings"] = _summarize_timings_for_ui(result.get("timings"))
    latent_grid, metadata = load_saved_latents(output_prefix)
    status = _format_extraction_status(result, output_prefix, model_name=model_name, video_path=video_path)
    latent_state = _latent_state(output_prefix, latent_grid, metadata)
    _log_gradio_step("extract", f"saved_prefix={output_prefix} latent_grid_shape={latent_grid.shape}")
    return (
        status,
        str(output_prefix),
        _serialize_json(payload),
        latent_state,
        None,
        "",
        "{}",
        None,
        None,
        "",
        "{}",
    )


def load_latents_step(
    latent_prefix_text: str,
    latent_npy_file: str | None,
    latent_metadata_file: str | None,
    latent_state: dict[str, Any] | None,
):
    _log_gradio_step("load_latents", f"prefix={latent_prefix_text or latent_state.get('output_prefix') if latent_state else 'none'}")
    if latent_npy_file and latent_metadata_file:
        session_dir = _create_session_dir("vjepa2-latents-")
        output_prefix = session_dir / "latents"
        shutil.copy2(latent_npy_file, output_prefix.with_suffix(".npy"))
        shutil.copy2(latent_metadata_file, output_prefix.with_suffix(".metadata.json"))
    else:
        output_prefix = _normalize_prefix(latent_prefix_text, (".npy", ".metadata.json"))
        if output_prefix is None and latent_state and latent_state.get("output_prefix"):
            output_prefix = Path(latent_state["output_prefix"])
        if output_prefix is None:
            return (
                _format_hint_status("Latents not loaded", "Provide a latent prefix or both latent files before loading latents."),
                latent_prefix_text or "",
                "{}",
                latent_state,
                None,
                "",
                "{}",
                None,
                None,
                _format_hint_status("RGB videos not ready", "Load latents first, then compute or load a projection before rendering videos."),
                "{}",
            )

    latent_grid, metadata = load_saved_latents(output_prefix)
    summary = summarize_latents(latent_grid)
    payload = {
        **_clean_latent_metadata_for_ui(metadata),
        "summary": summary,
        "latent_output_prefix": str(output_prefix),
    }
    status = _format_latent_status(output_prefix, metadata, summary)
    next_state = _latent_state(output_prefix, latent_grid, metadata)
    _log_gradio_step("load_latents", f"loaded prefix={output_prefix} latent_grid_shape={latent_grid.shape}")
    return (
        status,
        str(output_prefix),
        _serialize_json(payload),
        next_state,
        None,
        "",
        "{}",
        None,
        None,
        "",
        "{}",
    )


def toggle_projection_controls(projection_method: str):
    normalized = projection_method.strip().lower().replace("-", "_")
    return gr.update(visible=normalized in NEIGHBOR_TUNED_METHODS), gr.update(visible=normalized == "pca")


def compute_projection_step(
    latent_state: dict[str, Any] | None,
    projection_method: str,
    projection_pca_mode: str,
    projection_components: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    umap_random_state: float | None,
):
    _log_gradio_step("compute_projection", f"method={projection_method} components={projection_components}")
    if not latent_state or not latent_state.get("output_prefix"):
        selector_updates = _component_selector_updates(3)
        return (
            _format_hint_status("Projection not computed", "Load latents before computing a projection."),
            "",
            "{}",
            None,
            *selector_updates,
            None,
            _format_hint_status("Plot not ready", "Compute or load a projection before building a plot."),
            _format_hint_status("RGB videos not ready", "Compute or load a projection before generating RGB videos."),
            "{}",
        )

    latent_output_prefix = Path(latent_state["output_prefix"])
    latent_grid = latent_state.get("latent_grid")
    if latent_grid is None:
        latent_grid, _ = load_saved_latents(latent_output_prefix)
    settings = _normalize_projection_settings(
        projection_method,
        projection_pca_mode,
        projection_components,
        umap_n_neighbors,
        umap_min_dist,
        umap_metric,
        umap_random_state,
    )
    try:
        projection_bundle = compute_projection_bundle(latent_grid, **settings)
    except (RuntimeError, ValueError) as error:
        raise gr.Error(str(error)) from error
    projection_name = settings["method"]
    if projection_name == "pca" and settings.get("pca_mode") and settings["pca_mode"] != "global":
        projection_name = f"{settings['pca_mode']}_pca"
    projection_output_prefix = latent_output_prefix.parent / f"projection_{projection_name}_{projection_bundle['projection'].shape[1]}"
    artifacts = save_projection_artifacts(
        projection_output_prefix,
        projection_bundle,
        latent_output_prefix=latent_output_prefix,
    )
    projection_metadata = {
        key: value
        for key, value in projection_bundle.items()
        if key not in {"projection", "coordinates"}
    }
    projection_metadata.update(
        {
            "projection_output_prefix": str(projection_output_prefix),
            "projection_npz_path": str(artifacts.projection_path),
            "projection_metadata_path": str(artifacts.metadata_path),
        }
    )
    payload = {
        **projection_metadata,
    }
    status = _format_projection_status(projection_output_prefix, projection_metadata)
    selector_updates = _component_selector_updates(artifacts.projection_shape[1])
    _log_gradio_step("compute_projection", f"saved_prefix={projection_output_prefix} projection_shape={projection_bundle['projection'].shape}")
    return (
        status,
        str(projection_output_prefix),
        _serialize_json(payload),
        {
            **_projection_state(projection_output_prefix, projection_metadata),
            "projection": projection_bundle["projection"],
            "coordinates": projection_bundle["coordinates"],
        },
        *selector_updates,
        None,
        None,
        "",
        "{}",
    )


def load_projection_step(
    projection_prefix_text: str,
    projection_state: dict[str, Any] | None,
):
    _log_gradio_step("load_projection", f"prefix={projection_prefix_text or projection_state.get('output_prefix') if projection_state else 'none'}")
    output_prefix = _normalize_prefix(projection_prefix_text, (".projection.npz", ".projection.metadata.json"))
    if output_prefix is None and projection_state and projection_state.get("output_prefix"):
        output_prefix = Path(projection_state["output_prefix"])
    if output_prefix is None:
        selector_updates = _component_selector_updates(3)
        return (
            _format_hint_status("Projection not loaded", "Provide a projection prefix before loading a saved projection."),
            projection_prefix_text or "",
            "{}",
            projection_state,
            *selector_updates,
        )

    projection, coordinates, metadata = load_saved_projection(output_prefix)
    payload = {
        **metadata,
        "projection_output_prefix": str(output_prefix),
        "projection_shape": list(projection.shape),
    }
    status = _format_projection_status(output_prefix, metadata)
    selector_updates = _component_selector_updates(projection.shape[1])
    _log_gradio_step("load_projection", f"loaded prefix={output_prefix} projection_shape={projection.shape}")
    return (
        status,
        str(output_prefix),
        _serialize_json(payload),
        {
            **_projection_state(output_prefix, metadata),
            "projection": projection,
            "coordinates": coordinates,
            "metadata": metadata,
        },
        *selector_updates,
    )


def toggle_plot_dimensions(plot_dimensions: int):
    return gr.update(visible=int(plot_dimensions) == 3)


def build_plot_step(
    projection_state: dict[str, Any] | None,
    plot_dimensions: int,
    plot_max_points: int,
    plot_x_component: int,
    plot_y_component: int,
    plot_z_component: int | None,
):
    _log_gradio_step("build_plot", f"dimensions={plot_dimensions} components=({plot_x_component}, {plot_y_component}, {plot_z_component})")
    if not projection_state or not projection_state.get("output_prefix"):
        return None, _format_hint_status("Plot not ready", "Compute or load a projection before building a plot.")

    projection = projection_state.get("projection")
    coordinates = projection_state.get("coordinates")
    metadata = projection_state.get("metadata")
    if projection is None or coordinates is None or metadata is None:
        projection, coordinates, metadata = load_saved_projection(Path(projection_state["output_prefix"]))
    component_indices = [int(plot_x_component) - 1, int(plot_y_component) - 1]
    if int(plot_dimensions) == 3:
        if plot_z_component is None:
            return None, _format_hint_status("Plot not ready", "Choose a Z component for a 3D plot.")
        component_indices.append(int(plot_z_component) - 1)

    figure = build_projection_figure_from_data(
        projection,
        coordinates,
        method=metadata["method"],
        component_indices=tuple(component_indices),
        component_labels=metadata.get("component_labels"),
        max_points=max(100, int(plot_max_points)),
    )
    _log_gradio_step("build_plot", f"plotted_components={component_indices}")
    return figure, _format_plot_status(metadata["method"], component_indices, metadata.get("method_label"))


def create_rgb_videos_step(
    latent_state: dict[str, Any] | None,
    projection_state: dict[str, Any] | None,
    rgb_r_component: int,
    rgb_g_component: int,
    rgb_b_component: int | None,
    upscale_factor: int,
):
    _log_gradio_step("render_rgb", f"components=({rgb_r_component}, {rgb_g_component}, {rgb_b_component}) upscale={upscale_factor}")
    if not projection_state or not projection_state.get("output_prefix"):
        return (
            _format_hint_status("RGB videos not ready", "Compute or load a projection before generating RGB videos."),
            None,
            "{}",
        )
    if rgb_b_component is None:
        return (
            _format_hint_status("RGB videos not ready", "RGB rendering needs at least 3 projected components."),
            None,
            "{}",
        )

    projection_output_prefix = Path(projection_state["output_prefix"])
    projection = projection_state.get("projection")
    projection_metadata = projection_state.get("metadata")
    if projection is None or projection_metadata is None:
        projection, _, projection_metadata = load_saved_projection(projection_output_prefix)
    latent_output_prefix_text = None
    if latent_state and latent_state.get("output_prefix"):
        latent_output_prefix_text = latent_state["output_prefix"]
    elif projection_metadata.get("latent_output_prefix"):
        latent_output_prefix_text = projection_metadata["latent_output_prefix"]
    if not latent_output_prefix_text:
        return (
            _format_hint_status("RGB videos not ready", "Could not locate the latent metadata required for side-by-side rendering."),
            None,
            "{}",
        )

    latent_output_prefix = Path(latent_output_prefix_text)
    latent_metadata = None
    if latent_state and latent_state.get("metadata") is not None:
        latent_metadata = latent_state["metadata"]
    if latent_metadata is None:
        latent_metadata = _load_latent_metadata(latent_output_prefix)
    rgb_components = (int(rgb_r_component) - 1, int(rgb_g_component) - 1, int(rgb_b_component) - 1)
    output_dir = projection_output_prefix.parent / "renders"
    artifacts = create_visualizations_from_projection(
        projection=projection,
        latent_grid_shape=projection_metadata["latent_grid_shape"],
        metadata=latent_metadata,
        output_dir=output_dir,
        rgb_components=rgb_components,
        method=projection_metadata["method"],
        component_labels=projection_metadata.get("component_labels"),
        upscale_factor=max(1, int(upscale_factor)),
    )
    payload = {
        "projection_output_prefix": str(projection_output_prefix),
        "latent_output_prefix": str(latent_output_prefix),
        "rgb_components": [component + 1 for component in rgb_components],
        "artifacts": {
            "latent_video_path": str(artifacts.latent_video_path),
            "side_by_side_video_path": str(artifacts.side_by_side_video_path),
            "latent_video_shape": list(artifacts.latent_video_shape),
            "side_by_side_video_shape": list(artifacts.side_by_side_video_shape),
            "display_fps": artifacts.display_fps,
        },
    }
    status = _format_render_status(
        projection_metadata["method"],
        rgb_components,
        artifacts.latent_video_path,
        artifacts.side_by_side_video_path,
        projection_metadata.get("method_label"),
    )
    _log_gradio_step("render_rgb", f"side_by_side={artifacts.side_by_side_video_path}")
    return status, str(artifacts.side_by_side_video_path), _serialize_json(payload)


def prepare_tracking_step(latent_state: dict[str, Any] | None, tracking_frame_index: int | str | None = 0):
    _log_gradio_step("tracking_prepare", f"frame_index={tracking_frame_index}")
    if not latent_state or not latent_state.get("output_prefix"):
        return (
            gr.update(choices=[], value=None, interactive=False),
            None,
            _format_hint_status("Patch similarity not ready", "Load latents first, then show the first frame for tracking."),
            "{}",
            None,
            None,
        )

    latent_output_prefix = Path(latent_state["output_prefix"])
    latent_grid = latent_state.get("latent_grid")
    if latent_grid is None:
        latent_grid, _ = load_saved_latents(latent_output_prefix)

    metadata = latent_state.get("metadata")
    if metadata is None:
        metadata = _load_latent_metadata(latent_output_prefix)

    try:
        source_frames, display_fps, source_frame_indices = load_aligned_source_frames(metadata, latent_grid.shape)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise gr.Error(str(error)) from error

    frame_choices = _tracking_frame_choices(source_frame_indices)
    selected_frame_index = int(tracking_frame_index) if tracking_frame_index not in (None, "") else 0
    selected_frame_index = max(0, min(selected_frame_index, len(source_frames) - 1))
    selected_video_frame = source_frame_indices[selected_frame_index] if source_frame_indices else selected_frame_index

    payload = {
        "latent_output_prefix": str(latent_output_prefix),
        "video_path": metadata.get("video_path"),
        "latent_grid_shape": list(latent_grid.shape),
        "source_frame_shape": list(source_frames.shape),
        "source_frame_indices": list(source_frame_indices),
        "selected_frame_index": selected_frame_index,
        "selected_video_frame_index": selected_video_frame,
        "display_fps": display_fps,
    }
    tracking_state = {
        "latent_output_prefix": str(latent_output_prefix),
        "source_frames": source_frames,
        "display_fps": display_fps,
        "latent_grid_shape": list(latent_grid.shape),
        "source_frame_indices": list(source_frame_indices),
        "selected_frame_index": selected_frame_index,
    }
    return (
        gr.update(choices=frame_choices, value=selected_frame_index, interactive=True),
        source_frames[selected_frame_index],
        _format_tracking_ready_status(latent_grid.shape, display_fps),
        _serialize_json(payload),
        tracking_state,
        None,
    )


def select_patch_similarity_step(
    latent_state: dict[str, Any] | None,
    tracking_state: dict[str, Any] | None,
    evt: gr.SelectData,
):
    _log_gradio_step("tracking_select", f"click_index={getattr(evt, 'index', None)}")
    if evt is None or getattr(evt, "index", None) is None:
        return (
            None,
            _format_hint_status("Patch similarity not ready", "Click inside the first frame to select a latent patch."),
            None,
            "{}",
            tracking_state,
        )
    if not latent_state or not latent_state.get("output_prefix"):
        return (
            None,
            _format_hint_status("Patch similarity not ready", "Load latents first, then show the first frame for tracking."),
            None,
            "{}",
            tracking_state,
        )

    latent_output_prefix = Path(latent_state["output_prefix"])
    latent_grid = latent_state.get("latent_grid")
    if latent_grid is None:
        latent_grid, _ = load_saved_latents(latent_output_prefix)

    metadata = latent_state.get("metadata")
    if metadata is None:
        metadata = _load_latent_metadata(latent_output_prefix)

    source_frames = None
    if tracking_state and tracking_state.get("latent_output_prefix") == str(latent_output_prefix):
        source_frames = tracking_state.get("source_frames")
    if source_frames is None:
        source_frames, display_fps, source_frame_indices = load_aligned_source_frames(metadata, latent_grid.shape)
    else:
        display_fps = float(tracking_state.get("display_fps", 0.0) or 0.0)
        source_frame_indices = list(tracking_state.get("source_frame_indices", []))

    selected_frame_index = int(tracking_state.get("selected_frame_index", 0) if tracking_state else 0)
    selected_frame_index = max(0, min(selected_frame_index, len(source_frames) - 1))
    selected_video_frame = source_frame_indices[selected_frame_index] if source_frame_indices else selected_frame_index

    click_index = evt.index
    if not isinstance(click_index, (tuple, list)) or len(click_index) < 2:
        raise gr.Error("Image selection did not provide click coordinates.")
    click_xy = (int(click_index[0]), int(click_index[1]))
    token_index = map_click_to_latent_token(click_xy, source_frames[selected_frame_index].shape, latent_grid.shape, time_index=selected_frame_index)
    preview_frame = annotate_selected_patch(source_frames[selected_frame_index], token_index, latent_grid.shape)

    output_dir = latent_output_prefix.parent / "tracking"
    artifacts = create_patch_similarity_video(
        latent_grid=latent_grid,
        metadata=metadata,
        output_dir=output_dir,
        token_index=token_index,
        source_frames=source_frames,
    )
    payload = {
        "latent_output_prefix": str(latent_output_prefix),
        "click_xy": [int(click_xy[0]), int(click_xy[1])],
        "selected_token": {
            "t": int(token_index[0]),
            "h": int(token_index[1]),
            "w": int(token_index[2]),
        },
        "selected_frame_index": int(selected_frame_index),
        "selected_video_frame_index": int(selected_video_frame),
        "display_fps": artifacts.display_fps,
        "similarity_video_path": str(artifacts.similarity_video_path),
        "similarity_video_shape": list(artifacts.similarity_video_shape),
        "similarity_range": [artifacts.similarity_min, artifacts.similarity_max],
    }
    next_tracking_state = {
        "latent_output_prefix": str(latent_output_prefix),
        "source_frames": source_frames,
        "display_fps": display_fps if display_fps > 0 else artifacts.display_fps,
        "latent_grid_shape": list(latent_grid.shape),
        "selected_token": list(token_index),
        "click_xy": list(click_xy),
        "selected_frame_index": int(selected_frame_index),
        "selected_video_frame_index": int(selected_video_frame),
    }
    return (
        preview_frame,
        _format_tracking_result_status(
            token_index,
            click_xy,
            artifacts.similarity_video_path,
            frame_index=selected_frame_index,
            video_frame_index=selected_video_frame,
        ),
        str(artifacts.similarity_video_path),
        _serialize_json(payload),
        next_tracking_state,
    )


def build_demo() -> gr.Blocks:
    description = (
        "Run the V-JEPA 2.1 pipeline in independent stages: extract latents, load latent files, "
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

        gr.Markdown("## 1. Extract latents from video")
        with gr.Row():
            video_input = gr.Video(label="Input video", sources=["upload"])
            with gr.Column():
                model_input = gr.Dropdown(choices=MODEL_CHOICES, value=DEFAULT_MODEL_NAME, label="Model")
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
        latent_prefix_input = gr.Textbox(label="Latent prefix (.npy / .metadata.json stem)")
        with gr.Accordion("Extraction metadata", open=False):
            extraction_metadata_output = gr.Code(label="Extraction metadata", language="json", value="{}")
        with gr.Accordion("System fit estimate", open=False):
            preflight_metadata_output = gr.Code(label="System fit estimate", language="json", value="{}")

        gr.Markdown("## 2. Load latent space")
        with gr.Row():
            latent_npy_input = gr.File(label="Latent grid (.npy)", file_types=[".npy"], type="filepath")
            latent_metadata_input = gr.File(label="Latent metadata (.json)", file_types=[".json"], type="filepath")
        load_latents_button = gr.Button("Load latents")
        latent_status_output = gr.Markdown()
        with gr.Accordion("Latent summary", open=False):
            latent_metadata_output = gr.Code(label="Latent summary", language="json", value="{}")

        gr.Markdown("## 3. Compute or load a projection")
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

        gr.Markdown("## 4. Build a plot from chosen projection components")
        with gr.Row():
            plot_dimensions_input = gr.Radio([2, 3], value=3, label="Plot dimensions")
            plot_max_points_input = gr.Slider(minimum=500, maximum=12000, step=500, value=4000, label="Max plotted points")
            plot_x_component_input = gr.Dropdown(choices=[1, 2, 3], value=1, label="X component")
            plot_y_component_input = gr.Dropdown(choices=[1, 2, 3], value=2, label="Y component")
            plot_z_component_input = gr.Dropdown(choices=[1, 2, 3], value=3, label="Z component")
        build_plot_button = gr.Button("Build plot")
        plot_status_output = gr.Markdown()
        plot_output = gr.Plot(label="Latent space plot")

        gr.Markdown("## 5. Create RGB latent videos from any 3 projection components")
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

        with gr.Tabs():
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
                tracking_preview_output = gr.Image(label="First frame (click to choose a patch)", type="numpy")
                tracking_video_output = gr.Video(label="Patch similarity heatmap video")
                with gr.Accordion("Patch similarity metadata", open=False):
                    tracking_metadata_output = gr.Code(label="Patch similarity metadata", language="json", value="{}")

        projection_method_input.change(
            fn=toggle_projection_controls,
            inputs=[projection_method_input],
            outputs=[umap_controls, pca_controls],
        )
        plot_dimensions_input.change(
            fn=toggle_plot_dimensions,
            inputs=[plot_dimensions_input],
            outputs=[plot_z_component_input],
        )

        estimate_button.click(
            fn=estimate_limits_step,
            inputs=[video_input, model_input, crop_height_input, crop_width_input, frames_input, sample_fps_input, start_second_input, device_input],
            outputs=[preflight_status_output, preflight_metadata_output],
        )

        extract_button.click(
            fn=extract_latents_step,
            inputs=[video_input, model_input, crop_height_input, crop_width_input, frames_input, sample_fps_input, start_second_input, device_input],
            outputs=[
                extraction_status_output,
                latent_prefix_input,
                extraction_metadata_output,
                latent_state,
                projection_state,
                projection_prefix_input,
                projection_metadata_output,
                plot_output,
                side_by_side_output,
                render_status_output,
                render_metadata_output,
            ],
            trigger_mode="once",
            concurrency_limit=1,
        )

        load_latents_button.click(
            fn=load_latents_step,
            inputs=[latent_prefix_input, latent_npy_input, latent_metadata_input, latent_state],
            outputs=[
                latent_status_output,
                latent_prefix_input,
                latent_metadata_output,
                latent_state,
                projection_state,
                projection_prefix_input,
                projection_metadata_output,
                plot_output,
                side_by_side_output,
                render_status_output,
                render_metadata_output,
            ],
        )

        compute_projection_button.click(
            fn=compute_projection_step,
            inputs=[
                latent_state,
                projection_method_input,
                projection_pca_mode_input,
                projection_components_input,
                umap_n_neighbors_input,
                umap_min_dist_input,
                umap_metric_input,
                umap_random_state_input,
            ],
            outputs=[
                projection_status_output,
                projection_prefix_input,
                projection_metadata_output,
                projection_state,
                plot_dimensions_input,
                plot_x_component_input,
                plot_y_component_input,
                plot_z_component_input,
                rgb_r_component_input,
                rgb_g_component_input,
                rgb_b_component_input,
                plot_output,
                plot_status_output,
                render_status_output,
                render_metadata_output,
            ],
        )

        load_projection_button.click(
            fn=load_projection_step,
            inputs=[projection_prefix_input, projection_state],
            outputs=[
                projection_status_output,
                projection_prefix_input,
                projection_metadata_output,
                projection_state,
                plot_dimensions_input,
                plot_x_component_input,
                plot_y_component_input,
                plot_z_component_input,
                rgb_r_component_input,
                rgb_g_component_input,
                rgb_b_component_input,
            ],
        )

        build_plot_button.click(
            fn=build_plot_step,
            inputs=[
                projection_state,
                plot_dimensions_input,
                plot_max_points_input,
                plot_x_component_input,
                plot_y_component_input,
                plot_z_component_input,
            ],
            outputs=[plot_output, plot_status_output],
        )

        create_rgb_button.click(
            fn=create_rgb_videos_step,
            inputs=[
                latent_state,
                projection_state,
                rgb_r_component_input,
                rgb_g_component_input,
                rgb_b_component_input,
                upscale_factor_input,
            ],
            outputs=[render_status_output, side_by_side_output, render_metadata_output],
        )

        prepare_tracking_button.click(
            fn=prepare_tracking_step,
            inputs=[latent_state, tracking_frame_input],
            outputs=[
                tracking_frame_input,
                tracking_preview_output,
                tracking_status_output,
                tracking_metadata_output,
                tracking_state,
                tracking_video_output,
            ],
        )

        tracking_frame_input.change(
            fn=prepare_tracking_step,
            inputs=[latent_state, tracking_frame_input],
            outputs=[
                tracking_frame_input,
                tracking_preview_output,
                tracking_status_output,
                tracking_metadata_output,
                tracking_state,
                tracking_video_output,
            ],
        )

        tracking_preview_output.select(
            fn=select_patch_similarity_step,
            inputs=[latent_state, tracking_state],
            outputs=[
                tracking_preview_output,
                tracking_status_output,
                tracking_video_output,
                tracking_metadata_output,
                tracking_state,
            ],
        )

    return demo
