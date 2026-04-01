from __future__ import annotations

import json
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

import gradio as gr

from .extractor import MODEL_SPECS, estimate_extraction_requirements, extract_latents, probe_video, select_frame_indices
from .visualization import (
    build_projection_figure_from_data,
    compute_projection_bundle,
    create_visualizations_from_projection,
    load_saved_latents,
    load_saved_projection,
    save_projection_artifacts,
    summarize_latents,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = ROOT / "testvideo.mp4"
VENDOR_REPO = ROOT / "vendor" / "vjepa2"
CHECKPOINT_DIR = ROOT / "checkpoints"
APP_OUTPUT_DIR = ROOT / ".gradio_outputs"
DEFAULT_DEVICE = "mps" if platform.system() == "Darwin" else "auto"

MODEL_LABELS = {
    "vit_base_384": "ViT-B/16 · 80M · 384",
    "vit_large_384": "ViT-L/16 · 300M · 384",
    "vit_giant_384": "ViT-g/16 · 1B · 384",
    "vit_gigantic_384": "ViT-G/16 · 2B · 384",
}

MODEL_CHOICES = [
    (MODEL_LABELS.get(model_name, model_name), model_name)
    for model_name in sorted(MODEL_SPECS.keys())
]


def _resolve_video_path(video_file: str | None) -> Path:
    if video_file:
        return Path(video_file).resolve()
    if DEFAULT_VIDEO.exists():
        return DEFAULT_VIDEO.resolve()
    raise gr.Error("Upload a video or provide an example video first.")


def _create_session_dir(prefix: str) -> Path:
    APP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=APP_OUTPUT_DIR))


def _normalize_prefix(prefix_text: str | None, suffixes: Sequence[str]) -> Path | None:
    if not prefix_text or not prefix_text.strip():
        return None
    candidate = Path(prefix_text.strip()).expanduser().resolve()
    candidate_text = str(candidate)
    for suffix in suffixes:
        if candidate_text.endswith(suffix):
            return Path(candidate_text[: -len(suffix)])
    return candidate


def _serialize_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2)


def _load_latent_metadata(output_prefix: Path) -> dict[str, Any]:
    return json.loads(output_prefix.with_suffix(".metadata.json").read_text(encoding="utf-8"))


def _format_extraction_status(
    result: dict[str, Any],
    output_prefix: Path,
    *,
    model_name: str | None = None,
    video_path: Path | None = None,
) -> str:
    outputs = result.get("outputs", {})
    model_value = result.get("model") or model_name or "unknown"
    video_value = result.get("video_path") or (str(video_path) if video_path is not None else "unknown")
    return "\n".join(
        [
            "## Latents extracted",
            f"- output prefix: `{output_prefix}`",
            f"- device: `{result.get('device', 'unknown')}`",
            f"- model: `{model_value}`",
            f"- video: `{video_value}`",
            f"- latent tensor: `{outputs.get('npy', output_prefix.with_suffix('.npy'))}`",
            "- next: use **Load latents** to inspect the latent grid before projecting it.",
        ]
    )


def _format_preflight_status(preflight: dict[str, Any], video_meta: dict[str, Any], frame_indices: Sequence[int]) -> str:
    risk_level = str(preflight["risk_level"]).upper()
    range_text = f"{frame_indices[0]}..{frame_indices[-1]}" if frame_indices else "n/a"
    lines = [
        "## System fit estimate",
        f"- risk: `{risk_level}`",
        f"- device: `{preflight['device']}`",
        f"- selected frames: `{len(frame_indices)}` (`{range_text}`)",
        f"- source video: `{video_meta['frame_count']} frames @ {video_meta['fps']:.3f} fps, {video_meta['width']}x{video_meta['height']}`",
        f"- patch tokens: `{preflight['token_count']}` = `{preflight['time_patches']} x {preflight['height_patches']} x {preflight['width_patches']}`",
        f"- latent tensor size: `{preflight['latent_tensor_mib']:.1f} MiB`",
        f"- attention pressure vs 16f/384 baseline: `{preflight['quadratic_token_factor_vs_16f_384']:.2f}x`",
    ]

    mps_limit = preflight.get("mps_recommended_max_memory_gib")
    if mps_limit is not None:
        lines.append(f"- MPS recommended memory cap: `{mps_limit:.2f} GiB`")

    if preflight["device"].startswith("mps"):
        if preflight["risk_level"] == "high":
            lines.append("- warning: this configuration is likely to hit Apple MPS memory limits; reduce frames or crop size, or switch to CPU.")
        elif preflight["risk_level"] == "medium":
            lines.append("- warning: this configuration may be unstable on Apple MPS; watch for OOM during forward pass or CPU copy.")
        else:
            lines.append("- note: the saved `.npy` file is not the main memory cost; peak memory happens during attention and MPS synchronization.")

    return "\n".join(lines)


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
    return _format_preflight_status(preflight, video_meta, frame_indices), _serialize_json(payload)


def _format_latent_status(output_prefix: Path, metadata: dict[str, Any], summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "## Latents loaded",
            f"- prefix: `{output_prefix}`",
            f"- latent grid: `{metadata['latent_grid_shape']}`",
            f"- input tensor: `{metadata['input_tensor_shape']}`",
            f"- raw tokens: `{metadata['raw_token_shape']}`",
            f"- patch norm mean/std: `{summary['patch_norm_mean']:.4f}` / `{summary['patch_norm_std']:.4f}`",
            "- next: compute PCA or UMAP with the projection controls.",
        ]
    )


def _format_projection_status(output_prefix: Path, metadata: dict[str, Any]) -> str:
    labels = metadata.get("component_labels", [])
    return "\n".join(
        [
            "## Projection ready",
            f"- prefix: `{output_prefix}`",
            f"- method: `{metadata['method']}`",
            f"- components: `{metadata['settings']['n_components']}`",
            f"- labels: `{', '.join(labels)}`",
            "- next: build a plot and/or generate RGB videos from any 3 projected components.",
        ]
    )


def _format_plot_status(method: str, component_indices: Sequence[int]) -> str:
    selection = ", ".join(f"C{index + 1}" for index in component_indices)
    return "\n".join(
        [
            "## Plot updated",
            f"- method: `{method}`",
            f"- plotted components: `{selection}`",
        ]
    )


def _format_render_status(method: str, rgb_components: Sequence[int], latent_video_path: Path, side_by_side_video_path: Path) -> str:
    component_text = ", ".join(f"C{index + 1}" for index in rgb_components)
    return "\n".join(
        [
            "## RGB videos created",
            f"- method: `{method}`",
            f"- RGB components: `{component_text}`",
            f"- latent video: `{latent_video_path}`",
            f"- side-by-side video: `{side_by_side_video_path}`",
        ]
    )


def _format_hint_status(title: str, message: str) -> str:
    return "\n".join([f"## {title}", f"- {message}"])


def _normalize_projection_settings(
    projection_method: str,
    projection_components: int | float,
    umap_n_neighbors: int | float,
    umap_min_dist: float,
    umap_metric: str,
    umap_random_state: float | None,
) -> dict[str, Any]:
    return {
        "method": projection_method.strip().lower(),
        "n_components": max(2, int(projection_components)),
        "umap_n_neighbors": max(2, int(umap_n_neighbors)),
        "umap_min_dist": float(umap_min_dist),
        "umap_metric": umap_metric,
        "umap_random_state": None if umap_random_state in (None, "") else int(umap_random_state),
    }


def _component_selector_updates(component_count: int):
    choices = list(range(1, component_count + 1))
    plot_dimensions_value = 3 if component_count >= 3 else 2
    z_value = 3 if component_count >= 3 else None
    return (
        gr.update(choices=[2, 3] if component_count >= 3 else [2], value=plot_dimensions_value),
        gr.update(choices=choices, value=1),
        gr.update(choices=choices, value=2 if component_count >= 2 else 1),
        gr.update(choices=choices, value=z_value),
        gr.update(choices=choices, value=1),
        gr.update(choices=choices, value=2 if component_count >= 2 else 1),
        gr.update(choices=choices, value=z_value),
    )


def _projection_state(output_prefix: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "output_prefix": str(output_prefix),
        "method": metadata["method"],
        "component_count": int(metadata["settings"]["n_components"]),
        "latent_output_prefix": metadata.get("latent_output_prefix"),
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
    video_path = _resolve_video_path(video_file)
    session_dir = _create_session_dir("vjepa2-extract-")
    local_video_path = session_dir / video_path.name
    if video_path != local_video_path:
        shutil.copy2(video_path, local_video_path)

    output_prefix = session_dir / "latents"
    progress(0.1, desc="Running V-JEPA encoder")
    try:
        result = extract_latents(
            video_path=local_video_path,
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
        "video_path": str(local_video_path),
        "latent_output_prefix": str(output_prefix),
        "latent_npy_path": str(output_prefix.with_suffix(".npy")),
        "latent_metadata_path": str(output_prefix.with_suffix(".metadata.json")),
    }
    status = _format_extraction_status(result, output_prefix, model_name=model_name, video_path=local_video_path)
    latent_state = {"output_prefix": str(output_prefix), "session_dir": str(session_dir)}
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
        **metadata,
        "summary": summary,
        "latent_output_prefix": str(output_prefix),
    }
    status = _format_latent_status(output_prefix, metadata, summary)
    next_state = {"output_prefix": str(output_prefix), "session_dir": str(output_prefix.parent)}
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
    return gr.update(visible=projection_method.strip().lower() == "umap")


def compute_projection_step(
    latent_state: dict[str, Any] | None,
    projection_method: str,
    projection_components: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    umap_random_state: float | None,
):
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
    latent_grid, _ = load_saved_latents(latent_output_prefix)
    settings = _normalize_projection_settings(
        projection_method,
        projection_components,
        umap_n_neighbors,
        umap_min_dist,
        umap_metric,
        umap_random_state,
    )
    projection_bundle = compute_projection_bundle(latent_grid, **settings)
    projection_output_prefix = latent_output_prefix.parent / f"projection_{settings['method']}_{projection_bundle['projection'].shape[1]}"
    artifacts = save_projection_artifacts(
        projection_output_prefix,
        projection_bundle,
        latent_output_prefix=latent_output_prefix,
    )
    _, _, metadata = load_saved_projection(projection_output_prefix)
    payload = {
        **metadata,
        "projection_output_prefix": str(projection_output_prefix),
        "projection_npz_path": str(artifacts.projection_path),
        "projection_metadata_path": str(artifacts.metadata_path),
    }
    status = _format_projection_status(projection_output_prefix, metadata)
    selector_updates = _component_selector_updates(artifacts.projection_shape[1])
    return (
        status,
        str(projection_output_prefix),
        _serialize_json(payload),
        _projection_state(projection_output_prefix, metadata),
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

    projection, _, metadata = load_saved_projection(output_prefix)
    payload = {
        **metadata,
        "projection_output_prefix": str(output_prefix),
        "projection_shape": list(projection.shape),
    }
    status = _format_projection_status(output_prefix, metadata)
    selector_updates = _component_selector_updates(projection.shape[1])
    return (
        status,
        str(output_prefix),
        _serialize_json(payload),
        _projection_state(output_prefix, metadata),
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
    if not projection_state or not projection_state.get("output_prefix"):
        return None, _format_hint_status("Plot not ready", "Compute or load a projection before building a plot.")

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
    return figure, _format_plot_status(metadata["method"], component_indices)


def create_rgb_videos_step(
    latent_state: dict[str, Any] | None,
    projection_state: dict[str, Any] | None,
    rgb_r_component: int,
    rgb_g_component: int,
    rgb_b_component: int | None,
    upscale_factor: int,
):
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
    )
    return status, str(artifacts.side_by_side_video_path), _serialize_json(payload)


def build_demo() -> gr.Blocks:
    description = (
        "Run the V-JEPA 2.1 pipeline in independent stages: extract latents, load latent files, "
        "compute PCA or UMAP projections, build plots, and generate RGB videos from any 3 projected components."
    )

    with gr.Blocks(title="V-JEPA 2.1 Latent Explorer") as demo:
        gr.Markdown("# V-JEPA 2.1 Latent Explorer")
        gr.Markdown(description)

        latent_state = gr.State(value=None)
        projection_state = gr.State(value=None)

        gr.Markdown("## 1. Extract latents from video")
        with gr.Row():
            video_input = gr.Video(label="Input video", sources=["upload"])
            with gr.Column():
                model_input = gr.Dropdown(choices=MODEL_CHOICES, value="vit_large_384", label="Model")
                crop_height_input = gr.Slider(minimum=256, maximum=768, step=128, value=256, label="Crop height")
                crop_width_input = gr.Slider(minimum=256, maximum=768, step=128, value=256, label="Crop width")
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
        extraction_metadata_output = gr.Code(label="Extraction metadata", language="json", value="{}")
        preflight_metadata_output = gr.Code(label="System fit estimate", language="json", value="{}")

        gr.Markdown("## 2. Load latent space")
        with gr.Row():
            latent_npy_input = gr.File(label="Latent grid (.npy)", file_types=[".npy"], type="filepath")
            latent_metadata_input = gr.File(label="Latent metadata (.json)", file_types=[".json"], type="filepath")
        load_latents_button = gr.Button("Load latents")
        latent_status_output = gr.Markdown()
        latent_metadata_output = gr.Code(label="Latent summary", language="json", value="{}")

        gr.Markdown("## 3. Compute or load a projection")
        with gr.Row():
            with gr.Column():
                projection_method_input = gr.Radio(["PCA", "UMAP"], value="PCA", label="Projection method")
                projection_components_input = gr.Slider(minimum=2, maximum=5, step=1, value=5, label="Projected components")
                with gr.Group(visible=False) as umap_controls:
                    umap_n_neighbors_input = gr.Slider(minimum=2, maximum=200, step=1, value=15, label="UMAP neighbors")
                    umap_min_dist_input = gr.Slider(minimum=0.0, maximum=0.99, step=0.01, value=0.1, label="UMAP min_dist")
                    umap_metric_input = gr.Dropdown(
                        choices=["euclidean", "cosine", "manhattan", "chebyshev", "correlation"],
                        value="euclidean",
                        label="UMAP metric",
                    )
                    umap_random_state_input = gr.Number(value=42, precision=0, label="UMAP random state")
                compute_projection_button = gr.Button("Compute projection")
            with gr.Column():
                projection_prefix_input = gr.Textbox(label="Projection prefix (.projection.npz stem)")
                load_projection_button = gr.Button("Load saved projection")

        projection_status_output = gr.Markdown()
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
        render_metadata_output = gr.Code(label="Render metadata", language="json", value="{}")

        projection_method_input.change(
            fn=toggle_projection_controls,
            inputs=[projection_method_input],
            outputs=[umap_controls],
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

    return demo
