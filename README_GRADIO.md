# Gradio UI guide

This document describes the current Gradio app in this workspace, the staged latent-space workflow it exposes, and the component-local modules that support it.

For a VOS-only explanation grounded in both the current code and the V-JEPA 2.1 paper, see `src/vjepa2_latents/gradio_components/segmentation/README.md`.
For a patch-similarity tracking-only explanation, see `src/vjepa2_latents/gradio_components/tracking/README.md`.

## Purpose

The app provides a browser UI for exploring V-JEPA 2.1 latents without opening the notebook.

It supports a staged workflow where each step can run independently:

1. estimate whether an extraction configuration is likely to fit the selected device
2. either extract new latents from a video or load a previously saved latent run
3. compute PCA, `umap-learn`, or `mlx-vis` projections with custom parameters
  - PCA supports global, spatial-only, and temporal-only modes
4. load previously saved projection artifacts
5. build a 2D or 3D interactive plot from chosen projection components
6. generate RGB latent videos from any selected 3 projection components
7. generate a side-by-side source/latent comparison video
8. click a patch in the first frame and render a cosine-similarity tracking heatmap over time
9. click one foreground point and one background point, then render a KNN-based binary segmentation mask video

## Files

### `app.py`

Thin launcher that:

- adds `src/` to `sys.path`
- imports `build_demo()` from `src/vjepa2_latents/gradio_app.py`
- constructs the app as `demo`
- launches the server when run directly

### `src/vjepa2_latents/gradio_app.py`

Main application layer and UI entrypoint.

Current responsibilities:

- define the staged Gradio UI
- keep loaded latent grids and projections in `gr.State` between button clicks
- wire the per-stage Gradio callbacks and return values
- estimate extraction pressure before a run with `estimate_limits_step()`
- delegate the merged latent-source flow to the modular latent-source package
- compute or load reusable PCA, `umap-learn`, or `mlx-vis` projection artifacts
- support global, spatial-only, and temporal-only PCA modes
- update component selectors dynamically based on projected dimensionality
- build 2D or 3D plots from arbitrary selected components
- create RGB videos from arbitrary selected projected components
- show a click-to-track first-frame preview for dense patch similarity
- map first-frame click coordinates onto latent tokens at `t=0`
- render cosine-similarity heatmaps over time as MP4 videos
- show a promptable VOS preview for one green foreground click and one red background click
- classify all latent tokens with a tiny foreground/background KNN in latent space
- render binary segmentation-mask overlays over time as MP4 videos
- keep the metadata-heavy outputs collapsed by default so the main controls stay uncluttered
- return status and JSON metadata for each stage separately
- emit concise `[vjepa2]` console progress logs for the main Gradio stages

### `src/vjepa2_latents/gradio_components/latent_source/`

Modular latent-source package for the merged first UI section.

Current responsibilities:

- define latent-source-specific constants such as default video/model/device values
- build the first Gradio section UI in its own module
- keep extraction and latent-loading callbacks outside the main UI file
- keep latent-source-specific formatting, path, and session helpers local to the component
- scan `.gradio_outputs/` for reusable latent runs
- build metadata-rich labels for saved latent selections
- refresh the saved-latent picker after new extractions or imported latent files
- toggle between the extract and load sub-views in the merged source section

### `src/vjepa2_latents/gradio_components/latent_source/extractor/`

Component-local extractor implementation package for the latent-source workflow.

Current responsibilities:

- keep extractor code grouped by responsibility beside the latent-source component
- expose the same public API through `src/vjepa2_latents/extractor.py`
- keep the CLI entrypoint available through `python -m vjepa2_latents.gradio_components.latent_source.extractor`

Package layout:

### `src/vjepa2_latents/gradio_components/latent_source/extractor/config.py`

- define `ModelSpec`, `MODEL_SPECS`, image-normalization constants, crop parsing, and device selection
- provide memory and token-estimation helpers used by preflight checks

### `src/vjepa2_latents/gradio_components/latent_source/extractor/checkpoint.py`

- validate, download, and load checkpoints
- expose vendored upstream `app/...` imports safely
- construct the upstream encoder and load pretrained weights

### `src/vjepa2_latents/gradio_components/latent_source/extractor/video.py`

- probe video metadata
- select frame windows and decode source frames
- apply resize, crop, and model-input preprocessing helpers
- prepare display-aligned source frames for visualization

### `src/vjepa2_latents/gradio_components/latent_source/extractor/tensor.py`

- run the encoder synchronously for accurate timing
- reshape flat tokens back into the latent spatiotemporal grid
- serialize `.npy`, optional `.pt`, and metadata outputs

### `src/vjepa2_latents/gradio_components/latent_source/extractor/pipeline.py`

- orchestrate end-to-end extraction
- expose `estimate_extraction_requirements()`
- define the CLI parser and `main()` entrypoint

### `src/vjepa2_latents/gradio_components/latent_source/extractor/utils/logging.py`

- keep shared extractor logging and timing helpers

The logging helpers now focus on structured `[vjepa2]` console output and timing summaries.

### `src/vjepa2_latents/gradio_utils.py`

Shared Gradio helper layer.

Current responsibilities:

- define the default video, model, crop, checkpoint, and output-directory constants
- define the model and projection choice lists used by the UI
- normalize file prefixes and projection settings
- format the stage-by-stage status strings and metadata JSON
- build the `gr.State` payloads for latents and projections
- keep the small console logging helper used by the Gradio callbacks

### `src/vjepa2_latents/gradio_components/projection/core.py`

Projection-specific helpers.

Current responsibilities:

- define the projection method registry and display-name normalization
- detect optional `umap-learn` and `mlx-vis` support
- flatten latent grids into feature rows and `(t, h, w)` coordinates
- compute PCA, UMAP, and optional `mlx-vis` projections
- persist reusable projection bundles and load them back from disk
- summarize latent tensors for UI display

### `src/vjepa2_latents/gradio_components/plot/core.py`

Plot-specific helpers.

Current responsibilities:

- build Plotly figures from latent tensors or saved projections
- validate selected component indices for 2D and 3D plots
- label axes using the projection-method-aware component naming helpers

### `src/vjepa2_latents/gradio_components/render/video.py`

Render and media-export helpers.

Current responsibilities:

- convert chosen projection components into RGB frames
- infer display FPS for latent-aligned video exports
- load and crop source frames so they align with the latent grid
- compose side-by-side videos with aspect-ratio-aware padding
- export browser-friendly `.mp4` outputs with `ffmpeg`

### `src/vjepa2_latents/gradio_components/tracking/core.py`

Tracking-specific similarity helpers.

Current responsibilities:

- map clicked image coordinates to latent token indices
- compute token-to-token cosine similarity volumes
- annotate the selected or peak-response patch on preview frames
- blend hotspot-focused similarity heatmaps over source frames
- export patch-similarity tracking videos

### `src/vjepa2_latents/extractor.py`

Public extractor API and orchestration layer reused by the UI.

It now re-exports the latent-source-local extractor package while keeping the existing import path stable.

Current responsibilities:

- expose the stable extractor API used by the Gradio app, tests, and CLI wrapper
- act as the compatibility facade for the extractor code now stored under the latent-source component

### `src/vjepa2_latents/gradio_components/segmentation/`

Reusable segmentation component package.

This folder now cleanly holds both the Gradio-tab wiring and the non-UI VOS domain logic that belongs with the segmentation feature.

Current responsibilities:

- keep VOS propagation and rendering logic in `core.py`
- keep VOS-specific status-formatting helpers in `status.py`
- keep the segmentation UI and callbacks beside the feature-specific core logic
- provide a focused `README.md` for the VOS feature area

The old top-level `video.py`, `visualization.py`, and `vos.py` files are removed so plotting, rendering, tracking, and segmentation logic now live beside their owning Gradio components.

The extractor stack still supports the same workflow features:

- rectangular crops via `normalize_crop_size()` and `parse_crop_size()`
- preflight memory and token estimates via `estimate_extraction_requirements()`
- synchronous encoder wall-time measurement via `run_encoder_synchronously()`
- per-stage timing capture for encoder setup, encoder run, latent reshaping, CPU copy, metadata write, and file writes
- safer checkpoint validation and redownload logic
- fallback checkpoint-key resolution across multiple checkpoint layouts
- display-frame preprocessing that matches model crop behavior
- optional omission of `.pt` output when `save_pt=False`
- vendored upstream `app/...` import compatibility even with a root-level `app.py`

### `src/vjepa2_latents/__init__.py`

Package exports were expanded so the visualization helpers are importable from `vjepa2_latents` directly.

### `tests/test_shape_math.py`

Focused extractor and shape-utility coverage for:

- rectangular crop parsing and normalization
- model registry coverage for all exposed backbones
- checkpoint-key fallback behavior
- extraction preflight estimation
- reshape sub-step timing capture
- cached checkpoint redownload behavior
- optional `.pt` skipping during output serialization
- output serialization timing metadata and logging

### `tests/test_visualization.py`

Focused visualization coverage for:

- PCA, UMAP, and optional `mlx-vis` projection helpers
- projection artifact round-tripping
- 2D and 3D figure generation
- RGB rendering from selected components
- patch-similarity click mapping, cosine similarity, and heatmap overlays
- side-by-side frame layout and aspect-ratio padding
- even-dimension video padding helpers

### `requirements.txt`

Relevant UI dependencies:

- `gradio>=5.0`
- `plotly>=5.18`
- `umap-learn>=0.5`
- `mlx-vis>=0.7` on Apple Silicon, with `mlx>=0.20.0`

## Architecture

The UI is intentionally layered on top of the existing extractor rather than creating a second inference path.

Current high-level flow:

```text
browser UI
  -> Gradio Blocks app
  -> estimate_limits_step(...) [optional]
  -> extract_latents_step(...) or load_latents_step(...)
  -> compute_projection_step(...) or load_projection_step(...)
  -> build_plot_step(...)
  -> create_rgb_videos_step(...)
  -> prepare_tracking_step(...)
  -> select_patch_similarity_step(...)
  -> prepare_segmentation_step(...)
  -> select_segmentation_prompt_step(...)
  -> run_segmentation_step(...)
  -> return stage-specific status + artifacts + metadata
```

## Current UI layout

The app is built in `build_demo()` and split into 4 main sections plus analysis tabs for tracking and VOS, with a merged latent-source section at the top.

### 1. Choose latent source

The first section now asks the user to choose between:

- `Extract new latents`
- `Load saved latents`

#### Extract new latents

Inputs:

- input video upload
- model dropdown, defaulting to `vit_base_384` / 80M
- crop height, defaulting to `384`
- crop width, defaulting to `384`
- frame count
- sample FPS
- start second
- device

Buttons:

- `Estimate system fit`
- `Extract latents`

Outputs:

- extraction status markdown
- preflight estimate markdown
- latent prefix text field auto-filled with the saved stem
- extraction metadata JSON, collapsed by default
- preflight estimate JSON, collapsed by default

Notes:

- if `testvideo.mp4` exists in the workspace root, it is exposed as a Gradio example
- on macOS, the default device is `mps`
- Gradio extraction calls `extract_latents(..., save_pt=False)`, so the UI path persists `.npy` and `.metadata.json` but skips `.pt`

#### Load saved latents

Inputs:

- saved latent-run picker populated from `.gradio_outputs/`
- optional latent `.npy` upload
- optional latent `.json` metadata upload

Saved-run labels include a timestamp plus available metadata such as:

- source video name
- model name
- frame count
- crop size
- latent-grid dimensions

Button:

- `Refresh saved runs`
- `Load latents`

Outputs:

- active latent prefix textbox
- latent summary status
- latent summary JSON, collapsed by default

### 2. Compute or load a projection

Inputs:

- projection method: `PCA`, `UMAP`, or `mlx-vis` reducers (`UMAP-MLX`, `t-SNE-MLX`, `PaCMAP-MLX`, `LocalMAP-MLX`, `TriMap-MLX`, `DREAMS-MLX`, `CNE-MLX`, `MMAE-MLX`)
- PCA mode: `Global PCA`, `Spatial-only PCA`, or `Temporal-only PCA`
- projected component count
- neighbor count for `UMAP`, `UMAP-MLX`, `PaCMAP-MLX`, and `LocalMAP-MLX`
- UMAP `min_dist`
- distance metric
- random state
- optional projection prefix textbox

Buttons:

- `Compute projection`
- `Load saved projection`

Outputs:

- projection status
- projection metadata JSON, collapsed by default
- dynamically updated component selectors for plotting and RGB rendering

Behavior:

- neighbor / manifold controls are hidden unless a neighbor-tuned reducer is selected
- PCA mode controls are hidden unless `PCA` is selected
- component pickers are repopulated to match the saved or computed projection dimensionality
- 3D plotting is disabled automatically when fewer than 3 components exist
- MLX reducers raise a clear install hint when `mlx-vis` is not available

### 3. Build a plot from chosen projection components

Inputs:

- 2D or 3D plot mode
- max plotted points, auto-expanded to the full projection size after compute/load
- animate over time from `t0` to `tn`
- chosen X/Y/Z components

Button:

- `Build plot`

Outputs:

- interactive Plotly figure
- plot status

Behavior:

- static mode shows the selected projection components as a single latent-space scatter
- animation mode adds Plotly play/pause controls and a time slider so you can watch embeddings evolve across successive time steps
- the animation reuses the same component selection and `max_points` setting, then groups points by time step for each frame

### 4. Create RGB latent videos from chosen projection components

Inputs:

- chosen R/G/B projection components
- upscale factor

Button:

- `Create RGB videos`

Outputs:

- side-by-side source/latent video
- render metadata JSON, collapsed by default

Note:

- the latent-only RGB video is still written to disk and reported in metadata, but the UI presents the side-by-side video as the main visual output

### 5. Patch Similarity / Dense Tracking

Inputs:

- loaded latent state from the earlier extract/load stages
- a frame selector for choosing one of the available source frames
- a click inside the selected source frame

Buttons / interactions:

- `Show first frame`
- change the frame selector to inspect a different source frame
- click directly on the selected frame image to choose a patch/object

Outputs:

- first-frame preview with the selected latent patch outlined
- frame selector populated with the available source-frame choices
- patch-similarity status
- heatmap-over-video `.mp4` showing cosine similarity for the selected token across all `(t, h, w)` positions
- tracking metadata JSON, collapsed by default

Behavior:

- the first frame is decoded with the same resize-and-center-crop pipeline used for model input display
- click coordinates are mapped to the latent token grid for the selected frame
- the selected token is compared to every token in the latent grid with cosine similarity
- similarity scores are normalized to emphasize the strongest positive responses and blended with per-pixel opacity so hotspots stand out clearly
- each frame also marks the best-matching patch location, while the first frame keeps the originally selected patch outlined in a brighter accent color
- Gradio extraction calls `extract_latents(..., save_pt=False)`, so the UI path persists `.npy` and `.metadata.json` but skips `.pt`

### 7. Foreground / Background VOS

For the dedicated VOS walkthrough, paper comparison, and code map, see `src/vjepa2_latents/gradio_components/segmentation/README.md`.

Inputs:

- loaded latent state from the earlier extract/load stages
- a frame selector for choosing one of the available source frames
- a radio that decides whether the next click sets the foreground or background prompt
- one green foreground click and one red background click on the selected frame
- top-`k` neighbor count, defaulting to the paper-style `5`

Buttons / interactions:

- `Show frame for VOS prompts`
- click the frame once for foreground and once for background
- `Run VOS segmentation`

Outputs:

- prompt-frame preview with green/red prompt markers
- VOS segmentation status
- binary segmentation-mask overlay `.mp4`
- segmentation metadata JSON, collapsed by default

Behavior:

- prompts are constrained to the first frame to match the paper's supervision setup
- clicked foreground/background latent tokens seed the initial label set
- weighted top-`k` cosine propagation uses paper-style defaults: `temperature=0.2`, `context_frames=15`, `spatial_radius=12`
- later frames propagate labels using the first frame plus a rolling bank of past-frame predictions
- the resulting binary mask is resized back onto the display frames and blended as a green segmentation overlay
- the first frame keeps the original foreground/background prompt markers so the segmentation source is obvious
- this remains a sparse-prompt demo, so it is closer to the paper's propagation mechanics than to its exact dense-mask benchmark protocol

These videos are encoded for direct browser playback with H.264 (`libx264`) and `yuv420p`.

### Patch-similarity artifacts

Saved by the tracking step under a `tracking/` directory beside the latent files:

- `patch_similarity_t<t>-h<h>-w<w>.mp4`
- `vos_segmentation_fg_t<t>-h<h>-w<w>_bg_t<t>-h<h>-w<w>.mp4`

Tracking metadata includes:

- clicked image coordinates
- selected latent token indices
- similarity value range
- peak patch location per frame when rendered
- rendered video path, shape, and playback FPS

Segmentation metadata includes:

- foreground/background click coordinates
- foreground/background latent token indices
- KNN neighbor count
- propagation temperature, context-frame count, and spatial radius
- per-frame foreground coverage ratio
- rendered video path, shape, and playback FPS

## Visualization details

### Plotting

Plotting is driven by `build_projection_figure_from_data()`.

That means the app can:

- build plots directly from saved projections
- choose any valid 2 components for 2D
- choose any valid 3 components for 3D
- animate the chosen embedding components across time with a frame slider from `t0` to `tn`
- reuse PCA or UMAP results without recomputation
- reuse saved `mlx-vis` projections without recomputation
- label PCA axes with explained-variance percentages when available
- isolate static structure with spatial-only PCA or motion signatures with temporal-only PCA

### RGB rendering

RGB rendering is driven by `projection_rgb_frames()` and `create_visualizations_from_projection()`.

That means the app can:

- use PCA, `umap-learn`, or `mlx-vis` projections
- use more than 3 projected components overall
- let the user choose any 3 projected components for RGB mapping
- reuse a saved projection instead of recomputing it
- write filenames that encode both method and component selection

### Patch similarity / dense tracking

Patch similarity is driven by a clicked first-frame token.

That means the app can:

- let the user probe a specific object or region interactively
- map that click into the latent token lattice at `t=0`
- compare one selected token with every token in the clip using cosine similarity
- visualize token affinity over time as a clearer hotspot-focused heatmap overlay
- show the best-matching patch at each frame and the original selection on the first frame
- approximate the paper-style non-parametric label-propagation intuition without fine-tuning

### Foreground / background VOS

The new VOS component uses the same latent grid but switches from single-token cosine scoring to a paper-aligned weighted top-`k` label-propagation scheme seeded by two clicks.

That means the app can:

- let the user provide one positive prompt and one negative prompt directly on the source frame
- map both prompts into first-frame latent tokens
- propagate binary labels with weighted cosine top-`k` matching, temperature scaling, temporal context, and a local spatial neighborhood
- render a binary object mask overlay that matches the paper's propagation mechanics more closely
- save the segmentation overlay video beside the other tracking artifacts

## `mlx-vis` integration notes

The `mlx-vis` integration is additive.

Current behavior:

- saved projection artifacts use the same `.projection.npz` and `.projection.metadata.json` format across PCA, `umap-learn`, and `mlx-vis`
- Plotly remains the interactive plotting layer for both 2D and 3D exploration
- RGB latent videos still run through the existing latent-grid renderer, so any saved projection can drive the same side-by-side workflow
- neighbor-count controls are reused for `UMAP-MLX`, `PaCMAP-MLX`, and `LocalMAP-MLX`
- `TriMap-MLX`, `DREAMS-MLX`, `CNE-MLX`, and `MMAE-MLX` currently use their library defaults plus the selected component count

### Side-by-side rendering

The side-by-side video preserves aspect ratio.

Current behavior:

- source frames go through the same resize-and-center-crop pipeline as model input
- source and latent frames are fitted into a shared panel size
- frames are scaled uniformly
- remaining space is padded instead of stretched
- odd frame dimensions are padded to even sizes before encoding

## Current limitations

Known constraints in the current version:

- only one video is processed at a time
- projection loading is prefix-based rather than direct `.projection.npz` and metadata upload
- `ffmpeg` must be available on the system, or `imageio-ffmpeg` must be installed so the app can find a compatible encoder
- no explicit cleanup policy exists for old `.gradio_outputs/` runs or tracking videos
- UMAP can still be slower than PCA for large latent sets
- RGB videos are analytic visualizations, not reconstructions of the source video
- the app waits for extraction to finish before returning final outputs

## Recommended improvements

Recommended next improvements, in priority order:

1. add direct projection file uploads for `.projection.npz` and `.projection.metadata.json`
2. expose latent, projection, and render artifacts as downloadable files
3. add optional stage chaining such as extract → load or project → plot
4. cache projections for identical latent prefixes and reducer settings
5. cache decoded tracking frames and rendered similarity videos by selected token
6. record timing and file-size metadata per stage
7. add source-frame alignment and letterbox controls
8. experiment with `mlx-vis` GPU animation exports for point-cloud-only previews when a 2D embedding is selected
9. add maintenance actions for `.gradio_outputs/`
10. add more analysis views such as temporal trajectories or patch-neighborhood comparisons

## Quick code map

- app bootstrap: `app.py`
- package exports: `src/vjepa2_latents/__init__.py`
- UI construction: `src/vjepa2_latents/gradio_app.py::build_demo`
- latent-source callbacks: `src/vjepa2_latents/gradio_components/latent_source/callbacks.py`
- saved-latent catalog: `src/vjepa2_latents/gradio_components/latent_source/catalog.py`
- latent-source config: `src/vjepa2_latents/gradio_components/latent_source/config.py`
- latent-source helpers: `src/vjepa2_latents/gradio_components/latent_source/helpers.py`
- latent-source UI builder: `src/vjepa2_latents/gradio_components/latent_source/ui.py`
- latent-source extractor package: `src/vjepa2_latents/gradio_components/latent_source/extractor/`
- latent-source extractor pipeline: `src/vjepa2_latents/gradio_components/latent_source/extractor/pipeline.py`
- latent-source extractor config helpers: `src/vjepa2_latents/gradio_components/latent_source/extractor/config.py`
- latent-source extractor checkpoint helpers: `src/vjepa2_latents/gradio_components/latent_source/extractor/checkpoint.py`
- latent-source extractor video helpers: `src/vjepa2_latents/gradio_components/latent_source/extractor/video.py`
- latent-source extractor tensor helpers: `src/vjepa2_latents/gradio_components/latent_source/extractor/tensor.py`
- latent-source extractor logging helpers: `src/vjepa2_latents/gradio_components/latent_source/extractor/utils/logging.py`
- extractor facade: `src/vjepa2_latents/extractor.py`
- estimate step: `src/vjepa2_latents/gradio_components/latent_source/callbacks.py::estimate_limits_step`
- extract step: `src/vjepa2_latents/gradio_components/latent_source/callbacks.py::extract_latents_step`
- latent loading step: `src/vjepa2_latents/gradio_components/latent_source/callbacks.py::load_latents_step`
- projection step: `src/vjepa2_latents/gradio_app.py::compute_projection_step`
- projection loading step: `src/vjepa2_latents/gradio_app.py::load_projection_step`
- plot step: `src/vjepa2_latents/gradio_app.py::build_plot_step`
- render step: `src/vjepa2_latents/gradio_app.py::create_rgb_videos_step`
- tracking preview step: `src/vjepa2_latents/gradio_components/tracking/callbacks.py::prepare_tracking_step`
- tracking click handler: `src/vjepa2_latents/gradio_components/tracking/callbacks.py::select_patch_similarity_step`
- segmentation preview step: `src/vjepa2_latents/gradio_components/segmentation/callbacks.py::prepare_segmentation_step`
- segmentation click handler: `src/vjepa2_latents/gradio_components/segmentation/callbacks.py::select_segmentation_prompt_step`
- segmentation runner: `src/vjepa2_latents/gradio_components/segmentation/callbacks.py::run_segmentation_step`
- shared Gradio helpers: `src/vjepa2_latents/gradio_utils.py`
- projection helpers: `src/vjepa2_latents/gradio_components/projection/core.py::compute_projection_bundle`
- optional MLX reducer helper: `src/vjepa2_latents/gradio_components/projection/core.py::compute_mlx_projection`
- saved projection loading: `src/vjepa2_latents/gradio_components/projection/core.py::load_saved_projection`
- projection summary helper: `src/vjepa2_latents/gradio_components/projection/core.py::summarize_latents`
- plot builder from saved data: `src/vjepa2_latents/gradio_components/plot/core.py::build_projection_figure_from_data`
- RGB rendering from selected components: `src/vjepa2_latents/gradio_components/render/video.py::projection_rgb_frames`
- side-by-side media export: `src/vjepa2_latents/gradio_components/render/video.py::create_visualizations_from_projection`
- click-to-token mapping: `src/vjepa2_latents/gradio_components/tracking/core.py::map_click_to_latent_token`
- dense similarity computation: `src/vjepa2_latents/gradio_components/tracking/core.py::cosine_similarity_volume`
- tracking video export: `src/vjepa2_latents/gradio_components/tracking/core.py::create_patch_similarity_video`
- VOS token classification: `src/vjepa2_latents/gradio_components/segmentation/core.py::knn_binary_segmentation_volume`
- segmentation video export: `src/vjepa2_latents/gradio_components/segmentation/core.py::create_segmentation_video`
- extractor preflight and crop helpers: `src/vjepa2_latents/extractor.py::estimate_extraction_requirements`, `normalize_crop_size`, `parse_crop_size`, `prepare_display_frames`

## Local usage

```zsh
cd /Users/pishty/ws/vjepa2.1
source .venv/bin/activate
python app.py
```

## Validation

Focused validation that matches the current changes:

```zsh
cd /Users/pishty/ws/vjepa2.1
env NUMBA_DISABLE_JIT=1 /Users/pishty/ws/vjepa2.1/.venv/bin/python -m pytest tests/test_gradio_app.py tests/test_visualization.py tests/test_shape_math.py -q
env NUMBA_DISABLE_JIT=1 /Users/pishty/ws/vjepa2.1/.venv/bin/python -m unittest -v tests.test_gradio_app tests.test_visualization tests.test_shape_math
/Users/pishty/ws/vjepa2.1/.venv/bin/python -m unittest -v tests.test_shape_math
/Users/pishty/ws/vjepa2.1/.venv/bin/python -c "from app import build_demo; demo = build_demo(); print(type(demo).__name__)"
```

The current focused extractor timing/unit test run passes `20` tests in `tests/test_shape_math.py`.

The current test suite also includes mocked coverage for MLX reducer dispatch and the missing-dependency error path.

## Summary

The app is now a modular latent exploration workflow rather than a single-shot demo.

It currently lets you:

- estimate likely memory pressure before extraction
- extract and reload latents
- compute and reload PCA, `umap-learn`, or `mlx-vis` projections
- choose arbitrary projection components for plots
- choose arbitrary projection components for RGB rendering
- create non-squashed side-by-side videos for source vs latent comparison
- click a first-frame patch and inspect dense cosine-similarity tracking over time
- click foreground/background prompts and inspect a binary KNN segmentation mask over time

That makes it much easier to iterate on latent analysis without rerunning the whole pipeline each time.
