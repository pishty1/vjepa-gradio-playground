# Gradio UI guide

This document describes the current Gradio app in this workspace, the staged latent-space workflow it exposes, and the related extractor and visualization changes that support it.

## Purpose

The app provides a browser UI for exploring V-JEPA 2.1 latents without opening the notebook.

It supports a staged workflow where each step can run independently:

1. estimate whether an extraction configuration is likely to fit the selected device
2. extract latents from a video
3. load previously saved latent artifacts
4. compute PCA, `umap-learn`, or `mlx-vis` projections with custom parameters
  - PCA supports global, spatial-only, and temporal-only modes
5. load previously saved projection artifacts
6. build a 2D or 3D interactive plot from chosen projection components
7. generate RGB latent videos from any selected 3 projection components
8. generate a side-by-side source/latent comparison video
9. click a patch in the first frame and render a cosine-similarity tracking heatmap over time

## Files

### `app.py`

Thin launcher that:

- adds `src/` to `sys.path`
- imports `build_demo()` from `src/vjepa2_latents/gradio_app.py`
- constructs the app as `demo`
- launches the server when run directly

### `src/vjepa2_latents/gradio_app.py`

Main application layer.

Current responsibilities:

- define the staged Gradio UI
- keep loaded latent grids and projections in `gr.State` between button clicks
- estimate extraction pressure before a run with `estimate_limits_step()`
- run extraction independently from visualization
- load saved latent `.npy` and `.metadata.json` artifacts
- compute or load reusable PCA, `umap-learn`, or `mlx-vis` projection artifacts
- support global, spatial-only, and temporal-only PCA modes
- update component selectors dynamically based on projected dimensionality
- build 2D or 3D plots from arbitrary selected components
- create RGB videos from arbitrary selected projected components
- show a click-to-track first-frame preview for dense patch similarity
- map first-frame click coordinates onto latent tokens at `t=0`
- render cosine-similarity heatmaps over time as MP4 videos
- keep the metadata-heavy outputs collapsed by default so the main controls stay uncluttered
- return status and JSON metadata for each stage separately

### `src/vjepa2_latents/visualization.py`

Visualization and media helpers.

Current responsibilities:

- load saved latent artifacts
- flatten latent grids into feature rows and `(t, h, w)` coordinates
- compute PCA projections with NumPy SVD
- compute UMAP projections when `umap-learn` is available
- compute Apple-Silicon `mlx-vis` projections when the optional dependency is installed
- save and load reusable projection bundles
- build Plotly figures from either latent tensors or saved projections
- convert chosen projection components into RGB frames
- compose side-by-side videos
- map clicked image coordinates to latent token indices
- compute token-to-token cosine similarity volumes
- blend hotspot-focused similarity heatmaps back onto source frames and export them as MP4 videos
- preserve aspect ratio in the side-by-side panels by padding instead of squashing
- write browser-friendly `.mp4` outputs with `ffmpeg` using `libx264` and `yuv420p`

### `src/vjepa2_latents/extractor.py`

Existing extraction pipeline reused by the UI.

It was also extended to support the Gradio workflow more cleanly:

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
- synchronous encoder-run timing helper
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
  -> return stage-specific status + artifacts + metadata
```

## Current UI layout

The app is built in `build_demo()` and split into 6 main sections, with a preflight estimate action inside the extraction section.

### 1. Extract latents from video

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

### 2. Load latent space

Inputs:

- latent prefix textbox
- optional latent `.npy` upload
- optional latent `.json` metadata upload

Button:

- `Load latents`

Outputs:

- latent summary status
- latent summary JSON, collapsed by default

### 3. Compute or load a projection

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

### 4. Build a plot from chosen projection components

Inputs:

- 2D or 3D plot mode
- max plotted points
- chosen X/Y/Z components

Button:

- `Build plot`

Outputs:

- interactive Plotly figure
- plot status

### 5. Create RGB latent videos from chosen projection components

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

### 6. Patch Similarity / Dense Tracking

Inputs:

- loaded latent state from the earlier extract/load stages
- a click inside the first displayed source frame

Buttons / interactions:

- `Show first frame`
- click directly on the first frame image to choose a patch/object

Outputs:

- first-frame preview with the selected latent patch outlined
- patch-similarity status
- heatmap-over-video `.mp4` showing cosine similarity for the selected token across all `(t, h, w)` positions
- tracking metadata JSON, collapsed by default

Behavior:

- the first frame is decoded with the same resize-and-center-crop pipeline used for model input display
- click coordinates are mapped to the latent token grid at `t=0`
- the selected token is compared to every token in the latent grid with cosine similarity
- similarity scores are normalized to emphasize the strongest positive responses and blended with per-pixel opacity so hotspots stand out clearly
- each frame also marks the best-matching patch location, while the first frame keeps the originally selected patch outlined in a brighter accent color

## What changed in the current implementation

The current workspace goes beyond the original one-button prototype.

Implemented so far:

- staged extraction, loading, projection, plotting, and rendering
- reusable latent loading from saved files
- reusable projection loading from saved projection artifacts
- PCA and UMAP projection support with configurable reducer parameters
- global, spatial-only, and temporal-only PCA modes
- optional `mlx-vis` reducer support for Apple Silicon (`UMAP`, `t-SNE`, `PaCMAP`, `LocalMAP`, `TriMap`, `DREAMS`, `CNE`, `MMAE`)
- projection artifact persistence via `.projection.npz` and `.projection.metadata.json`
- arbitrary component selection for plotting
- arbitrary 3-component selection for RGB video generation
- separate render step for videos after a projection is available
- click-driven dense tracking from a selected first-frame patch
- preflight system-fit estimation before extraction
- lazy UMAP usage through optional dependency detection
- safer extraction status formatting that matches the real extractor return payload
- default extraction inputs now favor the 80M base model with `384 × 384` crops
- metadata panels are collapsed by default to keep the UI focused on the workflow
- rectangular crop support via separate crop height and crop width controls
- source-side display preprocessing that matches the model crop before side-by-side rendering
- aspect-ratio-preserving side-by-side rendering so the source video is no longer squashed
- broader model exposure with `vit_base_384`, `vit_large_384`, `vit_giant_384`, and `vit_gigantic_384`
- safer checkpoint handling through validation, fallback key lookup, and invalid-cache redownload
- synchronous encoder execution timing so the reported encoder duration includes async device completion
- extraction metadata that records encoder setup timings, major-phase timings, and output serialization timings
- in-memory latent and projection state so plot/render tweaks avoid reloading from disk when possible
- browser-native MP4 rendering through `ffmpeg` with `libx264` and `yuv420p`
- removal of the standalone latent-video panel from the UI so the side-by-side view is primary

## Artifact formats

### Latent artifacts

Extractor output format:

- `latents.npy`
- `latents.metadata.json`
- optionally `latents.pt` when `save_pt=True`

In the Gradio extraction path specifically, the app currently saves:

- `latents.npy`
- `latents.metadata.json`

### Projection artifacts

Saved by the projection step:

- `projection_<method>_<components>.projection.npz`
- `projection_<method>_<components>.projection.metadata.json`

Projection metadata includes:

- projection method
- human-readable method label
- component count
- component labels
- latent grid shape
- projection settings
- optional latent prefix back-reference

Latent metadata now also includes a `timings` block with:

- encoder forward-pass wall time measured synchronously
- encoder setup timings and extraction major-phase timings
- reshape total time plus reshape sub-step timings
- output serialization timings for CPU copy, NumPy write, metadata write, and optional `.pt` write

The encoder timing payload now includes:

- `device_executes_asynchronously`
- `measured_synchronously`
- `forward_run_seconds`
- `total_wall_seconds`

### Render artifacts

Saved by the render step under a `renders/` directory beside the projection files:

- `latent_<method>_rgb_c<r>-<g>-<b>.mp4`
- `latent_<method>_side_by_side_c<r>-<g>-<b>.mp4`

These videos are encoded for direct browser playback with H.264 (`libx264`) and `yuv420p`.

### Patch-similarity artifacts

Saved by the tracking step under a `tracking/` directory beside the latent files:

- `patch_similarity_t<t>-h<h>-w<w>.mp4`

Tracking metadata includes:

- clicked image coordinates
- selected latent token indices
- similarity value range
- peak patch location per frame when rendered
- rendered video path, shape, and playback FPS

## Visualization details

### Plotting

Plotting is driven by `build_projection_figure_from_data()`.

That means the app can:

- build plots directly from saved projections
- choose any valid 2 components for 2D
- choose any valid 3 components for 3D
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
- estimate step: `src/vjepa2_latents/gradio_app.py::estimate_limits_step`
- extract step: `src/vjepa2_latents/gradio_app.py::extract_latents_step`
- latent loading step: `src/vjepa2_latents/gradio_app.py::load_latents_step`
- projection step: `src/vjepa2_latents/gradio_app.py::compute_projection_step`
- projection loading step: `src/vjepa2_latents/gradio_app.py::load_projection_step`
- plot step: `src/vjepa2_latents/gradio_app.py::build_plot_step`
- render step: `src/vjepa2_latents/gradio_app.py::create_rgb_videos_step`
- tracking preview step: `src/vjepa2_latents/gradio_app.py::prepare_tracking_step`
- tracking click handler: `src/vjepa2_latents/gradio_app.py::select_patch_similarity_step`
- projection helpers: `src/vjepa2_latents/visualization.py::compute_projection_bundle`
- optional MLX reducer helper: `src/vjepa2_latents/visualization.py::compute_mlx_projection`
- saved projection loading: `src/vjepa2_latents/visualization.py::load_saved_projection`
- plot builder from saved data: `src/vjepa2_latents/visualization.py::build_projection_figure_from_data`
- RGB rendering from selected components: `src/vjepa2_latents/visualization.py::projection_rgb_frames`
- side-by-side media export: `src/vjepa2_latents/visualization.py::create_visualizations_from_projection`
- click-to-token mapping: `src/vjepa2_latents/visualization.py::map_click_to_latent_token`
- dense similarity computation: `src/vjepa2_latents/visualization.py::cosine_similarity_volume`
- tracking video export: `src/vjepa2_latents/visualization.py::create_patch_similarity_video`
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

That makes it much easier to iterate on latent analysis without rerunning the whole pipeline each time.
