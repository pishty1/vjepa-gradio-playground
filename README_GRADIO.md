# Gradio UI guide

This document describes the current Gradio app in this workspace, the staged latent-space workflow it exposes, and the related extractor and visualization changes that support it.

## Purpose

The app provides a browser UI for exploring V-JEPA 2.1 latents without opening the notebook.

It supports a staged workflow where each step can run independently:

1. estimate whether an extraction configuration is likely to fit the selected device
2. extract latents from a video
3. load previously saved latent artifacts
4. compute PCA or UMAP projections with custom parameters
5. load previously saved projection artifacts
6. build a 2D or 3D interactive plot from chosen projection components
7. generate RGB latent videos from any selected 3 projection components
8. generate a side-by-side source/latent comparison video

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
- keep latent state and projection state between button clicks
- estimate extraction pressure before a run with `estimate_limits_step()`
- run extraction independently from visualization
- load saved latent `.npy` and `.metadata.json` artifacts
- compute or load reusable PCA or UMAP projection artifacts
- update component selectors dynamically based on projected dimensionality
- build 2D or 3D plots from arbitrary selected components
- create RGB videos from arbitrary selected projected components
- return status and JSON metadata for each stage separately

### `src/vjepa2_latents/visualization.py`

Visualization and media helpers.

Current responsibilities:

- load saved latent artifacts
- flatten latent grids into feature rows and `(t, h, w)` coordinates
- compute PCA projections with NumPy SVD
- compute UMAP projections when `umap-learn` is available
- save and load reusable projection bundles
- build Plotly figures from either latent tensors or saved projections
- convert chosen projection components into RGB frames
- compose side-by-side videos
- preserve aspect ratio in the side-by-side panels by padding instead of squashing
- write `.mp4` outputs with OpenCV

### `src/vjepa2_latents/extractor.py`

Existing extraction pipeline reused by the UI.

It was also extended to support the Gradio workflow more cleanly:

- rectangular crops via `normalize_crop_size()` and `parse_crop_size()`
- preflight memory and token estimates via `estimate_extraction_requirements()`
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
- cached checkpoint redownload behavior
- optional `.pt` skipping during output serialization

### `tests/test_visualization.py`

Focused visualization coverage for:

- PCA and UMAP projection helpers
- projection artifact round-tripping
- 2D and 3D figure generation
- RGB rendering from selected components
- side-by-side frame layout and aspect-ratio padding
- even-dimension video padding helpers

### `requirements.txt`

Relevant UI dependencies:

- `gradio>=5.0`
- `plotly>=5.18`
- `umap-learn>=0.5`

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
  -> return stage-specific status + artifacts + metadata
```

## Current UI layout

The app is built in `build_demo()` and split into 5 main sections, with a preflight estimate action inside the extraction section.

### 1. Extract latents from video

Inputs:

- input video upload
- model dropdown
- crop height
- crop width
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
- extraction metadata JSON
- preflight estimate JSON

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
- latent summary JSON

### 3. Compute or load a projection

Inputs:

- projection method: `PCA` or `UMAP`
- projected component count
- UMAP neighbors
- UMAP `min_dist`
- UMAP metric
- UMAP random state
- optional projection prefix textbox

Buttons:

- `Compute projection`
- `Load saved projection`

Outputs:

- projection status
- projection metadata JSON
- dynamically updated component selectors for plotting and RGB rendering

Behavior:

- UMAP-specific controls are hidden unless `UMAP` is selected
- component pickers are repopulated to match the saved or computed projection dimensionality
- 3D plotting is disabled automatically when fewer than 3 components exist

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
- render metadata JSON

Note:

- the latent-only RGB video is still written to disk and reported in metadata, but the UI presents the side-by-side video as the main visual output

## What changed in the current implementation

The current workspace goes beyond the original one-button prototype.

Implemented so far:

- staged extraction, loading, projection, plotting, and rendering
- reusable latent loading from saved files
- reusable projection loading from saved projection artifacts
- PCA and UMAP projection support with configurable reducer parameters
- projection artifact persistence via `.projection.npz` and `.projection.metadata.json`
- arbitrary component selection for plotting
- arbitrary 3-component selection for RGB video generation
- separate render step for videos after a projection is available
- preflight system-fit estimation before extraction
- lazy UMAP usage through optional dependency detection
- safer extraction status formatting that matches the real extractor return payload
- rectangular crop support via separate crop height and crop width controls
- source-side display preprocessing that matches the model crop before side-by-side rendering
- aspect-ratio-preserving side-by-side rendering so the source video is no longer squashed
- broader model exposure with `vit_base_384`, `vit_large_384`, `vit_giant_384`, and `vit_gigantic_384`
- safer checkpoint handling through validation, fallback key lookup, and invalid-cache redownload
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
- component count
- component labels
- latent grid shape
- projection settings
- optional latent prefix back-reference

### Render artifacts

Saved by the render step under a `renders/` directory beside the projection files:

- `latent_<method>_rgb_c<r>-<g>-<b>.mp4`
- `latent_<method>_side_by_side_c<r>-<g>-<b>.mp4`

## Visualization details

### Plotting

Plotting is driven by `build_projection_figure_from_data()`.

That means the app can:

- build plots directly from saved projections
- choose any valid 2 components for 2D
- choose any valid 3 components for 3D
- reuse PCA or UMAP results without recomputation
- label PCA axes with explained-variance percentages when available

### RGB rendering

RGB rendering is driven by `projection_rgb_frames()` and `create_visualizations_from_projection()`.

That means the app can:

- use PCA or UMAP projections
- use more than 3 projected components overall
- let the user choose any 3 projected components for RGB mapping
- reuse a saved projection instead of recomputing it
- write filenames that encode both method and component selection

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
- generated videos use OpenCV `mp4v`, so Gradio may still re-wrap some outputs for browser playback
- no explicit cleanup policy exists for old `.gradio_outputs/` runs
- UMAP can still be slower than PCA for large latent sets
- RGB videos are analytic visualizations, not reconstructions of the source video
- the app waits for extraction to finish before returning final outputs

## Recommended improvements

Recommended next improvements, in priority order:

1. add direct projection file uploads for `.projection.npz` and `.projection.metadata.json`
2. expose latent, projection, and render artifacts as downloadable files
3. add optional stage chaining such as extract → load or project → plot
4. cache projections for identical latent prefixes and reducer settings
5. record timing and file-size metadata per stage
6. add source-frame alignment and letterbox controls
7. improve output encoding, optionally via `ffmpeg` when available
8. add maintenance actions for `.gradio_outputs/`
9. add more analysis views such as heatmaps or temporal trajectories

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
- projection helpers: `src/vjepa2_latents/visualization.py::compute_projection_bundle`
- saved projection loading: `src/vjepa2_latents/visualization.py::load_saved_projection`
- plot builder from saved data: `src/vjepa2_latents/visualization.py::build_projection_figure_from_data`
- RGB rendering from selected components: `src/vjepa2_latents/visualization.py::projection_rgb_frames`
- side-by-side media export: `src/vjepa2_latents/visualization.py::create_visualizations_from_projection`
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
env NUMBA_DISABLE_JIT=1 /Users/pishty/ws/vjepa2.1/.venv/bin/python -m pytest tests/test_visualization.py tests/test_shape_math.py -q
env NUMBA_DISABLE_JIT=1 /Users/pishty/ws/vjepa2.1/.venv/bin/python -m unittest -v tests.test_visualization tests.test_shape_math
/Users/pishty/ws/vjepa2.1/.venv/bin/python -c "from app import build_demo; demo = build_demo(); print(type(demo).__name__)"
```

The stored test log in `.tmp/test_output.log` shows `28` focused tests passing for the current extractor and visualization coverage.

## Summary

The app is now a modular latent exploration workflow rather than a single-shot demo.

It currently lets you:

- estimate likely memory pressure before extraction
- extract and reload latents
- compute and reload PCA or UMAP projections
- choose arbitrary projection components for plots
- choose arbitrary projection components for RGB rendering
- create non-squashed side-by-side videos for source vs latent comparison

That makes it much easier to iterate on latent analysis without rerunning the whole pipeline each time.
