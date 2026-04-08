# Patch similarity / dense tracking guide

This document explains the patch similarity / dense tracking part of this workspace from the code outward.

It focuses on the current implementation in `src/vjepa2_latents/gradio_components/tracking/`, how that code is wired into the Gradio app, what data flows through the component, and what the generated tracking video actually represents.

It is intentionally narrower than the root `README.md` and covers only the patch-similarity tracking workflow.

## Scope

The tracking component is an interactive latent-space probe built on top of extracted V-JEPA 2.1 latents.

It currently lets you:

1. load or extract latents in the main latent-source section
2. open the `Patch Similarity / Dense Tracking` tab
3. choose one of the display-aligned source frames that correspond to the latent time steps
4. click a visible object region or patch in that frame
5. map that click into a latent token at `(t, h, w)`
6. compare the selected token with every latent token in the clip using cosine similarity
7. render an `.mp4` heatmap video that highlights where similar tokens appear over time

This is a visualization and exploration tool, not a learned tracker.

## Folder layout

The tracking-specific code lives in `src/vjepa2_latents/gradio_components/tracking/`:

- `src/vjepa2_latents/gradio_components/tracking/__init__.py`: lazy package exports for the public tracking API
- `src/vjepa2_latents/gradio_components/tracking/callbacks.py`: Gradio callback wiring for frame preparation and click handling
- `src/vjepa2_latents/gradio_components/tracking/core.py`: click-to-token mapping, cosine similarity computation, overlay rendering, annotation, and video export
- `src/vjepa2_latents/gradio_components/tracking/helpers.py`: tracking-specific status strings and frame-choice labels
- `src/vjepa2_latents/gradio_components/tracking/ui.py`: the Gradio tab definition
- `src/vjepa2_latents/gradio_components/tracking/README.md`: this document

The component also depends on nearby shared modules:

- `src/vjepa2_latents/gradio_app.py`: wires the tracking tab into `build_demo()`
- `src/vjepa2_latents/gradio_components/projection/core.py`: provides `flatten_latent_grid(...)`
- `src/vjepa2_latents/gradio_components/render/video.py`: provides source-frame alignment, playback-FPS inference, and MP4 writing
- `src/vjepa2_latents/gradio_utils.py`: provides generic status formatting, metadata loading, JSON serialization, and console logging helpers

## Code map

Main tracking entry points in this repo:

- UI setup: `src/vjepa2_latents/gradio_app.py::build_demo`
- Tracking tab UI: `src/vjepa2_latents/gradio_components/tracking/ui.py::build_tracking_tab`
- Tracking frame preparation: `src/vjepa2_latents/gradio_components/tracking/callbacks.py::prepare_tracking_step`
- Patch click handler: `src/vjepa2_latents/gradio_components/tracking/callbacks.py::select_patch_similarity_step`
- Click-to-token mapping: `src/vjepa2_latents/gradio_components/tracking/core.py::map_click_to_latent_token`
- Similarity computation: `src/vjepa2_latents/gradio_components/tracking/core.py::cosine_similarity_volume`
- Patch annotation: `src/vjepa2_latents/gradio_components/tracking/core.py::annotate_selected_patch`
- Heatmap overlay rendering: `src/vjepa2_latents/gradio_components/tracking/core.py::similarity_heatmap_frames`
- Tracking video export: `src/vjepa2_latents/gradio_components/tracking/core.py::create_patch_similarity_video`
- Shared latent flattening: `src/vjepa2_latents/gradio_components/projection/core.py::flatten_latent_grid`
- Shared source-frame alignment: `src/vjepa2_latents/gradio_components/render/video.py::load_aligned_source_frames`
- Shared MP4 writing: `src/vjepa2_latents/gradio_components/render/video.py::write_video`

## UI behavior in the current app

The tracking tab is built by `build_tracking_tab()`.

It currently contains:

- a short Markdown explanation
- a `Frame to track` dropdown populated only after latents are ready
- a `Show first frame` button that initializes the preview and tracking state
- an image preview that accepts click events
- a video output for the generated heatmap overlay
- a collapsed metadata panel with JSON details about the current selection and output

The button label says `Show first frame`, but the underlying callback already supports selecting any available aligned frame through the dropdown. In practice the flow is:

1. click `Show first frame` to prepare tracking and populate the dropdown
2. optionally switch to another aligned frame with the dropdown
3. click in the preview image to launch the similarity render

## Data dependencies

The tracking component expects a `latent_state` produced by the latent-source workflow.

The important fields are:

- `output_prefix`: prefix for the saved latent artifacts
- `latent_grid`: latent tensor with shape `[1, t, h, w, d]`
- `metadata`: latent metadata including `video_path`, `frame_indices`, `tubelet_size`, `crop_size`, and source-video FPS

If `latent_grid` or `metadata` are missing from the in-memory state, the tracking callbacks fall back to the saved artifacts on disk.

## Shape conventions

The tracking code assumes the latent tensor shape is:

```text
[batch, time, grid_h, grid_w, embed_dim]
```

For visualization, the batch dimension must be `1`.

Two shape systems are involved at once:

- image-space frames: `[t, image_h, image_w, 3]`
- latent-space tokens: `[1, t, grid_h, grid_w, d]`

The entire job of the tracking component is to bridge those two spaces accurately and consistently.

## End-to-end flow

At a high level the current tracking path is:

```text
latent_state
  -> prepare_tracking_step(...)
  -> load aligned display frames matching latent time steps
  -> choose a preview frame
  -> click image coordinates (x, y)
  -> map click to latent token (t, h, w)
  -> cosine_similarity_volume(...)
  -> similarity_heatmap_frames(...)
  -> annotate peak patch per frame + selected patch on first frame
  -> write_video(...)
  -> return status + video path + metadata JSON
```

## How frame preparation works

`prepare_tracking_step(...)` is the callback used by both the button click and the frame-dropdown change event.

It performs these steps:

1. validate that latents have been loaded
2. resolve the latent grid and metadata, either from memory or from disk
3. call `load_aligned_source_frames(metadata, latent_grid.shape)`
4. derive the list of displayable frame choices
5. clamp the requested frame index into the valid range
6. return:
   - a Gradio dropdown update with valid frame choices
   - the selected preview frame
   - a readiness status string
   - metadata JSON describing the prepared state
   - a `tracking_state` cache used by later clicks
   - `None` for the video output so stale videos are cleared when a new frame is prepared

### Why the source frames are “aligned”

The component does not display arbitrary raw video frames.

Instead, `load_aligned_source_frames(...)` in `render/video.py`:

- derives the effective latent FPS from the saved frame indices and tubelet size
- selects one display frame per latent time step
- reads exactly those video frames from the source video
- applies the same display crop pipeline used elsewhere in the app through `prepare_display_frames(...)`

This matters because click coordinates are only meaningful if the preview frame and the latent grid refer to the same spatial crop.

## Tracking state and caching

`prepare_tracking_step(...)` stores a lightweight `tracking_state` dictionary in `gr.State`.

The current implementation caches:

- `latent_output_prefix`
- `source_frames`
- `display_fps`
- `latent_grid_shape`
- `source_frame_indices`
- `selected_frame_index`

`select_patch_similarity_step(...)` reuses these cached `source_frames` when the latent prefix matches.

That avoids decoding and recropping the same source frames again on each click. The Gradio test `tests/test_gradio_app.py::test_select_patch_similarity_step_uses_cached_tracking_frames` explicitly checks this behavior.

## How clicks map to latent tokens

The key spatial bridge is `map_click_to_latent_token(click_xy, image_shape, latent_grid_shape, time_index=...)`.

It converts image-space click coordinates into a latent token index `(t, h, w)`.

### The basic mapping

Given:

- click coordinates `(x, y)`
- preview image size `(image_h, image_w)`
- latent grid size `(grid_h, grid_w)`

the code:

- clamps the click to valid image bounds
- computes the token row from the normalized vertical position
- computes the token column from the normalized horizontal position
- clamps the time index to a valid latent time step

Conceptually, it is:

$$
h = \left\lfloor \frac{y}{H} \cdot grid_h \right\rfloor,
\qquad
w = \left\lfloor \frac{x}{W} \cdot grid_w \right\rfloor
$$

with boundary clamping.

### The swapped-coordinate safeguard

The function also contains a small robustness heuristic.

It evaluates both interpretations:

- `(x, y)` as reported
- `(y, x)` as a fallback

Then it chooses the token whose cell center is closer to the original click point.

This is a practical safeguard for coordinate-order quirks in UI event payloads and is covered by `tests/test_visualization.py::PatchSimilarityTests.test_maps_swapped_click_coordinates_to_the_visible_patch`.

## How cosine similarity is computed

`cosine_similarity_volume(latent_grid, token_index)` computes the dense response map.

It works as follows:

1. flatten the latent grid with `flatten_latent_grid(...)` into:
   - `features`: shape `[t * h * w, d]`
   - `coordinates`: shape `[t * h * w, 3]` containing `(t, h, w)` per row
2. find the feature row corresponding to the selected token index
3. $L_2$-normalize all feature vectors
4. compute cosine similarity between the selected token and every token in the clip
5. reshape the result back to `[t, h, w]`

Conceptually, for normalized feature vectors $\hat{f}_i$ and query $\hat{q}$:

$$
s_i = \hat{f}_i^\top \hat{q}
$$

The result is a dense similarity volume over all latent tokens, not just the selected frame.

### Important consequence

If you click a patch on frame `t=k`, the query token remains anchored to that frame, but the similarity search spans every frame in the latent clip.

So the exported video is best understood as:

- a token-affinity visualization over time
- not a hard object track with motion estimation, association logic, or temporal filtering

## How the heatmap overlay is rendered

`similarity_heatmap_frames(source_frames, similarity_volume, alpha=0.45)` turns the token-space response into display-space overlays.

The current implementation does several useful things to keep the result visually readable.

### 1. Ignore negative similarity for the heatmap intensity

Only positive similarity contributes to the hotspot strength:

- negative values are clipped to `0`
- the strongest positive responses are emphasized

This makes the overlay answer the question “where do strongly similar patches appear?” rather than showing positive and negative matches with equal visual weight.

### 2. Use percentile-based normalization

Instead of scaling directly from min to max, the code normalizes positive values using:

- the `70th` percentile as the lower anchor
- the `99.5th` percentile as the upper anchor

Then it applies a gamma-like exponent of `0.65`.

That means:

- weak background similarity is suppressed
- medium-to-strong hotspots stand out more clearly
- a few extreme values are less likely to flatten the rest of the map

### 3. Resize token heatmaps back to image resolution

Each `[grid_h, grid_w]` similarity map is resized to the current frame size using bilinear interpolation.

This is necessary because the overlay is displayed in image coordinates even though the similarities were computed in token coordinates.

### 4. Apply a color map and per-pixel alpha blend

The normalized heatmap is colorized with OpenCV’s `COLORMAP_TURBO`, converted to RGB, and blended over the source frame.

The alpha contribution is modulated per pixel as a function of heat strength, so stronger hotspots appear more vivid while weak regions remain closer to the source frame.

## How patch annotations work

`annotate_selected_patch(...)` draws a rectangle corresponding to one latent token cell on a frame.

It computes the image-space cell bounds from the latent grid dimensions and the frame size, then draws a rectangular outline.

The tracking export uses two annotation styles:

- **per-frame peak patch**: cyan-like `(48, 255, 255)` rectangle with thickness `3`
- **original selected patch on frame 0**: orange-like `(255, 96, 64)` rectangle with thickness `4`

This distinction helps the video show both:

- where the maximum similarity lands in each frame
- which patch the user originally selected as the query

## How the tracking video is built

`create_patch_similarity_video(...)` orchestrates the full export.

It performs these steps:

1. create the output directory if needed
2. resolve aligned source frames if they were not already supplied
3. infer display FPS from latent metadata
4. compute the dense cosine similarity volume
5. render blended heatmap frames
6. find the peak-similarity token in every frame
7. annotate those peak tokens on all frames
8. redraw the originally selected token on frame `0`
9. encode the result as browser-friendly H.264 MP4 via `write_video(...)`

The filename currently follows this pattern:

```text
patch_similarity_t{t+1}_h{h+1}_w{w+1}.mp4
```

The suffix is `1`-based for readability in the saved artifact name, even though the internal token indices are `0`-based.

## What metadata is returned to the UI

After a click, `select_patch_similarity_step(...)` returns JSON metadata that currently includes:

- `latent_output_prefix`
- `click_xy`
- `selected_token` with `t`, `h`, and `w`
- `selected_frame_index`
- `selected_video_frame_index`
- `display_fps`
- `similarity_video_path`
- `similarity_video_shape`
- `similarity_range`

This metadata is shown in the collapsed `Patch similarity metadata` panel.

## Saved artifact layout

Tracking outputs are written beside the latent artifacts under a sibling `tracking/` directory:

```text
<latent parent>/tracking/
  patch_similarity_t<t>_h<h>_w<w>.mp4
```

More precisely, the directory is built as:

```python
latent_output_prefix.parent / "tracking"
```

So if the latent prefix is:

```text
.gradio_outputs/session_001/latents/run_a
```

the tracking video will be written under:

```text
.gradio_outputs/session_001/latents/tracking/
```

## Error handling and guardrails

The tracking code currently protects against several invalid states.

### In the callbacks

- if latents are not loaded, the UI returns a hint status instead of trying to run tracking
- if the click event has no coordinates, the UI returns a hint or raises a clear Gradio error
- if source frames cannot be loaded, file and shape errors are surfaced as `gr.Error`

### In the core helpers

- `map_click_to_latent_token(...)` validates click shape, image shape, and latent-grid rank
- `cosine_similarity_volume(...)` validates latent-grid rank and token bounds
- `annotate_selected_patch(...)` validates RGB frame shape and latent-grid rank
- `similarity_heatmap_frames(...)` validates frame/similarity dimensions and time-step agreement

These checks keep failures closer to the source of the problem.

## Relationship to the rest of the app

The tracking component is deliberately downstream of latent extraction.

It does not:

- decode video independently of the latent metadata contract
- compute a second set of features
- maintain a separate representation of time or crop geometry

Instead it reuses:

- the saved latent tensor
- the saved extraction metadata
- the shared aligned-frame loader
- the shared video writer

That keeps the tracking view consistent with the rest of the app and avoids silently drifting away from the crop or frame-sampling scheme used during extraction.

## What the tests currently verify

Tracking-specific coverage is spread across both visualization and Gradio workflow tests.

### Unit-style visualization coverage

`tests/test_visualization.py` currently checks:

- click-to-token mapping on normal coordinates
- click-to-token mapping under swapped coordinate order
- cosine similarity values for a small synthetic latent tensor
- heatmap overlay generation and patch annotation behavior

### Callback-level Gradio coverage

`tests/test_gradio_app.py` currently checks:

- `prepare_tracking_step(...)` returns frame choices, preview frames, metadata, and cached tracking state
- `select_patch_similarity_step(...)` reuses cached aligned frames instead of reloading them
- the selected frame index is correctly folded into the chosen token index

## Current limitations

Known constraints of the current implementation:

1. **Single-query interaction**
   - Each click generates one query token and one output video.
   - There is no multi-point aggregation or multi-object comparison view.

2. **Cosine affinity, not identity tracking**
   - The method visualizes latent similarity, not persistent object identity.
   - It can highlight semantically similar regions, not only the same physical patch.

3. **One latent token per click**
   - There is no local averaging or prompt neighborhood expansion around the selected patch.

4. **No persistent artifact index**
   - Videos are written to disk, but there is no cleanup, deduplication, or gallery management yet.

5. **Heatmap scaling is heuristic**
   - The percentile normalization is intentionally visual and may not be ideal for quantitative comparison across runs.

6. **Preview-button wording is slightly outdated**
   - The tab can work on any aligned frame, even though the button is labeled `Show first frame`.

## Practical workflow

1. extract or load latents
2. open `Patch Similarity / Dense Tracking`
3. click `Show first frame`
4. optionally choose another frame from `Frame to track`
5. click the object region or patch you want to probe
6. inspect the preview annotation, status text, metadata, and exported heatmap video

## If you want to extend this component

Natural next improvements would be:

- caching rendered videos by `(latent prefix, token index, alpha)`
- supporting multiple prompt points and blended queries
- exposing download links for saved tracking artifacts
- adding side-by-side “raw frame / heatmap / peak token” comparison layouts
- letting the user adjust overlay alpha and normalization aggressiveness
- adding trajectory summaries for the per-frame peak patch positions

## Summary

The current tracking component is a clean latent-affinity explorer.

It takes one user-selected patch, maps it into a V-JEPA 2.1 token, computes cosine similarity to every other token in the clip, and renders the result as a browser-ready heatmap video with per-frame peak annotations.

That makes it useful for answering questions like:

- where else does this patch representation appear?
- how stable is this latent feature over time?
- which regions stay most similar to a selected object part across the clip?