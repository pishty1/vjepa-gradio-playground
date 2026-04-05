# VOS guide

This document explains the Video Object Segmentation (VOS) part of this workspace from two angles:

- what the current code in `src/vjepa2_latents/gradio_components/segmentation/` and the related package entry points actually do
- how that behavior relates to the V-JEPA 2.1 paper

It is intentionally narrower than `README_GRADIO.md` and focuses only on the segmentation workflow.

## Scope

The VOS UI in this workspace is an interactive demo built on top of extracted V-JEPA 2.1 latent tokens.

It currently lets you:

1. load or extract latents
2. open the VOS tab in the Gradio UI
3. use the first frame as the prompt frame
4. click one foreground point and one background point
5. propagate those labels through the clip with weighted top-`k` cosine matching
6. render a binary mask overlay video

## Folder layout

The VOS-specific code now lives in `src/vjepa2_latents/gradio_components/segmentation/`:

- `src/vjepa2_latents/gradio_components/segmentation/__init__.py`: segmentation exports
- `src/vjepa2_latents/gradio_components/segmentation/callbacks.py`: Gradio VOS callback wiring
- `src/vjepa2_latents/gradio_components/segmentation/core.py`: segmentation dataclass, prompt annotation, propagation, rendering, and video export
- `src/vjepa2_latents/gradio_components/segmentation/status.py`: VOS-specific status text helpers
- `src/vjepa2_latents/gradio_components/segmentation/ui.py`: VOS tab UI construction
- `src/vjepa2_latents/gradio_components/segmentation/README.md`: this document

The package-level callers remain stable:

- `src/vjepa2_latents/gradio_app.py` still wires the UI
- `src/vjepa2_latents/visualization.py` still re-exports public visualization helpers
- `src/vjepa2_latents/video.py` still holds non-VOS video utilities and imports the segmentation helpers for compatibility
- `src/vjepa2_latents/gradio_utils.py` still exposes the VOS status helpers through imports

## Paper reference

The paper referenced for this behavior is:

- `V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning`
- arXiv: `https://arxiv.org/abs/2603.14482`
- VOS details are described in Appendix `8.3 Video Object Segmentation Tracking`

## Code map

Main VOS entry points in this repo:

- UI setup: `src/vjepa2_latents/gradio_app.py::build_demo`
- VOS frame preparation: `src/vjepa2_latents/gradio_components/segmentation/callbacks.py::prepare_segmentation_step`
- VOS prompt click handling: `src/vjepa2_latents/gradio_components/segmentation/callbacks.py::select_segmentation_prompt_step`
- VOS run step: `src/vjepa2_latents/gradio_components/segmentation/callbacks.py::run_segmentation_step`
- Prompt-to-token mapping: `src/vjepa2_latents/video.py::map_click_to_latent_token`
- Core propagation: `src/vjepa2_latents/gradio_components/segmentation/core.py::knn_binary_segmentation_volume`
- Mask rendering: `src/vjepa2_latents/gradio_components/segmentation/core.py::segmentation_mask_frames`
- Video export: `src/vjepa2_latents/gradio_components/segmentation/core.py::create_segmentation_video`
- VOS status text: `src/vjepa2_latents/gradio_components/segmentation/status.py`

## How the current code works

### 1. Prompts come from the first frame only

The current implementation intentionally constrains VOS prompts to frame `t=0`.

That is enforced in two places:

- `prepare_segmentation_step(...)` always chooses the first aligned display frame as the prompt frame
- `knn_binary_segmentation_volume(...)` raises an error if the foreground/background prompt tokens do not come from `t=0`

This matches the paper's supervision setup, where labels are given on the first frame and then propagated forward.

### 2. Clicks are converted into latent patch tokens

When the user clicks the image in the VOS tab:

- `select_segmentation_prompt_step(...)` receives the click event from Gradio
- `map_click_to_latent_token(...)` converts image-space `(x, y)` into a latent token index `(t, h, w)`
- one click is stored as `foreground`
- one click is stored as `background`

The UI stores both the image-space clicks and the token-space indices in `segmentation_state`.

### 3. The propagation is cosine-based and weighted

The actual label propagation happens in `knn_binary_segmentation_volume(...)`.

At a high level, it does this:

1. flatten the latent grid into normalized patch features
2. treat the selected foreground token as one labeled exemplar and the selected background token as another
3. for each target token in each frame, build a memory bank of candidate labeled tokens
4. compute cosine similarity between the target token and the memory bank
5. keep the top-`k` neighbors
6. turn those similarities into weights with a temperature-scaled exponential
7. compute a weighted foreground/background vote
8. assign the token to the winning class

The implementation uses these defaults when the VOS video is generated:

- `top-k = 5`
- `temperature = 0.2`
- `context_frames = 15`
- `spatial_radius = 12`

These are chosen to mirror the paper's reported VOS propagation settings more closely than the earlier simplified two-token classifier.

### 4. The memory bank is temporal, not just static prompts

The current code does not compare every target token only against the two clicked prompts.

Instead, for frame `t > 0`, `_build_memory_for_frame(...)` in `knn_binary_segmentation_volume(...)` builds memory from:

- all tokens in the first frame, using the first frame's current probability labels
- up to `context_frames` previous predicted frames

So later frames use:

- the initial first-frame supervision
- a rolling bank of past propagated predictions

That is much closer to the paper's label-propagation setup than a pure two-point nearest-neighbor lookup.

### 5. Matching is locally constrained in space

For frames after the first one, the code applies a local spatial constraint:

- tokens from past predicted frames are only considered if they lie within a radius of `spatial_radius`
- first-frame memory stays globally available

This is implemented with the `radius` filtering logic inside `knn_binary_segmentation_volume(...)`.

This reflects the paper's use of a local neighborhood constraint during propagation.

### 6. Output is a binary overlay video

After token labels are predicted:

- `segmentation_mask_frames(...)` resizes the token mask to the source-frame resolution
- the foreground region is blended as a green overlay
- mask contours are drawn for visibility
- `annotate_prompt_points(...)` redraws the foreground/background prompt markers on the first frame
- `create_segmentation_video(...)` writes the result as an `.mp4`

## How this relates to the paper

### What matches the paper well

The current code now matches several important parts of the paper's VOS procedure:

- **First-frame supervision**: prompts come from frame `t=0`
- **Frozen feature propagation**: labels are propagated directly in latent feature space without fine-tuning
- **Cosine similarity**: matching is similarity-based in normalized latent space
- **Top-`k` weighted voting**: classification uses weighted neighbor aggregation rather than a single hard match
- **Temperature scaling**: similarities are turned into weights with `temperature = 0.2`
- **Temporal context**: previous frames contribute to later predictions
- **Local matching constraint**: propagation uses a spatial radius for past-frame memory

### What does not exactly match the paper

The current VOS UI is still a demo, not the exact benchmark protocol used in the paper.

Important differences:

1. **Sparse prompts instead of dense masks**
   - The paper uses first-frame segmentation masks.
   - This code uses one foreground click and one background click.

2. **Binary labels instead of multi-object labels**
   - The paper propagates object labels from one-hot mask assignments, which can represent multiple objects.
   - This code predicts only foreground vs background.

3. **Interactive UI instead of dataset evaluation**
   - The paper evaluates on benchmark datasets such as DAVIS 2017 and YouTube-VOS.
   - This code is designed for interactive latent inspection in Gradio.

4. **Prompt density is much lower**
   - The paper effectively starts with many labeled patches from the first-frame mask.
   - This code starts with two labeled tokens chosen by clicks.

Because of those differences, the current implementation should be understood as:

- **close to the paper's propagation mechanism**
- **not identical to the paper's exact evaluation protocol**

## Practical workflow in this repo

1. extract or load latents
2. open the `Foreground / Background VOS` tab
3. click `Show frame for VOS prompts`
4. select `Foreground (green)` and click the object of interest
5. select `Background (red)` and click nearby background
6. optionally change `Top-k neighbors`
7. click `Run VOS segmentation`

## If you want to match the paper even more closely

The next major step would be replacing sparse clicks with a dense first-frame mask input.

That would allow the UI to move closer to the paper's actual evaluation setup by:

- labeling many first-frame patches instead of just two
- supporting multiple objects instead of binary FG/BG
- propagating one-hot object assignments instead of a binary label