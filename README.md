# V-JEPA 2.1 latent explorer

This repo extracts patch-level V-JEPA 2.1 latents from video, saves them as reusable artifacts, and provides a Gradio app for projection, rendering, tracking, and promptable segmentation.

The project no longer depends on a checked-in clone of `facebookresearch/vjepa2`; encoder construction now happens through `torch.hub`, while checkpoints can be auto-downloaded or loaded from `checkpoints/`.

## What this project does

- extract V-JEPA 2.1 latent grids with shape `[batch, time, height, width, embed_dim]`
- estimate memory pressure before extraction, especially for Apple `mps`
- save reusable `.npy` and `.metadata.json` artifacts for later analysis
- compute PCA, `umap-learn`, and optional `mlx-vis` projections
- build interactive 2D and 3D Plotly views of the latent space
- render RGB latent videos and side-by-side source/latent videos
- probe dense patch similarity from a clicked frame location
- run sparse-prompt foreground/background video object segmentation

## Repository layout

### Entry points

- `app.py`: launches the Gradio app
- `extract_vjepa2_latents.py`: launches the extractor CLI

### Core package

- `src/vjepa2_latents/gradio_app.py`: builds the full Gradio UI
- `src/vjepa2_latents/gradio_utils.py`: shared Gradio helpers and status formatting
- `src/vjepa2_latents/__init__.py`: package-level exports for the main reusable APIs

### Latent extraction

- `src/vjepa2_latents/gradio_components/latent_source/`: latent source UI, callbacks, saved-run catalog, and config
- `src/vjepa2_latents/gradio_components/latent_source/extractor/`: extractor implementation
- `src/vjepa2_latents/gradio_components/latent_source/extractor/config.py`: model registry, crop parsing, device selection, memory estimates
- `src/vjepa2_latents/gradio_components/latent_source/extractor/checkpoint.py`: checkpoint download/validation and encoder loading via `torch.hub`
- `src/vjepa2_latents/gradio_components/latent_source/extractor/video.py`: frame probing, decoding, resize/crop, preprocessing
- `src/vjepa2_latents/gradio_components/latent_source/extractor/tensor.py`: encoder execution, token reshaping, output serialization
- `src/vjepa2_latents/gradio_components/latent_source/extractor/pipeline.py`: end-to-end extraction orchestration and CLI `main()`

### Analysis and visualization

- `src/vjepa2_latents/gradio_components/projection/`: projection compute/load helpers and UI
- `src/vjepa2_latents/gradio_components/plot/`: Plotly figure building
- `src/vjepa2_latents/gradio_components/render/`: RGB rendering and side-by-side video export
- `src/vjepa2_latents/gradio_components/tracking/`: patch similarity / dense tracking workflow
- `src/vjepa2_latents/gradio_components/segmentation/`: sparse-prompt VOS workflow

### Tests and notebooks

- `tests/test_shape_math.py`: extractor, timing, checkpoint, and shape coverage
- `tests/test_gradio_app.py`: Gradio workflow coverage
- `tests/test_visualization.py`: projection, rendering, tracking, and visualization coverage
- `inspectdata.ipynb`: notebook exploration of saved latent artifacts
- `playground.ipynb`: ad hoc experimentation notebook

## Setup

Create a virtual environment and install the project dependencies:

```zsh
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional notebook extras:

```zsh
python -m pip install matplotlib ipympl
```

`requirements.txt` already includes:

- `torch`, `torchvision`, `numpy`, `opencv-python`, `einops`, `timm`
- `gradio`, `plotly`, `umap-learn`
- `mlx` and `mlx-vis` on Apple Silicon only

## Quick start

### Dry-run the extractor

This validates preprocessing and latent shape math without downloading weights:

```zsh
python extract_vjepa2_latents.py --dry-run
```

### Extract latents from a video

```zsh
python extract_vjepa2_latents.py \
  --video testvideo.mp4 \
  --output-prefix skate_latents \
  --model vit_large_384
```

Example variations:

```zsh
python extract_vjepa2_latents.py \
  --video testvideo.mp4 \
  --output-prefix skate_latents_8fps \
  --model vit_large_384 \
  --sample-fps 8

python extract_vjepa2_latents.py \
  --video testvideo.mp4 \
  --output-prefix skate_latents_cpu \
  --model vit_large_384 \
  --device cpu

python extract_vjepa2_latents.py \
  --video testvideo.mp4 \
  --output-prefix skate_latents \
  --model vit_large_384 \
  --checkpoint-path checkpoints/vjepa2_1_vitl_dist_vitG_384.pt
```

Artifacts written by the extractor include:

- `<prefix>.npy`
- `<prefix>.metadata.json`
- optionally `<prefix>.pt`

## Gradio app

Run the browser UI locally:

```zsh
python app.py
```

The app exposes a staged workflow:

1. estimate extraction fit for the chosen device
2. extract new latents or load saved latent artifacts
3. compute or load projections
4. build interactive 2D or 3D plots
5. create latent RGB videos and side-by-side comparison videos
6. run click-based patch similarity tracking
7. run foreground/background prompt segmentation

### Current UI sections

- **Latent source**: extract or load latents, inspect metadata, reuse saved runs
- **Projection**: compute PCA, `UMAP`, or optional `mlx-vis` reducers
- **Plot**: explore chosen components in 2D or 3D Plotly views
- **Render**: build latent RGB videos and side-by-side outputs
- **Tracking**: click a patch and export cosine-similarity heatmap videos
- **Segmentation**: click one foreground and one background prompt and export overlay videos

## Architecture

The UI reuses the same extraction pipeline as the CLI rather than maintaining a separate inference path.

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
```

## Resolution and model notes

- official V-JEPA 2.1 checkpoints are published at `384` resolution
- the extractor also supports rectangular crops and smaller crops for experimentation
- inference still works because the upstream encoder supports RoPE-based variable-size inference
- the UI defaults to `vit_base_384` and uses `mps` on macOS when available

## Validation

Useful focused checks in this workspace:

```zsh
python -m unittest -v tests.test_shape_math
python -m unittest -v tests.test_gradio_app
python -m unittest -v tests.test_visualization
python -m unittest -v tests.test_gradio_app tests.test_visualization tests.test_shape_math
python -c "from app import build_demo; demo = build_demo(); print(type(demo).__name__)"
```

The recent loader fix also adds coverage for the `torch.hub` import-isolation path used to avoid collisions with the local `app.py` launcher.

## Related docs

- `src/vjepa2_latents/gradio_components/segmentation/README.md`: VOS-specific behavior and paper alignment
- `src/vjepa2_latents/gradio_components/tracking/README.md`: patch similarity / dense tracking details
- `README.md`: this single project-level guide

## Notes and limitations

- the first full extraction can be slow because checkpoint load and encoder forward pass are heavy
- the app processes one video at a time
- `ffmpeg` must be available, or `imageio-ffmpeg` must provide a compatible encoder
- RGB outputs are analytic latent visualizations, not reconstructions
- `timm` and some upstream PyTorch internals may emit non-fatal warnings during load/inference
- optional `mlx-vis` reducers are available only on Apple Silicon with the matching dependencies installed
