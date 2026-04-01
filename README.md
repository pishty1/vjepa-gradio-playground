# V-JEPA 2.1 latent extraction

This workspace extracts patch-level V-JEPA 2.1 latents from `testvideo.mp4` into a structured tensor with shape `[batch, time, height, width, embed_dim]`, then provides a notebook to inspect and visualize those latents.

## Current status

The full end-to-end pipeline has been implemented and verified in this workspace.

- official repo cloned into `vendor/vjepa2`
- extractor implemented in `src/vjepa2_latents/extractor.py`
- CLI entrypoint added in `extract_vjepa2_latents.py`
- tests added in `tests/test_shape_math.py`
- notebook inspector added in `inspectdata.ipynb`
- Hugging Face / Gradio app added in `app.py`
- notebook now includes UMAP projection and side-by-side video export
- successful latent extraction completed for `testvideo.mp4`

## What the extractor does

- reads exactly `16` frames from `testvideo.mp4`
- optionally samples frames at a target FPS, or uses consecutive frames
- resizes the short side to `256`, then center-crops to `256x256`
- normalizes pixels with ImageNet mean and std
- loads the frozen V-JEPA 2.1 encoder from the official codebase
- runs the encoder and reshapes flat patch tokens into a latent grid
- saves results as `.pt`, `.npy`, and `.metadata.json`
- prints progress logs for long stages like checkpoint load and encoder forward pass

## Important note about resolution

Official V-JEPA 2.1 checkpoints are released at `384` resolution, not `256`. This extractor still supports a `256x256` crop because the official encoder uses RoPE-enabled variable-size inference. That detail is recorded in `skate_latents.metadata.json`.

## Files

- `extract_vjepa2_latents.py`: CLI entrypoint
- `src/vjepa2_latents/extractor.py`: extraction pipeline and model loading
- `tests/test_shape_math.py`: unit tests for frame indexing and token reshaping
- `inspectdata.ipynb`: notebook for summary stats and latent visualizations
- `app.py`: Gradio entrypoint for a Hugging Face Spaces-style browser UI
- `src/vjepa2_latents/gradio_app.py`: app wiring for upload, extraction, and visualization
- `src/vjepa2_latents/visualization.py`: latent PCA plotting and RGB/side-by-side video rendering helpers
- `vendor/vjepa2`: official upstream clone from `facebookresearch/vjepa2`
- `checkpoints/`: cached checkpoints downloaded by the extractor

## Environment

Create a Python environment and install the local requirements.

```zsh
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you want to use the vendored repo directly as a package too, install it in editable mode after the requirements are present.

```zsh
python -m pip install -e vendor/vjepa2
```

The local environment used during validation also includes `matplotlib`, `ipympl`, and `umap-learn` for notebook plotting and UMAP projections.

If you want the full notebook feature set used in the current inspector, install those extras too:

```zsh
python -m pip install matplotlib ipympl umap-learn
```

For the browser UI / Hugging Face Spaces interface, install the app dependencies in `requirements.txt` as well. That file now includes `gradio` and `plotly`.

## Quick checks

Run the lightweight unit tests:

```zsh
python -m unittest tests/test_shape_math.py
```

Run a preprocessing-only dry run without downloading weights:

```zsh
python extract_vjepa2_latents.py --dry-run
```

The dry run should report:

- input shape: `[1, 3, 16, 256, 256]`
- latent shape: `[1, 8, 16, 16, 1024]`
- stripped tokens: `0`

## Extract latents

Default extraction from `testvideo.mp4`:

```zsh
python extract_vjepa2_latents.py \
  --video testvideo.mp4 \
  --output-prefix skate_latents \
  --model vit_large_384
```

Sample at `8` fps instead of taking consecutive frames:

```zsh
python extract_vjepa2_latents.py \
  --video testvideo.mp4 \
  --output-prefix skate_latents_8fps \
  --model vit_large_384 \
  --sample-fps 8
```

Use a custom local checkpoint instead of auto-downloading:

```zsh
python extract_vjepa2_latents.py \
  --video testvideo.mp4 \
  --output-prefix skate_latents \
  --model vit_large_384 \
  --checkpoint-path checkpoints/vjepa2_1_vitl_dist_vitG_384.pt
```

Force CPU if you want to avoid Apple `mps` during debugging:

```zsh
python extract_vjepa2_latents.py \
  --video testvideo.mp4 \
  --output-prefix skate_latents_cpu \
  --model vit_large_384 \
  --device cpu
```

## Browser UI

The workspace now includes a simple Hugging Face Spaces-style interface built with `gradio`.

It supports:

- uploading a video or selecting the bundled `testvideo.mp4` example
- running the existing V-JEPA 2.1 latent-extraction pipeline from the browser
- rendering an interactive `Plotly` PCA view of the latent space
- generating a latent RGB video from PCA-projected patch embeddings
- generating a side-by-side comparison video of source frames and latent RGB frames
- showing clear status updates and the saved run metadata

Run it locally with:

```zsh
python app.py
```

Then open the local Gradio URL in your browser.

## Completed extraction result

The default extraction has already been run successfully on `testvideo.mp4`.

- video metadata: `204` frames at `25.000` fps
- source resolution: `4096x2160`
- extracted frame range: `0..15`
- input tensor shape: `[1, 3, 16, 256, 256]`
- raw token shape: `[1, 2048, 1024]`
- latent shape: `[1, 8, 16, 16, 1024]`
- stripped leading tokens: `0`
- device used: `mps`

Files written by that run:

- `skate_latents.pt`
- `skate_latents.npy`
- `skate_latents.metadata.json`

For the default `16 x 256 x 256` clip, the grid math is:

- `time = 16 / 2 = 8`
- `height = 256 / 16 = 16`
- `width = 256 / 16 = 16`

So the expected output shape is:

- ViT-L: `[1, 8, 16, 16, 1024]`
- ViT-g: `[1, 8, 16, 16, 1408]`

## Notebook inspection workflow

The notebook `inspectdata.ipynb` is set up and has already been executed successfully.

It currently includes:

- loading `skate_latents.npy` or `skate_latents.pt`
- loading `skate_latents.metadata.json`
- printing tensor shape, dtype, frame indices, and norm summary
- plotting latent norm heatmaps over all `8` latent time steps
- projecting the latent vectors into RGB using the first `3` principal components
- projecting the latent vectors with `UMAP` and mapping those coordinates into RGB + intensity views
- generating a labeled side-by-side comparison video between source frames and latent-space UMAP frames
- plotting cosine-similarity maps against the center patch from latent frame `0`

Observed notebook summary from the extracted tensor:

- latent shape: `(1, 8, 16, 16, 1024)`
- latent dtype: `float32`
- patch norm mean: `62.6178`
- patch norm std: `5.5419`
- patch norm min: `41.8771`
- patch norm max: `73.5242`
- PCA explained variance: `PC1=0.1154`, `PC2=0.0703`, `PC3=0.0611`

Open the notebook in VS Code and run it interactively if you want to inspect or extend the views.

Recent notebook-generated artifacts include:

- `latent_space_pca_smooth_384.mp4`
- `latent_space_side_by_side_384.mp4`
- `latent_space_umap_384.mp4`
- `latent_space_umap_side_by_side_384.mp4`

## Validation completed

The following checks have already passed in this workspace:

- `python3 -m py_compile extract_vjepa2_latents.py src/vjepa2_latents/extractor.py tests/test_shape_math.py`
- `python3 -m unittest tests/test_shape_math.py`
- `python3 extract_vjepa2_latents.py --dry-run`
- `python3 extract_vjepa2_latents.py --video testvideo.mp4 --output-prefix skate_latents --model vit_large_384`

Focused checks for the browser UI can be run with:

- `python3 -m unittest tests/test_shape_math.py tests/test_visualization.py`

## Notes and warnings

- the first real extraction run may appear slow because checkpoint loading and the transformer forward pass are heavy
- the extractor now prints `[vjepa2] ...` progress messages to make long-running stages visible
- `timm` emits a deprecation warning from upstream imports; this does not block extraction
- PyTorch emits a `sdp_kernel` future warning from upstream internals; this also does not block extraction
- `umap-learn` may print an `n_jobs` warning when `random_state` is set; this does not block the notebook outputs
