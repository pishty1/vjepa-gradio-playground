# Training notes for `vendor/vjepa2/app/vjepa_2_1/train.py`

This document summarizes how training is implemented in the V-JEPA 2.1 code shipped in this workspace, based on these source files:

- `vendor/vjepa2/app/vjepa_2_1/train.py`
- `vendor/vjepa2/app/vjepa_2_1/utils.py`
- `vendor/vjepa2/app/vjepa_2_1/models/vision_transformer.py`
- `vendor/vjepa2/app/vjepa_2_1/models/predictor.py`
- `vendor/vjepa2/src/hub/backbones.py`
- `vendor/vjepa2/configs/train_2_1/*/*.yaml`
- `vendor/vjepa2/README.md`

It explains both the generic training loop and the concrete recipe implied by the shipped V-JEPA 2.1 configs.

## 1. What is being trained

The training job learns a self-supervised encoder using a JEPA-style masked prediction objective.

There are three model objects in the loop:

- `encoder`: the online encoder being optimized by gradient descent.
- `predictor`: receives online encoder features plus masks and predicts target features.
- `target_encoder`: a copy of the encoder updated with EMA, not by gradients.

Conceptually:

1. sample videos or images
2. mask token subsets
3. run the frozen/EMA target encoder to produce target features
4. run the online encoder + predictor to predict those features
5. compute prediction loss, and optionally context loss on visible tokens too
6. update the online encoder/predictor
7. update the target encoder with exponential moving average

This is the key V-JEPA 2.1 change highlighted in `vendor/vjepa2/README.md`: dense predictive loss plus deep self-supervision.

## 2. Training entrypoint

The main entrypoint is `main(args, resume_preempt=False)` in `vendor/vjepa2/app/vjepa_2_1/train.py`.

It expects a nested config dictionary with sections like:

- `meta`
- `mask`
- `model`
- `data`
- `img_data`
- `img_mask`
- `data_aug`
- `loss`
- `optimization`

The `folder` key at the top level is used for logs and checkpoints.

## 3. Runtime and distributed setup

At import time the script tries to do:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
```

This means that on SLURM each worker process is expected to see only its assigned GPU.

Other runtime setup:

- Python, NumPy, and PyTorch seeds are initialized.
- `torch.backends.cudnn.benchmark = True` is enabled.
- multiprocessing start method is set to `spawn` if possible.
- distributed setup is initialized via `src.utils.distributed.init_distributed()`.

The script then chooses:

- CPU if CUDA is unavailable
- `cuda:0` otherwise

Because `CUDA_VISIBLE_DEVICES` is narrowed per process, `cuda:0` usually means the local GPU for that rank.

## 4. Precision and mixed precision

From `meta.dtype`:

- `bfloat16` -> mixed precision enabled with `torch.bfloat16`
- `float16` -> mixed precision enabled with `torch.float16`
- anything else -> full precision `torch.float32`

The forward pass is wrapped in `torch.cuda.amp.autocast(...)`, and the backward pass uses `torch.cuda.amp.GradScaler()` when mixed precision is enabled.

## 5. Data sources and multi-modal training

The code supports training on both:

- video data from `data`
- image data from `img_data`

A notable part of `train.py` is that it can split distributed ranks between the two modalities.

### 5.1 Video branch

The video branch is driven by `data`:

- `dataset_type`
- `datasets`
- `datasets_weights`
- `dataset_fpcs`
- `batch_size`
- `tubelet_size`
- `fps`
- `crop_size`
- `patch_size`
- `num_workers`
- `pin_mem`

`dataset_fpcs` means the frames-per-clip groups used by the loader. In the shipped pretraining configs this is `[16, 16, 16]`, and in cooldown it becomes `[64, 64, 64]`.

### 5.2 Image branch

If `img_data` is present, some ranks are reassigned to image training. The key parameters are:

- `img_data.rank_ratio`
- `img_data.batch_size`
- `img_data.dataset_fpcs`
- `img_data.datasets`
- `img_data.datasets_weights`
- `img_data.num_workers`
- `img_mask`

The script computes:

- `img_world_size = int(world_size * img_rank_ratio)`
- `num_video_ranks = world_size - img_world_size`

Then it reassigns datasets and per-rank batch sizes so that some ranks read image data and the rest read video data.

For image ranks:

- `dataset_type`, `dataset_paths`, `dataset_fpcs`, `batch_size`, and `num_workers` are replaced with the image config.
- `cfgs_mask` can be replaced by `img_mask`.
- `lambda_value` for the context loss becomes `lambda_value_img`.

For video ranks:

- the original video config stays active.
- `lambda_value` becomes `lambda_value_vid`.

This matches the README statement that V-JEPA 2.1 uses multi-modal tokenizers and joint image/video training.

## 6. Data loading and masking

Data loading is initialized with:

- `src.datasets.data_manager.init_data(...)`
- `src.masks.multiseq_multiblock3d.MaskCollator(...)`

The mask collator is built from:

- `cfgs_mask`
- `dataset_fpcs`
- `crop_size`
- `patch_size`
- `tubelet_size`

The data loader yields nested samples that include:

- the clip tensors
- encoder masks (`masks_enc`)
- prediction masks (`masks_pred`)

Inside the training loop, `load_clips()` moves them to the active device.

### 6.1 Shipped mask recipe

The shipped V-JEPA 2.1 configs define two video masks:

- a smaller-block mask with `num_blocks: 8`, `spatial_scale: [0.15, 0.15]`
- a larger-block mask with `num_blocks: 2`, `spatial_scale: [0.7, 0.7]`

Both use:

- `aspect_ratio: [0.75, 1.5]`
- `temporal_scale: [1.0, 1.0]`
- `max_temporal_keep: 1.0`

The image mask uses a similar recipe with `num_blocks: 10` and small spatial scale.

So the public configs indicate a multi-mask training regime with both local/small and large masking patterns.

## 7. Data augmentation

The transform pipeline is created by `app.vjepa_2_1.transforms.make_transforms(...)`.

The shipped configs use:

- random resize aspect ratio: `[0.75, 1.35]`
- random resize scale: `[0.3, 1.0]`
- `motion_shift: false`
- `auto_augment: false`
- `reprob: 0.0`

## 8. Model construction

`train.py` calls `init_video_model(...)` from `vendor/vjepa2/app/vjepa_2_1/utils.py`.

That function builds:

- a backbone from `vendor/vjepa2/app/vjepa_2_1/models/vision_transformer.py`
- a predictor from `vendor/vjepa2/app/vjepa_2_1/models/predictor.py`

Then it wraps them with:

- `MultiSeqWrapper`
- `PredictorMultiSeqWrapper`

### 8.1 Backbone variants

The available factories include:

- `vit_base`
- `vit_large`
- `vit_giant_xformers`
- `vit_gigantic_xformers`

The training script maps model name to encoder embedding size:

- `vit_base` -> 768 indirectly through the backbone
- `vit_large` -> 1024
- `vit_giant_xformers` -> 1408
- `vit_gigantic_xformers` -> 1664

The `train.py` local variable `embed_dim_encoder` is used later when normalizing hierarchical targets.

### 8.2 Core model options used in V-JEPA 2.1

Important model flags include:

- `use_rope`
- `use_mask_tokens`
- `use_activation_checkpointing`
- `img_temporal_dim_size`
- `modality_embedding`
- `interpolate_rope`
- `pred_depth`
- `pred_embed_dim`
- `pred_num_heads`
- `has_cls_first`
- `normalize_predictor`
- `n_registers`
- `n_registers_predictor`

The shipped configs for V-JEPA 2.1 consistently set:

- `img_temporal_dim_size: 1`
- `interpolate_rope: true`
- `modality_embedding: true`
- `use_rope: true`
- `use_mask_tokens: true`
- `uniform_power: true`
- `use_activation_checkpointing: true`

### 8.3 Exact PyTorch module structure

This section describes the actual module classes used by the backbone and predictor.

#### Encoder input stem

The encoder input stem is either:

- `PatchEmbed`: `nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)`
- `PatchEmbed3D`: `nn.Conv3d(in_chans, embed_dim, kernel_size=(tubelet_size, patch_size, patch_size), stride=(tubelet_size, patch_size, patch_size))`

So for the usual video case, the very first projection is a 3D convolution that turns a clip of shape:

- `(B, 3, T, H, W)`

into patch tokens of shape:

- `(B, N, D)` where `D = embed_dim`
- `N = (T / tubelet_size) * (H / patch_size) * (W / patch_size)`

For the shipped `256px, 16f` recipe with `patch_size=16` and `tubelet_size=2`:

- `H / patch_size = 16`
- `W / patch_size = 16`
- `T / tubelet_size = 8`
- `N = 8 * 16 * 16 = 2048`

So the encoder sees:

- video tokens: `(B, 2048, D)`

For the cooldown `64f` stage at `256px`:

- `T / tubelet_size = 32`
- `N = 32 * 16 * 16 = 8192`

So the encoder token shape becomes:

- video tokens: `(B, 8192, D)`

For the image branch in the shipped configs, `img_temporal_dim_size=1`, so images are passed through `PatchEmbed3D` with `tubelet_size=1` and effectively become:

- `N = 1 * 16 * 16 = 256`
- image tokens: `(B, 256, D)`

#### Encoder transformer block

The encoder block class is `Block` in `vendor/vjepa2/app/vjepa_2_1/models/utils/modules.py`.

Each block has this structure:

```python
x = x + DropPath(Attention(LayerNorm(x)))
x = x + DropPath(MLP_or_SwiGLU(LayerNorm(x)))
```

Concretely the module members are:

- `norm1 = LayerNorm(dim)`
- `attn = RoPEAttention(...)` when `use_rope=True`, otherwise `Attention(...)`
- `drop_path = DropPath(...)` or `Identity()`
- `norm2 = LayerNorm(dim)`
- `mlp = MLP(...)` or `SwiGLUFFN(...)`

The token shape is preserved through a block:

- input: `(B, N, dim)`
- output: `(B, N, dim)`

#### Encoder attention submodule

When `use_rope=True`, the block uses `RoPEAttention`.

Its learnable projections are:

- `qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)`
- `proj = nn.Linear(dim, dim)`

The attention reshape is:

- input: `(B, N, dim)`
- after `qkv`: `(B, N, 3 * dim)`
- reshaped to: `(3, B, num_heads, N, head_dim)`
- where `head_dim = dim // num_heads`

Typical encoder head dimensions for the public V-JEPA 2.1 backbones are:

- `vit_base`: `dim=768`, `num_heads=12`, `head_dim=64`
- `vit_large`: `dim=1024`, `num_heads=16`, `head_dim=64`
- `vit_giant_xformers`: `dim=1408`, `num_heads=22`, `head_dim=64`
- `vit_gigantic_xformers`: `dim=1664`, `num_heads=26`, `head_dim=64`

So all of these variants keep a `64`-wide attention head.

#### Encoder MLP / feed-forward submodule

The encoder block chooses between:

- `MLP` for GELU-based blocks
- `SwiGLUFFN` for SiLU/SwiGLU-based blocks

The shipped V-JEPA 2.1 configs set `use_silu: false`, so the default path is the standard `MLP`.

`MLP` is exactly:

```python
fc1 = nn.Linear(in_features, hidden_features)
act = GELU()
drop = Dropout(drop)
fc2 = nn.Linear(hidden_features, out_features)
drop = Dropout(drop)
```

with:

- `in_features = dim`
- `out_features = dim`
- `hidden_features = int(dim * mlp_ratio)`

Typical encoder MLP widths are:

- `vit_base`: `768 -> 3072 -> 768`
- `vit_large`: `1024 -> 4096 -> 1024`
- `vit_giant_xformers`: `1408 -> 6144 -> 1408` because `1408 * 48 / 11 = 6144`
- `vit_gigantic_xformers`: `1664 -> 8192 -> 1664` because `1664 * 64 / 13 = 8192`

So the backbone blocks are structurally standard ViT blocks with LayerNorm, self-attention, and a large feed-forward network.

#### Predictor input projection

The predictor class is `VisionTransformerPredictor` in `vendor/vjepa2/app/vjepa_2_1/models/predictor.py`.

Before entering predictor blocks, encoder outputs are mapped into predictor width using `predictor_embed`.

There are two cases:

- one distillation level: `nn.Linear(embed_dim, predictor_embed_dim)`
- multiple levels: `nn.Sequential(nn.Linear(embed_dim * levels, embed_dim), act, nn.Linear(embed_dim, predictor_embed_dim))`

In the generic training path used by `train.py`, the typical setting is:

- `levels_predictor = 4`
- `pred_embed_dim = 384`

So the common predictor input projection is:

- `vit_base`: `Linear(3072, 768) -> act -> Linear(768, 384)`
- `vit_large`: `Linear(4096, 1024) -> act -> Linear(1024, 384)`
- `vit_giant_xformers`: `Linear(5632, 1408) -> act -> Linear(1408, 384)`
- `vit_gigantic_xformers`: `Linear(6656, 1664) -> act -> Linear(1664, 384)`

where the input width is `4 * embed_dim` because four hierarchical encoder outputs are concatenated.

#### Predictor mask tokens and sequence assembly

If `use_mask_tokens=True`, the predictor learns:

- `mask_tokens = nn.ParameterList([Parameter(1, 1, predictor_embed_dim) ...])`

with `num_mask_tokens` entries.

The shipped hub factories use `predictor_num_mask_tokens=8`, and the training code computes this more generally from the number of masks and frame-per-clip groups.

Inside the forward pass, the predictor builds a sequence:

1. projected context tokens `x` of shape `(B, N_ctxt, predictor_embed_dim)`
2. learned masked tokens `pred_tokens` of shape `(B, N_pred, predictor_embed_dim)`
3. concatenated sequence `(B, N_ctxt + N_pred, predictor_embed_dim)`

It then reorders that sequence to match token indices and runs it through the predictor transformer stack.

#### Predictor transformer block width

The predictor uses the same `Block` class as the encoder, but with:

- `dim = predictor_embed_dim`
- `num_heads = pred_num_heads`

In the shipped configs:

- `pred_embed_dim = 384`
- `pred_num_heads = 12`

so predictor attention uses:

- `head_dim = 384 / 12 = 32`
- `qkv = Linear(384, 1152)`
- `proj = Linear(384, 384)`

The predictor MLP uses `mlp_ratio=4.0`, so its feed-forward shape is:

- `384 -> 1536 -> 384`

This is true both for the 12-block predictor used with `vit_base` and the deeper 24-block predictor used in larger configs.

#### Predictor output heads

After the predictor blocks, the code applies:

- `predictor_norm = LayerNorm(predictor_embed_dim)`
- `predictor_proj = Linear(predictor_embed_dim, len(hierarchical_layers) * out_embed_dim)`
- and, if `return_all_tokens=True`, also `predictor_proj_context` with the same output width

In the generic `train.py` path, `teacher_embed_dim` is not passed into `init_video_model`, so:

- `out_embed_dim = embed_dim`

That means the common training-time predictor output widths are:

- `vit_base`: `Linear(384, 3072)`
- `vit_large`: `Linear(384, 4096)`
- `vit_giant_xformers`: `Linear(384, 5632)`
- `vit_gigantic_xformers`: `Linear(384, 6656)`

These output widths match the concatenated hierarchical target widths produced by the encoder during training.

If only one distilled level is used, the output width collapses to a single `embed_dim`-sized target.

#### Typical encoder and predictor stacks by model size

For the main V-JEPA 2.1 backbone families in this repo, the module structure is:

- `vit_base` encoder: patch embed -> `12` transformer blocks of width `768` -> hierarchical outputs from layers `[2, 5, 8, 11]`
- `vit_large` encoder: patch embed -> `24` blocks of width `1024` -> hierarchical outputs from `[5, 11, 17, 23]`
- `vit_giant_xformers` encoder: patch embed -> `40` blocks of width `1408` -> hierarchical outputs from `[9, 19, 29, 39]`
- `vit_gigantic_xformers` encoder: patch embed -> `48` blocks of width `1664` -> hierarchical outputs from `[11, 23, 37, 47]`

Typical shipped predictor stacks are:

- `vitb16` config: predictor depth `12`, width `384`, heads `12`
- `vitl16` config: predictor depth `24`, width `384`, heads `12`
- `vitG16` config: predictor depth `24`, width `384`, heads `12`

So the predictor is intentionally much narrower than the largest encoders and acts as the bottleneck that maps online features into the target feature space.

#### Hierarchical output shapes during training

When `training=True`, the encoder returns concatenated hierarchical outputs, not just the final block output.

So for a typical pretraining video batch at `256px, 16f`:

- `vit_base` target shape: `(B, 2048, 3072)`
- `vit_large` target shape: `(B, 2048, 4096)`
- `vit_giant_xformers` target shape: `(B, 2048, 5632)`
- `vit_gigantic_xformers` target shape: `(B, 2048, 6656)`

For the cooldown `64f` stage, only the token count changes:

- `vit_base` target shape: `(B, 8192, 3072)`
- `vit_large` target shape: `(B, 8192, 4096)`
- `vit_giant_xformers` target shape: `(B, 8192, 5632)`
- `vit_gigantic_xformers` target shape: `(B, 8192, 6656)`

For the image branch at `256px` with one temporal slice:

- `vit_base` target shape: `(B, 256, 3072)`
- `vit_large` target shape: `(B, 256, 4096)`
- `vit_giant_xformers` target shape: `(B, 256, 5632)`
- `vit_gigantic_xformers` target shape: `(B, 256, 6656)`

These are the feature tensors that the predictor is trying to match after masking is applied.

## 9. Deep self-supervision implementation

This is one of the most important V-JEPA 2.1 ideas.

In `vendor/vjepa2/app/vjepa_2_1/models/vision_transformer.py`, the backbone tracks hierarchical layers depending on depth.

Examples:

- depth 12 -> `[2, 5, 8, 11]`
- depth 24 -> `[5, 11, 17, 23]`
- depth 40 -> `[9, 19, 29, 39]`
- depth 48 -> `[11, 23, 37, 47]`

During training, when `training=True`, the encoder concatenates the selected hierarchical features along the channel dimension and returns them.

So instead of supervising only the last layer, the training target can represent several backbone stages at once.

### 9.1 Predictor-side distillation outputs

The predictor mirrors this behavior.

In `vendor/vjepa2/app/vjepa_2_1/models/predictor.py`:

- `all_hierarchical_layers` is selected from predictor depth
- `n_output_distillation` decides how many levels are used
- if more than one level is used, the predictor first mixes concatenated features and projects them
- the output projection size becomes `len(hierarchical_layers) * out_embed_dim`

This is how the code supports deep supervision across multiple internal levels.

## 10. Target encoder and online encoder roles

The script creates:

```python
encoder, predictor = init_video_model(...)
target_encoder = copy.deepcopy(encoder)
```

Then all three are wrapped in `DistributedDataParallel`, and `target_encoder.parameters()` are marked with `requires_grad = False`.

So the target encoder is a teacher network that stays frozen during backprop and is only updated by EMA.

## 11. Iterations per epoch and schedules

The number of iterations per epoch (`ipe`) comes from:

- `optimization.ipe` if explicitly set
- otherwise `len(unsupervised_loader)` or `unsupervised_loader.num_batches`

The shipped configs set:

- `ipe: 300`
- `ipe_scale: 1.25`

The optimizer and schedulers are created by `init_opt(...)` in `vendor/vjepa2/app/vjepa_2_1/utils.py`.

### 11.1 Optimizer

By default the code uses `torch.optim.AdamW`.

If `use_radamw` is enabled, it uses a custom `src.utils.adamw.AdamW`.

Parameters are split into four groups:

- encoder weights with weight decay
- predictor weights with weight decay
- encoder bias/norm parameters without decay
- predictor bias/norm parameters without decay

### 11.2 Learning-rate schedule

Two schedule families are supported:

- normal training -> `WarmupCosineSchedule`
- annealing/cooldown -> `LinearDecaySchedule`

The shipped pretrain configs use:

- `start_lr: 1e-4`
- `lr: 6e-4`
- `final_lr: 6e-4`
- `warmup: 40`

The shipped cooldown config uses:

- `is_anneal: true`
- `anneal_ckpt: .../latest.pth.tar`
- `resume_anneal: true`
- `warmup: 0`
- `final_lr: 1e-6`

### 11.3 Weight-decay schedule

Weight decay is always scheduled with `CosineWDSchedule`.

The shipped configs use:

- `weight_decay: 0.04`
- `final_weight_decay: 0.04`

### 11.4 EMA momentum schedule

The target encoder momentum is advanced once per iteration by a generator:

```python
ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
```

In the shipped configs both ends are the same:

- `ema: [0.99925, 0.99925]`

So in the provided recipes the EMA is effectively constant.

## 12. Concrete shipped recipes

The repo includes config folders:

- `vendor/vjepa2/configs/train_2_1/vitb16`
- `vendor/vjepa2/configs/train_2_1/vitl16`
- `vendor/vjepa2/configs/train_2_1/vitG16`

Each contains:

- `pretrain-256px-16f.yaml`
- `cooldown-256px-64f.yaml`

### 12.1 Stage 1: pretraining

For the shipped V-JEPA 2.1 configs, the first stage is a long pretraining run on 16-frame clips.

Common traits across the configs I inspected:

- mixed image/video training
- `crop_size: 256`
- `patch_size: 16`
- `tubelet_size: 2`
- `fps: 4`
- `predict_all: true`
- `weight_distance_loss: true`
- `dtype: bfloat16`
- `use_rope: true`
- `use_mask_tokens: true`
- `modality_embedding: true`
- `img_temporal_dim_size: 1`
- `epochs: 1000`
- `ipe: 300`

The datasets listed are weighted mixtures of:

- `k710_train_paths.csv`
- `ssv2_train_paths.csv`
- `howto_320p.csv`

Images come from:

- `imagenet1k.csv`

The rank split is typically:

- `img_data.rank_ratio: 0.5`

Meaning half the ranks train on image data and half on video data.

### 12.2 Stage 2: cooldown / anneal

The cooldown stage changes the clip length and learning-rate behavior.

In the `vitb16` cooldown config:

- `dataset_fpcs: [64, 64, 64]`
- `batch_size` is reduced
- `epochs: 40`
- `is_anneal: true`
- `resume_anneal: true`
- `anneal_ckpt` points to the pretraining checkpoint
- `lambda_progressive: false`
- `final_lr: 1e-6`

So the public recipe appears to be:

1. long pretraining on shorter clips
2. short anneal/cooldown on longer clips

I did not infer anything beyond that from the code; this is what the shipped configs explicitly show.

## 13. Forward pass in the training loop

Inside each iteration, `train_step()` performs:

1. `scheduler.step()`
2. `wd_scheduler.step()`
3. `forward_target(clips)`
4. `forward_context(clips)`
5. loss computation
6. backward and optimizer step
7. EMA update of `target_encoder`

### 13.1 `forward_target`

The target pass runs under `torch.no_grad()`:

```python
h = target_encoder(c, gram_mode=False, training_mode=True)
```

The returned hierarchical features are normalized before use.

If `levels_predictor > 1`, the code assumes the target is a concatenation of multiple `embed_dim_encoder`-sized chunks and applies `layer_norm` to each chunk separately before concatenating them back.

If only one level is used, it applies one `layer_norm` over the full last dimension.

### 13.2 `forward_context`

The online encoder is run on the masked context tokens, then the predictor produces:

- `z_pred`: predictions for masked target tokens
- `z_context`: predictions for visible/context tokens when `predict_all=True`

The code also switches modality to `image` when the temporal length matches `img_temporal_dim_size`.

If `normalize_predictor` is enabled, the predictor outputs are normalized chunk-wise too.

## 14. Losses

The training loss is built from a generic per-token distance:

```python
mean(abs(pred - target) ** loss_exp) / loss_exp
```

In the shipped configs `loss_exp: 1.0`, so this behaves like an L1-style feature matching loss.

### 14.1 Prediction loss

This always runs:

```python
loss_pred = loss_fn(z_pred, h, masks_pred, ...)
loss += loss_pred
```

It compares predicted masked tokens to target-encoder representations at the masked positions.

### 14.2 Context loss / dense predictive loss

If `predict_all` is enabled, the code also compares predicted context tokens against target features at the visible positions:

```python
loss_context = loss_fn(z_context, h, masks_enc, ...)
loss += loss_context * lambda_value_step
```

This is the code implementation of the README’s dense predictive loss idea: both masked and visible tokens contribute to the objective.

### 14.3 Distance-weighted context loss

If `weight_distance_loss=True`, the context loss is weighted using distances returned by:

- `compute_mask_distance(masks_pred, masks_enc, grid_size, offset_context_loss)`

In the shipped pretrain configs this is enabled.

This makes context loss contributions depend on the spatial/temporal relationship between prediction masks and encoder-visible masks.

### 14.4 Progressive lambda

The context loss can be scaled by a schedule:

- if `lambda_progressive=True`, `lambda_sched.value(epoch * ipe + itr)` is used
- otherwise a fixed `lambda_value` is used

The video/image branches can have different values:

- `lambda_value_vid`
- `lambda_value_img`

The shipped configs use `0.5` for both.

## 15. Backward pass and optimizer step

The backward section of the code does two things:

1. optionally suppresses the optimizer step when the loss spikes abnormally
2. otherwise performs the standard mixed-precision or full-precision update

### 15.1 Loss regulation

If `loss_reg_std_mult` is configured, the code tracks a rolling window of previous losses and computes:

- mean
- standard deviation
- `max_bound = mean + loss_reg_std_mult * std`

If the current loss exceeds that bound and enough history exists, it marks `run_step = False` and skips `optimizer.step()`.

This is a safety valve against unstable loss spikes.

### 15.2 Standard update path

If `run_step` is true:

- mixed precision:
  - `scaler.scale(loss).backward()`
  - `scaler.unscale_(optimizer)`
  - `scaler.step(optimizer)`
  - `scaler.update()`
- full precision:
  - `loss.backward()`
  - `optimizer.step()`

Then the code always calls:

```python
optimizer.zero_grad()
```

## 16. EMA teacher update

After the optimizer step, the target encoder is updated with momentum:

```python
m = min(next(momentum_scheduler), ema[1])
```

Then for each parameter pair:

```python
target = m * target + (1 - m) * online
```

implemented with `_foreach_mul_` and `_foreach_add_`.

This means the teacher is a smoothed copy of the online encoder, which stabilizes the target representation.

## 17. Logging, checkpointing, and resume behavior

### 17.1 Logging

The script logs:

- epoch
- iteration
- average loss
- per-FPC mask stats
- learning rate
- weight decay
- memory usage
- iteration / GPU / data times

A CSV log is written to:

- `folder/log_r{rank}.csv`

### 17.2 Checkpointing

Checkpoints are saved from rank 0 only. The saved state includes:

- `encoder`
- `predictor`
- `target_encoder`
- optimizer state
- scaler state
- epoch
- average loss
- batch size
- world size
- learning rate

The standard rolling checkpoint path is:

- `folder/latest.pth.tar`

### 17.3 Resume logic

Resume behavior is handled by `load_checkpoint(...)` in `utils.py`.

When a checkpoint is loaded, the code restores model/optimizer/scaler state, then advances:

- LR schedule
- WD schedule
- EMA schedule
- mask collator schedule

for `start_epoch * ipe` steps to recover the correct schedule phase.

### 17.4 Anneal/cooldown resume

If `is_anneal=True`:

- training can either start from `anneal_ckpt`
- or continue from `latest.pth.tar` when `resume_anneal=True`

This matches the two-stage public config layout.

## 18. Distilled smaller models: what the public code shows

The smaller public V-JEPA 2.1 checkpoints are:

- `vjepa2_1_vitb_dist_vitG_384.pt`
- `vjepa2_1_vitl_dist_vitG_384.pt`

From `vendor/vjepa2/src/hub/backbones.py`:

- `vjepa2_1_teacher_embed_dim = 1664`
- `vjepa2_1_vit_base_384(...)` uses `teacher_embed_dim=1664`
- `vjepa2_1_vit_large_384(...)` uses `teacher_embed_dim=1664`
- both use `n_output_distillation=1`
- both load the encoder from checkpoint key `ema_encoder`

So the public code strongly suggests:

- ViT-B and ViT-L were trained/distilled against a larger teacher with embedding width 1664, i.e. a ViT-G / ViT-Gigantic-sized teacher representation.
- the exported smaller public models use a single distilled output level in the hub factory path.

By contrast, the generic training code in `train.py` defaults to `levels_predictor = 4`, and the V-JEPA 2.1 README describes deep self-supervision at multiple intermediate layers. So the codebase supports both:

- multi-level hierarchical supervision during training
- distilled single-level export/load configurations for some released small checkpoints

I am intentionally keeping this phrasing conservative because the exact offline training orchestration for Meta’s internal runs is not fully spelled out in this repo.

## 19. What we can say confidently vs. what remains implicit

### 19.1 Confirmed by code/configs

The following are directly visible in this repo:

- training is self-supervised masked latent prediction with an EMA teacher
- V-JEPA 2.1 adds dense predictive loss on context tokens via `predict_all`
- hierarchical/deep supervision is implemented in the backbone and predictor
- the shipped configs mix image and video data by splitting ranks
- the shipped recipe uses long pretraining plus a short anneal/cooldown stage
- the public small checkpoints are named as distilled from a larger `vitG` teacher

### 19.2 Not fully specified in the public code

The following details are not completely recoverable from the training script alone:

- the exact internal launch tooling and SLURM wrappers used at Meta
- the exact production-scale dataset curation beyond the CSV names shown in configs
- whether the publicly released `384` checkpoints were trained entirely at 384 or converted/fine-tuned beyond the `256px` configs shipped here

The repo gives enough information to understand the algorithm and most of the recipe, but not every internal experiment detail.

## 20. Short end-to-end summary

A single V-JEPA 2.1 training iteration looks like this:

1. fetch a batch of video/image clips and corresponding masks
2. move clips and masks to the current device
3. run the EMA teacher (`target_encoder`) to get hierarchical target features
4. run the online `encoder` on visible/context tokens
5. run `predictor` to predict masked tokens and, optionally, context tokens too
6. compute masked prediction loss plus context loss
7. backprop through `encoder` and `predictor`
8. update optimizer and schedules
9. update `target_encoder` by EMA
10. log metrics and eventually checkpoint

That is the training process implemented by `vendor/vjepa2/app/vjepa_2_1/train.py`.
