# MapTR / MapTRv2 — working notes

This file exists to let a fresh Claude Code session pick up exactly where a
previous one left off, without re-deriving hard-won debugging context. Keep
it updated as work continues — especially the "Known environment gotchas"
section, since those bugs are easy to reintroduce by accident (e.g. editing
`sparse_encoder.py` without touching `sparse_block.py`'s parallel code path).

## Current branch / image state (as of 2026-07-28)

- Branch: `maptrv2`, pushed to `origin/maptrv2` at commit `522405e`.
- Docker image: `jhd0ck3r/maptrv2:latest` on Docker Hub, digest
  `sha256:683204d26aca96011734b6a2a206c74e2b425a4cde959184584a8ade6549acf5`.
  Built from `docker/Dockerfile` — CUDA 11.8 / torch 2.1.0 / Python 3.10,
  targets both Ampere (sm_86, e.g. local RTX 3070) and Hopper (sm_90, H100)
  via `TORCH_CUDA_ARCH_LIST="8.6 9.0+PTX"`.
- Local dev/test GPU: this environment has an RTX 3070 (sm_86). The actual
  cluster target is H100 (Hopper). User's personal machine has an RTX 5060
  (Blackwell) — different from both.
- **mmcv-full is now built from source** (`MMCV_WITH_OPS=1 pip install
  --no-binary mmcv-full`), not installed from OpenMMLab's prebuilt wheel
  index — see gotcha #12 below. This was needed because the prebuilt wheel
  had zero `sm_90` cubins and no PTX fallback at all, crashing every mmcv
  CUDA op (`ms_deform_attn`, `sigmoid_focal_loss`, etc.) on H100 with
  `RuntimeError: CUDA error: no kernel image is available for execution on
  the device` — invisible locally since `sm_86` was covered. Verified via
  `cuobjdump --list-elf`/`--list-ptx` on the installed `mmcv/_ext.*.so`:
  before the fix, 0 `sm_90` entries; after, 55/55 matching `sm_86`/`sm_90`
  cubins plus PTX. GKT and mmdetection3d's own ops were already correctly
  multi-arch (built locally by this same Dockerfile, so they picked up
  `TORCH_CUDA_ARCH_LIST` automatically) — this was an mmcv-only gap.

## What works right now

- **CARLA lidar-only MapTRv2 pipeline, verified end-to-end** against the
  local 259-tile test subset at `/home-local/johil9.nobkp/Documents/Code/carla`
  (note: lowercase `carla`, sibling dir to this repo):
  - `tools/maptrv2/custom_carla_map_converter.py` converts
    `reference_lines/*.json` + `manifest.json` into a map-annotation pkl.
  - `CustomCarlaLocalMapDataset` (`projects/mmdet3d_plugin/datasets/carla_offlinemap_dataset.py`)
    loads that pkl and produces real divider-class map-vector ground truth.
  - `projects/configs/maptrv2/maptrv2_carla_r50_24ep_lidar.py` trains a
    genuine LiDAR-only MapTRv2 (`modality='lidar'`, BEV built directly from
    the LiDAR `SparseEncoder`, no camera branch at all).
  - Ran `tools/train.py` for 1 full epoch (259 iters) + 1 eval pass:
    losses trended down normally (no NaN/Inf), checkpoint saved, and
    `CarlaMap_chamfer/mAP` computed successfully (~0.02 — meaningless as
    accuracy since train==val==the same 259 tiles, but confirms the whole
    loss/eval/checkpoint pipeline is wired correctly).

## How to re-run the smoke test

```bash
# From repo root, with the CARLA data + repo dirs bind-mounted (mounting
# subpaths only — never bind-mount the whole repo root or mmdetection3d/,
# it shadows compiled .so extensions and breaks imports):
docker run --rm --runtime=nvidia --shm-size=8g -e NVIDIA_VISIBLE_DEVICES=all -e PYTHONPATH=/MapTR \
  -v $(pwd)/projects:/MapTR/projects \
  -v $(pwd)/data:/MapTR/data \
  -v $(pwd)/tools:/MapTR/tools \
  -v $(pwd)/work_dirs:/MapTR/work_dirs \
  -v /home-local/johil9.nobkp/Documents/Code/carla:/home-local/johil9.nobkp/Documents/Code/carla:ro \
  -w /MapTR jhd0ck3r/maptrv2:latest python3 tools/train.py \
  projects/configs/maptrv2/maptrv2_carla_r50_24ep_lidar.py \
  --work-dir work_dirs/carla_lidar_dev \
  --cfg-options total_epochs=1 runner.max_epochs=1 evaluation.interval=1 data.workers_per_gpu=0
```

Notes:
- `PYTHONPATH=/MapTR` is required when calling `tools/train.py` directly
  (`dist_train.sh` sets this itself; running `train.py` bare does not).
- If you edit a file under `mmdetection3d/mmdet3d/ops/`, bind-mount only
  the *specific changed file(s)*, not the whole `ops/` directory — it
  contains compiled `.so` extensions alongside the source, and mounting
  the directory shadows them (`ImportError: cannot import name
  'ball_query_ext'` or similar).
- Regenerate the converter pkl if CARLA data changes:
  `python tools/maptrv2/custom_carla_map_converter.py --data-root <path> --out-dir data/carla/ --split test`

## Known environment gotchas (found this session — don't re-debug these)

These were discovered while upgrading the whole stack to CUDA 11.8/torch
2.1.0/Python 3.10 (for H100/Hopper support) and then building the CARLA
LiDAR-only path. Several are *silent* — they don't show up until a specific
code path actually executes (e.g. eval-mode, or a LiDAR-only forward pass),
so don't assume "it built fine" means "it actually works."

1. **Vendored `spconv` 1.x is completely broken under CUDA 11.8/sm_86+90.**
   `mmdetection3d/mmdet3d/ops/spconv/{conv,modules,pool,structure}.py` are
   ~2020-era CUDA kernels that crash (`cuda execution failed with error 2`,
   from `indice_cuda.cu`) on *any* real CUDA execution — verified failing
   even on a trivial 2-point/4x4x4 grid, so it's not a shape/data issue.
   **Fix**: `mmdetection3d/mmdet3d/ops/spconv/__init__.py` is now a thin
   shim re-exporting the maintained `spconv2` package (`spconv-cu118`,
   pinned in `docker/Dockerfile`) instead of the local broken files. The
   old files are left in place but unused/dead.

2. **spconv2's default `MaskImplicitGemm` algorithm can fail to find a
   kernel** ("`can't find suitable algorithm`") the first time a given
   input shape is seen — observed specifically at **eval time** (never
   during training on the *same* conv layers, presumably because training
   iterates enough shape variations to warm the tuner cache, while eval's
   first call on an unseen shape hits it cold).
   **Fix**: pin `algo=spconv.ConvAlgo.Native` (bypasses tuning). This had
   to be applied in **two separate places** — `make_sparse_convmodule()`
   in `sparse_block.py` (used by plain conv layers) AND directly inside
   `SparseBasicBlock.__init__`/`SparseBottleneck.__init__` in the same
   file (these build their own `conv1`/`conv2` via mmdet's generic
   `BasicBlock`/`Bottleneck.__init__`, a *completely separate* code path
   that bypasses `make_sparse_convmodule` — easy to fix one and think
   you're done).

3. **spconv2 forbids `sparse_tensor.features = ...` direct assignment**
   (`ValueError: you can't set feature directly, use
   'x = x.replace_feature(...)'`). Fixed in `SparseBasicBlock.forward`/
   `SparseBottleneck.forward` in `sparse_block.py`.

4. **`sparse_shape` axis order is `(x, y, z)`**, confirmed empirically by
   directly probing `mmdet3d.ops.Voxelization`'s output coords with a
   known asymmetric-resolution config (voxel_size `[0.1, 0.2, 0.4]` on a
   point at a known position) and checking which output column matched
   which axis. **Do not trust `SparseEncoder`'s own docstring**, which
   claims `coors` columns are `(batch_idx, z_idx, y_idx, x_idx)` — that
   describes something else; the actual `sparse_shape` passed to
   `SparseConvTensor` must be `[x_cells, y_cells, z_cells]`. Verified
   against the original nuScenes fusion config's own
   `sparse_shape=[300,600,41]` for a 30×60×8m range at 0.1/0.1/0.2m
   resolution (300=x, 600=y, 41=z — matches (x,y,z), not (z,y,x)).

5. **mmcv 1.7.2's single-GPU `MMDataParallel` is broken under torch 2.1.**
   `mmcv.parallel.Scatter.forward` calls PyTorch's private
   `torch.nn.parallel._functions._get_stream()` with a raw `int` device
   id; torch>=2.x's version of that function requires a `torch.device`
   object (`AttributeError: 'int' object has no attribute 'type'`).
   **Fix**: `docker/Dockerfile` `sed`-patches the installed
   `mmcv/parallel/_functions.py` after install to wrap the device id.

6. **`nuscenes-devkit` silently upgrades `shapely` past the pin.**
   `requirement.txt` pins `shapely==1.8.5.post1`, but `nuscenes-devkit`
   (installed after it in the Dockerfile, with an unpinned shapely dep)
   upgrades it to 2.x. Shapely 2.0 changed `STRtree.query()` to return
   integer indices instead of geometry objects — silently breaks
   `projects/mmdet3d_plugin/datasets/map_utils/tpfp_chamfer.py`'s chamfer
   distance eval (`AttributeError: 'numpy.int64' object has no attribute
   'intersects'`), but *only* when real (non-empty) predictions reach
   that code, i.e. only surfaces during a real eval pass, not synthetic
   empty-prediction tests. **Fix**: `docker/Dockerfile` re-pins
   `shapely==1.8.5.post1` as the very last install step.

7. **MapTRv2's test-time pipeline requires `MultiScaleFlipAug3D` wrapping**,
   even with `flip=False` and no real TTA. `MapTRv2.forward_test()`
   unconditionally indexes `img_metas[0][0][...]` (nested list), which
   only mmcv's test-time collation produces when the pipeline is wrapped
   in `MultiScaleFlipAug3D`. Without it: `KeyError: 0` (because
   `img_metas[0]` resolves to a plain dict, and `dict[0]` is a KeyError,
   not the expected list index). See `test_pipeline` in
   `carlasim_map.py`/`maptrv2_carla_r50_24ep_lidar.py` for the pattern.

8. **`bev_h_`/`bev_w_` must match the physical aspect ratio of
   `point_cloud_range`, AND must be explicitly passed to the dataset as
   `bev_size=(bev_h_, bev_w_)`.** These feed two independent things that
   must agree: the model's `seg_head` output shape, and the dataset's
   `gt_seg_mask` canvas size (`VectorizedAV2LocalMap`'s `canvas_size`).
   Mismatch → `ValueError: Target size (...) must be the same as input
   size (...)`. When we resized `point_cloud_range` from the old
   asymmetric 30×60m (nuScenes) to CARLA's square 25×25m, `bev_h_`/
   `bev_w_` (200×100) had to be resized to square too (100×100), and
   `bev_size` had to be threaded through `data.train/val/test` in the
   training config (the base `carlasim_map.py` deliberately doesn't set
   it — matches nuScenes convention of overriding model-shaped params in
   the derived training config, not the base data config).

9. **The dataset's own `aux_seg` param is separate from the model's
   `aux_seg_cfg`** and must be wired manually (`data.train.aux_seg=
   aux_seg_cfg` in the training config) — otherwise `gt_seg_mask` never
   gets produced even though the model expects it, crashing in
   `MapTRv2Head.loss()` with `TypeError: 'NoneType' object is not
   subscriptable`. Only needed for `train` (bev_seg is a training-time
   auxiliary loss, not used at eval).

10. **`tools/train.py` needs `data.shuffler_sampler`/
    `data.nonshuffler_sampler` set** (`dict(type='DistributedGroupSampler')`
    / `dict(type='DistributedSampler')`) — `mmdet_train.py`'s dataloader
    builder reads these directly off `cfg.data`, and they're not defaulted
    anywhere. Missing → `AttributeError: 'ConfigDict' object has no
    attribute 'shuffler_sampler'`.

11. **`extract_img_feat` in `maptrv2.py` had a real pre-existing bug**
    (`B = img.size(0)` executed *before* its own `if img is not None`
    check) — invisible in the original camera-only/fusion configs since
    `img` was never actually `None` there, but crashes immediately once a
    genuine LiDAR-only path passes `img=None`. Fixed by moving `B =
    img.size(0)` inside the `if` block. Similarly `MapTRv2Head.forward()`
    had a `mlvl_feats[-1].shape` read gated behind `use_aux_seg` but not
    also behind `pv_seg` specifically — fixed to only run when
    `pv_seg=True`.

12. **OpenMMLab's prebuilt `mmcv-full` wheel has no Hopper (sm_90) support
    and no PTX fallback**, unlike every other CUDA extension in this image
    (GKT, mmdetection3d's ops), which are compiled locally by this
    Dockerfile and correctly pick up `TORCH_CUDA_ARCH_LIST`. The wheel from
    `https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html`
    was built by OpenMMLab's own CI with an arch list that stops at
    `sm_86` — works fine locally on Ampere, but on an actual H100 crashes
    the first time *any* mmcv CUDA op runs (`ms_deform_attn`,
    `sigmoid_focal_loss`, etc.) with `RuntimeError: CUDA error: no kernel
    image is available for execution on the device`. **Fix**: install
    mmcv-full from source instead (`MMCV_WITH_OPS=1 pip install
    --no-cache-dir --no-binary mmcv-full "mmcv-full==1.7.2"`), which makes
    its `setup.py` read the Dockerfile's own `TORCH_CUDA_ARCH_LIST` like
    every other op does. Costs ~4-5 extra minutes of build time. Verify
    with `cuobjdump --list-elf`/`--list-ptx` on the installed
    `mmcv/_ext.*.so` if this ever needs re-checking after an mmcv version
    bump.

13. **`extract_lidar_feat` crashes with a bare `IndexError: index -1 is out
    of bounds for dimension 0 with size 0` on `coords[-1, 0]`** if
    voxelization produces zero total voxels for a batch — i.e. a tile's raw
    point cloud was empty, or every point in it fell outside
    `lidar_point_cloud_range`/`z_max=15.0`. First hit on the cluster's full
    4103-tile/5-town dataset (never reproduced against the local 259-tile
    single-town subset, which was hand-picked to have real GT/points).
    **Fix applied**: `extract_lidar_feat` (`maptrv2.py`) now takes an
    `img_metas` kwarg (threaded through from both `forward_train` and
    `simple_test`) and raises a clear `RuntimeError` naming the offending
    `sample_idx`(es) and each sample's raw point count before it would
    otherwise hit the opaque `IndexError` — makes it possible to identify
    the bad tile(s) from a single log line instead of just a crash. This
    does **not** fix the underlying data issue — see "full-dataset empty
    tiles" in Open items below for how to hunt down the actual culprit
    tile(s).

14. **`docker run` needs `--shm-size` explicitly raised for
    `data.workers_per_gpu > 0`**, or training crashes partway through (not
    immediately — took ~70-90 iterations locally) with `RuntimeError:
    DataLoader worker (pid ...) is killed by signal: Bus error. It is
    possible that dataloader's workers are out of shared memory.` Docker
    defaults `/dev/shm` to 64MB; PyTorch's multi-worker DataLoader passes
    tensors between worker processes and the main process through shared
    memory, and this dataset's point clouds are large enough (up to
    5,000,000 points/tile) to exhaust that default quickly. **Every
    verification run earlier in this project used `workers_per_gpu=0`**,
    which never touches this path at all — so this was completely
    invisible until the first real `workers_per_gpu>0` benchmark. **Fix**:
    add `--shm-size=8g` (or larger) to any `docker run` invocation that
    uses `workers_per_gpu>0` — see the updated smoke-test command above.
    If the actual cluster deployment runs via Singularity/Apptainer rather
    than Docker directly, note that Singularity's default behavior differs
    (it typically shares the host's real `/dev/shm` rather than an
    isolated small allocation) — this specific bug may not reproduce
    there, but hasn't been verified either way; if a Singularity run hits
    the same "Bus error"/shared-memory symptom, look for an equivalent
    `--shm-size`-style container option.

## Batch size / num_workers benchmarking (2026-07-29, local RTX 3070 8GB)

Benchmarked against the real full train set (4103 tiles, 5 towns,
`data/carla/carla_map_infos_train.pkl`) to inform settings for the
eventual H100 MIG 20GB / 16 CPU / 64GB RAM cluster target. Local GPU is
much smaller than the target, so **memory numbers are extrapolated
linearly and not yet verified on real H100 MIG hardware** — compute/
throughput numbers don't transfer across architectures at all and are
local-only reference points.

Measured (fp16, real training step, `log_config`'s reported `memory`):
- `samples_per_gpu=1`: ~3555 MiB, ~0.75-0.87s/iter
- `samples_per_gpu=2`: ~6731 MiB, ~0.73s/iter (2 samples/iter, i.e. ~0.37s/sample -- batching improves per-sample throughput)
- `samples_per_gpu=3`: OOMs on this 8GB card (needed >7.5GB)
- Linear fit from the 1/2 data points: ~379 MiB fixed + ~3176 MiB/sample
- `num_workers`: `data_time` drops from ~0.12s (0 workers) to ~0.005s
  (negligible) already at 4 workers; 8 and 16 workers showed **no further
  improvement** -- not CPU/data-loading bound past 4 workers on this
  16-core machine.
- Confirmed the dataset's occasional 5,000,000-raw-point tiles (18/4103,
  see gotcha #13's context) do **not** cause memory spikes -- voxelizer's
  `max_voxels=[90000,120000]` cap bounds memory regardless of raw point
  count, verified directly against one of these tiles.

Extrapolated to 20GB (20480 MiB), applying the linear fit:
- `samples_per_gpu=4`: ~13.1GB (64% of 20GB)
- `samples_per_gpu=5`: ~16.3GB (79% of 20GB)
- `samples_per_gpu=6`: ~19.4GB (95% of 20GB -- tight, don't trust without verifying)

**Recommendation given to user**: `samples_per_gpu=4-5`, `workers_per_gpu=4-8`,
plus `--shm-size=8g` (gotcha #14) and re-verifying batch size empirically
on the actual H100 MIG partition before a long run (a 10-20 iteration dry
run checking logged `memory` is enough). Also flagged: `lr=6e-4` and
`warmup_iters=500` in `maptrv2_carla_r50_24ep_lidar.py` were implicitly
tuned for `samples_per_gpu=1` (inherited from the nuScenes/AV2-derived
base configs) -- increasing batch size should come with linear LR scaling
(`new_lr ≈ 6e-4 * new_batch_size`), and `warmup_iters=500` should be
reconsidered since it's a much larger fraction of the first epoch at
higher batch sizes (iterations/epoch = 4103/samples_per_gpu drops
sharply as batch size grows).

## Viewing training outputs (2026-07-30)

No prediction visualizer existed for CARLA — the repo's existing ones
(`tools/maptrv2/av2_vis_pred.py`, `tools/maptrv2/nusc_vis_pred.py`,
`tools/maptr/vis_pred.py`, `tools/analysis_tools/visual.py`) are all
coupled to their source dataset's multi-camera setup or the
`nuscenes-devkit` SDK, and `tools/test.py --show` doesn't work either
since `CustomCarlaLocalMapDataset` never implements `show()`. Two new
tools, verified end-to-end against a real local checkpoint:

- `tools/maptrv2/carla_bev_vis.py` — runs inside the container (needs
  config+checkpoint+GPU). Renders GT-vs-predicted divider polylines
  overlaid in one PNG per sample, adapted from `av2_vis_pred.py`'s
  self-contained BEV-only plotting block (that script's camera-projection
  code is irrelevant here, CARLA is LiDAR-only). GT is read directly from
  `dataset.data_infos[i]['annotation']['divider']` rather than through the
  model-input pipeline, since `CustomCarlaLocalMapDataset`'s test-mode
  path never attaches `gt_bboxes_3d`/`gt_labels_3d` (only train-mode does,
  via `vectormap_pipeline`).
- `tools/maptrv2/webviewer.py` — standalone, runs *outside* the container
  (`pip install flask matplotlib seaborn tensorboard`), single file. Shows
  loss/eval curves, embedded TensorBoard, and the BEV gallery on one
  `localhost` page. **Gotcha found while testing**: the bare `tensorboard`
  command can hang forever on startup with no useful error if a stale
  system/apt-packaged TensorBoard shadows a real pip install on `$PATH`
  (hit this exact thing in this dev environment). Fixed by launching it as
  `sys.executable -m tensorboard.main` instead of trusting `$PATH` — do
  the same if adapting this pattern elsewhere.

## Open items / next steps

- **`carlasim_map.py`'s `ann_file_train` is currently stale/inconsistent.**
  It points at `data/carla/carla_map_infos_train.pkl` (someone — the user
  or a linter — changed this after the session's initial push away from
  reusing the test pkl), but no such file has been generated: only the
  259-tile `test` split exists locally, and the converter has only ever
  been run with `--split test`. Right now, running training against this
  config as-is will fail with a missing-file error. This needs
  reconciling — either generate a real `carla_map_infos_train.pkl` (e.g.
  by running the converter against a genuine train split once more data
  is available) or point `ann_file_train` back at the test pkl for further
  local dev. **Do not silently revert this — check with the user first**,
  it may reflect an intent to wire up a real train split. `ann_file_val`/
  `ann_file_test` still correctly point at the existing
  `carla_map_infos_test.pkl`.
- **Only the local 259-tile test subset exists** (`carla/test/`). The full
  remote-cluster dataset has 4103 tiles across 5 towns (`train`/`val`
  splits don't exist yet — will need a real split, e.g. by town or
  stride-based, before a real training run). Re-running the converter
  against the full dataset needs no code changes, just `--data-root`
  pointed at the right path and new `ann_file` names in the config.
- **Full-dataset run hit an empty-tile crash (see gotcha #13) not
  reproducible locally.** Before re-running training on the cluster, scan
  the full dataset's `manifest.json` for suspect tiles directly (much
  faster than waiting to hit them during training) — for each tile, check
  `n_points` for zero/very-low counts, and cross-check any survivors
  against the raw `.npz` block's actual `features` xyz range vs
  `lidar_point_cloud_range`/`z_max=15.0` in
  `maptrv2_carla_r50_24ep_lidar.py`. On the local subset, raw block
  `features` xyz is already tile-relative (verified: roughly matches
  `tile_radius=12.5` for x/y, and lands within `[-8, 15]` for z after
  each tile's own baked-in offset) — so this is not a world-vs-tile
  coordinate bug, but some tiles in the full/multi-town dataset may
  genuinely have near-zero LiDAR returns (e.g. sparse-geometry areas) or
  an unusually tall/deep local feature pushing all points outside the
  z-bounds tuned against the single local test town. If real, either
  filter such tiles out at converter time or widen
  `lidar_point_cloud_range`/`z_max` to a per-tile-safe margin.
- **Class taxonomy is divider-only, including *all* CARLA lane types**
  (driving/curb/sidewalk/border/restricted/parking/shoulder/stop/other
  all collapsed into one `divider` class) — this was a deliberate choice
  to maximize GT density on the tiny local test set. Before a real
  training run, revisit whether to filter to just `driving`-type lanes
  via the converter's `--lane-types` flag (see
  `tools/maptrv2/custom_carla_map_converter.py`).
- **`sparse_shape`/`lidar_bev_proj.in_channels` were measured empirically**
  for the *current* `lidar_point_cloud_range`/`lidar_voxel_size` in
  `maptrv2_carla_r50_24ep_lidar.py` (sparse_shape=[251,251,71],
  lidar_bev_proj.in_channels=384). If those ranges/resolutions change,
  re-measure via a dummy `extract_lidar_feat()` call rather than
  recomputing by hand (see gotcha #4 — easy to get axis order wrong).
- **No real accuracy signal yet** — the 1-epoch smoke test only confirms
  the pipeline is wired correctly, not that the model learns anything
  useful. A longer run (more epochs, real train/val split) is needed
  before drawing conclusions about model quality.
- The plan file from this work session (detailed step-by-step design) is
  at `/gel/usr/johil9/.claude/plans/when-running-the-dockerfile-mellow-russell.md`
  if more implementation detail/reasoning is needed than fits here.
