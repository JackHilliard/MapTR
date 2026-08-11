# MapTR / MapTRv2 — working notes

Lets a fresh session resume without re-deriving hard-won debugging context.
Keep it updated — especially "Known gotchas", whose bugs are easy to
reintroduce (e.g. editing `sparse_encoder.py` but not `sparse_block.py`'s
parallel code path). Gotcha numbers are referenced elsewhere in this file;
don't renumber them.

## Environment / state (as of 2026-08-07)

- Branch `maptrv2` → `origin/maptrv2` @ `7f68a8d`.
- Image `jhd0ck3r/maptrv2:latest`, digest
  `sha256:b42787a8e17b4705abf2623f3510fa7049ec31dda39bfc2a2ebc584e1252745e`
  (rebuilt 2026-08-07). Previous:
  `sha256:7d3c08f816cb0fcc2e5940047cc57cf14ba650cb4a33e553448a3c9c88a0ce76`
  — recorded because `:latest` is the only tag pushed, so this is the only
  rollback record.
- Built from `docker/Dockerfile`: CUDA 11.8 / torch 2.1.0 / Python 3.10,
  `TORCH_CUDA_ARCH_LIST="8.6 9.0+PTX"` (Ampere + Hopper).
- GPUs: local dev = RTX 3070 (sm_86); cluster target = H100 MIG 20GB /
  16 CPU / 64GB RAM; user's personal machine = RTX 5060 (Blackwell).
- Data: local subset at `/home-local/johil9.nobkp/Documents/Code/carla`
  (lowercase, sibling of this repo) — 259-tile `test` split. Full dataset
  is 4103 tiles / 5 towns; a 30-tile 60 m grid export lives at
  `/gel/usr/johil9/Documents/carla/Town10HD/grid_tiles`.

## The CARLA LiDAR-only pipeline (works end-to-end)

- `tools/maptrv2/custom_carla_map_converter.py` — `reference_lines/*.json`
  + `manifest.json` → map-annotation pkl.
- `CustomCarlaLocalMapDataset`
  (`projects/mmdet3d_plugin/datasets/carla_offlinemap_dataset.py`) — loads
  the pkl, emits divider-class map-vector GT.
- `projects/configs/maptrv2/maptrv2_carla_r50_24ep_lidar.py` — genuine
  LiDAR-only MapTRv2 (`modality='lidar'`, BEV straight from the LiDAR
  `SparseEncoder`, no camera branch).
- Verified: 1 epoch (259 iters) + eval, losses trend down, checkpoint
  saved, `CarlaMap_chamfer/mAP` computed. **No real accuracy signal yet** —
  that run had train==val==the same 259 tiles.

## Commands

Smoke test (bind-mount *subpaths only* — never the repo root or
`mmdetection3d/`, it shadows compiled `.so`s and breaks imports):

```bash
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

- `PYTHONPATH=/MapTR` is required when calling `tools/train.py` bare
  (`dist_train.sh` sets it itself).
- Editing a file under `mmdetection3d/mmdet3d/ops/`: bind-mount the
  *specific file*, not the directory (compiled `.so`s live alongside the
  source; `ImportError: cannot import name 'ball_query_ext'`).
- `--shm-size` is mandatory for `workers_per_gpu>0` — see gotcha #14.

Regenerate the pkl:
```bash
python tools/maptrv2/custom_carla_map_converter.py --data-root <path> --out-dir data/carla/ --split test
```

Generate predictions (training never persists them — see below):
```bash
python3 tools/test.py <work_dir>/<config>.py <work_dir>/latest.pth \
    --format-only --eval-options jsonfile_prefix=<work_dir>/results
```
Writes `<work_dir>/results/pts_bbox/carlamap_results.json`, schema
`{meta, results:[{sample_token, vectors:[{pts, pts_num, cls_name, type,
confidence_level}]}]}`; `pts` is `(num_pts_per_vec, 2)` already in the
tile-local BEV frame.

**Delete `data/carla/carla_map_gt.json` whenever the sample set or GT
frame changes** — `_format_gt()` skips regeneration if the file exists, so
eval silently scores against stale GT.

## Known gotchas (don't re-debug these)

Found while moving to CUDA 11.8 / torch 2.1.0 / Python 3.10 and building
the LiDAR-only path. Several are *silent* until a specific path executes
(eval mode, LiDAR-only forward) — "it built fine" ≠ "it works".

1. **Vendored `spconv` 1.x is broken under CUDA 11.8/sm_86+90.**
   `mmdetection3d/mmdet3d/ops/spconv/{conv,modules,pool,structure}.py`
   crash (`cuda execution failed with error 2`, `indice_cuda.cu`) on any
   real CUDA execution — even a 2-point/4×4×4 grid, so not a shape issue.
   **Fix**: `ops/spconv/__init__.py` is a shim re-exporting `spconv2`
   (`spconv-cu118`, pinned in the Dockerfile); old files are dead code.

2. **spconv2's default `MaskImplicitGemm` can fail to find a kernel**
   ("can't find suitable algorithm") on a first-seen input shape —
   observed at **eval time** only (training warms the tuner cache).
   **Fix**: pin `algo=spconv.ConvAlgo.Native` in **two places** in
   `sparse_block.py`: `make_sparse_convmodule()` *and* directly in
   `SparseBasicBlock.__init__`/`SparseBottleneck.__init__`, which build
   their own `conv1`/`conv2` via mmdet's generic `BasicBlock`/`Bottleneck`
   and bypass `make_sparse_convmodule` entirely.

3. **spconv2 forbids `sparse_tensor.features = ...`** (use
   `x.replace_feature(...)`). Fixed in `SparseBasicBlock.forward` /
   `SparseBottleneck.forward` in `sparse_block.py`.

4. **`sparse_shape` axis order is `(x, y, z)`** — i.e.
   `[x_cells, y_cells, z_cells]`. **Do not trust `SparseEncoder`'s
   docstring**, which claims `coors` is `(batch_idx, z, y, x)`. Confirmed
   by probing `Voxelization` output coords with an asymmetric voxel size
   (`[0.1, 0.2, 0.4]`), and cross-checked against the nuScenes fusion
   config's `sparse_shape=[300,600,41]` for a 30×60×8 m range at
   0.1/0.1/0.2 m.

5. **mmcv 1.7.2's `MMDataParallel` is broken under torch 2.1.**
   `mmcv.parallel.Scatter.forward` passes a raw `int` device id to torch's
   private `_get_stream()`, which now needs a `torch.device`
   (`AttributeError: 'int' object has no attribute 'type'`).
   **Fix**: the Dockerfile `sed`-patches `mmcv/parallel/_functions.py`.

6. **`nuscenes-devkit` silently upgrades `shapely` past the pin.**
   Shapely 2.0's `STRtree.query()` returns indices, not geometries, which
   breaks `datasets/map_utils/tpfp_chamfer.py`
   (`AttributeError: 'numpy.int64' object has no attribute 'intersects'`)
   — but only when non-empty predictions reach it, i.e. only in a real
   eval. **Fix**: the Dockerfile re-pins `shapely==1.8.5.post1` last.

7. **The test pipeline must be wrapped in `MultiScaleFlipAug3D`**, even
   with `flip=False`. `MapTRv2.forward_test()` unconditionally indexes
   `img_metas[0][0][...]`, and only that wrapper's collation produces the
   nested list. Without it: `KeyError: 0`. See `test_pipeline` in
   `carlasim_map.py` / `maptrv2_carla_r50_24ep_lidar.py`.

8. **`bev_h_`/`bev_w_` must match `point_cloud_range`'s aspect ratio AND
   be passed to the dataset as `bev_size=(bev_h_, bev_w_)`.** They feed
   the model's `seg_head` output shape and the dataset's `gt_seg_mask`
   canvas (`VectorizedAV2LocalMap.canvas_size`) independently; mismatch →
   `ValueError: Target size (...) must be the same as input size (...)`.
   CARLA's square 25×25 m range required 100×100, not nuScenes' 200×100.
   `bev_size` is threaded through `data.train/val/test` in the *training*
   config; the base `carlasim_map.py` deliberately omits it.

9. **The dataset's `aux_seg` is separate from the model's `aux_seg_cfg`**
   and must be wired manually (`data.train.aux_seg=aux_seg_cfg`), or
   `gt_seg_mask` is never produced → `TypeError: 'NoneType' object is not
   subscriptable` in `MapTRv2Head.loss()`. Train-only (eval doesn't use it).

10. **`tools/train.py` needs `data.shuffler_sampler` /
    `data.nonshuffler_sampler`** (`dict(type='DistributedGroupSampler')` /
    `dict(type='DistributedSampler')`) — `mmdet_train.py` reads them off `cfg.data` and
    nothing defaults them. Missing → `AttributeError: 'ConfigDict' object
    has no attribute 'shuffler_sampler'`.

11. **Pre-existing camera-assumption bugs in `maptrv2.py`**, invisible
    until a LiDAR-only path passes `img=None`: `extract_img_feat` ran
    `B = img.size(0)` *before* its own `if img is not None` check (moved
    inside), and `MapTRv2Head.forward()` read `mlvl_feats[-1].shape` gated
    on `use_aux_seg` but not on `pv_seg` (now gated on `pv_seg=True`).

12. **OpenMMLab's prebuilt `mmcv-full` wheel has no sm_90 cubins and no
    PTX fallback.** Fine on local Ampere; on a real H100 the first mmcv
    CUDA op (`ms_deform_attn`, `sigmoid_focal_loss`, …) dies with
    `RuntimeError: CUDA error: no kernel image is available for execution
    on the device`. GKT and mmdetection3d's ops were fine — they're built
    locally and pick up `TORCH_CUDA_ARCH_LIST`; this was mmcv-only.
    **Fix**: `MMCV_WITH_OPS=1 pip install --no-cache-dir --no-binary
    mmcv-full "mmcv-full==1.7.2"` (~4-5 min extra build). Verified via
    `cuobjdump --list-elf`/`--list-ptx` on `mmcv/_ext.*.so`: 0 → 55/55
    matching sm_86/sm_90 cubins plus PTX. Re-check after any mmcv bump.

13. **`extract_lidar_feat` → `IndexError: index -1 is out of bounds ...
    size 0` on `coords[-1, 0]`** when voxelization yields zero voxels
    (empty tile, or all points outside `lidar_point_cloud_range`). First
    hit on the full 4103-tile dataset; never on the hand-picked local
    subset. **Fix**: `extract_lidar_feat` takes `img_metas` (threaded from
    `forward_train` and `simple_test`) and raises a `RuntimeError` naming
    the offending `sample_idx`(es) and raw point counts. This is now only
    the last line of defence — see "Zero-voxel tiles are discarded".

14. **`docker run` needs `--shm-size=8g` for `workers_per_gpu>0`**, or
    training dies after ~70-90 iters with `RuntimeError: DataLoader worker
    ... killed by signal: Bus error ... out of shared memory`. Docker
    defaults `/dev/shm` to 64MB and this dataset's point clouds (up to
    5M points/tile) exhaust it. Invisible before the first
    `workers_per_gpu>0` benchmark, since every earlier run used 0.
    Singularity/Apptainer typically shares the host `/dev/shm`, so this may
    not reproduce there — unverified; if it does, find the equivalent flag.

## CARLA data facts

**GT frame is `offset`, not `tile_center`** (fixed 2026-08-03, `aaf4b46`).
Each `.npz` carries both: `offset` is what `features[:, 0:3]` is relative
to (verified exactly: `points - offset == features[:, :3]`), while
`tile_center` is the nominal geometric centre. The converter used to emit
`pts - tile_center`, misplacing GT by a mean 2.37 m across 400 train tiles
(max 7.58 m; 72% of tiles >1 m) against chamfer thresholds of
0.5/1.0/1.5 m — the likely
reason early mAP sat near 0.02. The converter now reads `offset` from each
`.npz` (lazy `np.load`, so only the small array is read) and records it as
`annotation_origin` per sample. Median GT-vertex → driving-surface
(`label == 0`) distance, old vs new: train 0.387 → 0.038 m, test
0.179 → 0.043 m. Instance counts unchanged (train 4103 tiles/15810,
test 259/1210). Re-confirmed on the 60 m grid export: 0.083 m via `offset`
vs 2.854 m via `tile_center`. **Any checkpoint trained before this fix is
not comparable — retrain.**

**Degenerate 5,000,000-point tiles.** 20 tiles have a manifest `n_points`
of exactly 5,000,000 (18 in train + 2 in town10hd); **15 are genuinely degenerate** (raw→effective
collapse below 1%), the other 5 reach 55k-67k distinct cells and are
merely oversampled. In `town01_tile_00000`, 4,994,233 of 5,000,000 points
share a *single* xy location; only 4,321 unique xy positions total. This
explains why `GridSamplePoints` collapses it to ~3,242 points and why the
voxelizer only found 718 occupied voxels there.

**98 tiles fall below the configured z range** (all town03/town05,
`z_min` down to −97.1 vs the config's −72). See Open items.

`.npz['labels']` holds per-point `lane_type_lookup` class ids (−1 =
unlabeled) — unused by the divider-only pipeline, but available for a real
multi-class taxonomy.

Polylines in the grid export carry a class (`class_id`/`class`, decoded via
the manifest's `class_lookup`): driving_centerline / curb / road_edge /
lane_divider / center_divider / sidewalk_edge / median / crosswalk. The
Town10HD export only *populates* driving_centerline (299 polylines).

## Zero-voxel tiles are discarded (2026-08-06)

A tile that voxelizes to zero voxels would take an unattended run down via
gotcha #13. Dropped in three places:

1. **Converter** counts per tile the points surviving both `z <= --z-max`
   and `--lidar-point-cloud-range` (whose xy default is now derived from
   tile geometry — see 2026-08-10 below; `GridSamplePoints` only *clamps*
   grid coords so it can't change this count). Tiles below `--min-lidar-points` (default 1) are dropped and
   listed in `carla_map_infos_<split>_dropped.json`; kept samples record
   `num_lidar_points` / `num_lidar_points_in_range`, and the pkl gains a
   `lidar_check` block recording the geometry used. `--no-lidar-check`
   restores the old fast path. The scan reads each block's full `features`
   array (~5 s for 259 tiles, ~95 s for 4103).
2. **`CustomCarlaLocalMapDataset.load_annotations`** re-filters on that
   count (`min_lidar_points` / `lidar_pc_range` ctor kwargs, wired in the
   training config's `data.train/val/test`), and warns if the pkl records
   no counts or a range disagreeing with the config. **Filtering must
   happen here, not in `__getitem__`** — it runs before `_set_group_flag()`,
   keeping `self.flag`, `len(self)`, `format_results()`'s length assert and
   `_format_bbox()`'s positional `data_infos[sample_id]` indexing
   consistent. Skipping in `__getitem__` only works in train mode and would
   silently desynchronise eval.
3. **`GridSamplePoints`** raises `EmptyLidarTileError` (in
   `pipelines/loading.py`) naming the tile; `prepare_train_data` catches it
   and returns `None` to resample, giving up after 100 consecutive skips so
   a globally wrong range fails fast. Test mode lets it propagate, per (2).
   Also fixed a latent `torch.unique(...).max()` crash on empty clouds.

**Verified**: both splits reproduce their baselines exactly (1210 / 15810
instances) with 0 tiles dropped under the current range; re-running train
with the *old* narrow z-range (`-10..18`) drops exactly the 6 town03
overpass tiles, confirming the criterion picks the right tiles.

## The converter is tile-size agnostic (2026-08-10)

`custom_carla_map_converter.py` had three hardcoded assumptions from the
original 25 m export. All gone, verified against the 25 m split export
(259 test / 4103 train), the 60 m grid export, and synthetic edge cases.
The pkl gains `tile_geometry` (tile_dir/tile_radius/tile_side),
`class_lookup` and `classes_kept`; every existing key is unchanged and
`CustomCarlaLocalMapDataset` loads new pkls as-is (259 samples,
`get_data_info` intact). Both real datasets reproduce their totals exactly
(test 1210, train 15810, 0 dropped) — a no-op on existing outputs.

1. **The out-of-bounds warning measured against the wrong origin** and
   fired on nearly every tile: it tested `|xy| > tile_radius + 1m`, but
   polylines are in the `offset` frame, where the tile centre sits at
   `tile_center - offset` — 1-2 m on 25 m tiles, **~17 m** on 60 m ones
   (the same fact behind the viewer's 2026-08-05 axis-cropping bug). It
   fired on 117/259 test, 1750/4103 train, 31/33 grid tiles, and scaled
   with tile size, so it would be ~100% on anything larger. Now compares
   against the tile's **own `bounds`** shifted into the offset frame:
   exact, no slack term, handles non-square tiles. (GeMap's copy at
   `~/Documents/Code/GeMap/tools/gemap/custom_carla_map_converter.py` only
   loosened the bound to `tile_radius + |tile_center - origin|`.) Fires on
   **0** tiles across all three real datasets, while a synthetic polyline
   5 m past the edge is still reported (as 4.00 m, after the 1 m margin) —
   quiet, not dead. Findings print as one ranked summary.
2. **Only `<data_root>/<split>/manifest.json` could be opened**, so the
   grid export (no split level, `grid_manifest.json`) couldn't be converted
   at all. `load_manifest()` now accepts either name at
   `<data_root>/<split>` or `<data_root>`, merging both with the viewer's
   rule (`manifest.json` wins, `grid_manifest.json` backfills).
   **`manifest.json` is the curated view, not a repackaging**: in Town10HD
   it lists 30 tiles vs grid_manifest's 33, and the 3 omitted have only
   `sidewalk_edge` polylines and no driving centerline — so preferring it
   is correct. Its dataset-level *counts* still aren't (`n_tiles: 4103`
   while listing 30), so nothing reads them.
3. **`--lidar-point-cloud-range` defaulted to a hardcoded ±12.5 xy**, which
   on a 60 m export silently measured in-range counts against a quarter of
   the tile. Now derived from the manifest's `tile_radius`/`tile_side` (or
   the widest per-tile `bounds`): the 25 m export still gets ±12.5, the
   60 m one ±30. z unchanged. An explicit flag overrides; the resolved
   range is printed and recorded in the pkl.

Two things that fell out:

- **Range-coverage diagnostic** — reports the median fraction of points
  kept and flags tiles under 90%, catching a range copied from a smaller
  export. It immediately showed something real: **46/259 test, 651/4103
  train and 8/30 grid tiles keep <90%** of their points, because the range
  is square around `offset` while the tile is displaced from it. Worst:
  69.3% (`town04_tile_01206`). Not a regression — it's the existing
  config's geometry — but a measurable argument for centring the range on
  `tile_center - offset`, or re-centring the cloud at load time.
- **`--lane-types` never worked; use `--classes`.** It compared
  `lane_type_lookup` *ids* against each polyline's `type` key, which holds
  a geometry kind (`'arc'`/`'straight'`), so any use of it matched nothing
  and silently produced an **empty pkl**. The real per-polyline taxonomy is
  `class_id`/`class` against the manifest's `class_lookup`, and only newer
  exports carry it. `--classes` takes ids or names and raises a clear error
  on a class-free export instead of silently emptying the output.
  Cross-checked: `--classes driving_centerline` on the grid export yields
  **299** instances, exactly the manifest's `polyline_counts_by_class`.
  `--lane-types` survives as a deprecated alias that warns.

## LiDAR voxelization was slow AND silently wrong (2026-07-30)

An H100 profile (`samples_per_gpu=10`) showed voxelization at **97.6% of
GPU compute** (`_Voxelization` + `point_to_voxelidx_kernel` +
`determin_voxel_num` = 29.167 s of 29.871 s self-CUDA over 20 iters) — not
the backbone, decoder, or losses. Cause: `voxelize()` in `maptrv2.py` calls
mmdet3d's legacy `Voxelization` op **once per sample in a Python loop**,
on tiles with up to 5M points (the converter aggregates unbounded scan
passes, unlike AV2/nuScenes' ~10 sweeps — our `max_num_points=10` /
`max_voxels=[90000,120000]` were copied verbatim and never re-tuned).
Grid size is *not* the cause (verified: wide vs narrow z-range at equal
point count is near-identical); cost scales with raw point count.

**Worse: the legacy kernel silently under-reports occupied voxels at large
point counts.** On a synthetic cloud with exactly 2000 known cells / 5M
points it reported 1,280 — missing 36%. A correctness bug, not just speed.

**Fix**: `GridSamplePoints` in
`projects/mmdet3d_plugin/datasets/pipelines/loading.py`, wired into the
config's train/test pipelines right after `LoadCarlaPointsFromFile` with
`grid_size=lidar_voxel_size` (no precision loss beyond what the voxelizer
already imposes). Collapses points to ~1 per occupied cell via integer
coordinate packing + one vectorized `torch.unique`, before `Voxelization`
sees them. On the worst-case tile: **26.4× faster** (43.7 ms vs 1154.9 ms)
and **100% occupied-voxel recovery** (718/718) — vs 8.6-22.4% for naive
random subsampling, which thins sparse regions like divider lines because
it is density-*proportional*; grid sampling is density-*uniform*.

**Still open**: not yet re-profiled on H100 to confirm the 97.6% figure
actually drops in the full loop. Re-run `tools/maptrv2/profile_train.py`.

## Batch size / workers (2026-07-29, local RTX 3070 8GB)

Against the full 4103-tile train set, fp16, real training steps
(`log_config`'s reported `memory`):

- `samples_per_gpu=1`: ~3555 MiB, ~0.75-0.87 s/iter
- `samples_per_gpu=2`: ~6731 MiB, ~0.73 s/iter (~0.37 s/sample)
- `samples_per_gpu=3`: OOM on 8GB
- Linear fit: ~379 MiB fixed + ~3176 MiB/sample → 4≈13.1GB, 5≈16.3GB,
  6≈19.4GB (95% of 20GB — too tight to trust unverified)
- `num_workers`: `data_time` 0.12 s → 0.005 s already at 4 workers; 8 and
  16 gave no further gain on this 16-core machine.
- 5M-point tiles do **not** spike memory — `max_voxels` bounds it.

**Memory numbers are extrapolated, not verified on H100 MIG; throughput
numbers don't transfer across architectures at all.** Recommendation:
`samples_per_gpu=4-5`, `workers_per_gpu=4-8`, `--shm-size=8g`, and a
10-20 iter dry run on the real MIG partition first.

### `lr=6e-4`'s reference batch is 32, not 1 (corrected 2026-08-11)

This section previously claimed `lr=6e-4` was tuned for `samples_per_gpu=1`
and that you should scale it as `6e-4 × batch`. **That is wrong and
dangerous** — it reads the *per-GPU* batch as the total. Every MapTR and
MapTRv2 config in this repo pairs `lr=6e-4` with `samples_per_gpu=4`, and
`README.md:58` states all experiments ran on **8 GPUs**: the tuned LR
corresponds to an **effective batch of 32**. The old rule over-scales by
32× (batch 22 → 1.3e-2 instead of ~5e-4) and would diverge immediately.

Scale from 32, in whichever direction batch moves:

- linear (`lr ∝ batch`, Goyal et al. 2017): `6e-4 × batch/32`
- sqrt (`lr ∝ √batch`, the variant usually argued for adaptive optimizers
  — this repo uses AdamW): `6e-4 × √(batch/32)`

Prefer **sqrt**; it's better motivated for AdamW and stays closer to a
known-good setting. The two agree closely near the reference batch
(batch 22 → 4.1e-4 linear vs 5.0e-4 sqrt) and only diverge far from it.
Both are heuristics — the linear rule was derived for SGD+momentum on
ImageNet and was never validated for AdamW. It does hold *internally*
here, which is weak corroboration: `maptr_nano_r18_110e.py` uses
`lr=4e-3` at `samples_per_gpu=24` (batch 192), and linear scaling from
6e-4/32 predicts 3.6e-3.

`warmup_iters=500` is inherited from the same bases and **is** an
implicit `samples_per_gpu`-dependent number, since it's counted in
iterations while `total_epochs` is counted in epochs. Larger batch →
fewer iters/epoch → warmup eats a larger fraction of the run. Upstream
nuScenes (~28k samples at batch 32, ~880 iters/epoch) spent ~0.6 of an
epoch warming up. On the 4103-tile CARLA train set at batch 22 that's
~187 iters/epoch, so an unchanged 500 spans ~2.7 of 24 epochs — set
`warmup_iters ≈ 110` to match the upstream fraction. General form:
`warmup_iters ≈ 0.6 × 4103/samples_per_gpu`.

## Resuming with an extended `total_epochs` shocks the LR schedule

Confirmed in the installed mmcv source (`lr_updater.py`):
`CosineAnnealingLrUpdaterHook.get_lr()` is **stateless**:
`progress/max_progress = runner.epoch/runner.max_epochs`, recomputed every
call. `runner.epoch` resumes correctly, but *extending* `total_epochs`
(24 → 48, the natural thing for "24 more epochs") doesn't extend the
schedule — it reshapes it. Epoch 24 goes from the end of the curve (LR ≈
`min_lr_ratio * base_lr`) to the midpoint of a fresh one (≈50% of peak),
shocking a converged model. This is an mmcv design limitation, not a repo
bug; **not fixed in code**.

**Workaround**: for a continuation phase use `--cfg-options
load_from=<checkpoint>` (weights only, *not* `--resume-from`) with its own
fresh, lower-peak LR schedule. For a seamless `--resume-from`, set
`total_epochs` to the full intended budget from the very start.

## Tooling

Training **never persists predictions** — the eval hook writes to
`osp.join('val', cfg.work_dir, <ctime>)` (`mmdet_train.py` ~line 177),
relative to the CWD *outside* the work_dir, which isn't bind-mounted.
`tools/test.py` needed three fixes to be usable:
1. The single-GPU path was disabled by a bare `assert False` (real lines
   commented out beneath). Re-enabled — it's the path that works.
2. The distributed path is broken under torch 2.1: mmcv 1.7.2's
   `MMDistributedDataParallel` hits `AttributeError: ... no attribute
   '_use_replicated_tensor_module'` in torch's `_run_ddp_forward`. Same
   family as gotcha #5. **Not fixed** — use single-GPU.
3. `--eval-options jsonfile_prefix=...` was overwritten by a hardcoded
   `test/<config>/<ctime>/` default. Changed to `setdefault`.

**Confidence scores**: a 1-epoch checkpoint scores only ~0.14-0.17, so a
0.3 threshold filters out *everything*. Check the range before concluding
a viewer or eval shows nothing.

**`tools/maptrv2/carla_bev_vis.py`** — in-container (config+checkpoint+GPU),
GT-vs-pred divider polylines per sample as PNG. Adapted from
`av2_vis_pred.py`'s BEV-only block. Reads GT directly from
`dataset.data_infos[i]['annotation']['divider']`, because
`CustomCarlaLocalMapDataset`'s test-mode path never attaches
`gt_bboxes_3d`/`gt_labels_3d` (only train mode does, via
`vectormap_pipeline`). The repo's other visualizers
(`tools/maptrv2/nusc_vis_pred.py`, `tools/maptr/vis_pred.py`,
`tools/analysis_tools/visual.py`) are all coupled to multi-camera setups or
`nuscenes-devkit`, and `tools/test.py --show` needs a `show()` the CARLA
dataset doesn't implement.

**`tools/maptrv2/webviewer.py`** — standalone, *outside* the container
(`pip install flask matplotlib seaborn tensorboard`). Loss/eval curves
(EMA-smoothed over faint raw), embedded TensorBoard, BEV gallery, dark
theme. Two things worth keeping:
- Launch TensorBoard as `sys.executable -m tensorboard.main`, never the
  bare `tensorboard` on `$PATH` — a stale apt-packaged copy shadowing the
  pip install hangs forever with no error. Its stderr now goes to a temp
  log shown on the page, with a `/tb_status` health check.
- `merge_log_dicts()` merges *all* `*.log.json` (mmcv writes a new one per
  (re)start), but **not** via "later file wins": a resumed run logs a
  near-empty checkpoint-save marker (`{'epoch': N, 'iter': M}`, no `loss`)
  for the epoch it resumed *from*, and a naive `dict.update()` let that
  replace the earlier file's complete data. It keeps whichever entry has
  more recorded iterations, per epoch.

**`tools/maptrv2/dataset_viewer.py`** — standalone CARLA browser (flask /
matplotlib / numpy, no torch or GPU). Two tabs:

*`?tab=browse`* — per-tile views: true RGB (`features[:, 3:6]`), lane
label, flat top-down, 1 m² density heat map, intensity; polylines
(per-class coloured, legend + checkbox filters) and log/linear density as
overlays. `--work-dir <dir>` overlays predictions (yellow dashed) over GT
(red solid), rescanning per request. Class-free result sets still draw
yellow dashed; older class-free exports render as one unclassified red set.

Manifest handling: discovery is a bounded-depth walk (data-root + 2 levels)
accepting `manifest.json` or `grid_manifest.json`. `grid_manifest.json`
has no `split`, no `towns`, no `tiles_per_town`, and its tiles have no
`town` key or name prefix (`tile_00000`) — split falls back to the manifest
directory name, town is inferred from `town_ply` (`Town10HD_full.ply` →
`Town10HD`). **Manifest counts are not trustworthy** (the packaged
`manifest.json` next to a 30-tile grid export claims `n_tiles: 4103`);
everything shown is recomputed from the tile list, and where both files
exist `manifest.json` wins with `grid_manifest.json` backfilling. **Tile
names are unique only within a directory** — every town has a
`tile_00000`, so tiles are keyed `<dataset>/<name>` and `/tile.png`
requires both `name` and `ds` (which also keeps `ds` from reaching an
unchecked filesystem join).

*`?tab=stats`* — exists so dataset-distribution questions stop being
answered by throwaway scripts. Two tiers:
- **manifest tier** (zero I/O, always shown): points/tile, points/m²,
  polylines/tile, per-class polyline counts.
- **deep tier** (opt-in button, background thread, `--scan-workers`
  default 8): effective post-grid-sampling point count, z extent,
  per-point label mix, `|tile_center − offset|`, polyline vertex counts
  and arc lengths. Reads every `.npz`; the whole 4362-tile / ~12 GB local
  set scans in ~60 s and reloads instantly from cache.

Cache: `~/.cache/maptr_dataset_viewer/` (`--stats-cache`), one file per
*dataset directory* so a `--data-root <root>` scan is reused by
`--data-root <root>/train`. Never written inside the dataset dir (`:ro` in
the container workflow). The key includes the grid size, so changing
`--scan-grid` invalidates rather than mixing incomparable numbers.

Two defaults that are easy to get wrong from this file alone:
- `--pc-range-z` defaults to **`-72 96`**, the *current* widened
  `lidar_point_cloud_range` z span — not the `[-8, 15]` gotcha #13
  mentions, which predates the town03 overpass fix. The old values flag
  3,399 of 4,362 tiles, i.e. pure noise.
- `--scan-grid` defaults to **`0.1 0.1 0.4`**, matching the anisotropic
  `lidar_voxel_size`, so "effective points" = what the model sees. It is
  deliberately *not* expected to match `GridSamplePoints` exactly — that
  transform phases its grid on `point_cloud_range`'s origin and clamps
  out-of-range coords, this one phases on the tile's own minimum
  (town01_tile_00000: 4,132 vs ~3,242). The 1200× collapse is the signal,
  not the last digit.

Suspect tiles appear in a ranked table with reason chips (`degenerate` /
`near-empty` / `no-GT` / `z-out-of-range` / `gt-outside-tile`), each
linking into the browse tab; also downloadable as `/stats.csv`. Note the
first full scan found **zero** `no-GT` or `near-empty` tiles, so the
gotcha #13 crash doesn't originate in the data. `|tile_center − offset|`
medians are 1.2-2.3 m per town — a direct measure of the old GT-frame
bug's cost, but deliberately *not* a suspect flag, since it's a property
of tile geometry (7.3 m median on the healthy 60 m grid export).

Charting conventions if extending the stats tab: distributions compared
across groups are horizontal box plots in a **single** hue (six towns is at
the categorical cap, and axis labels carry identity); categorical colour is
reserved for the two charts whose subject really is a category (polyline
class, lane label), reusing the browse tab's `CLASS_COLORS`/`LABEL_COLORS`;
the effective-vs-raw scatter uses emphasis (one muted hue + a status colour
for flagged points) rather than colouring by town.

## Transferable debugging lessons

- **`&gt_frame=` in an HTML attribute parses as `>_frame=`.** Browsers
  resolve known entity names (`&gt`) without the closing `;`, so
  `...&mode=density&gt_frame=offset...` reached the server as
  `mode="density>_frame=offset"` with `gt_frame` dropped. **curl can never
  reproduce this** — nothing HTML-parses the URL — so it survived rounds of
  "the server is provably correct". Fixed with `urlencode()` +
  `html.escape()`. **Test a web UI by parsing the emitted HTML
  (`html.parser`), not by curling URLs you constructed yourself.**
- **matplotlib in a threaded Flask server segfaults.** Needs *both* the OO
  `Figure`/`FigureCanvasAgg` API (never `pyplot`, a global state machine)
  *and* a `threading.Lock` around rendering — the OO API alone still
  crashed under concurrent load. Sequential curl never reproduces it; a
  browser loading a gallery does.
- **A colormap whose low end matches the page background looks blank.**
  With this dataset's density dynamic range the heat map appeared empty.
  Fixed with `inferno` + a log norm.
- **`fig.tight_layout()` must run after titles/labels exist**, or it
  reserves no room and clips them off the canvas (the `finish()` helper
  takes `tight=True` and orders it correctly).
- **Fractional `subplots_adjust` margins shrink in absolute terms** as
  variable-height figures get shorter, clipping the x-axis label off every
  one-group plot. Use `abs_margins()`, which takes inches.
- **Plot axes must be centred on `tile_center - offset`, not the origin.**
  Rendering happens in the `offset` frame where the tile is not centred on
  (0,0); `±tile_radius` axes cropped every plot — mild (~1.3 m) on 12.5 m
  tiles, severe (17 m, over half a radius) on 30 m grid tiles. Axes and
  density histogram bin edges are both centred correctly now.
- **`pkill -f "webviewer.py"` plus a relaunch in the *same* shell
  invocation self-matches** — the pattern matches the relaunch command's
  own text later in the invocation, killing the wrapper before it runs. Use
  separate tool calls.

## Open items / next steps

- **`lidar_point_cloud_range`'s z lower bound is too high.** 98 tiles have
  `z_min` below the configured −72, reaching −97.1 (town03). The config's
  comment cites an observed extreme of `[-66.90, 90.52]`, measured before
  the whole dataset was available. Out-of-range points are silently dropped
  by the voxelizer, so those tiles train on partial geometry. Either widen
  the range (and re-measure `sparse_shape` / `lidar_bev_proj.in_channels`
  per gotcha #4) or confirm the lost points are sub-surface and irrelevant.
  Reproduce via the stats tab's deep scan or its `/stats.csv` `z_min`.
- **Decide what to do with the 15 genuinely degenerate cap tiles** — named
  by the `degenerate` chip / `effective_over_raw` in `/stats.csv`. They
  cost a full load+decompress for ~4k usable points each.
- **`carlasim_map.py`'s `ann_file_train` is stale.** It points at
  `data/carla/carla_map_infos_train.pkl`, which has never been generated
  locally (the converter has only run with `--split test`), so training
  against this config as-is fails with a missing-file error. **Do not
  silently revert — check with the user first**; it may reflect intent to
  wire up a real train split. `ann_file_val`/`ann_file_test` are fine.
- **Only the local 259-tile test subset exists.** The full dataset is 4103
  tiles / 5 towns with no `train`/`val` split yet — one is needed (by town
  or stride) before a real run. Converting it needs no code changes, just
  `--data-root` and new `ann_file` names.
- **A new/larger dataset should be converted first and its
  `carla_map_infos_<split>_dropped.json` inspected** before a long run — a
  large drop count means the range is wrong, not that the data is bad.
- **Class taxonomy is divider-only**, collapsing *all* CARLA lane types
  (driving/curb/sidewalk/border/restricted/parking/shoulder/stop/other)
  into one `divider` class — deliberate, to maximise GT density on the tiny
  local set. Revisit filtering to driving lanes before a real run via the
  converter's `--classes` flag (e.g. `--classes driving_centerline`); the
  older `--lane-types` never matched anything — see the 2026-08-10 section.
- **`sparse_shape`/`lidar_bev_proj.in_channels` were measured empirically**
  for the *current* range/resolution (`[251,251,71]` and `384`). If either
  changes, re-measure with a dummy `extract_lidar_feat()` call rather than
  computing by hand — gotcha #4 makes axis order easy to get wrong.
- **No real accuracy signal yet.** A longer run with a real train/val split
  is needed before any conclusion about model quality.
- Detailed design/reasoning from the original work session:
  `/gel/usr/johil9/.claude/plans/when-running-the-dockerfile-mellow-russell.md`
