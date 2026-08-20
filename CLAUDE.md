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

15. **Changing `num_vec_one2one` without `bbox_coder.max_num` crashes at
    the first validation**, not at build time:
    `RuntimeError: selected index k out of range` from
    `nms_free_coder.py`'s `decode_single`, which does
    `cls_scores.view(-1).topk(max_num)`. `get_bboxes` decodes the
    **one2one branch only**, so the tensor holds
    `num_vec_one2one * num_map_classes` entries — with CARLA's single
    class that is just `num_vec_one2one`. The base pairs 50 with
    `max_num=50`; the HM configs cut queries to 25 and inherited the 50.
    Nothing validates the pair, so a full epoch trains before it dies.
    **Fix**: both HM configs derive
    `max_num = min(50, num_vec_one2one * num_map_classes)` and pass it as
    `pts_bbox_head.bbox_coder.max_num`. Note `max_num` also caps how many
    instances eval can ever score, so raising queries is safe while
    lowering them is not.

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

## GeMap-style `tile_center` frame (2026-08-13, merged to `maptrv2` 2026-08-18)

GeMap's copy of the converter centres tiles and polylines on `tile_center`,
not `offset`. That is available here as an opt-in frame, wired on **both**
sides so it cannot silently misalign the way the original `tile_center` bug
did (see "CARLA data facts"):

- `--gt-frame {offset,tile_center}` (default `offset`, a no-op). With
  `tile_center` the polylines are shifted by `tile['center']` and each sample
  records `lidar_recenter_shift = offset - tile_center`, plus `gt_frame` and
  `lidar_offset`; the pkl gains a top-level `gt_frame`.
- `LoadCarlaPointsFromFile(recenter=True)` adds that shift to the stored
  points (which are always written in the `offset` frame), after the `z_max`
  cut — that filter is defined on the stored z, and `count_points_in_range()`
  in the converter applies the same order.
- `get_data_info` threads `annotation_origin` / `gt_frame` /
  `lidar_recenter_shift` into the pipeline. An old pkl without the shift
  raises a clear error instead of loading misaligned.
- `projects/configs/maptrv2/maptrv2_carla_r50_24ep_lidar_tilecenter.py` sets
  both together, with its own `ann_file`s under `data/carla/tile_center/` and
  its **own `map_ann_file`** — sharing one `carla_map_gt.json` between frames
  would score one frame against the other's GT (`_format_gt()` never
  regenerates).

**The frames are equivalent up to a rigid shift, so this is not an accuracy
fix** — verified: GT in the two pkls differs by exactly the recorded shift
(max residual 0.0), and the median GT-vertex → driving-surface distance is
bit-identical (0.0568 m) in both. What it does change:

- **Range coverage goes from 46/259 tiles under 90% to median 100%.** The
  square `lidar_point_cloud_range` is centred on the origin, so in the
  `offset` frame it sits 1-2 m (25 m tiles) to ~17 m (60 m tiles) off the
  actual tile and crops it. `tile_center` is the tile's own centre, so it
  doesn't. This is the concrete argument the 2026-08-10 coverage diagnostic
  was pointing at.
- The model's input is re-centred, so **checkpoints are not comparable across
  frames** — retrain.

Verified on the 259-tile test split: 1210 instances / 0 dropped in both
frames, and a full 1-epoch train+eval under the tile-centre config runs clean
(mAP 0.0218 with train==val, so no accuracy conclusion).

### Tile-centre configs, per tile size (2026-08-18)

`maptrv2_carla_r50_24ep_lidar_tilecenter.py` (25 m),
`maptrv2_carla_r50_24ep_lidar_30m_tilecenter.py` (30 m, `tile_radius=15`) and
`maptrv2_carla_r50_24ep_lidar_30m_HM_tilecenter.py` (30 m + the polyline
geometry loss). Each is a thin `_base_` overlay on its own offset-frame
sibling and changes only four things: `recenter=True` on the loader in
**both** pipelines, and `ann_file`/`map_ann_file` redirected under a
`tile_center/` subdirectory.
Everything tile-size dependent — `sparse_shape`, `bev_h_`/`bev_w_`, the coder
and assigner ranges, the dataset's own `pc_range` — stays bound in the base's
namespace and is deliberately NOT re-derived. Verified through `Config.
fromfile`: the 30 m overlay resolves to `sparse_shape=[301,301,421]`,
`bev_size=(120,120)`, `pc_range=±15`, `recenter=True` on train *and* test, and
`aux_seg` intact (mmcv merges dicts recursively, so only the listed keys move).

**`evaluation` has to be restated in the overlay**, purely to re-point it at
the recentred `test_pipeline`. Neither HM config overrides `evaluation`, so
without that line it inherits the 30 m base's copy and scores in the `offset`
frame while training in the tile-centre one — silently. Verified through
`Config.fromfile`: `recenter=True` reaches `train_pipeline`,
`test_pipeline` **and** `evaluation.pipeline`, while `model`, `optimizer`,
`lr_config` and `total_epochs` are byte-identical to the base (`loss_pts`
stays `PolylineGeomLoss` mode `emd` at the tile-scaled weight 2.4 = 2.0 ×
30/25, so the HM tile-size derivation is untouched). The HM overlay shares
its base's `map_ann_file` name because it leaves `fixed_ptsnum_per_gt_line`
at 20 — note the comment above `data` in the HM configs still says 40, which
is **stale**; read the value. Two configs with *different* line counts must
not share a `map_ann_file`, since `_format_gt()` bakes the resampling into a
json it writes once and never regenerates.

Generate the pkl per split, into that same `tile_center/` directory. One pkl
serves both the plain and HM configs at a given tile size — the frame is a
property of the data, the loss is not:

```bash
python tools/maptrv2/custom_carla_map_converter.py \
    --data-root <30m export> --split test \
    --gt-frame tile_center --out-dir data/carla_30m/tile_center/
```

**Do not pass `--lidar-point-cloud-range`.** The converter derives xy from the
manifest's own `tile_radius`/`tile_side` (2026-08-10 section) and prints what
it resolved; that must equal the config's `lidar_point_cloud_range` or the
dataset warns at load. Confirmed end-to-end on the 60 m grid export, which is
the only non-25 m export on this machine: it resolves ±30 unaided, and range
coverage goes from **median 92.4% / 8 of 30 tiles under 90%** in the offset
frame to **median 100%** in the tile-centre one, with instance counts
unchanged (1246). That is the same effect measured on the 25 m split, and it
does not shrink as tiles grow — the range is square around the origin while
the tile is displaced from it.

**No `tile_radius=15` export exists locally**; `data/carla_30m/` is absent, so
both 30 m configs are still unexercised against real data.

### It reproduces GeMap's converter exactly (2026-08-18)

Both converters were run over the same 259-tile test split and their pkls
compared array by array: **max |GT_MapTRv2 − GT_GeMap| = 0.0** across all 1210
instances, same tiles, same `annotation_origin`. So the two are
interchangeable on this data and a GeMap checkpoint can be scored against
either pkl. Only the bookkeeping differs, which matters when reading a pkl
back:

| | MapTRv2 | GeMap |
|---|---|---|
| pkl-level key | `gt_frame` | `annotation_frame` |
| per-sample shift | `lidar_recenter_shift` = `offset − origin` | `recenter_shift` = `origin − offset` |

Note the **opposite sign**. Anything reading either file should detect the
frame *structurally* — compare `annotation_origin` against `tile_center` —
rather than trusting a key, which is what `dataset_viewer.pkl_gt_frame()`
does.

`tile_center_origin()` was hardened to match GeMap's `tile_origin()`: an
export that states `center` as `[x, y]`, or states only `bounds`, no longer
crashes. The z rule is GeMap's and is load-bearing — a 2D centre keeps the
block's own z, so the shift has zero z component and the cloud's z is left
untouched. `code_size=2` means z never reaches a regression target either
way. The 25 m export states 3D centres, so this changes nothing there (still
bit-identical); it only stops other exports failing.

## Multi-class map taxonomies: driving / curb (2026-08-20)

The pipeline was divider-only — *every* CARLA lane type collapsed into one
`divider` class. It now supports an arbitrary named taxonomy, wired on all
three sides (converter, dataset, config) so a mismatch is reported instead of
producing silently empty GT for a class.

Two configs use it, both 30 m tile-centre overlays that change the taxonomy
and *nothing* else:
`maptrv2_carla_r50_24ep_lidar_30m_tilecenter_2cls.py` and
`maptrv2_carla_r50_24ep_lidar_30m_HM_tilecenter_2cls.py`, with
`map_classes = ['driving', 'curb']` (label 0 = the export's
`driving_centerline` polylines, label 1 = its `curb` ones).

```bash
python tools/maptrv2/custom_carla_map_converter.py \
    --data-root <30m export> --split test --gt-frame tile_center \
    --map-classes driving=driving_centerline curb=curb \
    --out-dir data/carla/ --out-tag 30m_tc_2cls
```

- **`--map-classes name=export_class[,export_class...]`** emits one
  `annotation` key per named class instead of a single `divider`. Order is
  the label order. A bare `name` means `name=name`. Groups must be disjoint
  (the same polyline under two labels is unmatchable, so it raises), and
  `--classes` is refused alongside it — both filter the same polylines.
- **`--out-tag TAG`** appends to the output filename
  (`carla_map_infos_<split>_<TAG>.pkl`). Without it the filename is a
  function of `--split` alone, so writing a second dataset into
  `data/carla/` would **overwrite the existing 25 m offset-frame pkl with no
  warning**. That is why these configs, which live under `data/carla/` by
  request rather than in a `carla_30m/tile_center/` subdirectory, name their
  files `..._30m_tc_2cls.pkl`.
- The pkl gains `map_classes` (the names, in label order) and `class_groups`
  (name → export class ids).

**`VectorizedAV2LocalMap.CLASS2LABEL` is hardcoded to nuScenes/AV2 names**
(`divider`/`ped_crossing`/`boundary`/`centerline`), and
`gen_vectorized_samples` drops anything mapping to −1. A CARLA taxonomy of
its own would therefore vectorize to **zero instances, with no error at all**
— the failure this needed guarding. `VectorizedCarlaLocalMap`
(`carla_offlinemap_dataset.py`) subclasses it and builds the mapping from the
config's `map_classes` list: `{name: index}`. For `['divider']` that is
`{'divider': 0}`, i.e. identical to the base.

`CustomCarlaLocalMapDataset.load_annotations` now warns at load time when the
config's `map_classes` names a key the pkl lacks, when the pkl's declared
order differs (order *is* label order, so a permutation relabels every
instance), and when a configured class has zero instances anywhere.

Note `--map-classes` **drops** every export class not named: with
`driving=driving_centerline curb=curb`, the `road_edge`, `lane_divider`,
`center_divider`, `sidewalk_edge`, `median` and `crosswalk` polylines are
gone, not merged. On the Town10HD grid export that keeps 299 + 135 of 1251
polylines. To fold road edges into curbs, say `curb=curb,road_edge` and
reconvert — the grouping lives in the pkl, not the config.

**The local 25 m export cannot be split this way**: it has no `class_lookup`
at all, so its polylines carry no class and the converter raises rather than
emitting empty classes. Only the newer class-carrying exports (the 60 m grid
one, and presumably the 30 m one when it exists) can.

`aux_seg` stays at `seg_classes=1` in both configs. The BEV auxiliary seg
head's channel count comes from `aux_seg['seg_classes']`, independently of
`num_classes` (`maptrv2_head.py:242`), and the dataset rasterizes every class
into that one mask. Raising it is a separate experiment and changes the seg
head's weight shape.

**Verified**, on the 60 m grid export (the only class-carrying data here) and
the 259-tile 25 m split:

- Single-class output is **unchanged**: same 259 tiles / 1210 instances, every
  annotation array bit-identical to the pre-change converter (max |diff| 0.0),
  every per-sample key equal; the pkl gains only `map_classes` and
  `class_groups`. The dataset still loads it with `CLASS2LABEL ==
  {'divider': 0}`, 1210 label-0 instances, and no warnings.
- Two-class output: 299 `driving` + 135 `curb`, exactly the manifest's
  `polyline_counts_by_class`, disjoint on every tile.
- Both configs resolve through `Config.fromfile` with `num_classes=2` on the
  head *and* the coder, `map_classes` on all three data splits, `recenter=
  True` on train/test/`evaluation` pipelines, `pc_range` ±15 intact, and
  `model` minus the class knobs, `optimizer`, `lr_config`, `total_epochs` and
  both pipelines **byte-identical to their parents** (`loss_pts` stays
  `PolylineGeomLoss` in the HM one).
- Building the dataset on the two-class pkl yields labels {0: 299, 1: 135};
  the missing-key and wrong-order warnings both fire when deliberately
  mismatched.

**Gotcha #15 again, in the HM config.** `max_num` must be ≤
`num_vec_one2one * num_map_classes`, and the HM configs cut queries to 25. At
one class that pinned `max_num` to 25; at two the bound is 50, so the 2-class
HM config raises it back to the base's 50 — which also matters because
`max_num` caps how many instances an eval can ever score, and there are more
lines to find. Raising is the safe direction; lowering crashes at the first
validation, after a full epoch has trained.

Two host-side tools read the `divider` key directly and would have shown an
empty GT rather than an error; both now iterate the actual classes:
`tools/maptrv2/carla_bev_vis.py` (unions every class, drawn in one colour)
and `dataset_viewer.py`'s `load_gt_pkl` (via a new `pkl_map_classes()`, which
reads the pkl's declared order and falls back to `('divider',)` for older
files). Verified: 259 tiles / {0: 1210} on the old pkl, 30 tiles /
{0: 299, 1: 135} on the new one.

The viewer's **results tab now matches per class**, and while doing that a
separate, pre-existing eval bug came out (prediction resampling). Both are
written up under "Training-results tab" — read that before trusting any AP
this tab printed before 2026-08-20.

**Still divider-shaped**: the viewer's `shape_counts()` (the curve-vs-line
box plots) reconciles `n_arc + n_straight` from `reference_lines/*.json`
against the pkl's instance count, and a `--map-classes` pkl legitimately
holds fewer instances than the json holds polylines — and on the
`../carla_test` export it reads a DIFFERENT directory than the one converted
(`reference_lines/` vs `reference_curb_driving_lines/`), so the counts cannot
reconcile at all. Those tiles are dropped from the shape charts and reported
— the same behaviour a `--classes` subset already produced, safe but
unhelpful. The fix, if those charts are wanted on a multi-class run, is to
thread the pkl's `class_groups` and `tile_geometry.reference_dir` into that
filter; both are recorded for exactly this reason.

### The 30m curb/driving export (`../carla_test`, 2026-08-20)

The first real class-carrying CARLA dataset on this machine, and what the
`_2cls` configs were verified against: `../carla_test/test`, 3795 tiles,
7.7 GB, `tile_side` 30.0 (so `tile_radius` 15 -- exactly the 30m configs'
geometry, and the converter resolves `+-15` from the manifest unaided).

Two things about its layout that the converter had to learn:

1. **It ships TWO polyline directories, holding different GT.**
   `reference_lines/` is the historical unclassified set (~2.1 polylines/tile,
   `class_id` absent) and `reference_curb_driving_lines/` is the classified
   one (3.5/tile, every polyline carrying `class_id`/`class`). Same filenames
   in both. `--reference-dir` selects one explicitly; by default
   `reference_lines/` still wins, and only `--classes`/`--map-classes` -- which
   *need* a taxonomy -- falls back to a sibling that has one, printing what it
   picked. Recorded in the pkl as `tile_geometry.reference_dir`, because the
   two directories are not two spellings of one GT set.
2. **The manifest has no `class_lookup`.** The taxonomy is declared per tile,
   inside each polyline json (`"classes": {"0": "driving", "1": "curb"}`), so
   the converter reads it from there when the manifest is silent. All 3795
   tiles agree; a tile that disagreed would be listed by
   `report_class_conflicts()` rather than relabelled silently.

Its manifest also states `center` as a 2D `[x, y]` and gives tiles no
`bounds`, both of which the 2026-08-18 hardening already covers (2D centre ->
zero z shift; footprint falls back to the polyline json's `tile_bounds`).

Converted with::

    python tools/maptrv2/custom_carla_map_converter.py \
        --data-root ../carla_test --split test --gt-frame tile_center \
        --map-classes driving curb \
        --out-dir data/carla/ --out-tag 30m_tc_2cls

(`driving` and `curb` are the export's own class names, so the bare-name
shorthand applies -- no `name=export_class` needed.) Result: **3795 tiles, 0
dropped, 9458 driving + 3819 curb = 13277 instances**, matching a direct
count over the jsons exactly. Every tile has driving; 2025 of 3795 (53%) also
have curb, and 1770 are driving-only.

**The tile-centre frame is not optional on this export.** Converted in the
`offset` frame instead, the same data gives **75 tiles dropped outright**
(zero in-range points, i.e. gotcha #13 territory), median coverage 92.5%, and
**1282 of the surviving 3720 tiles keeping <90%** of their points -- one tile
keeps 0.0%. In `tile_center`: 0 dropped, median **100%**, worst tile 99.8%.
The reason is visible in the pkl: `|offset - tile_center|` has a median of
2.9 m but a **max of 117.6 m**, and 279 tiles exceed 15 m -- on this export
`offset` is evidently not the tile's own centroid for every block. This is
the 2026-08-13 coverage argument, an order of magnitude louder.

Frame sanity, measured the same way the original `tile_center` bug was
caught: GT vertex -> nearest LiDAR return, median over 60 random tiles,
**0.076 m** with the points recentred as `LoadCarlaPointsFromFile(
recenter=True)` does it, and `max |GT xy| = 15.00 m` exactly (the tile half
width). The recentring shift's z component is identically zero here, because
the manifest's 2D `center` makes z fall back to the block's own `offset`.

**There is no train split** -- this export is `test` only, so
`data.train.ann_file` in both `_2cls` configs points at a
`carla_map_infos_train_30m_tc_2cls.pkl` that does not exist yet. Convert one
(or repoint `data.train.ann_file`) before a real run; a train==val run scores
nothing meaningful.

#### First real 2-class run, and why `curb_AP` is 0 (2026-08-20)

A 1-epoch train+eval over a 64-tile subset of `../carla_test` (tiles that
carry *both* classes; 126 driving + 112 curb GT instances) runs clean end to
end under `maptrv2_carla_r50_24ep_lidar_30m_tilecenter_2cls.py`: 3326 MiB at
`samples_per_gpu=1`, 1.14 s/iter on the local RTX 3070, checkpoint saved,
and the eval reports **per class**::

    CarlaMap_chamfer/driving_AP: 0.0102   curb_AP: 0.0000   mAP: 0.0051
    | driving | 126 GT | 3200 dets |
    | curb    | 112 GT |    0 dets |

So the two-class chain works. **`curb` getting zero detections is not a
decoding bug**, and it is worth knowing why before reading it as one:

* `MapTRNMSFreeCoder.decode_single` takes `cls_scores.view(-1).topk(max_num)`
  and derives `labels = indexs % num_classes`, so label 1 is perfectly
  reachable; 3200 = 64 tiles x `max_num` 50, all of which landed on driving.
* Measured on the resulting checkpoint: mean sigmoid score **0.1865 driving
  vs 0.1694 curb**, and **0 of 400** one2one queries have curb as their
  argmax. Both classes are collapsed into the same narrow band an
  undertrained MapTR always produces (the 0.163-0.174 range documented under
  "Tooling"), with driving uniformly ~0.017 higher -- it is the majority
  class, 2.5:1 in this data.
* Because `max_num == num_vec_one2one`, a uniform bias of *any* size is
  enough for one class to take every slot: the top 50 of 100 flattened
  scores are all driving. Upstream nuScenes has the same shape (max_num 50,
  num_vec 50, 3 classes), so this is inherent to the decoding, not to this
  config.

**Read `curb_AP` only after the classifier has actually separated the
classes.** Re-check the score gap on a converged checkpoint before treating
a zero as a modelling result.

Reproduce (the subset pkl is just a filtered copy of the converted one)::

    docker run --rm --runtime=nvidia --shm-size=8g \
      -e NVIDIA_VISIBLE_DEVICES=all -e PYTHONPATH=/MapTR \
      -v $(pwd)/projects:/MapTR/projects -v $(pwd)/data:/MapTR/data \
      -v $(pwd)/tools:/MapTR/tools -v $(pwd)/work_dirs:/MapTR/work_dirs \
      -v /home-local/johil9.nobkp/Documents/Code/carla_test:/home-local/johil9.nobkp/Documents/Code/carla_test:ro \
      -w /MapTR jhd0ck3r/maptrv2:latest python3 tools/train.py \
      projects/configs/maptrv2/maptrv2_carla_r50_24ep_lidar_30m_tilecenter_2cls.py \
      --work-dir work_dirs/carla_2cls_smoke --cfg-options total_epochs=1 \
      runner.max_epochs=1 evaluation.interval=1 data.workers_per_gpu=0

Note the local RTX 3070 usually has a CARLA simulator on it (~1.9 GB), and
the first attempt at this OOM'd at 3.16 GiB with 30 MiB free. Check
`nvidia-smi` before blaming the config.

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
matplotlib / numpy, no torch or GPU; `open3d` optional, 3D tab only). Four
tabs:

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

### Training-results tab (2026-08-12)

*`?tab=results`* — scores a run's predictions **per tile** and asks what the
failures have in common. Needs `--work-dir` (for a `*result*.json`) plus an
eval GT.

**The GT can come from the converter pkl, not just `carla_map_gt.json`.**
This matters because `carla_map_gt.json` is *not shipped with a dataset* —
`_format_gt()` writes it to the config's `map_ann_file` the first time an
eval or `tools/test.py` runs, so a freshly converted dataset has none.
`carla_map_infos_<split>.pkl` exists as soon as the converter has run, so it
is searched first and auto-selected. `--gt-json` (alias `--gt`) takes either.

The pkl path **reproduces** `_format_gt()` rather than approximating it,
because that chain is almost an identity: `gen_vectorized_samples()` wraps
each annotation array in a `LineString`, keeping it if it has ≥2 points and
a class mapping to a label ≠ −1; `LiDARInstanceLines` stores that list
untouched (`self.instance_list = instance_line_list`); `_format_gt` writes
`np.array(list(gt_vec.coords))[:, :code_size]`. **No resampling, no clipping
to pc_range, no reordering** — just the pkl's own arrays, filtered by point
count and class, truncated to 2D. Verified against an existing
`carla_map_gt.json`: 259 tiles, 1210 instances, every coordinate
bit-identical, identical end-to-end mAP and TP total. The train pkl
independently reproduces 4103 tiles / 15810 instances.

The pkl is also the *safer* source when both exist: `_format_gt()` skips
regeneration if the json is already there, so a stale json silently scores
against the wrong GT (the warning at the top of this file), while the pkl is
rewritten every converter run. Loading it needs only `pickle` + `numpy` —
no mmcv, no mmdet3d — which is what keeps it host-side.

**Which GT covers which predictions is decided by token overlap**, not file
order: splits are disjoint tile sets and nothing in a results json records
its split, so defaulting to the first candidate would score test predictions
against the train pkl and report every tile as missing GT. A forced mismatch
names the file that would have worked instead.

**Why a per-tile score is even available.** `mean_ap.eval_map()` already
calls `custom_tpfp_gen()` **once per tile** and only the AP *aggregation* is
global. So the per-tile TP/FP vectors here are exactly the eval's; the only
thing added is running the AP formula over one tile's detections. That makes
"per-tile AP" a **local** score — that tile's own ranking against its own GT
— and *not* its contribution to the global AP, which isn't a well-defined
per-tile quantity (global ranking interleaves detections from every tile).
The global AP is computed too, with the real interleaved ranking, and shown
in the header specifically so it can be checked against the training log's
`CarlaMap_chamfer/*`. **The two will not be equal — that's not a bug.**

**The eval is reimplemented in numpy** (`chamfer_score_matrix`,
`tpfp_from_matrix`, `average_precision`) because the viewer runs on the
**host**: no torch, no mmdet3d, and neither shapely nor scipy is installed
there. Two deliberate deviations, both verified inert:
- GT resampling to 100 points uses arc-length interpolation instead of
  shapely's `LineString.interpolate` — identical, since shapely interpolates
  linearly between vertices.
- The STRtree 2 m-buffer prefilter is dropped and every pred×GT pair is
  scored. It cannot change a match: lines whose 2 m buffers miss are >4 m
  apart everywhere, so chamfer <-4, past even the -1.5 threshold. This also
  sidesteps gotcha #6 (shapely 2.0's STRtree break) entirely.

**Verified bit-exact against the real implementation**, not just
approximately: identical global AP at every threshold versus
`_evaluate_single` (i.e. `format_res_gt_by_classes` +
`custom_tpfp_gen`/`eval_map`) run inside the container on the same files —
**0.00008 / 0.00672 / 0.03118** on the 259-tile demo run, agreeing to 1.6e-10,
and 5.5e-12 on the two-class run. Re-run that comparison if the matching is
ever touched; `/gel/usr/johil9/.claude/jobs/*/tmp/resample_probe.py` is the
harness, and it drives the real dataset object rather than reassembling
eval_map's arguments by hand.

**CORRECTION (2026-08-20)**: this section previously recorded 0.00003 /
0.00311 / 0.02569 (mAP 0.00961) as the verified-identical numbers. They were
the viewer's, and they were **wrong** — understated by up to 2.2x. The
viewer resampled GT to 100 points but scored the raw 20-point predictions,
while `get_cls_results()` interpolates **both sides** to 100 whenever the
dataset's `eval_use_same_gt_sample_num_flag` is set, which every CARLA config
sets. Measuring a 20-point polyline against a 100-point one inflates chamfer,
so matches were lost: the effect was almost exactly a one-notch threshold
shift (the viewer's AP at 1.5 m equalled the real AP at 1.0 m). Fixed by
resampling predictions to `PRED_NUM_SAMPLE` too. Whatever the original
comparison checked, it was not this; the numbers above are re-measured.

Note the training log's epoch-1 mAP (0.01551) still does not equal the
results json's — different prediction sets, not a discrepancy in the maths.
The log's came from the training-time eval hook; the json came from a
separate `tools/test.py` pass.

**Matching is per CLASS** (2026-08-20), as `eval_map`'s own loop is: a
`driving` prediction can never be a true positive against a `curb` GT line.
The pooled version this replaced was not a rounding difference — a
deliberately mislabelled prediction laid *exactly* on a GT line of another
class scored AP **1.000** pooled and **0.000** per class. It needs the GT to
declare >1 class and the results json to carry matching `cls_name`s;
otherwise it falls back to pooling and `class_health()` says so in red.
`global_ap()` pools per class and then averages, which is what makes it
comparable to the log's `mAP`; per-tile AP averages only the classes present
in *that* tile, since dividing by the full class count would cap a
driving-only tile at 0.5 for reasons unrelated to the prediction. With one
class every number is unchanged — verified tile by tile, pooled versus split,
on all 259 test tiles (0 mismatches, identical global AP and TP total 286).
Note 286, not the 189 this file used to record: that number came from the
un-resampled predictions corrected above.
The global PR chart becomes one curve per class at 1.0 m when there is more
than one; three thresholds times N classes is unreadable, and curves are
never pooled across classes.

**`top_n` on the tile renderer** (new, also available to the browse tab).
The model emits a fixed `num_vec` (50) predictions *every* time, so drawing
all of them buries the GT under near-zero-confidence guesses. An absolute
score threshold can't fix this because calibration moves during training (a
1-epoch checkpoint tops out ~0.17, so 0.3 shows nothing and 0.1 shows all
50). The results tab defaults to `top_n = the tile's own GT count` — "the
model's best few guesses, as many as there are real lines" — and the legend
always states what was hidden (`top 9 of 50 by score`).

Sort/rank by any of a dozen metrics (AP, AP per threshold, TP count, recall,
precision, median matched chamfer, GT/pred counts, points, density); tiles
are sampled from the top/middle/bottom 20% bands. The sample is **seeded**
(`hashlib`, not `hash()` — the built-in is salted per process and would
reshuffle on every restart), so it survives a reload but re-rolls via
Shuffle, and specific tiles can be pinned per band. `better='low'` inverts
the banding for chamfer distance.

**Polyline count error = `n_pred_kept − n_gt`**, signed (0 = exactly right,
−1 = one too few, +1 = one too many), sorted by *distance from zero* so
"Best" means closest to correct in either direction (`better='zero'` in
`RMETRICS`, handled explicitly in `rank_rows`). Distinct from `n_tp`, which
is chamfer-**matched** instances — the two answer different questions and
both are kept. `n_tp`'s label was changed from "correctly predicted
polylines" to "matched polylines", since the old wording read as a count.

**The count only exists relative to a score threshold**, and this is not a
detail: the detection head emits a fixed `num_vec` (50) predictions for
*every* tile, so the raw count is a constant and `n_pred − n_gt` would be
`50 − n_gt`. Hence the threshold is now an **eval** parameter (part of
`eval_key`, so it invalidates the cache), not just a drawing one. AP, TP and
recall deliberately ignore it — AP integrates over every detection at every
operating point, and pre-thresholding would silently redefine it and stop it
matching the training log.

`count_health()` warns when the count metrics cannot mean anything: when the
threshold keeps all (or none) of the predictions on >95% of tiles, or when
the run's whole score range is narrower than 0.05. **The webviewer_demo
checkpoint hits both** — every confidence lies in 0.1632–0.1743, a spread of
0.011, so *no* threshold separates confident from unconfident predictions.
That is what an undertrained checkpoint looks like (confidences collapse to
one value); on a converged model the metric becomes meaningful. Reported
rather than worked around, since substituting some other definition of "how
many" would make the number look informative when it isn't.

**`count_error` vs `n_gt` is partly circular** — `count_error` contains
`−n_gt`, so it correlates negatively by construction (a constant predicted
count alone gives ρ = −1). Measured on the demo run: the circular version
reads **ρ = −0.84**, the honest predicted-vs-GT-count chart (`plot_counts`,
with the y=x diagonal, neither axis defined in terms of the other) reads
**ρ = −0.10**. The pairing carries an inline warning pointing at the
non-circular chart; don't quote the −0.84.

**Tile elevation (`z_median` / `z_mean`) is recorded in WORLD z, and that is
load-bearing.** `features` is stored relative to `offset`, and **`offset` is
the point cloud's centroid** — verified directly: `mean(features)` is 0 on
all three axes for every tile checked, to float32 rounding. So the
centroid-relative *mean* z is identically ~0 (measured spread across the
4103-tile train split: −0.07 to 0.17 m), and a chart against it would be a
chart against rounding noise. Adding `offset[2]` back gives two real
quantities: `z_mean` = elevation of the tile's centroid (≈ `offset[2]`),
`z_median` = elevation of its road surface. Sanity check that the conversion
is right: `z_median` clusters at exactly 0, which is CARLA's ground plane.

Their **difference is the vertical skew** — how far the road sits below the
centre of mass above it — and that difference is the only part of this the
model can see, since its input is centroid-relative. It is in `results.csv`
as `z_skew`; a trend against absolute `z_median` is about *where* a tile is,
not its height as such.

**`CACHE_VERSION` (in the viewer) must be bumped whenever `scan_tile()`
records a new field.** Without it a warm cache from an older build loads
happily, every tile looks already-scanned, and the charts needing the new
field render empty — the same silent-staleness class as the `carla_map_gt.
json` trap. A mismatch drops the cache and re-scans (~70 s for 4,362 tiles;
adding the two z stats cost no measurable time). Currently v3: v1 original,
v2 added z_median/z_mean, v3 moved them to world z. Relatedly, `/res.png`
now returns an explanatory placeholder image rather than a 404 when a chart
has nothing to draw, since the page has already emitted the `<img>` and a
404 renders as a broken-image icon that reads like a bug.

Scatters use **marker area ∝ coincident tile count**: several axis pairs are
small integers (GT count vs TP count is a lattice) and a plain scatter drew
259 tiles as ~40 visible dots, hiding its own distribution. Sizing by the
exact count beats jitter, which would invent positions on an axis whose
integer values are the whole point. Tiles the stats tab flags as suspect are
drawn in `CRITICAL`, so "are the bad tiles just the broken ones?" is
answerable by looking. Correlations are reported as **both** Spearman and
Pearson (`rankdata` is hand-rolled, average-tie — scipy isn't on the host,
and ties are heavy in these small-integer columns).

First real finding: at the 1-epoch checkpoint **nothing predicts quality** —
density, raw points, effective points and GT count all sit at |ρ| ≤ 0.05.
That is a statement about a barely-trained model, not about the data; re-run
it on a converged checkpoint before drawing any conclusion.

### Curve-vs-line splits, and what per-tile AP = 1.000 means (2026-08-17)

**The arc/straight tag lives only in `reference_lines/*.json`.** Each polyline
carries `type` (`'arc'` / `'straight'`, plus a redundant `is_arc`), and the
converter does **not** copy it into the pkl — `annotation['divider']` is bare
coordinate arrays. So `shape_counts()` reads it back per tile, applying the
converter's own filter (file order, drop `<2` points) so the counts add up to
the GT instances actually scored. Verified across the 259-tile test split: **0
tiles** where `n_arc + n_straight` disagreed with the pkl's instance count, and
624 arc + 586 straight = the known 1210 instances. A tile whose counts don't
reconcile is dropped from these charts and reported, rather than plotted
against a GT it doesn't describe — the case that produces it is a pkl
converted with a `--classes` subset.

Three box plots (`shape__arc` / `shape__straight` / `shape__all`), one box per
tile-count on the x axis, quality on the y — the one place in this file where
boxes are **vertical**, because the x axis is the count and that was the ask.

**They partition the tiles; they are not three views of all of them** (changed
2026-08-18). Every tile is curves-only, straights-only, or mixed, and appears
in exactly one chart — 45 / 65 / 149 on the test split, summing to 259. The
overlapping version could not answer "does the model struggle with curves",
because the curve chart's tiles were mostly full of straight lines too. The
comparison that is valid is curves-only against straights-only **at equal
counts**, with the mixed chart as the population in between.

**Still a per-TILE score, not per-instance.** A tile's AP covers all its
predictions; the partition removes the confound of the other kind being
present, it does not attribute a match to the geometry of the line it
matched.

**A column of one tile drew literally nothing** (fixed 2026-08-18) — measured,
not inferred: rendering an n=1 box and diffing against the background gives
**0** differing pixels. A box plot of a single value is a zero-height patch
whose only mark is a median line painted in `BG`, so the column read as
missing data rather than as a real measurement. Columns below
`SHAPE_MIN_TILES` (5) are now drawn as the individual tile values, as
**hollow** markers — hollow because that is the honest shape of the claim:
those are observations, not a distribution, and a filled mark would read as a
box seen edge-on. Points in a thin column are spread across it so equal values
don't hide behind each other; a single tile stays dead centre.

The **fliers had the same problem** and needed the same answer: at the stats
tab's 2.5 pt / 45%-alpha muted dot, the two outliers in the curves-only chart
(AP 0.88 and 0.71 — the best tiles in their columns) took a 3× zoom to find.
They now use the identical hollow-circle style, so one rule holds across the
whole chart: **a hollow circle is one tile**, whether it is an outlier inside
a box or a whole column. The stats tab keeps its tiny muted dot on purpose —
there a box summarises hundreds of tiles and the fliers are a cloud, whereas
these columns hold 7–27.

**Per-tile AP = 1.000 is arithmetic, not a bug** — the most-asked question of
this tab, now answered on the page by `ap_health()`. For a tile with G GT
lines, AP (mmdet's `'area'` mode over that tile's own detections) is 1.000
exactly when the **G highest-scoring of the head's 50 detections are all TPs at
every threshold**, 0.5 m included. The ~45 lower-scoring detections behind them
are all false positives and cost nothing, because max-interpolated precision
has already reached 1. With **G = 1** the score is quantised to `1/k` for k the
rank of the first match — 1.000, 0.500, 0.333, … or 0 — so a single-line tile
*cannot* express "mostly right". Measured on the 25m-V2 run: both AP-1.000
tiles have G = 1 and a matched chamfer of 0.29-0.30 m, i.e. genuinely inside
the strictest threshold. Tiles with G ≤ 2 are chipped `coarse AP` in the
gallery and each tile's caption now prints the per-threshold breakdown, which
is what makes the 1.000 legible. The global AP in the header is unaffected: its
ranking interleaves every tile, so a tile can't buy a perfect score off a small
GT set.

### 3D tab, and per-tile metrics on the browse captions (2026-08-18)

`?tab=view3d` opens one tile in an interactive **Open3D** window: tile picker,
colour scheme (RGB / lane label / height / intensity / uniform), frame,
polyline-class filter, GT/prediction toggles, a top-down preview of the same
selection, and the tile's full metric table.

**Open3D is imported ONLY in the child process that opens the window**, never
in the server, and this is a hard requirement rather than tidiness:
`draw_geometries()` runs its own event loop and blocks until the window
closes, and GUI toolkits demand the main thread — which a threaded Flask
worker is not. No lock can fix it, unlike the matplotlib case (`RENDER_LOCK`).
So a launch re-execs the file with `--o3d-spec <json>` (an `argparse.SUPPRESS`
flag) and returns immediately; the window is a separate process with its own
lifetime, several can be open at once, and closing one doesn't touch the
server. `launch_o3d()` waits 1.2 s and checks `poll()`, because a child that
dies instantly (no open3d, no display) must not report success — it surfaces
the child's own last line, e.g. `ModuleNotFoundError: No module named
'open3d'`, plus the install hint. **Install with plain `pip install open3d`,
never `--no-deps`**: 0.19 eagerly imports its ML chain and needs `plotly`,
`dash` and `addict`.

**Over an SSH tunnel from a cluster node, the 3D window cannot open**, and
this is now refused up front rather than failing silently. The window opens on
the machine running the *server*; `ssh -L` forwards HTTP only. Measured with
`DISPLAY` unset: Open3D prints a GLFW warning, draws nothing, and the process
**exits 0** — indistinguishable from success to any caller watching exit codes
or timing. So `launch_o3d()` pre-checks `DISPLAY`/`WAYLAND_DISPLAY` (0.02 s to
refuse, with an explanation), and success is confirmed by the child writing to
a **status file** after `create_window()` returns true — never inferred from
elapsed time, which cannot work in either direction here. `o3d_show()` uses
`Visualizer` explicitly rather than `draw_geometries()` precisely because
`create_window()` returns a bool and `draw_geometries()` does not. A real
launch reports success in ~2.6 s. `ssh -X` is not a reliable workaround either
(Open3D needs OpenGL 3.3+; indirect GLX usually cannot provide it) — run the
viewer on the desktop, or use VNC on the node. Every other tab works fine over
a tunnel.

`build_o3d_scene()` is pure numpy and imports nothing, which is what makes the
geometry testable on a host without open3d — only `o3d_show()` touches the
library. Verified against real Open3D 0.19 (installed to a throwaway
`--target` dir, not the user's env): 116,537-point cloud with colours, one
`LineSet` per polyline, a coordinate frame at the origin, bbox exactly ±12.5
in the `tile_center` frame, and an actual X window confirmed open via
`xdotool` while the server stayed responsive.

**Polylines are tube meshes, not LineSets.** Open3D's legacy visualiser
ignores `RenderOption.line_width` completely — measured by rendering one line
offscreen at width 1, 5 and 15 and counting pixels, which gave an identical
1-pixel-tall, 286-pixel line every time (OpenGL core profiles do not honour
`glLineWidth > 1`). A LineSet is therefore a hairline at any setting. `_tube()`
builds the mesh in numpy, carrying the rings along the curve by parallel
transport so it does not twist at corners; thickness is a UI control in
metres (default 0.30 m diameter). Verified by the same offscreen measurement:
4 → 10 → 26 pixels tall for radius 0.05 → 0.15 → 0.40.

**Everything is lifted clear of the road** (`line_lift`, a UI control,
default 0.5 m). A tube centred *on* the ground plane has half its volume
inside the densest part of the cloud, and the road returns draw over it from
most angles — the line is present but unreadable. The whole set moves by one
rigid amount, so GT and predictions stay comparable with each other; only
their height above the road is fictional, and the window's info line says so.

**Predictions are drawn on the GT plane** because the results json has no z.
Not a fudge: `reference_lines/*.json` stores 3D points whose z is CARLA's
ground plane (world z == 0), so in either frame GT *and* predictions land at
`-origin[2]` — the true height of the road surface relative to the cloud.
Measured: point-cloud z spans −7.69..14.84 and the polylines sit at −7.688.
`load_polylines(..., ndim=3)` keeps that z; every 2D caller still throws it
away.

`tile_metrics()` is the single source for per-tile numbers, in three tiers
(manifest / deep scan / selected run), and feeds **both** the 3D tab's table
and the browse tab's captions, so the two cannot drift apart. Browse captions
now carry all three tiers plus the suspect chips and a link into the 3D tab.
The eval tier reuses `evaluate_run()`'s cache, so a browse page pays the eval
once, not once per tile drawn.

**Watch the decorator when inserting a function before `index()`.**
`@app.route('/')` sat directly above `def index():`, so inserting
`view3d_page()` at that anchor silently moved the route onto the new function
and *every* tab rendered the 3D page. Caught only by asserting on the returned
`<title>` per tab — a check worth keeping in any page test.

### The viewer picks its frame off the GT file (2026-08-18)

A GeMap checkpoint drew its predictions centred while the viewer drew the
tile displaced — the two frames above, mixed. The viewer now has a **Tile
frame** picker (`--frame`, and a select on both the browse and results tabs)
with `auto` / `offset` / `tile_center`.

**`auto` detects the frame structurally**, by comparing each pkl sample's
`annotation_origin` against its `tile_center` — not by reading a declared
key, because MapTRv2 writes `gt_frame`, GeMap writes `annotation_frame` (with
the opposite shift sign), and older pkls declare neither. The origins are
always present and cannot disagree with the annotations they were subtracted
from. `carla_map_gt.json` carries no origin, so it returns None and the
caller falls back; that file's frame is the config's, not something recorded
in it.

Under `tile_center` the render shifts the POINTS by `offset − origin` (the
stored `features` are always in the offset frame) and feeds the same origin
to `load_polylines`, so points, GT and predictions land together. Verified:
the viewer's drawn GT is **bit-identical to the converter pkl's** across all
1210 instances (max |diff| 0.0), and `auto` renders byte-identical PNGs to an
explicit `tile_center` while `offset` differs.

Resolution order for `auto` with no GT selected (the browse tab): `--gt-json`
first, then `discover_gt()`. Explicit beats discovery, because `discover_gt()`
returns `data/carla`'s first pkl, which on a machine holding both frames is
just whichever sorts first. With no `--gt-json` and an offset-frame
`data/carla`, `auto` resolves to `offset` — old behaviour, byte-identical.

The results tab warns (`frame_health()`) when the frame being **drawn**
differs from the frame its GT was **built** in. What it cannot check is the
frame the predictions were trained in: nothing in a results json records
that, which is exactly why the symptom is confusing. The `gt` path also rides
along on every `/tile.png` URL so the image route can resolve `auto` the same
way the page did — validated against the discovered GT list first, since it
reaches an `open()`.

### Example tiles are centred on the origin, with a metric grid (2026-08-17)

Three changes to `_render_tile`, all about reading distance off a tile:

- **The view is centred on (0, 0), not on `tile_center - offset`.** That origin
  is the one frame the points, the GT and the predictions all share (and the
  training `point_cloud_range` is a box around it), so an asymmetric axis —
  the "−13 to +17" this used to produce — reads as a misalignment that isn't
  there. The 2026-08-10 cropping bug is avoided by *growing* the view radius to
  `tile_radius + |tile_center − offset|` instead of shifting the centre, so the
  whole tile still fits. The density heat map's 1 m² bins stay aligned to the
  tile, so its counts are untouched.
- **A 5 m major / 1 m minor grid, drawn ABOVE the point cloud** (`GRID_ZORDER`
  = 3, between the points at 2 and the polylines at 5). Under the points it is
  invisible on a 90k-point tile, which is exactly when you want a ruler. Done
  by hand with `vlines`/`hlines` because `ax.grid()`'s `zorder` isn't honoured
  and `set_axisbelow` only offers "all the way under". Suppressed in density
  mode, where a second 1 m rule phased off the origin instead of the tile edge
  reads as a rendering fault. `tick_steps()` steps the spacing up on larger
  exports so the 60 m grid tiles don't become 100 lines of haze.
- **Predictions are named `PRED_ZORDER` (7) over `GT_ZORDER` (5)** rather than
  bare numbers, so "pred always on top" survives future edits.

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

## The polyline geometry losses are tile-size parameterised (2026-08-10)

`maptrv2_carla_r50_24ep_lidar_HM.py` (the `PolylineGeomLoss`/`PolylineGeomCost`
config) no longer hardcodes its geometry constants. It declares a `tile_size`,
derived from its own `point_cloud_range`, and every tile-dependent knob is
computed from it. This matters because the tile size is genuinely not fixed —
the 25 m tiles and the 60 m `grid_tiles` export both exist.

**Why anything depends on it at all**: `loss_pts`/`pts_cost` see NORMALISED
coordinates (`normalize_2d_pts` divides by the pc_range extent), so a physical
error of d metres arrives as `d / tile_size`. Nothing warns you about this;
the loss just quietly means something different on a different tile.

**The part that is easy to get wrong** — the primitives do not share one
homogeneity degree, so a single common factor does not carry them across:

| quantity | degree | scaling applied |
|---|---|---|
| `exact_emd_loss`, all `pgf_*` terms | 1 | `loss_weight *= tile_scale` |
| CW-OT position (`squared=True`) | **2** | `cwot_w_pos *= tile_scale**(2-1)` |
| CW-OT direction (unit tangents) | **0** | `cwot_w_dir /= tile_scale` |
| `cwot_eps` | units of the **cost**, so 2 when squared | `/= tile_scale**degree` |

`cwot_eps` is the subtlest: it regularises against the transport cost
(`log_K = -cost/eps`), so scaling it as if it were a length leaves `cost/eps`
— the only thing Sinkhorn sees — drifting with tile size. Note also that
`PolylineGeomLoss` and `PolylineGeomCost` ship *different* `cwot_squared`
defaults (True vs False), hence different degrees; the config now states the
flag explicitly on both sides rather than relying on those defaults.

**Verified**: with the derivations in place, a fixed physical error costs
*exactly* the same at 25/60/100 m tiles in all three modes (0.00% spread,
driven through the real config file with only its `point_cloud_range` line
rewritten). With the old hardcoded constants the same error came out 4x
cheaper at 100 m. At the 25 m reference tile every derived value reproduces
its old constant bit-for-bit (2.0 / 0.05 / 1.0 / 0.5 / 1.0), so this changes
nothing for the current dataset — it only starts acting on a resized one.

**Separately, `pc_range` is now an optional arg on both classes** (default
`None` = off). It undoes the anisotropy that per-axis normalisation
introduces: `normalize_2d_pts` divides x and y by their *own* extents, so on a
NON-SQUARE pc_range every `torch.cdist` here measures in a stretched space and
over-weights error along the shorter axis. No config scalar can express that.
It is an exact no-op for CARLA's square tiles (verified bit-identical), and
only bites if a future export is rectangular. If you set it, set it on **both**
the loss and the cost — otherwise matching and supervision disagree about
what "close" means.

**Caveat**: the config restates `point_cloud_range` locally (mmcv evaluates
each config file in isolation, same reason `fixed_ptsnum_per_line` is
repeated). It must stay equal to the base's — everything else derived from it
(`bev_h_`/`bev_w_`, the coder, the assigner's pc_range, the LiDAR geometry) is
bound in the base's namespace and will NOT follow a change made in the HM
file. Actually resizing tiles means editing the base config, and re-measuring
`sparse_shape`/`lidar_bev_proj.in_channels` per gotcha #4.

## The GT order axis is 89.5% padding — and it broke EMD matching (2026-08-12)

Chasing low GPU utilisation on the HM configs turned up one structural fact
with two consequences, one a large speedup and one a real correctness bug.

**The fact.** `gt_shift_pts_pattern='v2'` (what all the CARLA configs use)
builds the GT order axis as `fixed_num - 1 = 19` slots. For a *closed*
polygon it fills them with cyclic shifts, but for an **open** polyline — which
every CARLA divider is — it emits exactly **2** real orders, forward and
flipped, and pads the other **17 with `padding_value = -10000`**
(`av2_offlinemap_dataset.py:290`, `shift_fixed_num_sampled_points_v2`). So
89.5% of the (gt, order) pairs the matching cost evaluates are padding.
Measured directly on real samples, not inferred.

**Consequence 1 — the cost was ~19x redundant.** `PolylineGeomCost` costed
every one of those pairs. Two exact redundancies:
* the 17 padding slices are identical to each other, so 1 suffices; and
* **EMD cannot distinguish the 2 real orders.** `exact_emd_loss` solves a
  balanced assignment over point *sets*, and forward/flipped are permutations
  of each other — verified bit-identical (spread 0.000e+00) on real GT.

`PolylineGeomCost(dedup_gt_slices=True)` (the default now) costs each
*distinct* slice once and scatters back: **bit-identical output**, 9.6x on the
cost alone, and **10-17 → 0.6-0.9 s/iter** end to end. `cost_mode='emd'` is
now about as cheap as `'pgf'`, which removes the main reason
`..._HM_pgfcost.py` existed. Note the permutation half applies to **EMD only**
— CW-OT reads tangents and PGF takes finite differences, so both see the
sequence and only the exact-duplicate (padding) collapse is valid there.

**Consequence 2 — `normalize_median` was normalising against garbage.** It
divided by the median of the *whole* cost matrix, which is 89.5% distances to
the `-10000` sentinel. The median therefore landed in the padding population
(measured: **565.7**), scaling real geometry costs down to **~0.0007** against
a `cls_cost` of weight 1.0. In `cost_mode='emd'`/`'cwot'` the assigner was
effectively matching on **classification score alone**, with geometry
contributing ~1/1400th — while paying 10-17 s/iter to compute it. The median
now ignores padding columns and real costs sit at ~1.0.

This changes matching behaviour, so **HM checkpoints trained before this are
not comparable** — geometry genuinely participates in assignment now.
`..._HM_pgfcost.py` was never affected: `pgf` is not median-normalised.

Padding is detected structurally — "every point in the slice is identical" —
rather than by comparing to -10000, because the cost sees *normalised*
coordinates and would otherwise need `pc_range` to reconstruct the sentinel.

**Still open**: the loss side is untouched (it runs on matched pairs, no order
axis, ~50 solves — not worth it). If `cwot` is ever wanted as a cost, it needs
a different fix: batching Sinkhorn over the pair axis on GPU, since its
per-pair Python loop is what makes it unusable, not the order axis.

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
- **`../carla_test` has no train split.** It is 3795 `test` tiles with the
  driving/curb taxonomy (2026-08-20 section) and is the only class-carrying
  dataset here with a real tile count. Both `_2cls` configs name a
  `carla_map_infos_train_30m_tc_2cls.pkl` that does not exist; convert one,
  or repoint `data.train.ann_file`, before any run whose numbers matter.
- **Only the local 259-tile test subset exists.** The full dataset is 4103
  tiles / 5 towns with no `train`/`val` split yet — one is needed (by town
  or stride) before a real run. Converting it needs no code changes, just
  `--data-root` and new `ann_file` names.
- **A new/larger dataset should be converted first and its
  `carla_map_infos_<split>_dropped.json` inspected** before a long run — a
  large drop count means the range is wrong, not that the data is bad.
- **The default taxonomy is still divider-only**, collapsing *all* CARLA
  lane types into one `divider` class — deliberate, to maximise GT density
  on the tiny local set. A real multi-class run now has two paths: filter to
  one class with `--classes driving_centerline`, or train a named taxonomy
  with `--map-classes` (see the 2026-08-20 section and the `_2cls` configs).
  Neither works on the local 25 m export, which carries no `class_lookup`.
  The older `--lane-types` never matched anything — see the 2026-08-10
  section.
- **`sparse_shape`/`lidar_bev_proj.in_channels` were measured empirically**
  for the *current* range/resolution (`[251,251,71]` and `384`). If either
  changes, re-measure with a dummy `extract_lidar_feat()` call rather than
  computing by hand — gotcha #4 makes axis order easy to get wrong.
- **No real accuracy signal yet.** A longer run with a real train/val split
  is needed before any conclusion about model quality.
- Detailed design/reasoning from the original work session:
  `/gel/usr/johil9/.claude/plans/when-running-the-dockerfile-mellow-russell.md`
