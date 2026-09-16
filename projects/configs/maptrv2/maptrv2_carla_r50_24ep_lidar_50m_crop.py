"""50 m CARLA tiles -> rotated 30 m crops, two classes (driving / curb).

A thin overlay on `maptrv2_carla_r50_24ep_lidar_30m.py`. The MODEL is the 30 m
one unchanged -- same `point_cloud_range` (+-15), `sparse_shape`, `bev_h_`/
`bev_w_`, coder and assigner ranges, colour-free `SparseEncoder(in_channels=3)`
-- because every crop the network sees IS a 30 m tile. What changes is where
those tiles come from:

* **The data are 50 m tiles on disk** (`tile_radius` 25 m), read straight from
  the export's `<split>/manifest.json` by `CustomCarla50mCropDataset` -- there
  is no converter pkl. Each `__getitem__` draws a uniform yaw about the tile
  centre plus a centre shift of up to 3.79 m (25 - 15*sqrt2, the slack a
  rotated 30 m square leaves inside a 50 m one, so every crop is fully
  populated), crops a 30 m square, and adds sigma = 0.02 m Gaussian noise to
  the points. GT polylines get the SAME pose and are clipped to the same box
  with the tiler's own `clip_polyline_to_box`.
* **Val/test crops are frozen by `crop_seed`**, not centred: `_format_gt()`
  caches the eval GT to `map_ann_file` once, so the crop each tile is scored
  on must be identical on every run. A *centred* crop would leave 24.4% of the
  test tiles with no GT at all (the manifest was filtered for a *reachable*
  non-empty crop, found by aiming); seeded aiming drops that to 0.4%. No yaw
  at eval, so map quality is not conflated with rotation robustness.
* **The crop frame replaces the tile_center frame.** The crop centre is the
  origin by construction, so `LoadCarla50mCrop` does the reframing itself and
  there is no `lidar_recenter_shift` / `recenter=True` in these pipelines.
  Points' z is re-zeroed per crop on its own 2nd-percentile ground.
* **Two classes, `driving` and `curb`**, in that (label) order -- the export's
  own names, exactly as `maptrv2_carla_r50_24ep_lidar_30m_2cls.py` uses them
  (labels come from `map_classes` order via `VectorizedCarlaLocalMap`).
  Curbs are ~30% of the polylines and appear in 24% of train / 34% of test
  tiles. Unlike the pkl-based 2cls config this one raises
  `aux_seg.seg_classes` to 2 as well, so the auxiliary BEV mask is per class;
  that is what the benchmark ran with and it is restated on BOTH sides
  (model `aux_seg` and `data.train.aux_seg`) as gotcha #9 requires.
* **Colour-free points**, per the project convention: `LoadCarla50mCrop`
  defaults to `load_dim=3, use_dim=3`, matching the inherited
  `in_channels=3`. The original benchmark (branch
  `worktree-carla50m-emdv2-tuned`) ran with the 4-channel strength input;
  its checkpoints are not loadable under this config, and its numbers should
  be re-measured before being quoted against runs from here.
* `samples_per_gpu=16`: every reported 50 m run used 16 on an 80 GB H100
  (22, the value the 30 m benchmark used, OOMs on 50 m crops at 68 GB).

Loss is the inherited `PtsL1Loss` -- this is the ORIG side of the
50 m benchmark; `..._50m_crop_EMDV2.py` changes exactly that one line.

Data layout expected under `data_root` (the export root, NOT `data/carla/`):

    <data_root>/<split>/manifest.json               tiles: [{name, center}]
    <data_root>/<split>/blocks/<name>.npz           points (world), features,
                                                    tile_center, tile_radius
    <data_root>/<split>/reference_lines/<name>_reference_lines.json

Set `data_root` to wherever the 50 m export is mounted (the benchmark used
`/data_carla50` on the cluster); the default below assumes a symlink or
bind-mount at `data/carla50m/`. The eval GT json is written next to the
other CARLA GT files and must not be shared with any other config.
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_30m.py']

# ---- data source -----------------------------------------------------------
dataset_type = 'CustomCarla50mCropDataset'
data_root = 'data/carla50m/'
ann_file_train = data_root + 'train/manifest.json'
ann_file_val = data_root + 'test/manifest.json'
ann_file_test = data_root + 'test/manifest.json'
# Own GT json: _format_gt() writes it once and never regenerates, and the
# frozen eval crops below are what it records.
map_ann_file = 'data/carla/carla_map_gt_50m_crop_2cls.json'

# ---- taxonomy --------------------------------------------------------------
map_classes = ['driving', 'curb']
num_map_classes = len(map_classes)

# Per-class auxiliary BEV mask (the 30 m base keeps seg_classes=1). The head
# builds its seg conv from aux_seg['seg_classes'] and the dataset rasterises
# one channel per class when it is >1 (av2_offlinemap_dataset.py), so the two
# must agree -- same dict on both sides below.
aux_seg_cfg = dict(
    use_aux_seg=True,
    bev_seg=True,
    pv_seg=False,
    seg_classes=num_map_classes,
    feat_down_sample=32,
    pv_thickness=1,
)

# ---- crop geometry ---------------------------------------------------------
# Must equal the base's lidar_point_cloud_range / lidar_voxel_size (restated
# because mmcv evaluates each config file in isolation and the pipelines
# below need them by name). crop_size is the base's tile size: +-15 m.
crop_size = 30.0
lidar_point_cloud_range = [-15.0, -15.0, -72.0, 15.0, 15.0, 96.0]
lidar_voxel_size = [0.1, 0.1, 0.4]

crop_common = dict(
    crop_size=crop_size,
    require_gt=True,
    gt_tries=32,
    min_polyline_len=2.0,
    # export class names, positionally paired with map_classes above
    gt_classes=('driving', 'curb'),
    reference_dir='reference_lines',
)
crop_train = dict(split='train', rotate=True, translate=True, noise_std=0.02,
                  **crop_common)
# Frozen by seed, no yaw, no noise -- see the docstring.
crop_eval = dict(split='test', rotate=False, translate=True, noise_std=0.0,
                 crop_seed=0, **crop_common)

# ---- pipelines -------------------------------------------------------------
# LoadCarla50mCrop replaces LoadCarlaPointsFromFile: it reads the world-frame
# points, applies the pose drawn by the dataset (so points and GT cannot
# decouple) and crops. load_dim/use_dim=3 is the colour-free convention and
# must match SparseEncoder.in_channels=3 in the base.
train_pipeline = [
    dict(type='LoadCarla50mCrop', coord_type='LIDAR', load_dim=3, use_dim=3),
    dict(
        type='GridSamplePoints',
        grid_size=lidar_voxel_size,
        point_cloud_range=lidar_point_cloud_range),
    dict(
        type='DefaultFormatBundle3D',
        with_gt=False,
        with_label=False,
        class_names=map_classes),
    dict(type='CustomCollect3D', keys=['points'])
]
test_pipeline = [
    dict(type='LoadCarla50mCrop', coord_type='LIDAR', load_dim=3, use_dim=3),
    dict(
        type='GridSamplePoints',
        grid_size=lidar_voxel_size,
        point_cloud_range=lidar_point_cloud_range),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1, 1),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                with_gt=False,
                with_label=False,
                class_names=map_classes),
            dict(type='CustomCollect3D', keys=['points'])
        ])
]

model = dict(
    pts_bbox_head=dict(
        num_classes=num_map_classes,
        aux_seg=aux_seg_cfg,
        # num_vec_one2one stays at 50, so the one2one branch flattens to
        # 50 * 2 = 100 and the inherited max_num=50 is within topk's range
        # (gotcha #15).
        bbox_coder=dict(num_classes=num_map_classes)))

data = dict(
    samples_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        map_classes=map_classes,
        aux_seg=aux_seg_cfg,
        **crop_train),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_val,
        map_ann_file=map_ann_file,
        pipeline=test_pipeline,
        map_classes=map_classes,
        **crop_eval),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_test,
        map_ann_file=map_ann_file,
        pipeline=test_pipeline,
        map_classes=map_classes,
        **crop_eval),
)

# Restated purely to re-point it at THIS test_pipeline; otherwise the eval
# hook inherits the base's copy and loads pkl-style points into a crop
# dataset (the same trap the tile-centre / no-colour overlays documented).
evaluation = dict(interval=2, pipeline=test_pipeline, metric='chamfer',
                  save_best='CarlaMap_chamfer/mAP', rule='greater')
