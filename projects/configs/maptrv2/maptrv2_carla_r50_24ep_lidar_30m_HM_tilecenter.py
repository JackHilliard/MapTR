"""GeMap-style tile-centred variant of the 30m HM (polyline geometry) config.

The frame overlay for `maptrv2_carla_r50_24ep_lidar_30m_HM.py`, standing to
it exactly as `maptrv2_carla_r50_24ep_lidar_30m_tilecenter.py` stands to the
plain 30m config. It changes the *frame* and nothing else: no loss, no
matching cost, no schedule, so a run under this file is comparable with its
offset-frame sibling on the loss/matching axis and differs only in where the
geometry is measured from.

Everything the HM config derives from its own `point_cloud_range` --
`tile_scale` and every geometry constant of `PolylineGeomLoss` /
`PolylineGeomCost` -- is deliberately NOT restated here. Those are bound in
that file's namespace, they are a function of tile SIZE, and the tile-centre
frame is a rigid translation that leaves the size untouched. Restating
`point_cloud_range` in this file would not re-derive them anyway (mmcv
evaluates each config in isolation), it would only look as though it had.

`offset` is the point cloud's own centroid, `tile_center` the tile's nominal
geometric centre. They differ by 1-2 m on the 25 m export and up to ~17 m on
the 60 m grid one, so both sides have to agree:

* the pkl must be generated with `--gt-frame tile_center`, which subtracts
  `tile_center` from the polylines and records the per-sample
  `lidar_recenter_shift = offset - tile_center`;
* `LoadCarlaPointsFromFile(recenter=True)` adds that shift to the stored
  points, which are always written in the `offset` frame.

Set one without the other and GT sits metres away from its own point cloud
(chamfer thresholds here are 0.5/1.0/1.5 m), so both live in this file.

Note `evaluation` is restated below purely to re-point it at the RECENTRED
`test_pipeline`. The HM config does not override `evaluation`, so without
this it would inherit the 30m base's copy, which loads points in the
`offset` frame -- training on one frame while scoring in the other, silently.

--- Generating the pkl ---

Shared with `maptrv2_carla_r50_24ep_lidar_30m_tilecenter.py`; one pkl serves
both, since the frame is a property of the data, not of the loss::

    python tools/maptrv2/custom_carla_map_converter.py \
        --data-root <30m export> --split test \
        --gt-frame tile_center \
        --out-dir data/carla/ --out-tag 30m_tc

    python tools/maptrv2/custom_carla_map_converter.py \
        --data-root <30m export> --split train \
        --gt-frame tile_center \
        --out-dir data/carla/ --out-tag 30m_tc

Do NOT pass `--lidar-point-cloud-range`: the converter derives xy from the
manifest's own `tile_radius`/`tile_side` and prints what it resolved, which
must equal `lidar_point_cloud_range` below or the dataset warns at load.

`map_ann_file` gets its own path because `_format_gt()` skips regeneration
when the json already exists, so sharing one across frames (or tile sizes)
would silently score one against the other's GT.

Checkpoints trained under this config are NOT comparable with `offset`-frame
ones -- the input geometry differs.
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_30m_HM.py']

# mmcv evaluates each config file in isolation, so the base's module-level
# names are not visible here and the values the pipelines need are restated.
# They must stay equal to the 30m base's -- everything else derived from them
# (bev_h_/bev_w_, sparse_shape, the coder and assigner ranges, the dataset's
# pc_range, and the HM file's tile_scale) is bound elsewhere and will NOT
# follow a change made here.
lidar_point_cloud_range = [-15.0, -15.0, -72.0, 15.0, 15.0, 96.0]
lidar_voxel_size = [0.1, 0.1, 0.4]
map_classes = ['divider']

data_root = 'data/carla/'
# The same files the non-HM tile-centre config uses, map_ann_file included:
# the frame is a property of the data, the loss is not, and this config
# leaves `fixed_ptsnum_per_gt_line` at the base's 20, so _format_gt() would
# write byte-identical GT either way. That mirrors how the offset-frame HM
# configs already share their base's map_ann_file.
#
# (Watch this if the HM line count ever diverges from the base's. The
# resampling is baked into the json, which _format_gt() writes once and
# never regenerates, so two configs with different fixed_ptsnum_per_gt_line
# must NOT share a path. The comment above `data` in
# maptrv2_carla_r50_24ep_lidar_30m_HM.py still says 40 -- it is stale, the
# value there is 20 -- so read the value, not the comment.)
#
# Kept in data/carla/ and separated by the converter's `--out-tag`, not by a
# per-variant subdirectory -- matching the non-HM sibling and the 2cls
# configs. These four lines must stay equal to that sibling's, since the two
# deliberately share one pkl.
ann_file_train = data_root + 'carla_map_infos_train_30m_tc.pkl'
ann_file_val = data_root + 'carla_map_infos_test_30m_tc.pkl'
ann_file_test = data_root + 'carla_map_infos_test_30m_tc.pkl'
map_ann_file = data_root + 'carla_map_gt_30m_tc.json'

# Same steps as the base, with recenter=True on the loader. mmcv replaces
# list-valued keys wholesale rather than merging them, so the pipelines are
# restated in full.
# --- actor augmentation ---------------------------------------------------
# Paste scanned CARLA vehicles/pedestrians into the tile at load time, carving
# the ground shadow each one removes. Set actor_catalogue to the scanned
# catalogue.json to enable (None disables). Runs after LoadCarlaPointsFromFile
# (in its tile-centred frame, which is why this lives in the tile_center
# configs) and before GridSamplePoints, so pasted points get the same voxel
# decimation as real ones. GT polylines are left untouched on purpose: the
# model must infer map elements hidden under traffic.
actor_catalogue = None
actor_paste = dict(
    type='CarlaActorPaste',
    catalogue=actor_catalogue,
    n_vehicles=(0, 5),
    n_pedestrians=(0, 6),
    prob=0.8)

train_pipeline = [
    dict(
        type='LoadCarlaPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        z_max=96.0,
        recenter=True),
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
    dict(
        type='LoadCarlaPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        z_max=96.0,
        recenter=True),
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

# Only the frame-dependent keys are set; mmcv merges dicts recursively, so
# the HM base's samples_per_gpu / workers_per_gpu and its per-split
# fixed_ptsnum_per_line carry through untouched.
data = dict(
    train=dict(ann_file=ann_file_train, pipeline=train_pipeline),
    val=dict(
        ann_file=ann_file_val,
        map_ann_file=map_ann_file,
        pipeline=test_pipeline),
    test=dict(
        ann_file=ann_file_test,
        map_ann_file=map_ann_file,
        pipeline=test_pipeline),
)

evaluation = dict(interval=2, pipeline=test_pipeline, metric='chamfer',
                  save_best='CarlaMap_chamfer/mAP', rule='greater')

if actor_catalogue is not None:
    train_pipeline.insert(1, actor_paste)
