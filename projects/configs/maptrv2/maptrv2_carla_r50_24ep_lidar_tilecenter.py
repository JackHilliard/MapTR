"""GeMap-style tile-centred variant of the CARLA LiDAR-only MapTRv2 config.

Identical to `maptrv2_carla_r50_24ep_lidar.py` in every model/schedule
respect. The only difference is the *frame*: GT polylines and the LiDAR
point cloud are both expressed relative to the tile's nominal geometric
centre (`tile_center`) instead of the block's `offset`, which is what
GeMap's copy of this converter does.

`offset` is the point cloud's own centroid, so the two origins differ by
1-2 m on the 25 m export and up to ~17 m on the 60 m grid one. Nothing here
changes that fact -- it only decides which of the two everything is
measured from, consistently on both sides:

* the pkl must be generated with `--gt-frame tile_center`, which subtracts
  `tile_center` from the polylines and records the per-sample
  `lidar_recenter_shift = offset - tile_center`;
* `LoadCarlaPointsFromFile(recenter=True)` adds that shift to the stored
  points, which are always written in the `offset` frame.

Set one without the other and GT sits metres away from its own point
cloud (chamfer thresholds here are 0.5/1.0/1.5 m), so both live in this
file rather than being toggled independently.

Generate the pkl into its own directory so the two frames never share an
`ann_file`::

    python tools/maptrv2/custom_carla_map_converter.py \\
        --data-root <path> --split test --gt-frame tile_center \\
        --out-dir data/carla/tile_center/

`map_ann_file` is likewise separate: `_format_gt()` skips regeneration when
the json already exists, so pointing both frames at one path would silently
score one against the other's GT.

Checkpoints trained under this config are NOT comparable with `offset`-frame
ones -- the input geometry differs.
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar.py']

# mmcv evaluates each config file in isolation, so the base's module-level
# names are not visible here and the values the pipelines need are restated.
# They must stay equal to the base's -- everything else derived from them
# (bev_h_/bev_w_, sparse_shape, the coder and assigner ranges) is bound in
# the base's namespace and will not follow a change made here. See CLAUDE.md's
# note on the HM config for the same caveat.
lidar_point_cloud_range = [-12.5, -12.5, -72.0, 12.5, 12.5, 96.0]
lidar_voxel_size = [0.1, 0.1, 0.4]
map_classes = ['divider']

data_root = 'data/carla/'
# Separate directory, so a `--gt-frame offset` pkl can coexist untouched.
ann_file_train = data_root + 'tile_center/carla_map_infos_train.pkl'
ann_file_val = data_root + 'tile_center/carla_map_infos_test.pkl'
ann_file_test = data_root + 'tile_center/carla_map_infos_test.pkl'
map_ann_file = data_root + 'tile_center/carla_map_gt.json'

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
