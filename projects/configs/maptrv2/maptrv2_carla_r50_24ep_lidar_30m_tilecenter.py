"""GeMap-style tile-centred variant of the 30m CARLA LiDAR-only config.

Stands to `maptrv2_carla_r50_24ep_lidar_30m.py` exactly as
`maptrv2_carla_r50_24ep_lidar_tilecenter.py` stands to the 25m config:
identical in every model/schedule respect, differing only in the *frame*
that GT polylines and the LiDAR point cloud are expressed in. Everything
tile-size dependent comes from the 30m base and is not re-derived here --
`sparse_shape`, `bev_h_`/`bev_w_`, the coder and assigner ranges and the
dataset's own `pc_range` all stay bound in that file's namespace.

`offset` is the point cloud's own centroid, and `tile_center` is the tile's
nominal geometric centre; the two differ by 1-2 m on the 25 m export and up
to ~17 m on the 60 m grid one. Nothing here changes that -- it only decides
which of the two everything is measured from, consistently on both sides:

* the pkl must be generated with `--gt-frame tile_center`, which subtracts
  `tile_center` from the polylines and records the per-sample
  `lidar_recenter_shift = offset - tile_center`;
* `LoadCarlaPointsFromFile(recenter=True)` adds that shift to the stored
  points, which are always written in the `offset` frame.

Set one without the other and GT sits metres away from its own point cloud
(chamfer thresholds here are 0.5/1.0/1.5 m), so both live in this file
rather than being toggled independently.

Verified equivalent to GeMap's own converter on the 25 m split: every
annotation array bit-identical, max |difference| 0.0 over 1210 instances.

The tile-centre frame matters MORE at 30 m than at 25 m, not less. The
square `lidar_point_cloud_range` is centred on the origin, so in the
`offset` frame it sits off the actual tile and crops it; on the 25 m export
that cost 46/259 tiles more than 10% of their points, and the displacement
does not shrink as tiles grow.

--- Generating the pkl ---

The 30m export needs its own converter run, into its own directory so the
two frames never share an `ann_file`::

    python tools/maptrv2/custom_carla_map_converter.py \
        --data-root <30m export> --split test \
        --gt-frame tile_center \
        --out-dir data/carla_30m/tile_center/

    python tools/maptrv2/custom_carla_map_converter.py \
        --data-root <30m export> --split train \
        --gt-frame tile_center \
        --out-dir data/carla_30m/tile_center/

`--lidar-point-cloud-range` is deliberately NOT passed: the converter
derives xy from the manifest's own `tile_radius`/`tile_side`, so a real 30m
export yields +-15 by itself. It prints the range it resolved, and it must
equal `lidar_point_cloud_range` below or the dataset warns at load time.
Pass it explicitly only to override a manifest that states the wrong
geometry.

Check the converter's output before training on it: a large
`carla_map_infos_<split>_dropped.json` means the range is wrong, not that
the data is bad, and the printed range-coverage line should now read a
median near 100% -- that improvement is the point of this frame.

`map_ann_file` is likewise separate: `_format_gt()` skips regeneration when
the json already exists, so pointing two frames (or two tile sizes) at one
path would silently score one against the other's GT.

Checkpoints trained under this config are NOT comparable with `offset`-frame
ones, nor with 25m ones -- the input geometry differs in both cases.
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_30m.py']

# mmcv evaluates each config file in isolation, so the base's module-level
# names are not visible here and the values the pipelines need are restated.
# They must stay equal to the base's -- everything else derived from them
# (bev_h_/bev_w_, sparse_shape, the coder and assigner ranges, the dataset's
# pc_range) is bound in the base's namespace and will NOT follow a change
# made here. Resizing tiles means editing the base, per CLAUDE.md.
lidar_point_cloud_range = [-15.0, -15.0, -72.0, 15.0, 15.0, 96.0]
lidar_voxel_size = [0.1, 0.1, 0.4]
map_classes = ['divider']

data_root = 'data/carla_30m/'
# Separate directory, so a `--gt-frame offset` 30m pkl can coexist untouched.
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

# Only the frame-dependent keys are set; mmcv merges dicts recursively, so
# data_root / pc_range / bev_size / lidar_pc_range / aux_seg and the rest of
# the 30m base's data config carry through untouched.
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
