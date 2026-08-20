"""Colour-free variant of the 30m tile-centred HM (polyline geometry) config.

Drops the point cloud's colour channel: the model sees pure geometry.

--- What colour normally is here ---

The CARLA `.npz` blocks store `features` as `(N, 6)` -- xyz plus RGB in
[0, 1] -- and `LoadCarlaPointsFromFile` collapses that RGB to ONE scalar
"strength" channel with ITU-R BT.709 luma weights
(`rgb @ [0.2126, 0.7152, 0.0722]`, inherited from the Pointcept
`CarlaSegDataset` this was ported from). Each point is therefore
`[x, y, z, strength]`, `GridSamplePoints` keeps whole points, `voxelize()`
mean-reduces them per voxel, and `SparseEncoder(in_channels=4)` consumes all
four. So colour IS used by default -- as brightness, with hue and saturation
already discarded.

This config removes even that. It is a two-line change that must be made on
BOTH sides at once:

* `use_dim=3` on the loader keeps only `[x, y, z]` (the loader always builds
  the strength column first and then selects, so this drops it after the
  fact -- `load_dim` stays 4);
* `in_channels=3` on the `SparseEncoder`, or the very first sparse conv gets
  a 3-channel input against a 4-channel weight and dies at the first
  iteration.

Nothing else moves. `sparse_shape` is a function of the range and voxel size,
not of the feature width, and `lidar_bev_proj.in_channels` (3200) is
`output_channels` x the z-extent after the encoder's downsampling -- also
independent of the input width. Both were re-verified against this exact
config with a dummy `extract_lidar_feat()` call rather than reasoned about,
per gotcha #4.

--- What this is for ---

Colour in CARLA is a rendering property, not a LiDAR return: a real sensor
gives intensity, not RGB, so a model that leans on it may not transfer.
Measured on `../carla_test`, the strength channel does carry variation (~940
distinct values, std 0.205), but on the 25m export -- the only one with
per-point labels -- mean strength by class is 0.592 / 0.622 / 0.600 / 0.592
against a within-class std of ~0.20, i.e. it barely separates road from
non-road on its own. Whether the network extracts anything from it in
spatial context is exactly what running this config against its parent
answers.

--- Notes ---

The annotation pkl is UNCHANGED and shared with the parent: colour is
dropped at load time, so the GT, the tiles and the frame are all identical.
`map_ann_file` is shared for the same reason -- `_format_gt()` writes GT
polylines, which do not depend on the input features, so the two configs
would produce byte-identical files.

`evaluation` is restated below purely to re-point it at the no-colour
`test_pipeline`. mmcv evaluated the parent's `evaluation` dict against the
parent's list object, so without this line the eval would load 4-channel
points into a 3-channel model -- a crash at the first validation, after a
full epoch has trained. Same trap the tile-centre overlays document.

Checkpoints are NOT comparable with the parent's: the first conv layer has a
different input width.
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_30m_HM_tilecenter.py']

# mmcv evaluates each config file in isolation, so the values the pipelines
# need are restated. They must stay equal to the parent chain's -- everything
# derived from them (bev_h_/bev_w_, sparse_shape, the coder and assigner
# ranges, the dataset's pc_range) is bound elsewhere and will NOT follow a
# change made here.
lidar_point_cloud_range = [-15.0, -15.0, -72.0, 15.0, 15.0, 96.0]
lidar_voxel_size = [0.1, 0.1, 0.4]
map_classes = ['divider']

# The one model change. `voxelize` needs no edit -- it carries no channel
# count -- and the mean-reduce in maptrv2.voxelize() just averages 3 columns
# instead of 4.
model = dict(lidar_encoder=dict(backbone=dict(in_channels=3)))

# Same steps as the parent, with use_dim=3 on the loader. mmcv replaces
# list-valued keys wholesale rather than merging them, so the pipelines are
# restated in full -- including the actor-augmentation wiring below, which
# would otherwise be lost from this branch of the config tree.
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
        # [nocolour] xyz only -- drops the RGB-derived strength channel.
        use_dim=3,
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
        use_dim=3,
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

# Only the pipelines are set; mmcv merges dicts recursively, so ann_file,
# map_ann_file, data_root, pc_range, bev_size, lidar_pc_range, aux_seg and
# map_classes all carry through from the parent untouched.
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)

evaluation = dict(interval=2, pipeline=test_pipeline, metric='chamfer',
                  save_best='CarlaMap_chamfer/mAP', rule='greater')

if actor_catalogue is not None:
    train_pipeline.insert(1, actor_paste)
