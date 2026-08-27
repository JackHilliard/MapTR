"""30m CARLA LiDAR config with the actor-paste augmentation ENABLED.

Identical to `maptrv2_carla_r50_24ep_lidar_30m.py` in every model / data /
schedule respect; the only change is that `CarlaActorPaste` is inserted into
the TRAIN pipeline (after `LoadCarlaPointsFromFile`, before
`GridSamplePoints`, so pasted and real points get the same voxel
decimation). The test/eval pipelines are untouched on purpose -- evaluation
stays on clean tiles, so a run under this config is directly comparable with
the parent and any difference is attributable to the augmentation alone.

The parent already carries the wiring (`actor_catalogue = None` +
`if actor_catalogue is not None: train_pipeline.insert(...)`), but that `if`
runs in the PARENT's namespace at config-eval time -- setting
`actor_catalogue` here cannot retroactively trigger it, and mmcv replaces
list-valued keys wholesale rather than merging them. Hence this file
restates the train pipeline in full with the paste step in place.

--- What the augmentation needs on disk ---

* `actor_catalogue` below: the `catalogue.json` written by
  `point2vector_data/carla_actor_scan.py`. The path follows the dataset
  convention -- relative to the repo root, `data/carla_assets/` beside
  `data/carla/` -- so it resolves the same way on the cluster and locally.
  Put (or symlink) the WHOLE scan directory there, not just the json: the
  catalogue references its `vehicles/*.npz` / `pedestrians/*.npz` relative
  to itself. On this machine the scan lives at
  `/gel/usr/johil9/Documents/carla/carla_assets/` (213 assets -- 115
  vehicles + 98 pedestrians). To point somewhere else at launch::

      --cfg-options data.train.pipeline.1.catalogue=/path/to/catalogue.json

  (index 1 = the paste step; `--cfg-options actor_catalogue=...` does NOT
  work, the value is baked into the pipeline at config-eval time.)
* Per-tile `placements/*_placements.json` sidecars next to the tiles
  (`<tile dir>/../placements`, from point2vector_data's tile_placements.py).
  **A tile without a sidecar is silently left un-augmented** -- the
  transform returns the sample unchanged rather than crashing. As of
  2026-08-27 the 25m export (`../carla`) has sidecars for both splits but
  the 30m `../carla_test` export has NONE, so against today's 30m data this
  config trains exactly like its parent. Generate the sidecars first, and
  spot-check with `tools/maptrv2/actor_aug_grid.py` (which renders the
  augmentation through these same dataloader stages).
* The `point2vector_data` checkout providing `actor_paste.py`: set
  `$POINT2VECTOR_DATA`, or keep the checkout as a sibling of this repo
  (mount it into the container -- the paste step imports it lazily at the
  first augmented sample, so a missing checkout fails at iteration 1, not
  at config parse).

The sidecars are written in the tile_center frame, which is what the
parent's `recenter=True` loader produces -- the frames agree by
construction under the current convention (the transform double-checks via
the sample's `gt_frame` either way).

GT polylines are deliberately left untouched by the paste: the model is
meant to infer map elements hidden under traffic. Checkpoints ARE
weight-compatible with the parent's (no shape changes anywhere); whether
the runs are comparable is a training-distribution question, which is the
experiment.
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_30m.py']

# mmcv evaluates each config file in isolation, so the values the pipeline
# needs are restated. They must stay equal to the parent's -- everything
# derived from them (bev_h_/bev_w_, sparse_shape, the coder and assigner
# ranges) is bound in the parent's namespace and will NOT follow a change
# made here.
lidar_point_cloud_range = [-15.0, -15.0, -72.0, 15.0, 15.0, 96.0]
lidar_voxel_size = [0.1, 0.1, 0.4]
map_classes = ['divider']

# The scanned actor catalogue, following the dataset path convention
# (relative to the repo root, beside data/carla/) -- see the docstring.
actor_catalogue = 'data/carla_assets/catalogue.json'

train_pipeline = [
    dict(
        type='LoadCarlaPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        # xyz only + tile_center frame, matching the parent exactly.
        use_dim=3,
        z_max=96.0,
        recenter=True),
    # The augmentation. Same knobs as the parent's (dormant) actor_paste
    # dict: up to 5 vehicles and 6 pedestrians per tile, 80% of samples
    # touched, ground shadows carved. placements_dir is left at its default
    # (`<tile dir>/../placements`).
    dict(
        type='CarlaActorPaste',
        catalogue=actor_catalogue,
        n_vehicles=(0, 5),
        n_pedestrians=(0, 6),
        prob=0.8),
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

# Train only -- val/test/evaluation keep the parent's clean pipelines.
data = dict(train=dict(pipeline=train_pipeline))
