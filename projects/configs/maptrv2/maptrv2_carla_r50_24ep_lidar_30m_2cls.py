"""Two-class (driving / curb) variant of the 30m CARLA LiDAR config.

A thin overlay on `maptrv2_carla_r50_24ep_lidar_30m.py` changing exactly one
axis: the map TAXONOMY. Everything else -- tile size, the tile_center frame,
the colour-free loaders, `sparse_shape`, `bev_h_`/`bev_w_`, the
coder/assigner ranges, the pipelines (actor augmentation included) and the
schedule -- stays bound in the parent and is deliberately not restated.

--- What changes ---

Every CARLA lane type used to collapse into ONE `divider` class, which was a
deliberate choice to maximise GT density on the tiny local subset. This
config instead trains two classes, in this order (the order IS the label
order):

    0 = driving   <- the export's `driving` polylines
                     (`driving_centerline` on the older grid export)
    1 = curb      <- the export's `curb` polylines

That requires four things to agree, and they are all set below:

* the pkl must carry an `annotation` dict with a `driving` and a `curb` key
  -- produced by the converter's `--map-classes` flag (command below);
* `data.train/val/test.map_classes` must list them in that order, because
  `VectorizedCarlaLocalMap` builds its label mapping from that list
  (the AV2 base hardcodes nuScenes names and would map both to -1, i.e.
  silently vectorize to zero instances -- hence the subclass);
* `pts_bbox_head.num_classes` and `bbox_coder.num_classes` must be 2;
* `map_ann_file` must be a NEW path -- `_format_gt()` skips regeneration
  whenever the file exists, so reusing the single-class json would score
  two-class predictions against one-class GT.

Note the classes are NOT a subset relabelling of the old `divider` set: with
`--map-classes` only the two NAMED export classes are kept, and anything else
the export carries is dropped rather than merged. `../carla_test` publishes
exactly these two, so nothing is lost there. To fold road edges into curbs,
say `curb=curb,road_edge` in the converter flag and regenerate; nothing in
this file changes, since the grouping lives in the pkl.

`aux_seg` is left at `seg_classes=1` on purpose. That is the BEV auxiliary
segmentation head, whose channel count is independent of `num_classes`
(maptrv2_head.py builds it from `aux_seg['seg_classes']`), and the dataset
rasterizes every class into the one mask. Raising it to 2 would be a
separate experiment, and it must then be raised on BOTH sides at once
(model `aux_seg` and `data.train.aux_seg` are the same dict here, so it
would be, but the seg head's weights change shape -- not comparable).

--- Generating the pkl ---

The class taxonomy is a property of the DATA, so this pkl is shared with
`maptrv2_carla_r50_24ep_lidar_30m_HM_2cls.py`, exactly as the single-class
pkl is shared with its HM sibling::

    python tools/maptrv2/custom_carla_map_converter.py \\
        --data-root ../carla_test --split test \\
        --map-classes driving curb \\
        --out-dir data/carla/ --out-tag 30m_tc_2cls

(`--gt-frame tile_center` is the converter's default.) `driving` and `curb`
are that export's OWN class names, so the bare-name shorthand applies; on an
export that spells them differently, name the mapping
(`--map-classes driving=driving_centerline curb=curb`). Note it also ships
several polyline directories -- an unclassified `reference_lines/` and
classified siblings -- and the converter picks the one whose taxonomy covers
the requested classes, printing the `[ref]` line that says so. Pass
`--reference-dir` to override.

**There is no train split yet.** `../carla_test` is test-only, so
`ann_file_train` below names a pkl that does not exist; convert one the same
way (`--split train`) or repoint it before a real run. Training with
train == val scores nothing meaningful.

Verified on `../carla_test` (3795 tiles, `tile_side` 30.0): 0 tiles dropped,
median 100% range coverage, 9458 `driving` + 3819 `curb` instances. The
tile-centre frame is doing real work there -- the same export converted in
the `offset` frame drops 75 tiles outright and leaves 1282 of the rest under
90% coverage, because `|offset - tile_center|` reaches 117 m on some blocks.

`--out-tag` is what makes `--out-dir data/carla/` safe: the output filename
is otherwise a function of `--split` alone, so a second dataset written
there would overwrite an existing pkl with no warning. The tag names what
differs from the plain 25m file (tile size, frame, taxonomy).

Do NOT pass `--lidar-point-cloud-range`: the converter derives xy from the
manifest's own `tile_radius`/`tile_side` and prints what it resolved, which
must equal the parent's `lidar_point_cloud_range` (+/-15 in xy) or the
dataset warns at load.

Check the converter's `[class]` lines before training: it prints an instance
count per map class and shouts if one is empty. A class-free export (the
local 25m one has no `class_lookup` at all) cannot be split this way and
the converter raises rather than emitting empty classes.

Checkpoints are not comparable with single-class ones -- the classification
head has a different output width and the GT is a different set of lines.
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_30m.py']

# Label order. Everything below, and the converter's --map-classes flag,
# must list them identically.
map_classes = ['driving', 'curb']
num_map_classes = len(map_classes)

# [2cls] Stored alongside the other CARLA pkls, kept apart by the tag in the
# filename -- see the --out-tag note in the docstring.
data_root = 'data/carla/'
ann_file_train = data_root + 'carla_map_infos_train_30m_tc_2cls.pkl'
ann_file_val = data_root + 'carla_map_infos_test_30m_tc_2cls.pkl'
ann_file_test = data_root + 'carla_map_infos_test_30m_tc_2cls.pkl'
# Its own GT json: _format_gt() writes this once and never regenerates, so
# sharing a path across taxonomies would silently score against the wrong GT.
map_ann_file = data_root + 'carla_map_gt_30m_tc_2cls.json'

# The pipelines are inherited untouched. `DefaultFormatBundle3D`'s
# `class_names` is the only place in them that mentions map classes, and it
# is inert here: formating.py only reads it under `with_gt and with_label`,
# both of which are False in these configs (GT vectors are attached by the
# dataset's own vectormap_pipeline, not by the transform pipeline). Not
# restating them keeps the actor-augmentation wiring in one file.

model = dict(
    pts_bbox_head=dict(
        num_classes=num_map_classes,
        # num_vec_one2one stays at the 30m base's 50, so the one2one
        # branch flattens to 50 * 2 = 100 scores and the inherited
        # max_num=50 is still within topk's range (gotcha #15). Restated
        # only because num_classes has to be, and the two live together.
        bbox_coder=dict(num_classes=num_map_classes)))

data = dict(
    train=dict(
        data_root=data_root,
        ann_file=ann_file_train,
        map_classes=map_classes),
    val=dict(
        data_root=data_root,
        ann_file=ann_file_val,
        map_ann_file=map_ann_file,
        map_classes=map_classes),
    test=dict(
        data_root=data_root,
        ann_file=ann_file_test,
        map_ann_file=map_ann_file,
        map_classes=map_classes),
)
