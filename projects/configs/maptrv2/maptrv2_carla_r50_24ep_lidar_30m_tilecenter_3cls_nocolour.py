"""Three-class (driving / curb / crosswalk) colour-free 30m tile-centre config.

A taxonomy overlay on `maptrv2_carla_r50_24ep_lidar_30m_tilecenter_nocolour.py`.
It changes the map classes and NOTHING else: the colour-free loader
(`use_dim=3`) and `SparseEncoder(in_channels=3)`, the tile-centred frame
(`recenter=True` on both pipelines and on `evaluation`), the tile size,
`sparse_shape`, `bev_h_`/`bev_w_`, the coder/assigner ranges, the actor
augmentation and the schedule all stay bound in the parent chain and are
deliberately not restated.

Chain: 30m -> 30m_tilecenter -> 30m_tilecenter_nocolour -> THIS.

Note this hangs off the *nocolour* config rather than off a
`..._3cls.py`, so there is no colour-carrying three-class sibling; the
two-class family is laid out the other way round (taxonomy first, colour
second). Either order composes to the same config -- both changes are
independent -- and this way the load-bearing two-line colour change
(`use_dim=3` + `in_channels=3`, which must move together or the first sparse
conv gets a 3-channel input against a 4-channel weight) lives in exactly one
file instead of being copied into a new branch of the tree.

--- READ THIS BEFORE RUNNING: no export here has a crosswalk class ---

The pkl these configs name CANNOT be generated from the data on this
machine today. Checked directly, every polyline json in every export:

    ../carla_test            30m, 3795 tiles  ->  driving 9458, curb 3819
                             taxonomy {"0": "driving", "1": "curb"}, all
                             3795 tiles agreeing. NO crosswalk.
    Town10HD grid_tiles      60m, 33 tiles    ->  crosswalk 32, and the full
                             8-class lookup -- but 60m tiles, so the wrong
                             geometry for this config chain entirely.
    ../carla (local 25m)     no `class_lookup` at all; cannot be split.

So the only export that HAS crosswalk is the wrong tile size, and the only
export at the right tile size does not have it.

This fails loudly rather than silently, which is the good case. Verified by
running it::

    ValueError: unknown class 'crosswalk'; this export has: 0=driving, 1=curb

`resolve_classes()` raises on a name the export's taxonomy does not carry --
deliberately, because the old `--lane-types` flag silently kept nothing and
produced an empty pkl with no indication why. You will hit this the moment
you try to convert, not after a night of training.

What this config is therefore FOR: a 30m export that publishes crosswalk
polylines, whenever one exists. Nothing about the file changes when it does
-- only the converter run below starts succeeding.

--- Generating the pkl (once a crosswalk-carrying 30m export exists) ---

The taxonomy is a property of the DATA, so this pkl is shared with
`maptrv2_carla_r50_24ep_lidar_30m_HM_tilecenter_3cls_nocolour.py`, exactly as
the two-class pair share theirs -- and shared with the parent's colour-
carrying self too, since colour is dropped at load time and does not touch
the GT::

    python tools/maptrv2/custom_carla_map_converter.py \\
        --data-root <30m export with crosswalk> --split test \\
        --gt-frame tile_center \\
        --map-classes driving curb crosswalk \\
        --out-dir data/carla/ --out-tag 30m_tc_3cls

The bare-name shorthand assumes the export spells them exactly `driving`,
`curb` and `crosswalk`; if it uses the older grid-export names, name the
mapping instead (`--map-classes driving=driving_centerline curb=curb
crosswalk=crosswalk`). To fold sidewalk edges into crosswalks, or road edges
into curbs, extend the group (`curb=curb,road_edge`) and reconvert -- the
grouping lives in the pkl, not in this file.

Do NOT pass `--lidar-point-cloud-range`: the converter derives xy from the
manifest's own `tile_radius`/`tile_side` and prints what it resolved, which
must equal the parent chain's `lidar_point_cloud_range` (+/-15 in xy) or the
dataset warns at load.

Check the converter's `[class]` lines before training -- it prints an
instance count per map class and shouts `is EMPTY` if a class matched
nothing. A class that resolves but never occurs is the one failure mode the
ValueError above does not cover.

**There is still no train split.** `../carla_test` is test-only, so
`ann_file_train` names a pkl that will not exist even once crosswalk data
arrives; convert one the same way (`--split train`) or repoint it. Training
with train == val scores nothing meaningful.

--- What has to agree, and does below ---

* the pkl carries an `annotation` dict with `driving`, `curb` and
  `crosswalk` keys, in that order (the converter records `map_classes`, and
  the dataset warns at load if the config's order differs -- order IS label
  order, so a permutation relabels every instance);
* `data.train/val/test.map_classes` lists them in that order, because
  `VectorizedCarlaLocalMap` builds its CLASS2LABEL from that list. The AV2
  base hardcodes nuScenes names and would map all three to -1, i.e.
  vectorize to zero instances with no error at all -- hence the subclass;
* `pts_bbox_head.num_classes` and `bbox_coder.num_classes` are 3;
* `map_ann_file` is a NEW path. `_format_gt()` skips regeneration whenever
  the file exists, so reusing the two-class json would score three-class
  predictions against two-class GT, silently.

`max_num` needs no change here. This chain keeps the 30m base's
`num_vec_one2one=50`, so the one2one branch flattens to 50 * 3 = 150 scores
and the inherited `max_num=50` is comfortably inside `topk`'s range
(gotcha #15). The HM sibling is the one that has to think about this.

`aux_seg` stays at `seg_classes=1`. The BEV auxiliary segmentation head's
channel count comes from `aux_seg['seg_classes']` independently of
`num_classes` (`maptrv2_head.py:242`), and the dataset rasterizes every class
into that one mask. Raising it is a separate experiment that changes the seg
head's weight shape.

The pipelines are inherited untouched. `DefaultFormatBundle3D`'s
`class_names` therefore still reads `['divider']` from the nocolour parent,
and that is inert: `formating.py` only reads it under `with_gt and
with_label`, both False throughout these configs, because GT vectors are
attached by the dataset's own `vectormap_pipeline` rather than by the
transform pipeline. The two-class overlay relies on exactly the same thing.
Not restating the pipelines is what keeps the colour and actor-augmentation
wiring in one place.

Checkpoints are not comparable with the one- or two-class configs (different
classification-head width, different GT line set), nor with any
colour-carrying config (different first-conv input width).
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_30m_tilecenter_nocolour.py']

# Label order. This list, the converter's --map-classes flag and the pkl's
# recorded `map_classes` must all agree, in this order.
map_classes = ['driving', 'curb', 'crosswalk']
num_map_classes = len(map_classes)

# [3cls] Stored alongside the other CARLA pkls, kept apart only by the tag in
# the filename -- the converter's output name is otherwise a function of
# --split alone, so an untagged second dataset would overwrite the 25m
# offset-frame original with no warning.
data_root = 'data/carla/'
ann_file_train = data_root + 'carla_map_infos_train_30m_tc_3cls.pkl'
ann_file_val = data_root + 'carla_map_infos_test_30m_tc_3cls.pkl'
ann_file_test = data_root + 'carla_map_infos_test_30m_tc_3cls.pkl'
# Its own GT json: _format_gt() writes this once and never regenerates.
map_ann_file = data_root + 'carla_map_gt_30m_tc_3cls.json'

# mmcv merges dicts recursively, so the parent's
# lidar_encoder.backbone.in_channels=3 survives this.
model = dict(
    pts_bbox_head=dict(
        num_classes=num_map_classes,
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
