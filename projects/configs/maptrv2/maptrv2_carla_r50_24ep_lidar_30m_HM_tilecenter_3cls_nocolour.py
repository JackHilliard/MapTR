"""Three-class (driving / curb / crosswalk) colour-free 30m tile-centre HM config.

A taxonomy overlay on
`maptrv2_carla_r50_24ep_lidar_30m_HM_tilecenter_nocolour.py`, standing to it
exactly as `maptrv2_carla_r50_24ep_lidar_30m_tilecenter_3cls_nocolour.py`
stands to the plain colour-free tile-centre config. It changes the map
classes and nothing else: the polyline geometry loss (`PolylineGeomLoss`),
the matching cost (`PolylineGeomCost`), every tile-size derivation behind
them, the colour-free loader and encoder, the tile-centred pipelines and the
schedule all stay bound in the parent chain.

Chain: 30m_HM -> 30m_HM_tilecenter -> 30m_HM_tilecenter_nocolour -> THIS.

None of the HM geometry constants are restated, and that is deliberate.
They are a function of tile SIZE -- `loss_pts`/`pts_cost` see NORMALISED
coordinates, so a physical error of d metres arrives as `d / tile_size` --
and the number of classes does not enter that derivation anywhere. Restating
`point_cloud_range` here would not re-derive them in any case: mmcv evaluates
each config file in isolation, so it would only look as though it had.

--- READ THIS BEFORE RUNNING: no export here has a crosswalk class ---

Same blocker as the non-HM sibling, whose docstring has the full survey.
In short: `../carla_test` (30m, 3795 tiles) publishes only
`{"0": "driving", "1": "curb"}`; the only export carrying crosswalk is the
60m Town10HD grid one (32 crosswalk polylines, 33 tiles), which is the wrong
tile geometry for this chain. Converting anyway fails immediately and
loudly -- verified::

    ValueError: unknown class 'crosswalk'; this export has: 0=driving, 1=curb

This config is ready for a 30m export that publishes crosswalks; nothing in
it changes when one appears.

--- Generating the pkl (once such an export exists) ---

Shared with `maptrv2_carla_r50_24ep_lidar_30m_tilecenter_3cls_nocolour.py`:
the taxonomy and the frame are properties of the data, the loss is not, and
this config leaves `fixed_ptsnum_per_gt_line` at the parent's 20, so
`_format_gt()` would write byte-identical GT either way::

    python tools/maptrv2/custom_carla_map_converter.py \\
        --data-root <30m export with crosswalk> --split test \\
        --gt-frame tile_center \\
        --map-classes driving curb crosswalk \\
        --out-dir data/carla/ --out-tag 30m_tc_3cls

Two configs with DIFFERENT line counts must not share a `map_ann_file`,
since `_format_gt()` bakes the resampling into a json it writes once and
never regenerates. Watch this if the HM line count ever diverges from the
parent's 20.

**There is still no train split** -- see the sibling's docstring.

--- max_num: the one thing that is not just "num_classes = 3" ---

The HM chain cuts `num_vec_one2one` from 50 to 25, and
`MapTRNMSFreeCoder.decode_single` does `cls_scores.view(-1).topk(max_num)`
over the ONE2ONE branch only, whose flattened size is
`num_vec_one2one * num_map_classes` (gotcha #15). At one class that is 25,
which is what the 1-class parent sets; at three it is 75, so the cap can be
raised back to the 30m base's 50 -- and should be, because `max_num` also
bounds how many instances an eval can ever score, and a three-class tile has
more lines to find. Raising it is the safe direction; lowering it below the
head's output is what crashes, and only at the first validation, after a
full epoch has already trained.

`min(50, ...)` rather than a bare 75: 50 is the base's value and there is no
evidence that decoding more than 50 instances per tile helps on 30m CARLA
tiles, so this raises the cap back to the family default rather than past it.

Everything else -- the `class_names` note, `aux_seg`, the pkl/`map_ann_file`
rules, and why the pipelines are not restated -- is identical to the non-HM
sibling. Read that file's docstring rather than trusting a summary here.

Checkpoints are not comparable with the one- or two-class HM configs, nor
with any colour-carrying config.
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_30m_HM_tilecenter_nocolour.py']

# Label order. This list, the converter's --map-classes flag and the pkl's
# recorded `map_classes` must all agree, in this order.
map_classes = ['driving', 'curb', 'crosswalk']
num_map_classes = len(map_classes)

# Restated locally only to compute max_num below: mmcv evaluates each config
# in isolation, so this must stay equal to num_vec_one2one in
# maptrv2_carla_r50_24ep_lidar_30m_HM.py, which is where it actually binds.
num_vec_one2one = 25
# See the max_num discussion in the docstring. 50 is the base 30m config's
# value; 25 * 3 = 75 is the head's one2one output width, so 50 is in range.
max_num = min(50, num_vec_one2one * num_map_classes)

# [3cls] Same files as the non-HM three-class config, map_ann_file included.
data_root = 'data/carla/'
ann_file_train = data_root + 'carla_map_infos_train_30m_tc_3cls.pkl'
ann_file_val = data_root + 'carla_map_infos_test_30m_tc_3cls.pkl'
ann_file_test = data_root + 'carla_map_infos_test_30m_tc_3cls.pkl'
map_ann_file = data_root + 'carla_map_gt_30m_tc_3cls.json'

# mmcv merges dicts recursively, so the parent's
# lidar_encoder.backbone.in_channels=3 survives this.
model = dict(
    pts_bbox_head=dict(
        num_classes=num_map_classes,
        bbox_coder=dict(num_classes=num_map_classes, max_num=max_num)))

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
