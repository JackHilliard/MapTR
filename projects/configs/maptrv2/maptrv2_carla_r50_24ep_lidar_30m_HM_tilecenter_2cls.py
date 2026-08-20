"""Two-class (driving / curb) variant of the 30m tile-centred HM config.

The taxonomy overlay for `maptrv2_carla_r50_24ep_lidar_30m_HM_tilecenter.py`,
standing to it exactly as `maptrv2_carla_r50_24ep_lidar_30m_tilecenter_2cls.py`
stands to the plain 30m tile-centre config. It changes the map classes and
nothing else: the polyline geometry loss (`PolylineGeomLoss`), the matching
cost (`PolylineGeomCost`), every tile-size derivation behind them, the
tile-centred pipelines and the schedule all stay bound in the parent chain.

None of the HM geometry constants are restated here, and that is deliberate
rather than an oversight. They are a function of tile SIZE (see CLAUDE.md:
`loss_pts`/`pts_cost` see NORMALISED coordinates, so a physical error of d
metres arrives as d / tile_size), and the number of classes does not enter
that derivation anywhere. Restating `point_cloud_range` here would not
re-derive them in any case -- mmcv evaluates each config file in isolation,
so it would only look as though it had.

--- What changes ---

    0 = driving   <- the export's `driving` polylines
                     (`driving_centerline` on the older grid export)
    1 = curb      <- the export's `curb` polylines

in that order, because the order is the label order:
`VectorizedCarlaLocalMap` builds its CLASS2LABEL from the config's
`map_classes` list. Read the full rationale, and the seg-head note, in the
non-HM sibling's docstring; the mechanics are identical.

The one thing that is NOT identical is `max_num`. The HM configs cut
`num_vec_one2one` from 50 to 25, and `MapTRNMSFreeCoder.decode_single` does
`cls_scores.view(-1).topk(max_num)` over the ONE2ONE branch only, whose
flattened size is `num_vec_one2one * num_map_classes` (gotcha #15). At one
class that was 25, which is what the parent sets; at two it is 50, so the
cap can be raised back to the base's 50 -- and should be, since `max_num`
also bounds how many instances an eval can ever score, and a two-class tile
has more lines to find. Raising it is the safe direction; lowering it below
the head's output is what crashes, and only at the first validation, after
a full epoch has trained.

--- Generating the pkl ---

Shared with `maptrv2_carla_r50_24ep_lidar_30m_tilecenter_2cls.py` -- the
taxonomy and the frame are properties of the data, the loss is not, and this
config leaves `fixed_ptsnum_per_gt_line` at the parent's 20, so `_format_gt()`
would write byte-identical GT either way::

    python tools/maptrv2/custom_carla_map_converter.py \\
        --data-root ../carla_test --split test \\
        --gt-frame tile_center \\
        --map-classes driving curb \\
        --out-dir data/carla/ --out-tag 30m_tc_2cls

`driving` and `curb` are that export's OWN class names, so the bare-name
shorthand applies; on an export that spells them differently, name the
mapping (`--map-classes driving=driving_centerline curb=curb`). Note it also
ships two polyline directories -- an unclassified `reference_lines/` and a
classified `reference_curb_driving_lines/` -- and the converter switches to
the classified one because `--map-classes` needs a taxonomy, printing the
`[ref]` line that says so. Pass `--reference-dir` to override.

**There is no train split yet.** `../carla_test` is test-only, so
`ann_file_train` below names a pkl that does not exist; convert one the same
way (`--split train`) or repoint it before a real run. Training with
train == val scores nothing meaningful.

(Watch the shared `map_ann_file` if the HM line count ever diverges from the
parent's. The resampling is baked into the json, which `_format_gt()` writes
once and never regenerates, so two configs with different
`fixed_ptsnum_per_gt_line` must NOT share a path.)

Do NOT pass `--lidar-point-cloud-range`: the converter derives xy from the
manifest's own `tile_radius`/`tile_side` and prints what it resolved, which
must equal the parent chain's `lidar_point_cloud_range` (+/-15 in xy) or the
dataset warns at load.

Checkpoints are not comparable with single-class ones, nor with offset-frame
or 25m ones.
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_30m_HM_tilecenter.py']

# Label order. Everything below, and the converter's --map-classes flag,
# must list them identically.
map_classes = ['driving', 'curb']
num_map_classes = len(map_classes)

# Restated because mmcv evaluates this file in isolation; it must stay equal
# to num_vec_one2one in maptrv2_carla_r50_24ep_lidar_30m_HM.py, which is
# where the reduced query count actually lives.
num_vec_one2one = 25
# See the max_num discussion in the docstring. 50 is the base 30m config's
# value and the cap this expression is bounded by.
max_num = min(50, num_vec_one2one * num_map_classes)

# [2cls] Stored alongside the other CARLA pkls rather than in a per-variant
# subdirectory; the tag in the filename is the only thing keeping them apart.
data_root = 'data/carla/'
ann_file_train = data_root + 'carla_map_infos_train_30m_tc_2cls.pkl'
ann_file_val = data_root + 'carla_map_infos_test_30m_tc_2cls.pkl'
ann_file_test = data_root + 'carla_map_infos_test_30m_tc_2cls.pkl'
map_ann_file = data_root + 'carla_map_gt_30m_tc_2cls.json'

# The pipelines are inherited untouched -- `DefaultFormatBundle3D`'s
# `class_names` is inert at with_gt=False/with_label=False, and not
# restating them keeps the actor-augmentation wiring in one file.

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
