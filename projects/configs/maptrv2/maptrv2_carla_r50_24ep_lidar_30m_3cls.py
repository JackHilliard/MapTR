"""Three-class (driving / curb / crosswalk) variant of the 30m CARLA config.

Stands to `maptrv2_carla_r50_24ep_lidar_30m.py` as the `_2cls` config does,
adding `crosswalk` as label 2. See that file's docstring for the taxonomy
mechanics (what must agree, and why); everything said there holds here with
three classes instead of two.

Crosswalk is SPARSE: 576 polylines over 257 of 3795 tiles on `../carla_test`
(0.2 per tile vs driving's 2.5). Expect the majority-class collapse the
2-class run documented, harder -- do not read `crosswalk_AP` before the
classifier separates the classes (see CLAUDE.md).

Generate the shared pkl (also used by the HM sibling)::

    python tools/maptrv2/custom_carla_map_converter.py \\
        --data-root ../carla_test --split test \\
        --map-classes driving curb crosswalk \\
        --out-dir data/carla/ --out-tag 30m_tc_3cls

(`--gt-frame tile_center` is the converter's default.) No `--reference-dir`
needed: the converter selects the polyline directory whose taxonomy covers
the requested classes (`reference_driving_curb_crosswalk` there) and prints
what it picked.

num_vec_one2one stays at the base's 50, so the one2one branch flattens to
50 x 3 = 150 scores and the inherited max_num=50 is untouched (gotcha #15).
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_30m.py']

map_classes = ['driving', 'curb', 'crosswalk']
num_map_classes = len(map_classes)

data_root = 'data/carla/'
ann_file_train = data_root + 'carla_map_infos_train_30m_tc_3cls.pkl'
ann_file_val = data_root + 'carla_map_infos_test_30m_tc_3cls.pkl'
ann_file_test = data_root + 'carla_map_infos_test_30m_tc_3cls.pkl'
map_ann_file = data_root + 'carla_map_gt_30m_tc_3cls.json'

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
