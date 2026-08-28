"""Two-class (driving / curb) variant of the 30m HM (polyline geometry) config.

Stands to `maptrv2_carla_r50_24ep_lidar_30m_HM.py` as
`maptrv2_carla_r50_24ep_lidar_30m_2cls.py` stands to the plain 30m config: a
thin overlay changing the map TAXONOMY and nothing else. See that file's
docstring for the taxonomy mechanics and the converter command -- the pkl is
SHARED with it (the taxonomy is a property of the data, the loss is not).

The one HM-specific subtlety is `max_num` (gotcha #15). The HM base cuts
`num_vec_one2one` to 25 and pairs it with `bbox_coder.max_num=25`, because
`decode_single` does `cls_scores.view(-1).topk(max_num)` over the one2one
branch, whose flattened size is `num_vec_one2one * num_map_classes` -- 25 at
one class. At two classes the bound is 50, so max_num is raised back to
`min(50, 25 * 2) = 50`. Raising is the safe direction (lowering crashes at
the first validation, after a full epoch has trained), and it also matters
because max_num caps how many instances an eval can ever score -- there are
more lines to find with two classes.
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_30m_HM.py']

map_classes = ['driving', 'curb']
num_map_classes = len(map_classes)

# Must equal the HM base's num_vec_one2one (restated -- mmcv evaluates each
# config file in isolation, so the base's module-level name is not visible).
num_vec_one2one = 25
max_num = min(50, num_vec_one2one * num_map_classes)

data_root = 'data/carla/'
ann_file_train = data_root + 'carla_map_infos_train_30m_tc_2cls.pkl'
ann_file_val = data_root + 'carla_map_infos_test_30m_tc_2cls.pkl'
ann_file_test = data_root + 'carla_map_infos_test_30m_tc_2cls.pkl'
map_ann_file = data_root + 'carla_map_gt_30m_tc_2cls.json'

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
