"""Three-class (driving / curb / crosswalk) variant of the 30m HM config.

Stands to `maptrv2_carla_r50_24ep_lidar_30m_HM.py` as
`maptrv2_carla_r50_24ep_lidar_30m_3cls.py` stands to the plain 30m config;
the pkl is shared with that file (see its docstring for the converter
command and the crosswalk sparsity caveat).

`max_num` follows gotcha #15 as in the HM `_2cls` config: the HM base's 25
queries flatten to 25 x 3 = 75 scores at three classes, so max_num is raised
from the base's 25 back to `min(50, 75) = 50`. Raising is the safe
direction; lowering crashes at the first validation.
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_30m_HM.py']

map_classes = ['driving', 'curb', 'crosswalk']
num_map_classes = len(map_classes)

# Must equal the HM base's num_vec_one2one (restated -- mmcv evaluates each
# config file in isolation, so the base's module-level name is not visible).
num_vec_one2one = 25
max_num = min(50, num_vec_one2one * num_map_classes)

data_root = 'data/carla/'
ann_file_train = data_root + 'carla_map_infos_train_30m_tc_3cls.pkl'
ann_file_val = data_root + 'carla_map_infos_test_30m_tc_3cls.pkl'
ann_file_test = data_root + 'carla_map_infos_test_30m_tc_3cls.pkl'
map_ann_file = data_root + 'carla_map_gt_30m_tc_3cls.json'

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
