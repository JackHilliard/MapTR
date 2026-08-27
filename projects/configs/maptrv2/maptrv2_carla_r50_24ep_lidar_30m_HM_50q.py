"""30m HM (polyline geometry) config with the query
budget restored to the non-HM baseline's: num_vec_one2one=50 (parent: 25).

The parent's cut to 25 mirrored the Pointcept config and was never tuned.
Three values are coupled to it and must move together:
  * num_vec_one2many = num_vec_one2one * 6 (the base config's convention),
  * bbox_coder.max_num = min(50, num_vec_one2one * num_map_classes) --
    gotcha #15: max_num also caps how many instances eval can score, and
    the parent's 25 would silently keep doing so,
  * transformer decoder transformerlayers num_vec, which the parent set to
    its own 25.
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_30m_HM.py']

num_vec_one2one = 50
num_vec_one2many = num_vec_one2one * 6
num_map_classes = 1
max_num = min(50, num_vec_one2one * num_map_classes)

model = dict(
    pts_bbox_head=dict(
        num_vec_one2one=num_vec_one2one,
        num_vec_one2many=num_vec_one2many,
        bbox_coder=dict(max_num=max_num),
        transformer=dict(
            decoder=dict(
                transformerlayers=dict(num_vec=num_vec_one2one))),
    ),
)
