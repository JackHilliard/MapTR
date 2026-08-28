"""30m HM (polyline geometry) config combining the
two rebalancing ablations: classification weight 2.0 (see `_cls2`) and the
EMD point loss weight doubled to 4.0 x tile_scale (see `_pts2x`). Comments
and caveats live in those two single-variable siblings.
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_30m_HM.py']

ref_tile_size = 25.0
tile_size = 30.0
loss_pts_weight = 4.0 * (tile_size / ref_tile_size)   # = 4.8; parent: 2.4

model = dict(
    pts_bbox_head=dict(
        loss_cls=dict(loss_weight=2.0),
        loss_pts=dict(loss_weight=loss_pts_weight),
    ),
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                cls_cost=dict(type='FocalLossCost', weight=2.0)))),
)
