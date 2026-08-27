"""Colour-free 30m tile-centred HM (polyline geometry) config with the
classification weight restored to the non-HM baseline's 2.0 (parent: 1.0,
mirroring Pointcept's loss_ce_weight).

cls_cost.weight must move with it: mmdet's DETRHead.__init__ asserts the
loss and matcher classification weights are identical. Note that also
shifts the MATCHING balance -- geometry:class goes from 1:1 to 1:2, since
the EMD cost is median-normalised to ~1.0.
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_30m_HM_tilecenter_nocolour.py']

model = dict(
    pts_bbox_head=dict(
        loss_cls=dict(loss_weight=2.0),
    ),
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                cls_cost=dict(type='FocalLossCost', weight=2.0)))),
)
