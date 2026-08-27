"""Colour-free 30m tile-centred HM (polyline geometry) config with the EMD
point loss weight doubled: base 2.0 -> 4.0 before tile scaling.

4.0 is the value the HM base's own Pointcept-comparison note derives for
matching their effective gradient magnitude (their coord_scale maps the
tile to [-1,1], MapTRv2's normalisation to [0,1], so an identical physical
error yields a half-sized number here). The x1.2 tile factor is the
parent's tile_scale = 30/25 (scale_loss_weight_with_tile=True), restated
because mmcv evaluates this file in isolation.

The matching-cost weight is untouched: in cost_mode='emd' it is inert
(median-normalised inside PolylineGeomCost), per the HM base's comment.
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_30m_HM_tilecenter_nocolour.py']

ref_tile_size = 25.0
tile_size = 30.0
loss_pts_weight = 4.0 * (tile_size / ref_tile_size)   # = 4.8; parent: 2.4

model = dict(
    pts_bbox_head=dict(
        loss_pts=dict(loss_weight=loss_pts_weight),
    ),
)
