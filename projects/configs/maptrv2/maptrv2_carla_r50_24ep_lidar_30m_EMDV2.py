"""30 m pkl data, monotone-OT (`emdv2`) geometry loss.

`maptrv2_carla_r50_24ep_lidar_30m.py` with `loss_pts` swapped for
`PolylineGeomLoss(mode='emdv2')` and nothing else -- the same one-line change
`..._50m_crop_EMDV2.py` makes on the crop data; see that file for what
`emdv2` is and why the weight is 5.0 * sqrt(2) = 7.07.

On the 30 m tile-centre data this reached PARITY with `PtsL1Loss` (mAP 0.7474
vs 0.7485; AP@0.5 deltas over nine evals oscillate about zero, last-three
mean -0.001) and beat plain `mode='emd'` by a consistent ~2x on AP@0.5, so
the scale/permutation diagnosis holds. Adding the uniform-spacing regulariser
(`emdv2_spacing_w=0.05`, "EMDV2S") stayed at parity too. The win over L1 only
appeared on the 50 m rotated crops with a second class.

Unlike the `_HM` family this keeps the base's `OrderedPtsL1Cost` matching,
`loss_dir=0.005`, `num_vec_one2one=50` and `cls_cost` -- it is a loss
swap, not the HM recipe.
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_30m.py']

point_cloud_range = [-15.0, -15.0, -30.0, 15.0, 15.0, 20.0]
loss_pts_weight = 7.07

model = dict(
    pts_bbox_head=dict(
        loss_pts=dict(
            _delete_=True,
            type='PolylineGeomLoss',
            mode='emdv2',
            loss_weight=loss_pts_weight,
            emdv2_pointwise_reduction='sum',
            emdv2_spacing_w=0.0,
            pc_range=point_cloud_range)))
