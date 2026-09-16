"""30 m single-tile, two classes (driving / curb), `emdv2` geometry loss.

The single-tile BASELINE for the multi-tile experiments: exactly
`maptrv2_carla_r50_24ep_lidar_30m_2cls.py` with `loss_pts` swapped for
`PolylineGeomLoss(mode='emdv2')`, the same one-line change
`..._30m_EMDV2.py` makes on the single-class data (see it, and
`..._50m_crop_EMDV2.py`, for what `emdv2` is and why the weight is
5.0 * sqrt(2) = 7.07). Data, frame, colour-free loaders, schedule and
matching are all inherited.

Its neighbourhood sibling is `..._3x3_2cls_EMDV2.py`; a run of each on the
same tiles is the "does a bigger tile help" comparison, once the
neighbourhood predictions are clipped back to +-15 m with
`tools/maptrv2/stitch_results.py --clip-to-tile` so both are scored on the
same per-tile GT.
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_30m_2cls.py']

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
