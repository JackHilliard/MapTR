"""San Jose 50 m crops with the tuned monotone-OT recipe (`emdv2`, lr 3e-4).

`maptrv2_sanjose_r50_24ep_lidar_50m_crop.py` plus exactly what
`maptrv2_carla_r50_24ep_lidar_50m_crop_EMDV2_LR3e-4.py` adds on top of its
own base: `loss_pts = PolylineGeomLoss(mode='emdv2')` at weight 7.07 with
`pointwise_reduction='sum'`, and `lr = 3e-4`. That pairing was the best of
the CARLA 50 m sweep (88.4 mAP vs 83.2 for ordered L1; see that file).
Nothing here has been tuned on San Jose -- it is the CARLA recipe applied
as-is so the two datasets are compared under one setting.
"""

_base_ = ['./maptrv2_sanjose_r50_24ep_lidar_50m_crop.py']

# Must equal the base's point_cloud_range (restated -- config isolation).
point_cloud_range = [-15.0, -15.0, -30.0, 15.0, 15.0, 20.0]

# 5.0 (PtsL1Loss) * sqrt(C=2): PtsL1Loss sums P*C elementwise |d| terms,
# this sums P L2 norms, so parity needs sqrt(C).
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

optimizer = dict(type='AdamW', lr=3e-4, weight_decay=0.01)
