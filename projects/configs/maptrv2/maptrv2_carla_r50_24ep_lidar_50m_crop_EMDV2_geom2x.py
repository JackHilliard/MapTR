"""EMDV2 sweep point: geometry weight x2 (14.14) -> 76.27 mAP. Divider chamfer was 0.339 m vs curb 0.174 m and EMDV2 had traded divider precision for recall; more geometry weight did not help.

See `..._50m_crop_EMDV2_LR3e-4.py` for the full table. Baseline
(85.3 mAP): loss_pts 7.07, lr 6e-4. One factor changed per file.
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_50m_crop_EMDV2.py']

model = dict(pts_bbox_head=dict(loss_pts=dict(loss_weight=14.14)))
