"""EMDV2 sweep point: lr 3e-4 combined with geometry weight 3.54 -> 86.28 mAP, worse than lr 3e-4 alone.

See `..._50m_crop_EMDV2_LR3e-4.py` for the full table. Baseline
(85.3 mAP): loss_pts 7.07, lr 6e-4. One factor changed per file.
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_50m_crop_EMDV2_LR3e-4.py']

model = dict(pts_bbox_head=dict(loss_pts=dict(loss_weight=3.54)))
