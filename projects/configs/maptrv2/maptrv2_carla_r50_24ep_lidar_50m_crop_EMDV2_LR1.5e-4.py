"""EMDV2 sweep point: lr x0.25 (1.5e-4) -> 85.51 mAP; 3e-4 is the optimum, lower is not better.

See `..._50m_crop_EMDV2_LR3e-4.py` for the full table. Baseline
(85.3 mAP): loss_pts 7.07, lr 6e-4. One factor changed per file.
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_50m_crop_EMDV2.py']

optimizer = dict(type='AdamW', lr=1.5e-4, weight_decay=0.01)
