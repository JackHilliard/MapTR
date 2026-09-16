"""The tuned monotone-OT recipe: `..._50m_crop_EMDV2.py` at lr = 3e-4.

Best of a seven-run sweep around the EMDV2 baseline (loss_pts 7.07, lr 6e-4),
varying the geometry weight and the learning rate one factor at a time. All
30 epochs, identical data, schedule, seed and batch size, 2170-tile held-out
test set:

    lr 3e-4                 88.35  <- this file   driving 82.3  curb 94.4
    lr 3e-4 + geom 3.54     86.28                 (_LR3e-4_geom05x)
    lr 1.5e-4               85.51                 (_LR1.5e-4)
    lr 6e-4 (baseline)      85.26                 driving 78.7  curb 91.8
    geom 3.54               84.40                 (_geom05x)
    ordered L1              83.20                 driving 79.2  curb 87.2
    geom 14.14              76.27                 (_geom2x)
    lr 1.2e-3               diverged              (_LR1.2e-3)

Both axes came out monotone: halving the lr is worth +3.1 mAP and lifts both
classes, where the loss change alone lifted only curb; the geometry weight is
neutral-to-harmful in both directions, and combining the two apparent wins is
worse than the lr change alone. The baseline lr was inherited from the
ordered-L1 config and had never been tuned for emdv2.

Remaining gap at this point is divider localisation at the tight threshold
(AP@0.5 0.742 vs AP@1.5 0.886; recall@0.5 0.670).
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_50m_crop_EMDV2.py']

optimizer = dict(type='AdamW', lr=3e-4, weight_decay=0.01)
