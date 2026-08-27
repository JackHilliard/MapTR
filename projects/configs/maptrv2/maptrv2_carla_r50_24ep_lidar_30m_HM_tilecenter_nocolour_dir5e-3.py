"""Colour-free 30m tile-centred HM (polyline geometry) config with a
non-zero direction loss: loss_dir weight 0.005 (parent: 0.0, mirroring
Pointcept, where the term is dead code).

Part of a sweep {0.005, 0.01, 0.1, 0.5}. The EMD point loss is
permutation-invariant, so with loss_dir at 0.0 nothing in the HM objective
constrains vertex ORDER at all and the predictions zigzag (2026-08-23
section of CLAUDE.md); this is the fix-at-source test. 0.005 is the non-HM
baseline's value, but HM loss magnitudes are ~15x smaller overall, hence
the larger candidates. Judge by AP@0.5 and by rendering, not mAP alone --
the chamfer eval's order-sensitive resampling currently flatters the
zigzag by ~0.04 mAP, so part of a genuine fix will not show in the score.
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_30m_HM_tilecenter_nocolour.py']

model = dict(
    pts_bbox_head=dict(
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
    ),
)
