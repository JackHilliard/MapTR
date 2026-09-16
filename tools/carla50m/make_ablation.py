"""EMDV2 ablation: one factor at a time around the proven baseline."""
from mmcv import Config
SRC='/MapTR2/projects/configs/maptrv2/maptrv2_carla50m_EMDV2.py'
VARIANTS={
 # name        loss_pts   lr        rationale
 'A_geom2x':   (14.14,    6e-4, 'geometry x2 -- divider chamfer 0.339m vs curb 0.174m; '
                                'EMDV2 traded divider precision (82.9 vs L1 87.3) for recall'),
 'B_geom05x':  (3.54,     6e-4, 'geometry x0.5 -- control for A; if A helps, direction matters'),
 'C_lr05x':    (7.07,     3e-4, 'lr x0.5 -- gentler optimisation'),
 'D_lr2x':     (7.07,    1.2e-3, 'lr x2 -- baseline lr was inherited from the L1 config, never tuned for emdv2'),
}
for name,(lw,lr,why) in VARIANTS.items():
    cfg=Config.fromfile(SRC)
    cfg.model['pts_bbox_head']['loss_pts']['loss_weight']=lw
    cfg.optimizer['lr']=lr
    # every reported run used 16; the inherited 22 OOMs on 50 m crops
    # (68 GB of 81). Pin it so the config reproduces without cfg-options.
    cfg.data['samples_per_gpu']=16
    hdr=(f"# EMDV2 ablation {name}: loss_pts={lw}, lr={lr}\n"
         f"# {why}\n"
         f"# Baseline (85.3% mAP): loss_pts=7.07, lr=6e-4. One factor changed.\n")
    out=f'/MapTR2/projects/configs/maptrv2/abl_{name}.py'
    open(out,'w').write(hdr+cfg.pretty_text)
    print(f"  {name:<12} loss_pts={lw:<6} lr={lr}")
