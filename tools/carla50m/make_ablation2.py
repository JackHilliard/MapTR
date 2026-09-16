"""Round 2: both axes were monotone toward 'less'. Push further, and combine."""
from mmcv import Config
SRC='/MapTR2/projects/configs/maptrv2/maptrv2_carla50m_EMDV2.py'
VARIANTS={
 'E_lr025x':  (7.07, 1.5e-4, 'lr x0.25 -- C (3e-4) beat baseline by +11.0 mAP at ep12; '
                             'is 3e-4 the optimum or is lower still better?'),
 'F_lr05_geom05x': (3.54, 3e-4, 'combine the two winning directions: C lr 3e-4 (+11.0) '
                                'with B geom 3.54 (+0.6)'),
}
for name,(lw,lr,why) in VARIANTS.items():
    cfg=Config.fromfile(SRC)
    cfg.model['pts_bbox_head']['loss_pts']['loss_weight']=lw
    cfg.optimizer['lr']=lr
    # every reported run used 16; the inherited 22 OOMs on 50 m crops
    # (68 GB of 81). Pin it so the config reproduces without cfg-options.
    cfg.data['samples_per_gpu']=16
    hdr=(f"# EMDV2 ablation round 2 {name}: loss_pts={lw}, lr={lr}\n# {why}\n"
         f"# Round 1 @ep12: C(lr3e-4) .7840 > B(geom3.54) .6798 > base .6738 > "
         f"A(geom14.14) .5430 > D(lr1.2e-3) diverged\n")
    open(f'/MapTR2/projects/configs/maptrv2/abl_{name}.py','w').write(hdr+cfg.pretty_text)
    print(f"  {name:<16} loss_pts={lw:<6} lr={lr}")
