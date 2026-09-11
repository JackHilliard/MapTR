import sys, warnings, numpy as np, torch
sys.path.insert(0,'/MapTR2'); warnings.filterwarnings('ignore')
from mmcv import Config
from mmdet3d.datasets import build_dataset
import projects.mmdet3d_plugin  # noqa: F401
from projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head import normalize_2d_pts
cfg=Config.fromfile('/MapTR2/projects/configs/maptrv2/maptrv2_carla50m_ORIG.py')
head=cfg.model['pts_bbox_head']; pr=cfg.get('point_cloud_range')
pad=cfg.data.train.get('padding_value')
if pad is None: pad=head.get('padding_value')
if pad is None: pad=-10000.0   # LiDARInstanceLines default
pad=float(pad)
print(f"  padding_value = {pad}   pc_range xy = {pr[0]},{pr[1]} .. {pr[3]},{pr[4]}")
print(f"  (pad - pc_range[0]) / patch_w = {(pad - pr[0])/(pr[3]-pr[0]):.5f}  <- the -332.83 seen")
ds=build_dataset(cfg.data.train)
lo=hi=None; npad=0; ntot=0
for i in (5,11,23,37,52,64,71,88):
    ex=ds[i]
    if ex is None: continue
    G=ex['gt_bboxes_3d'].data.shift_fixed_num_sampled_points_v2
    N=normalize_2d_pts(G, pr)
    m=(G<=pad+1.0)                      # padded slots
    npad+=int(m.sum()); ntot+=int(m.numel())
    V=N[~m]
    lo=V.min().item() if lo is None else min(lo,V.min().item())
    hi=V.max().item() if hi is None else max(hi,V.max().item())
print(f"  padded entries: {npad}/{ntot} ({100*npad/ntot:.1f}%) -- masked out by the head's matcher")
print(f"  normalised REAL GT: min {lo:.6f}  max {hi:.6f}")
print(f"  VERDICT: {'IN [0,1] -- normalisation correct' if lo>=-1e-4 and hi<=1+1e-4 else 'OUT OF RANGE'}")
