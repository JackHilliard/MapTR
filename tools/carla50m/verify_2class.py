import sys, warnings, numpy as np, collections
sys.path.insert(0,'/MapTR2'); warnings.filterwarnings('ignore')
from mmcv import Config
from mmdet3d.datasets import build_dataset
import projects.mmdet3d_plugin  # noqa: F401
cfg=Config.fromfile('/MapTR2/projects/configs/maptrv2/maptrv2_carla50m_ORIG.py')
print(f"  map_classes={cfg.map_classes}  head num_classes={cfg.model['pts_bbox_head']['num_classes']}  "
      f"seg_classes={cfg.model['pts_bbox_head']['aux_seg']['seg_classes']}")
ds=build_dataset(cfg.data.train)
print(f"  train dataset MAPCLASSES={ds.MAPCLASSES}  NUM={ds.NUM_MAPCLASSES}")
rng=np.random.default_rng(5); lab=collections.Counter(); ninst=0; both=0; n=0
for i in rng.choice(len(ds), 60, replace=False):
    ex=ds[int(i)]
    if ex is None: continue
    L=ex['gt_labels_3d'].data.numpy(); n+=1
    for v in L: lab[int(v)]+=1
    ninst+=len(L)
    if len(set(L.tolist()))>1: both+=1
print(f"  samples {n}   GT instances {ninst}   labels seen: "
      f"{ {cfg.map_classes[k]: v for k,v in sorted(lab.items())} }")
print(f"  samples containing BOTH classes: {both}/{n}")
G=ds[int(rng.choice(len(ds)))]
print(f"  VERDICT: {'BOTH CLASSES PRESENT in the loss targets' if len(lab)==2 else 'ONLY ONE CLASS -- still broken'}")
# seg mask channels
for i in rng.choice(len(ds), 5, replace=False):
    ex=ds[int(i)]
    if ex is None or 'gt_seg_mask' not in ex: continue
    M=ex['gt_seg_mask'].data.numpy()
    print(f"  aux BEV seg mask shape {M.shape}  (expect {len(cfg.map_classes)} channels)  "
          f"lit per channel {[int(M[c].sum()) for c in range(M.shape[0])]}")
    break
