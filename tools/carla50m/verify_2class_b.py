import sys, warnings, json, numpy as np, collections
sys.path.insert(0,'/MapTR2'); warnings.filterwarnings('ignore')
from mmcv import Config
from mmdet3d.datasets import build_dataset
import projects.mmdet3d_plugin  # noqa: F401
cfg=Config.fromfile('/MapTR2/projects/configs/maptrv2/maptrv2_carla50m_ORIG.py')
ds=build_dataset(cfg.data.train)
print("A. aux BEV seg: does the BOUNDARY channel light on a tile that has curbs?")
rng=np.random.default_rng(2); found=0
for i in rng.choice(len(ds), 200, replace=False):
    ex=ds[int(i)]
    if ex is None or 'gt_seg_mask' not in ex: continue
    L=ex['gt_labels_3d'].data.numpy()
    if 1 not in L: continue
    M=ex['gt_seg_mask'].data.numpy()
    lit=[int(M[c].sum()) for c in range(M.shape[0])]
    print(f"   tile with labels {dict(collections.Counter(L.tolist()))} -> lit per channel {lit}")
    found+=1
    if found==3: break
print(f"   VERDICT: {'BOUNDARY CHANNEL IS DRAWN' if found and lit[1]>0 else 'boundary channel dead'}")

print("\nB. rebuild eval GT cache (2 classes)")
dv=build_dataset(cfg.data.val)
print(f"   val {len(dv)} samples, map_ann_file={dv.map_ann_file}")
dv._format_gt()
g=json.load(open(dv.map_ann_file))['GTs']
c=collections.Counter()
for t in g:
    for v in t['vectors']: c[v['cls_name']]+=1
tot=sum(c.values())
print(f"   cache: {len(g)} tiles, {tot} polylines -> {dict(c)}")
for k,v in c.items(): print(f"      {k}: {v} ({100*v/tot:.1f}%)")
print(f"   tiles with >=1 boundary: {sum(1 for t in g if any(v['cls_name']=='boundary' for v in t['vectors']))}/{len(g)}")
