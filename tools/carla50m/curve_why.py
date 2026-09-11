import sys, os, json, glob, warnings, numpy as np
sys.path.insert(0,'/MapTR2'); sys.path.insert(0,'/MapTR2/projects/mmdet3d_plugin/datasets')
warnings.filterwarnings('ignore')
from mmcv import Config
from mmdet3d.datasets import build_dataset
import projects.mmdet3d_plugin  # noqa
from projects.mmdet3d_plugin.datasets.carla50m_crop_dataset import LoadCarla50mCrop
from carla50m_metrics import _resample as _rs
cfg=Config.fromfile('/MapTR2/projects/configs/maptrv2/maptrv2_carla50m_ORIG.py')
ds=build_dataset(cfg.data.val); idx={i['sample_idx']:k for k,i in enumerate(ds.data_infos)}
ld=LoadCarla50mCrop()
gt_by={g['sample_token']:g for g in json.load(open('/MapTR2/data/carla50m_map_gt_2cls.json'))['GTs']}
def curvature(vecs):
    b=0.0
    for v in vecs:
        p=np.asarray(v['pts'],float)[:,:2]
        if len(p)<3: continue
        a,e=p[0],p[-1]; ae=e-a; L=np.linalg.norm(ae)
        if L>1e-6: b=max(b,float(np.abs(np.cross(ae/L,p-a)).max()))
    return b
# is LiDAR density the discriminator on curved tiles?
rows=[]
for tok in list(gt_by)[:900]:
    vs=gt_by[tok]['vectors']
    if len(vs)<3 or curvature(vs)<2.0: continue
    i=idx.get(tok)
    if i is None: continue
    r=dict(ds.get_data_info(i)); ld(r); n=len(r['points'].tensor)
    # min spacing between GT lanes -> how close together they run
    G=[_rs(v['pts']) for v in vs]
    dmin=9e9
    for a in range(len(G)):
        for b in range(a+1,len(G)):
            d=np.linalg.norm(G[a][:,None,:]-G[b][None,:,:],axis=-1).min()
            dmin=min(dmin,d)
    rows.append((tok,n,len(vs),curvature(vs),dmin))
import statistics as st
pts=[r[1] for r in rows]
print(f"curved tiles: {len(rows)}   LiDAR points/crop: median {int(st.median(pts)):,}  "
      f"min {min(pts):,}  max {max(pts):,}")
lo=sorted(rows,key=lambda r:r[1])[:8]
hi=sorted(rows,key=lambda r:-r[1])[:8]
print(f"\n  sparsest 8 curved tiles: pts {[f'{r[1]//1000}k' for r in lo]}")
print(f"  densest  8 curved tiles: pts {[f'{r[1]//1000}k' for r in hi]}")
print(f"\n  lane separation (min gap between GT lanes):")
print(f"    sparsest: median {st.median([r[4] for r in lo]):.2f} m")
print(f"    densest : median {st.median([r[4] for r in hi]):.2f} m")
print(f"    all     : median {st.median([r[4] for r in rows]):.2f} m")
