import sys, os, json, glob, warnings, numpy as np, statistics as st
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
R={}
for run in ('c50m_ORIG','c50m_EMDV2'):
    f=sorted(glob.glob(f'/MapTR2/work_dirs/{run}/*/pts_bbox/carlamap_results.json'),key=os.path.getmtime)[-1]
    R[run]={r['sample_token']:r for r in json.load(open(f))['results']}
def curvature(vecs):
    b=0.0
    for v in vecs:
        p=np.asarray(v['pts'],float)[:,:2]
        if len(p)<3: continue
        a,e=p[0],p[-1]; ae=e-a; L=np.linalg.norm(ae)
        if L>1e-6: b=max(b,float(np.abs(np.cross(ae/L,p-a)).max()))
    return b
def rec(tok,res):
    h=n=0
    for ci in (0,1):
        gv=[v for v in gt_by[tok]['vectors'] if int(v.get('type',0))==ci]
        if not gv: continue
        n+=len(gv)
        pv=[v for v in res[tok]['vectors'] if int(v.get('type',0))==ci
            and float(v['confidence_level'])>0.4 and len(v['pts'])>=2]
        if not pv: continue
        P=np.stack([_rs(v['pts']) for v in pv]); G=np.stack([_rs(v['pts']) for v in gv])
        d=np.linalg.norm(P[:,None,:,None,:]-G[None,:,None,:,:],axis=-1)
        ch=0.5*(d.min(3).mean(2)+d.min(2).mean(2)); arg=ch.argmin(1); best=ch.min(1)
        sc=np.array([float(v['confidence_level']) for v in pv]); taken=set()
        for i in np.argsort(-sc):
            if best[i]<=1.5 and arg[i] not in taken: taken.add(int(arg[i])); h+=1
    return h,n
rows=[]
for tok in list(gt_by)[:900]:
    vs=gt_by[tok]['vectors']
    if len(vs)<2 or tok not in R['c50m_ORIG']: continue
    G=[_rs(v['pts']) for v in vs]
    dmin=9e9
    for a in range(len(G)):
        for b in range(a+1,len(G)):
            dmin=min(dmin,float(np.linalg.norm(G[a][:,None,:]-G[b][None,:,:],axis=-1).min()))
    h,n=rec(tok,R['c50m_ORIG'])
    if n: rows.append((dmin,curvature(vs),len(vs),h/n))
print(f"tiles: {len(rows)}\n")
print("A. recall vs LANE SEPARATION (min gap between GT lanes)")
for lo,hi,lab in ((-1,0.01,'touching (0 m)'),(0.01,1.5,'0-1.5 m'),(1.5,3,'1.5-3 m'),(3,99,'>3 m')):
    s=[r for r in rows if lo<r[0]<=hi]
    if s: print(f"   {lab:<16} tiles {len(s):>4}  lanes/tile {st.mean([r[2] for r in s]):4.1f}  "
                f"curvature {st.median([r[1] for r in s]):4.2f} m  recall {100*st.mean([r[3] for r in s]):5.1f}%")
print("\nB. recall vs CURVATURE (same tiles)")
for lo,hi,lab in ((-1,0.5,'straight'),(0.5,2,'gentle'),(2,99,'curved')):
    s=[r for r in rows if lo<r[1]<=hi]
    if s: print(f"   {lab:<16} tiles {len(s):>4}  lanes/tile {st.mean([r[2] for r in s]):4.1f}  "
                f"separation {st.median([r[0] for r in s]):4.2f} m  recall {100*st.mean([r[3] for r in s]):5.1f}%")
