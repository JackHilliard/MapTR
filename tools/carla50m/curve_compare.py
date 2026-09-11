"""Is curve performance really better on the 30m dataset, or is it selection?
Bin test tiles by curvature on BOTH datasets and measure divider recall."""
import sys, json, glob, os, warnings, numpy as np
sys.path.insert(0,'/MapTR2/projects/mmdet3d_plugin/datasets'); warnings.filterwarnings('ignore')
from carla50m_metrics import _resample as _rs

def curvature(vecs):
    best=0.0
    for v in vecs:
        p=np.asarray(v['pts'],float)[:,:2]
        if len(p)<3: continue
        a,b=p[0],p[-1]; ab=b-a; L=np.linalg.norm(ab)
        if L<1e-6: continue
        best=max(best, float(np.abs(np.cross(ab/L,p-a)).max()))
    return best

def recall_by_curvature(gt_path, res_path, cls_id, tag, limit=700):
    GT=json.load(open(gt_path))['GTs']; gt_by={g['sample_token']:g for g in GT}
    res={r['sample_token']:r for r in json.load(open(res_path))['results']}
    bins={'straight (<0.5m)':[], 'gentle (0.5-2m)':[], 'curved (>2m)':[]}
    ncur=[]
    for tok in list(gt_by)[:limit]:
        if tok not in res: continue
        gv=[v for v in gt_by[tok]['vectors'] if int(v.get('type',0))==cls_id]
        if not gv: continue
        c=curvature(gv); ncur.append(c)
        pv=[v for v in res[tok]['vectors'] if int(v.get('type',0))==cls_id
            and float(v['confidence_level'])>0.4 and len(v['pts'])>=2]
        hit=0
        if pv:
            P=np.stack([_rs(v['pts']) for v in pv]); G=np.stack([_rs(v['pts']) for v in gv])
            d=np.linalg.norm(P[:,None,:,None,:]-G[None,:,None,:,:],axis=-1)
            ch=0.5*(d.min(3).mean(2)+d.min(2).mean(2))
            arg=ch.argmin(1); best=ch.min(1); taken=set()
            sc=np.array([float(v['confidence_level']) for v in pv])
            for i in np.argsort(-sc):
                if best[i]<=1.5 and arg[i] not in taken: taken.add(int(arg[i])); hit+=1
        k='straight (<0.5m)' if c<0.5 else ('gentle (0.5-2m)' if c<2 else 'curved (>2m)')
        bins[k].append((hit,len(gv)))
    print(f"\n=== {tag} ===")
    print(f"  tiles sampled {len(ncur)}   curvature: median {np.median(ncur):.2f} m  p90 {np.percentile(ncur,90):.2f} m")
    for k,v in bins.items():
        if not v: print(f"   {k:<18} (none)"); continue
        h=sum(a for a,_ in v); n=sum(b for _,b in v)
        lanes=np.mean([b for _,b in v])
        print(f"   {k:<18} tiles {len(v):>4}  lanes/tile {lanes:4.1f}  recall {h}/{n} = {100*h/n:5.1f}%")

J='/MapTR2'
# 30 m dataset (single class 'divider'), best model there = ordered L1 (ORIG)
recall_by_curvature(f'{J}/data30/carla_map_gt.json',
                    f'{J}/data30/orig_results.json', 0, '30 m dataset — ordered L1')
# 50 m dataset, divider class, ordered L1
recall_by_curvature('/MapTR2/data/carla50m_map_gt_2cls.json',
    sorted(glob.glob('/MapTR2/work_dirs/c50m_ORIG/*/pts_bbox/carlamap_results.json'),
           key=os.path.getmtime)[-1], 0, '50 m dataset — ordered L1 (divider)')
