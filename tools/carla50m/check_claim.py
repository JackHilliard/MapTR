import sys, json, glob, os, warnings, numpy as np
sys.path.insert(0,'/MapTR2/projects/mmdet3d_plugin/datasets'); warnings.filterwarnings('ignore')
from carla50m_metrics import _resample
CLS=['divider','boundary']
GT=json.load(open('/MapTR2/data/carla50m_map_gt_2cls.json'))['GTs']
gt_by={g['sample_token']:g for g in GT}
R={}
for run in ('c50m_ORIG','c50m_EMDV2'):
    f=sorted(glob.glob(f'/MapTR2/work_dirs/{run}/*/pts_bbox/carlamap_results.json'),key=os.path.getmtime)[-1]
    R[run]={r['sample_token']:r for r in json.load(open(f))['results']}
picks=json.load(open('/MapTR2/tools/carla50m/figs/picks.json'))
print(f"{'tile':<34} {'class':<9} {'GT':>3} | {'L1 pred':>7} {'L1 hit':>6} | {'OT pred':>7} {'OT hit':>6}")
print('-'*84)
tot={'c50m_ORIG':[0,0],'c50m_EMDV2':[0,0]}; totgt=[0,0]
for tok in picks:
    for ci,cn in enumerate(CLS):
        gv=[v for v in gt_by[tok]['vectors'] if int(v.get('type',0))==ci]
        totgt[ci]+=len(gv)
        row=[]
        for run in ('c50m_ORIG','c50m_EMDV2'):
            pv=[v for v in R[run][tok]['vectors'] if int(v.get('type',0))==ci
                and float(v['confidence_level'])>0.4 and len(v['pts'])>=2]
            hit=0
            if gv and pv:
                P=np.stack([_resample(v['pts']) for v in pv]); G=np.stack([_resample(v['pts']) for v in gv])
                d=np.linalg.norm(P[:,None,:,None,:]-G[None,:,None,:,:],axis=-1)
                ch=0.5*(d.min(3).mean(2)+d.min(2).mean(2))
                arg=ch.argmin(1); best=ch.min(1); taken=set()
                sc=np.array([float(v['confidence_level']) for v in pv])
                for i in np.argsort(-sc):
                    if best[i]<=1.5 and arg[i] not in taken: taken.add(int(arg[i])); hit+=1
            row += [len(pv), hit]; tot[run][ci]+=hit
        print(f"{tok[-28:]:<34} {cn:<9} {len(gv):>3} | {row[0]:>7} {row[1]:>6} | {row[2]:>7} {row[3]:>6}")
print('-'*84)
for ci,cn in enumerate(CLS):
    print(f"{'TOTAL over these 4 tiles':<34} {cn:<9} {totgt[ci]:>3} | "
          f"{'':>7} {tot['c50m_ORIG'][ci]:>6} | {'':>7} {tot['c50m_EMDV2'][ci]:>6}")
