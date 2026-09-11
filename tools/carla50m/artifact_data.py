"""Metrics + visual comparison figures for the ORIG vs EMDV2 artifact."""
import sys, os, json, glob, warnings, numpy as np
sys.path.insert(0,'/MapTR2/projects/mmdet3d_plugin/datasets')
sys.path.insert(0,'/MapTR2')
warnings.filterwarnings('ignore')
from carla50m_metrics import precision_and_chamfer, _resample
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly

CLS=['divider','boundary']; OUT='/MapTR2/tools/carla50m/figs'; os.makedirs(OUT,exist_ok=True)
GT=json.load(open('/MapTR2/data/carla50m_map_gt_2cls.json'))['GTs']
gt_by={g['sample_token']:g for g in GT}
RUNS={}
for run in ('c50m_ORIG','c50m_EMDV2'):
    f=sorted(glob.glob(f'/MapTR2/work_dirs/{run}/*/pts_bbox/carlamap_results.json'),
             key=os.path.getmtime)[-1]
    RUNS[run]=json.load(open(f))['results']
    print(f"{run}: {len(RUNS[run])} tiles from {f.split('/')[-3]}")

# ---------------- metrics ----------------
M={}
for run,res in RUNS.items():
    M[run]=precision_and_chamfer(res, GT, CLS)
    print(f"  {run} metrics done")
json.dump(M, open(f'{OUT}/metrics.json','w'), indent=1)

# ---------------- per-tile scoring to choose illustrative tiles ----------------
def tile_stats(res, tok_set, thr=1.0):
    st={}
    by={r['sample_token']:r for r in res}
    for tok in tok_set:
        if tok not in by: continue
        g=gt_by[tok]; r=by[tok]; per={}
        for ci,_ in enumerate(CLS):
            gv=[v for v in g['vectors'] if int(v.get('type',0))==ci]
            pv=[v for v in r['vectors'] if int(v.get('type',0))==ci
                and float(v['confidence_level'])>0.4 and len(v['pts'])>=2]
            if not gv: per[ci]=(0,0,None); continue
            if not pv: per[ci]=(0,len(gv),None); continue
            P=np.stack([_resample(v['pts']) for v in pv]); G=np.stack([_resample(v['pts']) for v in gv])
            d=np.linalg.norm(P[:,None,:,None,:]-G[None,:,None,:,:],axis=-1)
            ch=0.5*(d.min(3).mean(2)+d.min(2).mean(2))
            arg=ch.argmin(1); best=ch.min(1); taken=set(); md=[]
            for i in np.argsort(-np.array([float(v['confidence_level']) for v in pv])):
                if best[i]<=thr and arg[i] not in taken:
                    taken.add(int(arg[i])); md.append(best[i])
            per[ci]=(len(md), len(gv), float(np.mean(md)) if md else None)
        st[tok]=per
    return st

toks=[t for t in list(gt_by)[:900]
      if any(int(v.get('type',0))==1 for v in gt_by[t]['vectors'])
      and len(gt_by[t]['vectors'])>=4]
so=tile_stats(RUNS['c50m_ORIG'], toks); se=tile_stats(RUNS['c50m_EMDV2'], toks)
score=[]
for t in toks:
    if t not in so or t not in se: continue
    o,e=so[t],se[t]
    ngt=sum(o[c][1] for c in (0,1))
    if ngt<3: continue
    rec_o=sum(o[c][0] for c in (0,1))/ngt; rec_e=sum(e[c][0] for c in (0,1))/ngt
    score.append((rec_e-rec_o, rec_e, t))
score.sort(reverse=True)
# 2 tiles where EMDV2 recovers most vs ORIG, then 2 dense tiles both do well on,
# so the figure shows the difference AND the typical case -- not only wins.
picks=[t for _,_,t in score[:2]]
for _,rec,t in sorted(score, key=lambda x:-x[1]):
    if t not in picks and len(gt_by[t]['vectors'])>=5:
        picks.append(t)
    if len(picks)>=4: break
picks=list(dict.fromkeys(picks))[:4]
print("  picked tiles:", picks)

# ---------------- figures ----------------
BG='#0f1419'; MUT='#8b95a3'; FG='#eef1f4'
C_GT={0:'#7d8590', 1:'#7d8590'}
C_PR={0:'#4dd4c0', 1:'#f2a65a'}
def panel(ax, tok, res=None, title=''):
    g=gt_by[tok]
    for v in g['vectors']:
        p=np.asarray(v['pts'])[:,:2]
        ax.plot(p[:,0],p[:,1],c='#5a6672',lw=3.2,alpha=.95,zorder=1,solid_capstyle='round')
    if res is not None:
        by={r['sample_token']:r for r in res}
        for v in by[tok]['vectors']:
            if float(v['confidence_level'])<=0.4 or len(v['pts'])<2: continue
            p=np.asarray(v['pts'])[:,:2]; ci=int(v.get('type',0))
            ax.plot(p[:,0],p[:,1],c=C_PR[ci],lw=1.9,zorder=3)
    ax.add_patch(MplPoly([[-15,-15],[15,-15],[15,15],[-15,15]],closed=True,
                         fill=False,ec='#2c333c',lw=1.2))
    ax.set_xlim(-16,16); ax.set_ylim(-16,16); ax.set_aspect('equal')
    ax.set_facecolor(BG); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color('#2c333c')
    ax.set_title(title, color=FG, fontsize=10.5, pad=6)

fig,axes=plt.subplots(len(picks),3,figsize=(11.4,3.75*len(picks)),facecolor=BG)
if len(picks)==1: axes=axes[None,:]
for r,tok in enumerate(picks):
    ng=len(gt_by[tok]['vectors'])
    nb=sum(1 for v in gt_by[tok]['vectors'] if int(v.get('type',0))==1)
    panel(axes[r,0], tok, None, 'ground truth' if r==0 else '')
    panel(axes[r,1], tok, RUNS['c50m_ORIG'], 'ordered L1' if r==0 else '')
    panel(axes[r,2], tok, RUNS['c50m_EMDV2'], 'monotone OT (EMDV2)' if r==0 else '')
    axes[r,0].set_ylabel(f"{tok.split('_tile_')[0][-12:]}\n{ng} lanes ({nb} curb)",
                         color=MUT, fontsize=8, rotation=0, ha='right', va='center', labelpad=42)
fig.suptitle('Test-set predictions  ·  grey = ground truth, teal = divider, amber = curb  ·  conf > 0.4',
             color=FG, fontsize=11.5, y=.995)
fig.tight_layout(rect=[0,0,1,.975])
fig.savefig(f'{OUT}/fig_preds.png', dpi=130, facecolor=BG); plt.close(fig)
print(f"  wrote fig_preds.png ({os.path.getsize(OUT+'/fig_preds.png')/1e6:.2f} MB)")
