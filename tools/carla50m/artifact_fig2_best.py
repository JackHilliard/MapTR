"""Prediction figure with LiDAR background and class-coloured GT."""
import sys, os, json, glob, warnings, numpy as np
sys.path.insert(0,'/MapTR2'); sys.path.insert(0,'/MapTR2/projects/mmdet3d_plugin/datasets')
warnings.filterwarnings('ignore')
from mmcv import Config
from mmdet3d.datasets import build_dataset
import projects.mmdet3d_plugin  # noqa
from projects.mmdet3d_plugin.datasets.carla50m_crop_dataset import LoadCarla50mCrop
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly

OUT='/MapTR2/tools/carla50m/figs'
CLS=['divider','boundary']
cfg=Config.fromfile('/MapTR2/projects/configs/maptrv2/maptrv2_carla50m_ORIG.py')
ds=build_dataset(cfg.data.val)                      # seeded, frozen eval crops
idx={info['sample_idx']:i for i,info in enumerate(ds.data_infos)}
loader=LoadCarla50mCrop()
GT=json.load(open('/MapTR2/data/carla50m_map_gt_2cls.json'))['GTs']
gt_by={g['sample_token']:g for g in GT}
RUNS={}
for run in ('c50m_ORIG','abl_C_lr05x'):
    f=sorted(glob.glob(f'/MapTR2/work_dirs/{run}/*/pts_bbox/carlamap_results.json'),
             key=os.path.getmtime)[-1]
    RUNS[run]={r['sample_token']:r for r in json.load(open(f))['results']}

PICKS=json.load(open(f'{OUT}/picks.json')) if os.path.exists(f'{OUT}/picks.json') else None
if PICKS is None:
    PICKS=['town12_chunk_28_tile_00121','town12_chunk_22_tile_00064']
print('tiles:', PICKS)

BG='#0f1419'; FG='#eef1f4'; MUT='#8b95a3'
GT_C ={0:'#37514c', 1:'#544733'}      # GT: class-tinted but desaturated --
                                       # bright = prediction, dull = ground truth
PR_C ={0:'#4dd4c0', 1:'#f2a65a'}      # predictions : bright teal / bright amber
PT_C ='#39434e'                        # LiDAR

def points_for(tok):
    i=idx.get(tok)
    if i is None: return None
    r=dict(ds.get_data_info(i)); loader(r)
    return r['points'].tensor.numpy()

def recovered(tok, res, ci, thr=1.5):
    gv=[v for v in gt_by[tok]['vectors'] if int(v.get('type',0))==ci]
    if res is None or tok not in res or not gv: return None, len(gv)
    pv=[v for v in res[tok]['vectors'] if int(v.get('type',0))==ci
        and float(v['confidence_level'])>0.4 and len(v['pts'])>=2]
    if not pv: return 0, len(gv)
    from carla50m_metrics import _resample as _rs
    P=np.stack([_rs(v['pts']) for v in pv]); G=np.stack([_rs(v['pts']) for v in gv])
    d=np.linalg.norm(P[:,None,:,None,:]-G[None,:,None,:,:],axis=-1)
    ch=0.5*(d.min(3).mean(2)+d.min(2).mean(2))
    arg=ch.argmin(1); best=ch.min(1); taken=set(); hit=0
    sc=np.array([float(v['confidence_level']) for v in pv])
    for i in np.argsort(-sc):
        if best[i]<=thr and arg[i] not in taken: taken.add(int(arg[i])); hit+=1
    return hit, len(gv)

def panel(ax, tok, res, title, show_gt=True, show_pred=True):
    P=points_for(tok)
    if P is not None and len(P):
        sel=np.random.default_rng(0).choice(len(P), min(60000,len(P)), replace=False)
        ax.scatter(P[sel,0],P[sel,1],s=.09,c=PT_C,linewidths=0,zorder=0)
    if show_gt:
        for v in gt_by[tok]['vectors']:
            p=np.asarray(v['pts'])[:,:2]; ci=int(v.get('type',0))
            ax.plot(p[:,0],p[:,1],c=GT_C[ci],lw=4.2,alpha=.95,zorder=1,solid_capstyle='round')
    if show_pred and res is not None and tok in res:
        for v in res[tok]['vectors']:
            if float(v['confidence_level'])<=0.4 or len(v['pts'])<2: continue
            p=np.asarray(v['pts'])[:,:2]; ci=int(v.get('type',0))
            ax.plot(p[:,0],p[:,1],c=PR_C[ci],lw=1.9,zorder=3)
    ax.add_patch(MplPoly([[-15,-15],[15,-15],[15,15],[-15,15]],closed=True,fill=False,ec='#2c333c',lw=1.2))
    ax.set_xlim(-15.6,15.6); ax.set_ylim(-15.6,15.6); ax.set_aspect('equal')
    ax.set_facecolor(BG); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color('#2c333c')
    if title: ax.set_title(title, color=FG, fontsize=10.5, pad=6)
    if res is not None:
        hd,nd=recovered(tok,res,0); hb,nb=recovered(tok,res,1)
        bb=dict(facecolor='#0f1419', edgecolor='#2c333c', boxstyle='square,pad=0.34', alpha=.92)
        ax.text(.975,.025, f"divider {hd}/{nd}\ncurb    {hb}/{nb}",
                transform=ax.transAxes, fontsize=8.4, family='monospace',
                color=PR_C[0], va='bottom', ha='right', linespacing=1.5, bbox=bb, zorder=6)
        # curb line recoloured when incomplete -- overlay just that line
        ax.text(.975,.025, f"divider {hd}/{nd}\ncurb    {hb}/{nb}".split(chr(10))[1],
                transform=ax.transAxes, fontsize=8.4, family='monospace',
                color=PR_C[1] if hb==nb else '#e8697c', va='bottom', ha='right', zorder=7,
                fontweight='bold' if hb<nb else 'normal')

n=len(PICKS)
fig,axes=plt.subplots(n,3,figsize=(11.6,3.85*n),facecolor=BG)
if n==1: axes=axes[None,:]
for r,tok in enumerate(PICKS):
    vs=gt_by[tok]['vectors']
    nd=sum(1 for v in vs if int(v.get('type',0))==0); nb=len(vs)-nd
    panel(axes[r,0], tok, None, 'ground truth' if r==0 else '')
    panel(axes[r,1], tok, RUNS['c50m_ORIG'], 'ordered L1' if r==0 else '')
    panel(axes[r,2], tok, RUNS['abl_C_lr05x'], 'monotone OT \u00b7 lr 3e-4' if r==0 else '')
    axes[r,0].set_ylabel(f"{nd} divider\n{nb} curb", color=MUT, fontsize=8.5,
                         rotation=0, ha='right', va='center', labelpad=30)
from matplotlib.lines import Line2D
leg=[Line2D([],[],color=GT_C[0],lw=4.6,label='GT divider'),
     Line2D([],[],color=GT_C[1],lw=4.6,label='GT curb'),
     Line2D([],[],color=PR_C[0],lw=2.0,label='predicted divider'),
     Line2D([],[],color=PR_C[1],lw=2.0,label='predicted curb'),
     Line2D([],[],marker='o',color=PT_C,lw=0,markersize=4,label='LiDAR')]
fig.legend(handles=leg, loc='upper center', ncol=5, frameon=False,
           bbox_to_anchor=(.5,.995), labelcolor=MUT, fontsize=9.5)
fig.tight_layout(rect=[0,0,1,.955])
fig.savefig(f'{OUT}/fig_preds3_best.png', dpi=132, facecolor=BG); plt.close(fig)
print('wrote fig_preds3_best.png %.2f MB'%(os.path.getsize(OUT+'/fig_preds3_best.png')/1e6))
