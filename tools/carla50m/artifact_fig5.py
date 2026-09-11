"""Curved tiles across the full performance range: best AND worst."""
import sys, os, json, glob, warnings, numpy as np
sys.path.insert(0,'/MapTR2'); sys.path.insert(0,'/MapTR2/projects/mmdet3d_plugin/datasets')
warnings.filterwarnings('ignore')
from mmcv import Config
from mmdet3d.datasets import build_dataset
import projects.mmdet3d_plugin  # noqa
from projects.mmdet3d_plugin.datasets.carla50m_crop_dataset import LoadCarla50mCrop
from carla50m_metrics import _resample as _rs
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from matplotlib.lines import Line2D

OUT='/MapTR2/tools/carla50m/figs'
cfg=Config.fromfile('/MapTR2/projects/configs/maptrv2/maptrv2_carla50m_ORIG.py')
ds=build_dataset(cfg.data.val); idx={i['sample_idx']:k for k,i in enumerate(ds.data_infos)}
loader=LoadCarla50mCrop()
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
        if L<1e-6: continue
        b=max(b,float(np.abs(np.cross(ae/L,p-a)).max()))
    return b
def recovered(tok,res,ci,thr=1.5):
    gv=[v for v in gt_by[tok]['vectors'] if int(v.get('type',0))==ci]
    if not gv: return None,0
    pv=[v for v in res[tok]['vectors'] if int(v.get('type',0))==ci
        and float(v['confidence_level'])>0.4 and len(v['pts'])>=2]
    if not pv: return 0,len(gv)
    P=np.stack([_rs(v['pts']) for v in pv]); G=np.stack([_rs(v['pts']) for v in gv])
    d=np.linalg.norm(P[:,None,:,None,:]-G[None,:,None,:,:],axis=-1)
    ch=0.5*(d.min(3).mean(2)+d.min(2).mean(2)); arg=ch.argmin(1); best=ch.min(1)
    sc=np.array([float(v['confidence_level']) for v in pv]); taken=set(); hit=0
    for i in np.argsort(-sc):
        if best[i]<=thr and arg[i] not in taken: taken.add(int(arg[i])); hit+=1
    return hit,len(gv)

# all curved tiles (>2 m deviation), scored by total recall across both classes
scored=[]
for tok in list(gt_by)[:1400]:
    if tok not in R['c50m_ORIG']: continue
    vs=gt_by[tok]['vectors']
    if len(vs)<3 or curvature(vs)<2.0: continue
    tot=0; n=0
    for ci in (0,1):
        for run in ('c50m_ORIG','c50m_EMDV2'):
            h,g=recovered(tok,R[run],ci)
            if h is not None: tot+=h; n+=g
    if n: scored.append((tot/n, curvature(vs), len(vs), tok))
scored.sort(reverse=True)
picks=[(t,'best') for _,_,_,t in scored[:2]] + [(t,'worst') for _,_,_,t in scored[-2:]]
print('curved tiles found:', len(scored))
print(' best :', [f"{s:.2f}" for s,_,_,_ in scored[:2]])
print(' worst:', [f"{s:.2f}" for s,_,_,_ in scored[-2:]])
print(f' recall range across curved tiles: {scored[-1][0]*100:.0f}% .. {scored[0][0]*100:.0f}%')
json.dump({'range':[scored[-1][0],scored[0][0]],'n':len(scored)}, open(f'{OUT}/curve_range.json','w'))

BG='#0f1419';FG='#eef1f4';MUT='#8b95a3'
GT_C={0:'#37514c',1:'#544733'}; PR_C={0:'#4dd4c0',1:'#f2a65a'}; PT='#39434e'
def panel(ax,tok,res,title=''):
    i=idx.get(tok)
    if i is not None:
        r=dict(ds.get_data_info(i)); loader(r); P=r['points'].tensor.numpy()
        if len(P):
            s=np.random.default_rng(0).choice(len(P),min(55000,len(P)),replace=False)
            ax.scatter(P[s,0],P[s,1],s=.09,c=PT,linewidths=0,zorder=0)
    for v in gt_by[tok]['vectors']:
        p=np.asarray(v['pts'])[:,:2]
        ax.plot(p[:,0],p[:,1],c=GT_C[int(v.get('type',0))],lw=4.2,alpha=.95,zorder=1,solid_capstyle='round')
    if res is not None:
        for v in res[tok]['vectors']:
            if float(v['confidence_level'])<=0.4 or len(v['pts'])<2: continue
            p=np.asarray(v['pts'])[:,:2]
            ax.plot(p[:,0],p[:,1],c=PR_C[int(v.get('type',0))],lw=1.9,zorder=3)
        hd,nd=recovered(tok,res,0); hb,nb=recovered(tok,res,1)
        bb=dict(facecolor='#0f1419',edgecolor='#2c333c',boxstyle='square,pad=0.34',alpha=.92)
        txt=f"divider {hd}/{nd}"+(f"\ncurb    {hb}/{nb}" if nb else "")
        ax.text(.975,.025,txt,transform=ax.transAxes,fontsize=8.2,family='monospace',
                color=PR_C[0],va='bottom',ha='right',linespacing=1.5,bbox=bb,zorder=6)
        if nb:
            ax.text(.975,.025,f"curb    {hb}/{nb}",transform=ax.transAxes,fontsize=8.2,family='monospace',
                    color=PR_C[1] if hb==nb else '#e8697c',va='bottom',ha='right',zorder=7,
                    fontweight='bold' if hb<nb else 'normal')
    ax.add_patch(MplPoly([[-15,-15],[15,-15],[15,15],[-15,15]],closed=True,fill=False,ec='#2c333c',lw=1.2))
    ax.set_xlim(-15.6,15.6);ax.set_ylim(-15.6,15.6);ax.set_aspect('equal')
    ax.set_facecolor(BG);ax.set_xticks([]);ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_color('#2c333c')
    if title: ax.set_title(title,color=FG,fontsize=10.5,pad=6)

fig,axes=plt.subplots(4,3,figsize=(11.6,15.4),facecolor=BG)
for r,(tok,kind) in enumerate(picks):
    panel(axes[r,0],tok,None,'ground truth' if r==0 else '')
    panel(axes[r,1],tok,R['c50m_ORIG'],'ordered L1' if r==0 else '')
    panel(axes[r,2],tok,R['c50m_EMDV2'],'monotone OT' if r==0 else '')
    vs=gt_by[tok]['vectors']; nd=sum(1 for v in vs if int(v.get('type',0))==0)
    col='#4dd4c0' if kind=='best' else '#e8697c'
    axes[r,0].set_ylabel(f"{kind}\n{nd} divider\n{len(vs)-nd} curb\ncurv {curvature(vs):.1f} m",
                         color=col,fontsize=8.5,rotation=0,ha='right',va='center',labelpad=30)
leg=[Line2D([],[],color=GT_C[0],lw=4.6,label='GT divider'),Line2D([],[],color=GT_C[1],lw=4.6,label='GT curb'),
     Line2D([],[],color=PR_C[0],lw=2.0,label='predicted divider'),Line2D([],[],color=PR_C[1],lw=2.0,label='predicted curb'),
     Line2D([],[],marker='o',color=PT,lw=0,markersize=4,label='LiDAR')]
fig.legend(handles=leg,loc='upper center',ncol=5,frameon=False,bbox_to_anchor=(.5,.994),labelcolor=MUT,fontsize=9.5)
fig.tight_layout(rect=[0,0,1,.968])
fig.savefig(f'{OUT}/fig_curves.png',dpi=128,facecolor=BG);plt.close(fig)
print('wrote fig_curves.png %.2f MB'%(os.path.getsize(OUT+'/fig_curves.png')/1e6))
