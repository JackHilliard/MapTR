"""Render the 50m->30m crop augmentation: world view (tile + rotated crop box)
next to the crop the network actually receives."""
import sys, os, warnings, numpy as np
sys.path.insert(0,'/MapTR2'); warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from projects.mmdet3d_plugin.datasets.carla50m_crop_dataset import (
    CustomCarla50mCropDataset, LoadCarla50mCrop)

ROOT='/data_carla50'; OUT='/MapTR2/tools/carla50m/figs'; os.makedirs(OUT, exist_ok=True)
def build(split, **kw):
    return CustomCarla50mCropDataset(
        data_root=ROOT, ann_file=f'{ROOT}/{split}/manifest.json', split=split,
        pipeline=[], classes=[], map_classes=['divider'], gt_classes=('driving',),
        pc_range=[-15.,-15.,-2.,15.,15.,24.], fixed_ptsnum_per_line=20, code_size=2,
        bev_size=(120,120), eval_use_same_gt_sample_num_flag=True, min_lidar_points=1,
        lidar_pc_range=[-15.,-15.,-72.,15.,15.,96.],
        modality=dict(use_lidar=True,use_camera=False),
        aux_seg=dict(use_aux_seg=True,bev_seg=True,pv_seg=False,seg_classes=1,
                     feat_down_sample=32,pv_thickness=1),
        test_mode=(split!='train'), filter_empty_gt=False, **kw)

tr = build('train')
ld = LoadCarla50mCrop()
BG='#0f1318'; FG='#eef1f4'; MUT='#8b95a3'; GT_C='#3fbfae'; BOX='#e8697c'

def world_panel(ax, name, centre, R, shift, half=15.0, ds=None):
    ds = ds if ds is not None else tr
    with np.load(os.path.join(ds.blocks_dir, f'{name}.npz')) as d:
        W = d['points'].astype(np.float64)
    sel = np.random.default_rng(0).choice(len(W), min(40000,len(W)), replace=False)
    ax.scatter(W[sel,0]-centre[0], W[sel,1]-centre[1], s=.12, c='#4a5568', linewidths=0)
    for pl in ds._polys[name]:
        p = pl['points']
        ax.plot(p[:,0]-centre[0], p[:,1]-centre[1], c=GT_C, lw=1.6, alpha=.95)
    # crop box: corners in crop frame -> world-relative
    c = np.array([[-half,-half],[half,-half],[half,half],[-half,half]])
    wc = (c + shift) @ R                       # inverse of (x-centre)@R.T - shift
    ax.add_patch(MplPoly(wc, closed=True, fill=False, ec=BOX, lw=2.0))
    a = (np.array([[0,0],[half*.7,0]]) + shift) @ R
    ax.annotate('', xy=a[1], xytext=a[0], arrowprops=dict(arrowstyle='->', color=BOX, lw=1.8))
    ax.set_xlim(-26,26); ax.set_ylim(-26,26); ax.set_aspect('equal')

def crop_panel(ax, info):
    r = dict(info); ld(r); P = r['points'].tensor.numpy()
    sel = np.random.default_rng(0).choice(len(P), min(40000,len(P)), replace=False)
    ax.scatter(P[sel,0], P[sel,1], s=.12, c='#4a5568', linewidths=0)
    for g in info['annotation']['divider']:
        ax.plot(g[:,0], g[:,1], c=GT_C, lw=2.0)
        ax.scatter(g[:,0], g[:,1], s=7, c=GT_C, zorder=3)
    ax.add_patch(MplPoly([[-15,-15],[15,-15],[15,15],[-15,15]], closed=True,
                         fill=False, ec=BOX, lw=1.6))
    ax.set_xlim(-16.5,16.5); ax.set_ylim(-16.5,16.5); ax.set_aspect('equal')
    return len(P), len(info['annotation']['divider'])

def style(ax, title, sub=''):
    ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_color('#2c333c')
    ax.tick_params(colors=MUT, labelsize=7)
    ax.set_title(title, color=FG, fontsize=10, pad=6)
    if sub: ax.text(.5,-.11, sub, transform=ax.transAxes, ha='center',
                    color=MUT, fontsize=7.5)

if os.environ.get('ONLY_FIG3'): pass
if not os.environ.get('ONLY_FIG3'):
    # ---- FIG 1: four tiles, world (tile + rotated box) vs crop ----
    rng = np.random.default_rng(7)
    idxs = [int(i) for i in rng.choice(len(tr), 4, replace=False)]
    fig, axes = plt.subplots(2, 4, figsize=(17, 9), facecolor=BG)
    for k, i in enumerate(idxs):
        info = tr.get_data_info(i); name = info['sample_idx']
        th = float(np.arctan2(info['crop_R'][1,0], info['crop_R'][0,0]))
        world_panel(axes[0,k], name, info['crop_centre'], info['crop_R'], info['crop_shift'])
        style(axes[0,k], f'{name[:26]}', f'50 m tile · yaw {np.degrees(th):.0f}° · shift {np.linalg.norm(info["crop_shift"]):.2f} m')
        n,g = crop_panel(axes[1,k], info)
        style(axes[1,k], '30 m crop fed to the network', f'{n:,} pts · {g} GT lines')
    fig.suptitle('50 m tile (top, red = rotated 30 m crop box)  ->  crop the model receives (bottom)',
                 color=FG, fontsize=13, y=.97)
    fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(f'{OUT}/fig1_crops.png', dpi=115, facecolor=BG); plt.close(fig)
    print('fig1 done')

    # ---- FIG 2: same tile, six independent draws ----
    i = idxs[0]
    fig, axes = plt.subplots(2, 6, figsize=(21, 7.4), facecolor=BG)
    for k in range(6):
        info = tr.get_data_info(i); name = info['sample_idx']
        th = float(np.arctan2(info['crop_R'][1,0], info['crop_R'][0,0]))
        world_panel(axes[0,k], name, info['crop_centre'], info['crop_R'], info['crop_shift'])
        style(axes[0,k], f'draw {k+1}', f'yaw {np.degrees(th):.0f}°')
        n,g = crop_panel(axes[1,k], info)
        style(axes[1,k], '', f'{n:,} pts · {g} GT')
    fig.suptitle(f'Same tile, six independent draws — every epoch sees a different crop  ({name})',
                 color=FG, fontsize=13, y=.97)
    fig.tight_layout(rect=[0,0,1,.93]); fig.savefig(f'{OUT}/fig2_draws.png', dpi=115, facecolor=BG); plt.close(fig)
    print('fig2 done')


# ---- FIG 3: train (random) vs eval (frozen seed) ----
te = build('test', rotate=False, translate=True, noise_std=0.0, require_gt=True, crop_seed=0)
fig, axes = plt.subplots(2, 4, figsize=(17, 9), facecolor=BG)
for k in range(4):
    info = te.get_data_info(k)
    world_panel(axes[0,k], info['sample_idx'], info['crop_centre'], info['crop_R'], info['crop_shift'], ds=te)
    style(axes[0,k], f'{info["sample_idx"][:26]}', f'eval · yaw 0° · shift {np.linalg.norm(info["crop_shift"]):.2f} m')
    n,g = crop_panel(axes[1,k], info)
    style(axes[1,k], 'frozen eval crop', f'{n:,} pts · {g} GT lines')
fig.suptitle('Eval crops: frozen by seed (identical every run), aimed at GT — not centred',
             color=FG, fontsize=13, y=.97)
fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(f'{OUT}/fig3_eval.png', dpi=115, facecolor=BG); plt.close(fig)
print('fig3 done')
for f in ('fig1_crops','fig2_draws','fig3_eval'):
    print(f'  {f}.png  {os.path.getsize(f"{OUT}/{f}.png")/1e6:.2f} MB')
