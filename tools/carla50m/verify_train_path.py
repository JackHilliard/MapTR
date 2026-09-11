"""DOUBLE CHECK: verify crops+rotation on the tensors the MODEL receives,
through the real config, real pipeline, real dataloader -- not a hand-built probe."""
import sys, warnings, numpy as np, torch
sys.path.insert(0,'/MapTR2'); warnings.filterwarnings('ignore')
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet.datasets import build_dataloader
import projects.mmdet3d_plugin  # noqa: F401

CFG='/MapTR2/projects/configs/maptrv2/maptrv2_carla50m_ORIG.py'
cfg=Config.fromfile(CFG)
print("="*80); print("0. CONFIG ACTUALLY USED BY THE RUNNING JOB".center(80)); print("="*80)
tr_cfg=cfg.data.train
print(f"  type={tr_cfg['type']}  split={tr_cfg.get('split')}")
print(f"  rotate={tr_cfg.get('rotate')} translate={tr_cfg.get('translate')} "
      f"noise_std={tr_cfg.get('noise_std')} crop_size={tr_cfg.get('crop_size')} "
      f"crop_seed={tr_cfg.get('crop_seed','<none>')}")
print(f"  pipeline: {[s['type'] for s in tr_cfg['pipeline']]}")
print(f"  pc_range(dataset)={tr_cfg['pc_range']}   fixed_ptsnum={tr_cfg['fixed_ptsnum_per_line']}")
va=cfg.data.val
print(f"  VAL rotate={va.get('rotate')} translate={va.get('translate')} "
      f"noise_std={va.get('noise_std')} crop_seed={va.get('crop_seed')}")

ds=build_dataset(cfg.data.train)
print(f"\n  train dataset built: {len(ds)} samples")

print("\n"+"="*80); print("1. THROUGH THE REAL PIPELINE (post GridSample / FormatBundle)".center(80)); print("="*80)
rng=np.random.default_rng(11)
al=[]; nb_pts=[]; nb_gt=[]; outside_gt=0; ptsnum=set()
for i in rng.choice(len(ds), 30, replace=False):
    ex = ds[int(i)]                      # <-- the exact __getitem__ training uses
    if ex is None:                       # filter_empty_gt dropped it
        continue
    P = ex['points'].data.numpy()        # after GridSamplePoints
    G = ex['gt_bboxes_3d'].data.fixed_num_sampled_points.numpy()  # what the loss sees
    nb_pts.append(len(P)); nb_gt.append(len(G)); ptsnum.add(G.shape[1])
    if np.abs(G[:,:,:2]).max() > 15.0+1e-3: outside_gt += 1
    V=G.reshape(-1,G.shape[-1])[:,:2]
    d=np.linalg.norm(V[:,None,:]-P[None,:,:2],axis=-1).min(1)
    al.append(d)
al=np.concatenate(al)
print(f"  samples inspected: {len(nb_pts)}   points/sample median {int(np.median(nb_pts)):,}")
print(f"  GT instances/sample median {np.median(nb_gt):.1f}   points per instance: {ptsnum} (fixed_ptsnum=20)")
print(f"  GT vertex -> nearest VOXELIZED point: median {np.median(al):.4f} m  "
      f"p95 {np.percentile(al,95):.4f}  max {al.max():.4f}")
print(f"  GT outside the +-15 m box: {outside_gt}/{len(nb_pts)}")
print(f"  VERDICT: {'ALIGNED through the full pipeline' if np.median(al)<0.25 else 'MISALIGNED'}")

print("\n"+"="*80); print("2. AUGMENTATION IS LIVE ACROSS EPOCHS (same index, repeated)".center(80)); print("="*80)
sig=[]
for _ in range(6):
    ex=ds[7]
    if ex is None: continue
    G=ex['gt_bboxes_3d'].data.fixed_num_sampled_points.numpy()
    sig.append((len(G), float(G[:,:,0].mean()), float(G[:,:,1].mean())))
uniq=len(set(sig))
print(f"  index 7 fetched 6x -> {uniq} distinct GT layouts")
for s in sig: print(f"     n_gt={s[0]}  mean xy=({s[1]:+.3f},{s[2]:+.3f})")
print(f"  VERDICT: {'RE-DRAWN each access' if uniq>1 else 'STATIC -- augmentation NOT live'}")

print("\n"+"="*80); print("3. VAL IS FROZEN (same index, repeated)".center(80)); print("="*80)
dv=build_dataset(cfg.data.val)
a=dv.data_infos[3]['annotation']['divider']; b=dv.data_infos[3]['annotation']['divider']
i1=dv.get_data_info(3); i2=dv.get_data_info(3)
same=np.array_equal(i1['crop_R'],i2['crop_R']) and np.array_equal(i1['crop_shift'],i2['crop_shift'])
print(f"  val pose identical across get_data_info calls: {same}")
print(f"  val yaw = 0: {np.allclose(i1['crop_R'], np.eye(2))}")
print(f"  val n_gt at idx3: {len(a)}")
