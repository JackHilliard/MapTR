"""Verify the 50m->30m crop augmentation: alignment, coverage, determinism."""
import sys, types, importlib, numpy as np
MJ='/MapTR2'; sys.path.insert(0, MJ)
import warnings; warnings.filterwarnings('ignore')

from projects.mmdet3d_plugin.datasets.carla50m_crop_dataset import (
    CustomCarla50mCropDataset, LoadCarla50mCrop)

ROOT='/data_carla50'
PC=[-15.0,-15.0,-2.0,15.0,15.0,24.0]   # dataset-block pc_range, as in maptrv2_carla_ORIG.py

def build(split, **kw):
    return CustomCarla50mCropDataset(
        data_root=ROOT, ann_file=f'{ROOT}/{split}/manifest.json', split=split,
        pipeline=[], classes=[], map_classes=['divider'], gt_classes=('driving',),
        pc_range=PC, fixed_ptsnum_per_line=20, code_size=2, bev_size=(120,120),
        eval_use_same_gt_sample_num_flag=True, min_lidar_points=1,
        lidar_pc_range=[-15.0,-15.0,-72.0,15.0,15.0,96.0],
        modality=dict(use_lidar=True, use_camera=False),
        aux_seg=dict(use_aux_seg=True, bev_seg=True, pv_seg=False,
                     seg_classes=1, feat_down_sample=32, pv_thickness=1),
        test_mode=(split!='train'), filter_empty_gt=False, **kw)

print("="*78); print("1. BUILD".center(78)); print("="*78)
tr = build('train')
te = build('test', rotate=False, translate=False, noise_std=0.0)
print(f"  train: {len(tr)} samples   test: {len(te)} samples")
print(f"  train max_shift={tr.max_shift:.3f} m   need={tr._need:.3f} m (half*sqrt2)")
print(f"  test  max_shift={te.max_shift:.3f} m   need={te._need:.3f} m (no rotation)")
print(f"  test has precomputed annotation: {'annotation' in te.data_infos[0]}")
print(f"  train has precomputed annotation: {'annotation' in tr.data_infos[0]} (must be False)")

loader = LoadCarla50mCrop()

print("\n"+"="*78); print("2. POINT/GT ALIGNMENT  (the invariant that matters)".center(78)); print("="*78)
print("  GT centrelines lie ON the road, so if points and GT received the same")
print("  pose, every GT vertex sits close to LiDAR returns. A decoupled rotation")
print("  would scatter them metres apart.")
rng=np.random.default_rng(0)
d_all=[]; occ=[]; npts=[]; ngt=[]
idxs=rng.choice(len(tr), 40, replace=False)
for i in idxs:
    info = tr.get_data_info(int(i))
    res = dict(info); loader(res)
    P = res['points'].tensor.numpy()
    npts.append(len(P))
    G = info['annotation']['divider']
    ngt.append(len(G))
    if len(P)==0 or not G: continue
    # nearest LiDAR point for each GT vertex (xy)
    V=np.concatenate(G,0)
    d=np.linalg.norm(V[:,None,:2]-P[None,:,:2],axis=-1).min(1)
    d_all.append(d)
    # quadrant occupancy -> crop fully populated, no empty corners
    q=[((P[:,0]>0)&(P[:,1]>0)),((P[:,0]<=0)&(P[:,1]>0)),
       ((P[:,0]>0)&(P[:,1]<=0)),((P[:,0]<=0)&(P[:,1]<=0))]
    occ.append(sum(1 for m in q if m.sum()>0))
d=np.concatenate(d_all)
print(f"  GT vertex -> nearest LiDAR point, over {len(idxs)} random train crops:")
print(f"     median {np.median(d):.3f} m | p90 {np.percentile(d,90):.3f} | max {d.max():.3f}")
print(f"     fraction > 1.0 m: {100*(d>1.0).mean():.2f}%   > 2.0 m: {100*(d>2.0).mean():.2f}%")
print(f"  VERDICT: {'ALIGNED' if np.median(d)<0.5 else 'MISALIGNED -- points and GT disagree'}")
print(f"\n  crop occupancy: all 4 quadrants populated in {sum(1 for o in occ if o==4)}/{len(occ)} crops")
print(f"  points/crop: median {int(np.median(npts)):,}  min {min(npts):,}  max {max(npts):,}")
print(f"  GT lines/crop: median {np.median(ngt):.1f}  min {min(ngt)}  max {max(ngt)}  empty: {sum(1 for g in ngt if g==0)}")

print("\n"+"="*78); print("3. EXTENT: everything inside the 30 m box".center(78)); print("="*78)
bad_p=bad_g=0
for i in idxs[:20]:
    info=tr.get_data_info(int(i)); res=dict(info); loader(res)
    P=res['points'].tensor.numpy()
    if len(P) and (np.abs(P[:,:2]).max() > 15.0+1e-3): bad_p+=1
    for g in info['annotation']['divider']:
        if np.abs(g[:,:2]).max() > 15.0+1e-3: bad_g+=1
print(f"  crops with points outside +-15 m: {bad_p}/20")
print(f"  GT polylines outside +-15 m     : {bad_g}")

print("\n"+"="*78); print("4. ROTATION really is uniform".center(78)); print("="*78)
th=[tr.get_data_info(int(i))['crop_R'] for i in rng.choice(len(tr),300,replace=False)]
ang=np.array([np.arctan2(R[1,0],R[0,0]) for R in th])
ang=(ang+2*np.pi)%(2*np.pi)
h,_=np.histogram(ang,bins=8,range=(0,2*np.pi))
print(f"  yaw histogram over 300 draws (8 bins, expect ~37.5 each): {h.tolist()}")
print(f"  min {ang.min():.2f} max {ang.max():.2f} rad   mean {ang.mean():.2f} (expect ~pi={np.pi:.2f})")

print("\n"+"="*78); print("5. EVAL DETERMINISM".center(78)); print("="*78)
a=[te.get_data_info(i) for i in range(5)]
b=[te.get_data_info(i) for i in range(5)]
same=all(np.array_equal(x['crop_R'],y['crop_R']) and
         len(x['annotation']['divider'])==len(y['annotation']['divider']) and
         all(np.array_equal(u,v) for u,v in zip(x['annotation']['divider'],y['annotation']['divider']))
         for x,y in zip(a,b))
print(f"  test get_data_info identical across calls: {same}")
tr_a=tr.get_data_info(0)['crop_R']; tr_b=tr.get_data_info(0)['crop_R']
print(f"  train get_data_info varies across calls  : {not np.array_equal(tr_a,tr_b)} (must be True)")
print(f"  test yaw is identity                     : {np.allclose(a[0]['crop_R'],np.eye(2))}")
