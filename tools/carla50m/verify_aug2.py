import sys, numpy as np, warnings
MJ='/MapTR2'; sys.path.insert(0,MJ); warnings.filterwarnings('ignore')
from projects.mmdet3d_plugin.datasets.carla50m_crop_dataset import (
    CustomCarla50mCropDataset, LoadCarla50mCrop)
ROOT='/data_carla50'
def build(split,**kw):
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

tr=build('train'); ld=LoadCarla50mCrop()
rng=np.random.default_rng(1)

print("A. HOW FAR outside +-15 m, and is it the post-crop noise?")
ex=[]
for i in rng.choice(len(tr),15,replace=False):
    r=dict(tr.get_data_info(int(i))); ld(r); P=r['points'].tensor.numpy()
    ex.append(np.abs(P[:,:2]).max()-15.0)
ex=np.array(ex)
print(f"   max overshoot per crop: median {np.median(ex):.4f} m  max {ex.max():.4f} m")
print(f"   noise_std={tr.noise_std} -> 4 sigma = {4*tr.noise_std:.3f} m")
print(f"   VERDICT: {'post-crop Gaussian noise, as in the reference' if ex.max()<6*tr.noise_std else 'REAL BUG'}")
tr0=build('train', noise_std=0.0)
r=dict(tr0.get_data_info(0)); LoadCarla50mCrop()(r); P=r['points'].tensor.numpy()
print(f"   same check with noise_std=0: max |xy| = {np.abs(P[:,:2]).max():.6f} m (must be <=15)")

print("\nB. EMPTY-GT RATE (filter_empty_gt would drop these in training)")
n=200; empt=0; ngt=[]
for i in rng.choice(len(tr),n,replace=False):
    g=tr.get_data_info(int(i))['annotation']['divider']; ngt.append(len(g)); empt+= (len(g)==0)
print(f"   empty crops: {empt}/{n} ({100*empt/n:.1f}%)   GT lines/crop mean {np.mean(ngt):.2f} max {max(ngt)}")

print("\nC. POINTS PER CROP (batch-size / memory planning)")
np_=[]
for i in rng.choice(len(tr),25,replace=False):
    r=dict(tr.get_data_info(int(i))); ld(r); np_.append(len(r['points'].tensor))
np_=np.array(np_)
print(f"   median {int(np.median(np_)):,}  mean {int(np_.mean()):,}  p90 {int(np.percentile(np_,90)):,}  max {int(np_.max()):,}")

print("\nD. GT LENGTH SANITY (clip should not leave stubs)")
L=[]
for i in rng.choice(len(tr),60,replace=False):
    for g in tr.get_data_info(int(i))['annotation']['divider']:
        L.append(np.linalg.norm(np.diff(g[:,:2],axis=0),axis=1).sum())
L=np.array(L)
print(f"   polyline length: min {L.min():.2f} m (min_polyline_len={tr.min_polyline_len})  "
      f"median {np.median(L):.2f}  max {L.max():.2f}")
print(f"   any below threshold: {(L < tr.min_polyline_len - 1e-6).sum()}")
