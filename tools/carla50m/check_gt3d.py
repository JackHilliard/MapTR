import sys, numpy as np, warnings, torch
sys.path.insert(0,'/MapTR2'); warnings.filterwarnings('ignore')
from projects.mmdet3d_plugin.datasets.carla50m_crop_dataset import CustomCarla50mCropDataset
from projects.mmdet3d_plugin.datasets.av2_offlinemap_dataset import LiDARInstanceLines
ROOT='/data_carla50'
ds=CustomCarla50mCropDataset(
    data_root=ROOT, ann_file=f'{ROOT}/test/manifest.json', split='test',
    pipeline=[], classes=[], map_classes=['divider'], gt_classes=('driving',),
    pc_range=[-15.,-15.,-2.,15.,15.,24.], fixed_ptsnum_per_line=20, code_size=2,
    bev_size=(120,120), eval_use_same_gt_sample_num_flag=True, min_lidar_points=1,
    lidar_pc_range=[-15.,-15.,-72.,15.,15.,96.],
    modality=dict(use_lidar=True,use_camera=False),
    aux_seg=dict(use_aux_seg=True,bev_seg=True,pv_seg=False,seg_classes=1,
                 feat_down_sample=32,pv_thickness=1),
    rotate=False, translate=False, noise_std=0.0, require_gt=False,
    test_mode=True, filter_empty_gt=False)
print(f"built: {len(ds)} test samples")
shapes=set(); zr=[]
for i in range(50):
    for g in ds.data_infos[i]['annotation']['divider']:
        shapes.add(g.shape[1]); zr.append((g[:,2].min(), g[:,2].max()))
print(f"  GT column counts seen: {shapes}  (must be {{3}})")
z=np.array(zr); print(f"  GT z after re-zero: min {z[:,0].min():.2f}  max {z[:,1].max():.2f}  (pc_range z is [-2, 24])")
# the exact call that crashed
ex = ds.vectormap_pipeline({}, ds.data_infos[0])
t = ex['gt_bboxes_3d'].data.fixed_num_sampled_points
print(f"  fixed_num_sampled_points OK -> {tuple(t.shape)}")
print(f"  xy range: [{t[:,:,:2].min():.2f}, {t[:,:,:2].max():.2f}]  z range: [{t[:,:,2].min():.2f}, {t[:,:,2].max():.2f}]")
print(f"  labels: {ex['gt_labels_3d'].data.tolist()[:8]}")
