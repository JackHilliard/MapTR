import sys, numpy as np, warnings
sys.path.insert(0,'/MapTR2'); warnings.filterwarnings('ignore')
from projects.mmdet3d_plugin.datasets.carla50m_crop_dataset import CustomCarla50mCropDataset
ROOT='/data_carla50'
def build(**kw):
    return CustomCarla50mCropDataset(
        data_root=ROOT, ann_file=f'{ROOT}/test/manifest.json', split='test',
        pipeline=[], classes=[], map_classes=['divider'], gt_classes=('driving',),
        pc_range=[-15.,-15.,-2.,15.,15.,24.], fixed_ptsnum_per_line=20, code_size=2,
        bev_size=(120,120), eval_use_same_gt_sample_num_flag=True, min_lidar_points=1,
        lidar_pc_range=[-15.,-15.,-72.,15.,15.,96.],
        modality=dict(use_lidar=True,use_camera=False),
        aux_seg=dict(use_aux_seg=True,bev_seg=True,pv_seg=False,seg_classes=1,
                     feat_down_sample=32,pv_thickness=1),
        test_mode=True, filter_empty_gt=False, **kw)

def rate(tag, **kw):
    ds=build(**kw)
    n=[len(i['annotation']['divider']) if 'annotation' in i
       else len(ds.get_data_info(k)['annotation']['divider'])
       for k,i in enumerate(ds.data_infos)]
    e=sum(1 for x in n if x==0)
    print(f"   {tag:52s} empty {e:4d}/{len(n)} ({100*e/len(n):5.1f}%)  mean GT/tile {np.mean(n):.2f}")
    return ds
rate("A. centred, no aiming (was configured)", rotate=False, translate=False, noise_std=0.0, require_gt=False)
rate("B. seeded aim, no yaw  (proposed eval)", rotate=False, translate=True, noise_std=0.0, require_gt=True, crop_seed=0)
rate("C. seeded aim + yaw", rotate=True, translate=True, noise_std=0.0, require_gt=True, crop_seed=0)

print("\n   reproducibility of B across two builds:")
d1=build(rotate=False, translate=True, noise_std=0.0, require_gt=True, crop_seed=0)
d2=build(rotate=False, translate=True, noise_std=0.0, require_gt=True, crop_seed=0)
same=all(np.array_equal(a['crop_shift'],b['crop_shift']) and
         len(a['annotation']['divider'])==len(b['annotation']['divider'])
         for a,b in zip(d1.data_infos,d2.data_infos))
print(f"   identical across independent builds: {same}")
print(f"   max_shift with yaw off: {d1.max_shift:.2f} m (3.79 m when yaw is on)")
