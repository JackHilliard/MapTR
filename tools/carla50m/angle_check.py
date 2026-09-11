"""Quantitative rotation check: for each crop, the angle of every GT polyline
in the CROP frame must equal its world angle + yaw (mod 180)."""
import sys, warnings, numpy as np
sys.path.insert(0,'/MapTR2'); warnings.filterwarnings('ignore')
from projects.mmdet3d_plugin.datasets.carla50m_crop_dataset import CustomCarla50mCropDataset
ROOT='/data_carla50'
ds = CustomCarla50mCropDataset(
    data_root=ROOT, ann_file=f'{ROOT}/train/manifest.json', split='train',
    pipeline=[], classes=[], map_classes=['divider'], gt_classes=('driving',),
    pc_range=[-15.,-15.,-2.,15.,15.,24.], fixed_ptsnum_per_line=20, code_size=2,
    bev_size=(120,120), eval_use_same_gt_sample_num_flag=True, min_lidar_points=1,
    lidar_pc_range=[-15.,-15.,-72.,15.,15.,96.],
    modality=dict(use_lidar=True,use_camera=False),
    aux_seg=dict(use_aux_seg=True,bev_seg=True,pv_seg=False,seg_classes=1,
                 feat_down_sample=32,pv_thickness=1),
    test_mode=False, filter_empty_gt=False)

def ang(p):
    """Principal direction of a polyline (deg, mod 180) via total displacement."""
    d = p[-1,:2] - p[0,:2]
    return np.degrees(np.arctan2(d[1], d[0])) % 180.0

rng = np.random.default_rng(3); errs = []
for i in rng.choice(len(ds), 120, replace=False):
    info = ds.get_data_info(int(i))
    R = info['crop_R']; yaw = np.degrees(np.arctan2(R[1,0], R[0,0]))
    centre = info['crop_centre']; shift = info['crop_shift']
    for g in info['annotation']['divider']:
        if len(g) < 2: continue
        a_crop = ang(g)
        # map the crop polyline back to world and measure there
        w = g[:, :2] @ R + shift @ R + centre     # inverse of (w-centre)@R.T - shift
        a_world = ang(w)
        d = (a_crop - (a_world + yaw)) % 180.0
        errs.append(min(d, 180.0 - d))
errs = np.array(errs)
print(f"  GT polylines checked: {len(errs)}")
print(f"  |crop_angle - (world_angle + yaw)| mod 180:")
print(f"     max {errs.max():.6f} deg   mean {errs.mean():.6f}   p99 {np.percentile(errs,99):.6f}")
print(f"  VERDICT: {'ROTATION EXACT' if errs.max() < 1e-3 else 'ROTATION INCONSISTENT'}")
