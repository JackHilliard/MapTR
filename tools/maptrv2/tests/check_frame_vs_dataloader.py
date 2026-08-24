"""Does the side view's z equal the z the dataloader feeds the model?

Compares, per tile:
  viewer   : features[:, 2] + tile_frame(block, 'tile_center').shift[2]
  loader   : features[:, 2] cut at z_max, then + lidar_recenter_shift[2]
and then how much of that survives lidar_point_cloud_range's z clipping in
the voxelizer, which the viewer does not draw at all.
"""
import os, pickle, sys, random
for _p in ('/home-local/johil9.nobkp/Documents/Code/MapTR/.claude/worktrees/'
           'reorder-results/tools/maptrv2', '/MapTR/tools/maptrv2'):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
        break
import numpy as np
import dataset_viewer as V

PKL = ('/home-local/johil9.nobkp/Documents/Code/MapTR/data/carla/'
       'carla_map_infos_test_30m_tc_3cls.pkl')
Z_MAX = 96.0
PC_Z = (-72.0, 96.0)

blob = pickle.load(open(PKL, 'rb'))
samples = blob['samples']
shifts = np.array([s['lidar_recenter_shift'] for s in samples], dtype=np.float64)
print(f'{len(samples)} samples')
print(f'lidar_recenter_shift z: min {shifts[:,2].min():.6g} '
      f'max {shifts[:,2].max():.6g}  '
      f'(nonzero on {int((shifts[:,2] != 0).sum())} tiles)')
print(f'                  x: |max| {np.abs(shifts[:,0]).max():.2f} m, '
      f'y: |max| {np.abs(shifts[:,1]).max():.2f} m')

DATA = '/home-local/johil9.nobkp/Documents/Code/carla_test'
V.STATE.update({'data_root': DATA, 'datasets': {}, 'class_lookup': {}})
ds = V.discover_datasets(DATA)
V.STATE['datasets'] = ds
_idx, _grp = V.build_index(ds)
V.STATE['tiles'], V.STATE['groups'] = _idx, _grp
V.STATE['tiles_by_uid'] = {t['_uid']: t for t in _idx}

random.seed(0)
sel = random.sample(samples, 40)
maxdiff = 0.0
cut_by_zmax = 0
out_of_range = 0
total = 0
worst = None
for s in sel:
    name = s['token']
    blk = V.load_block(name, 'test')
    if blk is None:
        continue
    feat = blk['features']
    origin, shift, _c = V.tile_frame(blk, 'tile_center', name, 'test')

    viewer_z = feat[:, 2] + shift[2]

    # what LoadCarlaPointsFromFile does, in ITS order
    keep = feat[:, 2] <= Z_MAX
    loader_z = feat[keep, 2] + np.asarray(s['lidar_recenter_shift'])[2]

    d3 = np.abs(np.asarray(shift, float)
                - np.asarray(s['lidar_recenter_shift'], float)).max()
    maxdiff = max(maxdiff, float(d3))
    cut_by_zmax += int((~keep).sum())
    n_out = int(((loader_z < PC_Z[0]) | (loader_z > PC_Z[1])).sum())
    out_of_range += n_out
    total += len(feat)
    frac = n_out / max(len(loader_z), 1)
    if worst is None or frac > worst[1]:
        worst = (name, frac, float(viewer_z.min()), float(viewer_z.max()))

print(f'\n40 tiles, {total:,} points')
print(f'max |viewer shift - pkl lidar_recenter_shift| (all 3 axes) = {maxdiff:.3g}')
print(f'points dropped by z_max={Z_MAX}: {cut_by_zmax:,} '
      f'({100*cut_by_zmax/total:.3f}%)')
print(f'points outside pc_range z {PC_Z}: {out_of_range:,} '
      f'({100*out_of_range/total:.3f}%)')
print(f'worst tile by out-of-range fraction: {worst[0]} {100*worst[1]:.3f}%, '
      f'viewer z {worst[2]:.2f}..{worst[3]:.2f}')
