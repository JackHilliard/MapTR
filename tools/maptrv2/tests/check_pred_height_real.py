import os, sys
for _p in ('/home-local/johil9.nobkp/Documents/Code/MapTR/.claude/worktrees/'
           'reorder-results/tools/maptrv2', '/MapTR/tools/maptrv2'):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
        break
import numpy as np
import dataset_viewer as V

DATA = '/home-local/johil9.nobkp/Documents/Code/carla_test'
WD = '/results'
V.STATE.update({
    'data_root': DATA, 'max_points': 150000, 'work_dir': WD,
    'results_cache': {}, 'gt_cache': {}, 'shape_cache': {}, 'frame_cache': {},
    'frame': 'auto', 'gt_json': None, 'deep': {},
    'scan_grid': (0.1, 0.1, 0.4), 'scan_workers': 2, 'scan_stride': 1,
    'pc_range_z': (-72.0, 96.0), 'num_pts_per_vec': 20,
    'cache_dir': '/tmpx/realcache', 'tags_file': None,
})
ds = V.discover_datasets(DATA)
V.STATE['datasets'] = ds
tiles, groups = V.build_index(ds)
V.STATE['tiles'], V.STATE['groups'] = tiles, groups
V.STATE['tiles_by_uid'] = {t['_uid']: t for t in tiles}
V.STATE['lane_types'] = V.merged_lookup(ds, 'lane_type_lookup')
V.STATE['class_lookup'] = V.merged_lookup(ds, 'class_lookup')
V.STATE['class_choices'] = V.class_choices(ds)
V.STATE['class_summary'] = V.class_summary(ds)
V.STATE['class_ids'] = {v: int(k) for k, v in V.STATE['class_lookup'].items()}

found = V.discover_results(WD)
print('result sets:', list(found))
rp = list(found.values())[0]
res = V.load_results(rp)

# a tile that has both GT and predictions, and a modest point count
pick = None
for t in tiles[:400]:
    if t['name'] not in res:
        continue
    blk = V.load_block(t['name'], t['_ds'])
    if blk is None:
        continue
    origin, _s, _c = V.tile_frame(blk, 'tile_center')
    if V.load_polylines(t['name'], origin, t['_ds'], ndim=3):
        pick = t
        break
print('tile:', pick['_uid'], 'preds:', len(res[pick['name']]))

blk = V.load_block(pick['name'], pick['_ds'])
origin, shift, _c = V.tile_frame(blk, 'tile_center')
xyz = blk['features'][:, :3] + shift
gt3 = V.load_polylines(pick['name'], origin, pick['_ds'], ndim=3)
fb = V._pred_fallback_z(gt3, xyz)
gz = np.concatenate([pl[:, 2] for pl, _c, _n in gt3])
print(f'GT z range   {gz.min():.3f} .. {gz.max():.3f}  (spread {gz.ptp():.3f} m)')
print(f'cloud z span {xyz[:,2].min():.2f} .. {xyz[:,2].max():.2f}'
      f'   (-origin[2] would be {-origin[2]:.1f})')
allz = []
for pts, _s, _c in res[pick['name']]:
    zz = V.gt_z_at(np.asarray(pts)[:, :2], gt3, fb)
    allz.append(zz)
allz = np.concatenate(allz)
print(f'pred z range {allz.min():.3f} .. {allz.max():.3f}  '
      f'(spread {allz.ptp():.3f} m)')
assert xyz[:, 2].min() <= allz.min() and allz.max() <= xyz[:, 2].max(), 'off axis!'
print('every predicted vertex lies INSIDE the cloud z range: ok')
print(f'pred z stays within the GT z range: '
      f'{gz.min() - 1e-6 <= allz.min() and allz.max() <= gz.max() + 1e-6}')

out = '/tmpx/realviews'
os.makedirs(out, exist_ok=True)
for view in ('top', 'front'):
    buf = V.render_tile(pick['name'], mode='points', show_polylines=True,
                        point_size=1.5, max_points=150000, ds=pick['_ds'],
                        frame='tile_center', view=view,
                        results_path=rp, score_thresh=0.0, top_n=3)
    with open(os.path.join(out, f'pred_{view}.png'), 'wb') as f:
        f.write(buf.getvalue())
    print(f'wrote pred_{view}.png')
