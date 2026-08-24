import os, sys, time
for _p in ('/home-local/johil9.nobkp/Documents/Code/MapTR/.claude/worktrees/'
           'reorder-results/tools/maptrv2', '/MapTR/tools/maptrv2'):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
        break
import numpy as np
import dataset_viewer as V

DATA = '/home-local/johil9.nobkp/Documents/Code/carla_test'
V.STATE.update({
    'data_root': DATA, 'max_points': 150000, 'work_dir': None,
    'results_cache': {}, 'gt_cache': {}, 'shape_cache': {}, 'frame_cache': {},
    'frame': 'tile_center', 'gt_json': None, 'deep': {},
    'scan_grid': (0.1, 0.1, 0.4), 'scan_workers': 2, 'scan_stride': 1,
    'pc_range_z': (-72.0, 96.0), 'num_pts_per_vec': 20,
    'cache_dir': '/tmpx/realcache', 'tags_file': None,
    'density_max': None, 'density_ceiling_cache': None,
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

t0 = time.time()
vmax, src = V.density_ceiling()
print(f'ceiling = {vmax:,.0f} pts/m^2  ({src})  in {time.time()-t0:.1f}s')
t1 = time.time()
again = V.density_ceiling()
print(f'second call {time.time()-t1:.3f}s (cached) -> {again[0]:,.0f}')
assert again[0] == vmax

# stability across "restarts": rebuild STATE cache and re-derive
V.STATE['density_ceiling_cache'] = None
third, _ = V.density_ceiling()
print(f'after cache clear: {third:,.0f}  same={third == vmax}')
assert third == vmax, 'ceiling is not reproducible across restarts!'

# render a few tiles shared vs per-tile and confirm the pixels differ
out = '/tmpx/realviews'
os.makedirs(out, exist_ok=True)
picks = [tiles[0], tiles[7], tiles[19]]
for t in picks:
    for tag, dm in (('shared', vmax), ('pertile', 0.0)):
        buf = V.render_tile(t['name'], mode='density', show_polylines=False,
                            point_size=1.5, max_points=150000, ds=t['_ds'],
                            frame='tile_center', view='top', density_max=dm)
        with open(os.path.join(out, f'dens_{tag}_{t["name"]}.png'), 'wb') as f:
            f.write(buf.getvalue())
    blk = V.load_block(t['name'], t['_ds'])
    origin, shift, center = V.tile_frame(blk, 'tile_center', t['name'], t['_ds'])
    xy = blk['features'][:, :2] + shift[:2]
    r = float(blk['tile_radius'])
    H, _x, _y = V.density_hist(xy, center, r)
    print(f'{t["name"]}: max cell {H.max():,.0f}  '
          f'({100*H.max()/vmax:.1f}% of the shared ceiling)')
    blk.close()
print('wrote shared/per-tile pairs to', out)
