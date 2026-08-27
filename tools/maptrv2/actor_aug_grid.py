"""Render a grid of tiles showing the CarlaActorPaste augmentation.

Column 0 is the tile as the dataloader emits it WITHOUT actors (the ground
truth); the remaining columns are independently seeded draws of the
augmentation. Rows are distinct tiles, auto-picked for placement richness
(most vehicle+pedestrian candidates). Each cell runs the real dataloader
stages -- LoadCarlaPointsFromFile(recenter=True) -> CarlaActorPaste ->
GridSamplePoints -- so what is drawn is exactly what the model would see.

Run inside the training container (needs mmcv/mmdet3d and the plugin), e.g.::

    docker run --rm -e PYTHONPATH=/MapTR -e POINT2VECTOR_DATA=/p2v \
      -v $(pwd)/projects:/MapTR/projects -v $(pwd)/tools:/MapTR/tools \
      -v $(pwd)/data:/MapTR/data \
      -v <carla export>:<carla export>:ro \
      -v <point2vector_data checkout>:/p2v:ro \
      -v <actor scan dir>:/assets:ro -v <outdir>:/out \
      -w /MapTR jhd0ck3r/maptrv2:latest python3 tools/maptrv2/actor_aug_grid.py \
      --catalogue /assets/catalogue.json \
      --placements <carla export>/test/placements --out /out/actor_aug_grid.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from mmcv import Config
from mmdet3d.datasets import build_dataset  # noqa: F401  (registers plugin path via config)

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('--config',
                default='projects/configs/maptrv2/maptrv2_carla_r50_24ep_lidar.py')
ap.add_argument('--catalogue', required=True,
                help="the actor scan's catalogue.json")
ap.add_argument('--placements', required=True,
                help='directory of *_placements.json sidecars')
ap.add_argument('--out', default='actor_aug_grid.png')
ap.add_argument('--n-tiles', type=int, default=4)
ap.add_argument('--n-aug', type=int, default=5)
ap.add_argument('--seed', type=int, default=1000)
args = ap.parse_args()
CFG, CATALOGUE, PLACEMENTS, OUT = (args.config, args.catalogue,
                                   args.placements, args.out)
N_TILES, N_AUG = args.n_tiles, args.n_aug

cfg = Config.fromfile(CFG)
# plugin import
import importlib, sys
sys.path.insert(0, '.')
importlib.import_module(cfg.plugin_dir.rstrip('/').replace('/', '.'))

from mmdet3d.datasets.builder import DATASETS
from mmdet.datasets import build_dataset as build_ds
ds_cfg = cfg.data.test.copy()
ds_cfg['pipeline'] = []
dataset = build_ds(ds_cfg)
print('dataset:', len(dataset), 'tiles')

from projects.mmdet3d_plugin.datasets.pipelines.loading import (
    LoadCarlaPointsFromFile, GridSamplePoints)
from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import CarlaActorPaste

load = LoadCarlaPointsFromFile(coord_type='LIDAR', load_dim=4, use_dim=4,
                               z_max=96.0, recenter=True)
grid = GridSamplePoints(grid_size=cfg.lidar_voxel_size,
                        point_cloud_range=cfg.lidar_point_cloud_range)

# --- pick 4 tiles with rich placements (both classes available) -------------
scored = []
for i in range(len(dataset)):
    info = dataset.get_data_info(i)
    name = os.path.splitext(os.path.basename(info['pts_filename']))[0]
    p = os.path.join(PLACEMENTS, name + '_placements.json')
    if not os.path.exists(p):
        continue
    c = json.load(open(p)).get('candidates', [])
    nv = sum(1 for x in c if x['cls'] == 'vehicle')
    np_ = sum(1 for x in c if x['cls'] == 'pedestrian')
    scored.append((min(nv, np_), nv, np_, i, name))
scored.sort(reverse=True)
# spread the picks: best, then every ~8th among the top ranks for variety
picks = [scored[0]] + [scored[k] for k in range(8, 8 * N_TILES, 8)
         if k < len(scored)]
picks = picks[:N_TILES]
print('picked tiles:', [(s[4], 'veh_cands=%d' % s[1], 'ped_cands=%d' % s[2]) for s in picks])

# --- run the loader stages ---------------------------------------------------
def run(idx, paste=None):
    r = dict(dataset.get_data_info(idx))
    r = load(r)
    meta = []
    if paste is not None:
        r = paste(r)
        meta = r.get('pasted_actors', [])
    r = grid(r)
    return r['points'].tensor.numpy(), meta

cells = []  # rows of (points, meta, n_veh, n_ped)
for t, (_, _, _, idx, name) in enumerate(picks):
    row = [(run(idx)[0], [], name)]
    for v in range(N_AUG):
        paste = CarlaActorPaste(
            catalogue=CATALOGUE,
            placements_dir=PLACEMENTS,
            n_vehicles=(1, 5),
            n_pedestrians=(1, 6),
            prob=1.0,
            seed=args.seed + 37 * t + v,   # unique rng per (tile, version) cell
        )
        pts, meta = run(idx, paste)
        row.append((pts, meta, name))
    cells.append(row)
    print(name, 'augs:', ['%dv+%dp' % (sum(1 for m in mm if m['cls'] == 'vehicle'),
                                       sum(1 for m in mm if m['cls'] == 'pedestrian'))
                          for _, mm, _ in row[1:]])

# --- draw --------------------------------------------------------------------
BG, FG = '#0d1117', '#e6edf3'
fig = Figure(figsize=(3.1 * (N_AUG + 1), 3.1 * N_TILES), dpi=150)
fig.patch.set_facecolor(BG)
axes = fig.subplots(N_TILES, N_AUG + 1, squeeze=False)

for r, row in enumerate(cells):
    gt_info = dataset.get_data_info(picks[r][3])
    polys = gt_info['annotation']['divider']
    for c, (pts, meta, name) in enumerate(row):
        ax = axes[r][c]
        ax.set_facecolor(BG)
        z = pts[:, 2]
        lo, hi = np.percentile(z, [1, 99.5])
        ax.scatter(pts[:, 0], pts[:, 1], c=np.clip(z, lo, hi), s=0.25,
                   cmap='cividis', linewidths=0, rasterized=True)
        for pl in polys:
            pl = np.asarray(pl)
            ax.plot(pl[:, 0], pl[:, 1], color='#f85149', lw=0.9, alpha=0.85,
                    zorder=4)
        nv = np_ = 0
        for m in meta:
            veh = m['cls'] == 'vehicle'
            nv += veh
            np_ += not veh
            ax.add_patch(Circle((m['x'], m['y']), 2.6 if veh else 1.2,
                                fill=False, lw=1.3, zorder=5,
                                edgecolor='#ffa657' if veh else '#39c5cf'))
        ax.set_xlim(-12.5, 12.5)
        ax.set_ylim(-12.5, 12.5)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color('#30363d')
        if r == 0:
            ax.set_title('ground truth' if c == 0
                         else 'augmentation %d' % c, color=FG, fontsize=11)
        if c == 0:
            ax.set_ylabel(name, color=FG, fontsize=9)
        else:
            ax.text(0.02, 0.02, '%dv + %dp' % (nv, np_),
                    transform=ax.transAxes, color=FG, fontsize=8,
                    va='bottom', ha='left')

fig.suptitle('CarlaActorPaste augmentation — points as the dataloader emits them '
             '(after grid sampling); orange = pasted vehicle, cyan = pasted pedestrian, '
             'red = GT divider', color=FG, fontsize=12, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT, facecolor=BG)
print('wrote', OUT)
