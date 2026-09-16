"""Multi-tile CARLA samples: a tile together with the tiles around it.

``CustomCarlaLocalMapDataset`` infers every 30 m tile on its own. This
dataset builds, for every tile in the converter pkl, ONE sample covering the
tile and all of its footprint-overlapping neighbours -- the "3x3 grid" (on
this export the tiles sit on a 24 m stride with a 20% overlap, so a full
grid is 30 + 2 x 24 = 78 m across, i.e. ``neighbourhood_radius`` 39). The
point clouds of the neighbours are shifted into the centre tile's frame and
concatenated by ``LoadCarlaNeighbourhoodPoints``; the GT is stitched from
each tile's own reference-line json by OpenDRIVE ``road_id`` (see
``map_utils/polyline_stitch.py``) so a line crossing a tile edge is one
instance rather than two overlapping clipped copies.

Nothing about the model knows this happened: it sees a bigger tile. That is
deliberate -- it makes the neighbourhood sample the direct test of "infer on
a larger tile / on several tiles at once", with every tile-size knob in the
config (``point_cloud_range``, ``sparse_shape``, ``bev_h_``) grown the way
the 25 m -> 30 m derivation grew them. The BEVFormer temporal path
(``prev_bev``) is NOT an alternative here: on the LiDAR modality the BEV
comes straight from the sparse encoder and never enters the temporal
self-attention (``transformer.py`` ``get_bev_features``, ``modality ==
'lidar'`` branch), so "tiles as video frames" has no hook to attach to
without a model change.

Two things worth knowing before reading numbers off it:

* **Neighbourhoods are irregular.** The export keeps only tiles that carry
  GT, so most tiles have 2-3 overlapping neighbours, not 8 (measured over
  ``../carla_test``: mean 2.7, 106 tiles have none, 3 of 3795 have a full
  3x3; stitched GT per sample mean 5.6, max 33). The
  ``neighbourhood_radius`` box is therefore mostly empty of points AND of
  GT -- consistent with each other, since both come from the same tiles.
* **The eval is over the neighbourhood, not the tile.** ``_format_gt``
  writes the stitched GT, so ``CarlaMap_chamfer/*`` scores every line in
  the +-R box, and a line in the overlap zone is scored once per
  neighbourhood it appears in. To compare with single-tile numbers, clip
  the predictions back to the centre +-15 m box with
  ``tools/maptrv2/stitch_results.py --clip-to-tile`` and score them against
  the per-tile pkl.
"""

import collections
import json
import os

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmdet.datasets import DATASETS
from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.points import get_points_type

from .carla_offlinemap_dataset import CustomCarlaLocalMapDataset
from .map_utils.polyline_stitch import (Piece, clip_to_box, merge_grouped,
                                        shift_polylines)
from .pipelines.loading import LoadCarlaPointsFromFile


@DATASETS.register_module()
class CustomCarlaNeighbourhoodDataset(CustomCarlaLocalMapDataset):
    """One sample per tile, covering the tile and its overlapping neighbours.

    Args:
        neighbourhood_radius (float): half-extent of the sample box, in the
            centre tile's frame. Must equal the config's
            ``point_cloud_range`` xy half-extent. 39 fits a full 3x3 of 30 m
            tiles on the export's 24 m stride.
        neighbour_max_offset (float | None): a tile is a neighbour when both
            |dx| and |dy| between the two tile centres are below this. None
            means the tile side (footprints overlap or touch), which is the
            3x3 rule on a lattice and the only sensible rule off one.
        gt_merge_tol (float): geometric tolerance for joining two tiles'
            clipped copies of one ``road_id``. 5 cm: the export's per-tile
            copies agree to ~4 cm, and nothing else comes close.
        min_polyline_len (float): stitched GT shorter than this after
            clipping to the box is dropped (corner stubs).
        reference_dir (str | None): polyline directory name under the tile
            directory; None takes the one the pkl records
            (``tile_geometry.reference_dir``), which is the directory the
            pkl's own GT came from.
        include_neighbours (bool): False makes this the single-tile dataset
            again (centre points only, centre GT stitched to nothing) --
            for A/B runs that differ only in the box size.
    """

    def __init__(self,
                 *args,
                 neighbourhood_radius=39.0,
                 neighbour_max_offset=None,
                 gt_merge_tol=0.05,
                 min_polyline_len=1.0,
                 reference_dir=None,
                 include_neighbours=True,
                 **kwargs):
        # Read by load_annotations, which the base __init__ calls.
        self.neighbourhood_radius = float(neighbourhood_radius)
        self.neighbour_max_offset = neighbour_max_offset
        self.gt_merge_tol = gt_merge_tol
        self.min_polyline_len = min_polyline_len
        self.reference_dir = reference_dir
        self.include_neighbours = include_neighbours
        self._json_cache = {}
        self.neighbourhood_stats = {}
        super().__init__(*args, **kwargs)
        pc = self.pc_range
        half = (pc[3] - pc[0]) / 2.0
        if abs(half - self.neighbourhood_radius) > 1e-6 or \
                abs((pc[4] - pc[1]) / 2.0 - self.neighbourhood_radius) > 1e-6:
            raise ValueError(
                f'{self.__class__.__name__}: neighbourhood_radius='
                f'{self.neighbourhood_radius} but pc_range={list(pc)} has '
                f'xy half-extent {half}. The GT is clipped to the '
                'neighbourhood box and the model regresses against '
                'pc_range; they must be the same box.')

    # ------------------------------------------------------------------
    def load_annotations(self, ann_file):
        blob = mmcv.load(ann_file, file_format='pkl')
        pkl_root = blob.get('data_root')
        self._pkl_data_root = pkl_root if (
            pkl_root and os.path.isdir(pkl_root)) else None
        samples = sorted(blob['samples'], key=lambda e: e['sample_idx'])
        self._check_map_classes(samples, blob.get('map_classes'))
        samples = self._filter_empty_lidar_tiles(samples,
                                                 blob.get('lidar_check'))
        geom = blob.get('tile_geometry') or {}
        self._tile_dir = self._resolve_tile_dir(blob, geom)
        self._ref_dir = self.reference_dir or geom.get('reference_dir') \
            or 'reference_lines'
        self._class_of_export_id = self._export_class_map(blob)
        side = geom.get('tile_side') or (
            2.0 * geom['tile_radius'] if geom.get('tile_radius') else 30.0)
        max_off = self.neighbour_max_offset
        if max_off is None:
            max_off = float(side)

        origins = []
        for s in samples:
            if s.get('gt_frame', 'offset') != 'tile_center':
                raise ValueError(
                    f'{self.__class__.__name__} needs a tile_center-frame '
                    f'pkl (sample {s["sample_idx"]} is in the '
                    f'{s.get("gt_frame", "offset")!r} frame): tile centres '
                    'are what neighbours are found by, and the stitched GT '
                    'is expressed relative to them.')
            origins.append(np.asarray(s['annotation_origin'], dtype=np.float64))
        origins = np.stack(origins) if origins else np.zeros((0, 3))

        n_nbrs, n_gt, n_gt_tile = [], [], []
        for i, s in enumerate(samples):
            if self.include_neighbours and len(samples) > 1:
                d = np.abs(origins[:, :2] - origins[i, :2])
                mask = (d[:, 0] < max_off) & (d[:, 1] < max_off)
                mask[i] = False
                nbr_idx = np.flatnonzero(mask)
            else:
                nbr_idx = np.zeros(0, dtype=int)
            nbrs = []
            for j in nbr_idx:
                t = samples[j]
                nbrs.append(dict(
                    index=int(j),
                    sample_idx=t['sample_idx'],
                    lidar_path=t['lidar_path'],
                    # stored points are relative to the block's own offset;
                    # centre-frame = stored + offset_j - origin_i
                    points_shift=(np.asarray(t['lidar_offset'], np.float64)
                                  - origins[i]).astype(np.float32),
                    gt_shift=(origins[j] - origins[i]).astype(np.float32),
                ))
            s['neighbours'] = nbrs
            s['annotation_tile'] = s['annotation']
            s['annotation'] = self._stitched_annotation(
                s, [samples[j] for j in nbr_idx], origins[i])
            n_nbrs.append(len(nbrs))
            n_gt.append(sum(len(v) for v in s['annotation'].values()))
            n_gt_tile.append(sum(len(v) for v in s['annotation_tile'].values()))

        if samples:
            hist = collections.Counter(n_nbrs)
            self.neighbourhood_stats = dict(
                n_tiles=len(samples), max_offset=max_off,
                neighbours_mean=float(np.mean(n_nbrs)),
                neighbours_hist={int(k): int(v) for k, v in sorted(hist.items())},
                gt_per_sample_mean=float(np.mean(n_gt)),
                gt_per_sample_max=int(np.max(n_gt)),
                gt_per_tile_mean=float(np.mean(n_gt_tile)),
                reference_dir=self._ref_dir, tile_dir=self._tile_dir)
            print_log(
                f'{self.__class__.__name__}: {len(samples)} tiles, '
                f'neighbours per tile mean {np.mean(n_nbrs):.2f} '
                f'(hist {dict(sorted(hist.items()))}); stitched GT per '
                f'sample mean {np.mean(n_gt):.1f} / max {np.max(n_gt)} '
                f'(per single tile {np.mean(n_gt_tile):.1f}); box '
                f'+-{self.neighbourhood_radius} m, GT from '
                f'{self._tile_dir}/{self._ref_dir}', logger='current')
        return samples

    # ------------------------------------------------------------------
    def _resolve_tile_dir(self, blob, geom):
        cands = [geom.get('tile_dir')]
        root = self.raw_data_root or self._pkl_data_root or self.data_root
        if root and blob.get('split'):
            cands.append(os.path.join(root, blob['split']))
        if root:
            cands.append(root)
        for c in cands:
            if c and os.path.isdir(c):
                return c
        raise FileNotFoundError(
            f'{self.__class__.__name__}: cannot find the tile directory '
            f'holding the reference-line jsons (tried {cands}). Pass '
            'raw_data_root=<export root> to the dataset.')

    def _export_class_map(self, blob):
        """export class id/name -> config class name, via the pkl's
        ``class_groups`` (name -> export ids). A pkl without it is the
        single-class ``divider`` conversion, where everything maps to
        the one class."""
        groups = blob.get('class_groups') or {}
        lookup = blob.get('class_lookup') or {}
        out = {}
        self._catch_all_class = None
        for name, ids in groups.items():
            if name not in self.MAPCLASSES:
                continue
            if ids is None:
                # the converter's "no --map-classes" case: every export
                # class collapsed into this one name (historically
                # 'divider')
                self._catch_all_class = name
                continue
            for cid in ids:
                out[int(cid)] = name
                if str(cid) in lookup:
                    out[lookup[str(cid)]] = name
            out[name] = name
        if not out and self._catch_all_class is None:
            if len(self.MAPCLASSES) != 1:
                raise ValueError(
                    f'{self.__class__.__name__}: the pkl records no '
                    f'class_groups but the config asks for '
                    f'{list(self.MAPCLASSES)}; regenerate it with '
                    '--map-classes so export classes can be mapped.')
            self._catch_all_class = self.MAPCLASSES[0]
        return out

    def _tile_pieces(self, sample):
        """World-frame reference-line pieces of one tile, cached."""
        name = sample['sample_idx']
        if name in self._json_cache:
            return self._json_cache[name]
        path = os.path.join(self._tile_dir, self._ref_dir,
                            f'{name}_reference_lines.json')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'{self.__class__.__name__}: no reference-line json for '
                f'tile {name} at {path}; the stitched GT needs the '
                'export\'s per-polyline road_id, which the pkl does not '
                'carry.')
        with open(path) as f:
            blob = json.load(f)
        lookup = blob.get('classes') or {}
        pieces = []
        for k, pl in enumerate(blob.get('polylines', [])):
            pts = np.asarray(pl.get('points', []), dtype=np.float64)
            if pts.ndim != 2 or len(pts) < 2:
                continue
            cname = pl.get('class')
            cid = pl.get('class_id')
            if cname is None and cid is not None:
                cname = lookup.get(str(cid))
            cls = self._class_of_export_id.get(
                int(cid) if cid is not None else None)
            if cls is None:
                cls = self._class_of_export_id.get(cname)
            if cls is None:
                cls = self._catch_all_class
            if cls is None:
                continue  # an export class this taxonomy drops
            pieces.append(Piece(points=pts, cls=cls,
                                road_id=str(pl.get('road_id', k)),
                                source=name))
        self._json_cache[name] = pieces
        return pieces

    def _stitched_annotation(self, centre, neighbours, origin):
        pieces = list(self._tile_pieces(centre))
        for t in neighbours:
            pieces.extend(self._tile_pieces(t))
        merged = merge_grouped(pieces, key=lambda p: (p['cls'], p['road_id']),
                               tol=self.gt_merge_tol)
        r = self.neighbourhood_radius
        box = (-r, -r, r, r)
        ann = {cls: [] for cls in self.MAPCLASSES}
        for m in merged:
            local = shift_polylines([m['points']], -origin)[0]
            for sub in clip_to_box([local], box, min_len=self.min_polyline_len):
                ann[m['cls']].append(np.asarray(sub, dtype=np.float32))
        return ann

    # ------------------------------------------------------------------
    def get_data_info(self, index):
        d = super().get_data_info(index)
        info = self.data_infos[index]
        pts_base = (self.raw_data_root or self._pkl_data_root
                    or self.data_root or '')
        d['neighbours'] = [
            dict(nb, lidar_path=os.path.join(pts_base, nb['lidar_path']))
            for nb in info['neighbours']
        ]
        d['neighbourhood_radius'] = self.neighbourhood_radius
        return d


@PIPELINES.register_module()
class LoadCarlaNeighbourhoodPoints(LoadCarlaPointsFromFile):
    """``LoadCarlaPointsFromFile`` plus the neighbours' clouds.

    Each neighbour is loaded exactly as the centre is (same ``z_max`` cut on
    its stored z, same column selection), shifted by its ``points_shift``
    into the centre frame, and concatenated. Points outside the
    neighbourhood box are dropped here rather than left for the voxelizer,
    since up to nine tiles' worth of points would otherwise ride through
    ``GridSamplePoints``.
    """

    def __call__(self, results):
        results = super().__call__(results)
        centre = results['points'].tensor.numpy()
        clouds = [centre]
        for nb in results.get('neighbours', []):
            pts = self._load_points(nb['lidar_path'])
            pts[:, :3] += np.asarray(nb['points_shift'], dtype=np.float32)[:3]
            clouds.append(pts[:, self.use_dim])
        points = np.concatenate(clouds, axis=0)
        r = results.get('neighbourhood_radius')
        if r is not None:
            keep = (np.abs(points[:, 0]) <= r) & (np.abs(points[:, 1]) <= r)
            points = points[keep]
        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(
            points, points_dim=points.shape[-1], attribute_dims=None)
        results['num_neighbours'] = len(results.get('neighbours', []))
        return results
