#!/usr/bin/env python3
"""Stitch per-tile map predictions across tile boundaries, and score it.

Host-side (json + numpy, no torch / mmdet3d / container -- like
``dataset_viewer.py`` and ``reorder_results.py``). Two jobs:

1. **Cross-tile consensus / smoothing** of single-tile predictions. Every
   tile is inferred alone, so a line crossing a tile edge is predicted twice
   -- once per tile, each copy clipped to its own box, overlapping the other
   by up to 6 m on this export (30 m tiles, 24 m stride). This lifts every
   prediction into the world frame (pkl ``annotation_origin``), merges the
   copies of one line from different tiles (``polyline_stitch.merge_pieces``:
   same class, overlapping or abutting within ``--merge-tol``, joined only
   at a line's extremity), optionally blends the overlap zone by confidence
   (``--consensus``) and smooths (``--smooth``), then clips every merged line
   back into each tile it crosses and writes an ordinary
   ``carlamap_results.json`` the standard eval can score. Nothing about the
   model changes; this is the "amend each tile from its neighbours" idea in
   its cheapest form, and its AP delta is the go/no-go for a learned version.

2. **``--clip-to-tile R``** for predictions made on a larger box than the
   GT they are to be scored on (a 50 m tile run scored on 30 m GT, say):
   clips each sample's predictions to the centre +-R box. No merging.

``--eval`` scores before and after with ``dataset_viewer``'s numpy chamfer
eval, which is verified bit-exact against the container's ``eval_map``.

Examples::

    # consensus + smoothing on single-tile predictions
    python3 tools/maptrv2/stitch_results.py \
        <run>/pts_bbox/carlamap_results.json \
        --gt data/carla/carla_map_infos_test_30m_tc_2cls.pkl \
        -o <run>/stitched/pts_bbox/carlamap_results.json \
        --consensus --smooth 1 --score-thresh 0.3 --eval

    # clip a larger-box run's predictions to +-15 m before scoring on 30 m GT
    python3 tools/maptrv2/stitch_results.py <run>/pts_bbox/carlamap_results.json \
        --gt data/carla/carla_map_infos_test_30m_tc_2cls.pkl \
        -o <run>/tile15/pts_bbox/carlamap_results.json --clip-to-tile 15 --eval

Measured on the converged 2-class run (CLAUDE.md, "Multi-tile inference
feasibility"): with every prediction allowed to merge, mAP drops 0.04 --
the head's ~93% junk instances join onto real lines -- and with
``--score-thresh 0.3`` freezing them out, consensus and smoothing are
neutral (-0.001). Always pass a threshold.

Score semantics after stitching: a tile's piece of a merged line keeps the
confidence that tile's own prediction had (its member); a piece of a line
the tile never predicted itself -- a neighbour's line extended into it --
gets the merged line's (max member) confidence. So consensus never lowers
a tile's own scores, and only geometry moves.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))


def _load_stitch():
    # projects.mmdet3d_plugin cannot be imported without torch; the module
    # is pure numpy, so load the file by path.
    path = os.path.join(REPO, 'projects', 'mmdet3d_plugin', 'datasets',
                        'map_utils', 'polyline_stitch.py')
    spec = importlib.util.spec_from_file_location('polyline_stitch', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ps = _load_stitch()


# ---------------------------------------------------------------------------
def load_pkl(path):
    with open(path, 'rb') as f:
        blob = pickle.load(f)
    samples = blob['samples']
    origins = {s['sample_idx']: np.asarray(s['annotation_origin'], np.float64)
               for s in samples}
    geom = blob.get('tile_geometry') or {}
    radius = geom.get('tile_radius') or (
        geom['tile_side'] / 2.0 if geom.get('tile_side') else None)
    classes = list(blob.get('map_classes') or ['divider'])
    return blob, origins, radius, classes


def load_results(path):
    with open(path) as f:
        blob = json.load(f)
    return blob


def results_to_pieces(blob, origins, score_thresh=0.0):
    """World-frame pieces from a results json; also returns the tokens seen
    (so tiles that predicted nothing still get an (empty) entry back).

    Predictions below ``score_thresh`` are kept but marked ``frozen``: they
    go back to their own tile untouched and never merge with anything. The
    head emits 50 instances per tile of which ~3.5 are real lines, so
    without a threshold the ~93% junk joins onto, and extends, the good
    lines of neighbouring tiles."""
    pieces, tokens, skipped = [], [], 0
    for entry in blob['results']:
        tok = entry['sample_token']
        tokens.append(tok)
        o = origins.get(tok)
        if o is None:
            skipped += 1
            continue
        for v in entry['vectors']:
            pts = np.asarray(v['pts'], dtype=np.float64)
            if pts.ndim != 2 or len(pts) < 2:
                continue
            sc = float(v.get('confidence_level', 1.0))
            pieces.append(ps.Piece(points=pts[:, :2] + o[:2], score=sc,
                                   cls=v.get('cls_name', 'divider'),
                                   type=int(v.get('type', 0)), source=tok,
                                   frozen=sc < score_thresh))
    return pieces, tokens, skipped


def suppress_duplicates(pieces, tol):
    """Within-tile NMS: drop a prediction whose symmetric chamfer distance
    to a higher-scoring kept prediction of the same tile and class is below
    ``tol``. The head emits a fixed 25/50 instances per tile and a real line
    often comes out two or three times at slightly different confidences;
    stitching those across tiles would join each copy separately."""
    by_tile = {}
    for p in pieces:
        by_tile.setdefault((p['source'], p['cls']), []).append(p)
    keep = []
    for group in by_tile.values():
        group.sort(key=lambda p: -p['score'])
        kept = []
        for p in group:
            dup = False
            for q in kept:
                d1, _, _ = ps.polyline_distance(p['points'], q['points'])
                d2, _, _ = ps.polyline_distance(q['points'], p['points'])
                if 0.5 * (d1.mean() + d2.mean()) < tol:
                    dup = True
                    break
            if not dup:
                kept.append(p)
        keep.extend(kept)
    return keep


def parse_smooth(spec):
    """'chaikin', 'chaikin:2', 'savgol:window=9,order=3', 'spline:lam=8' ->
    (method, params). A bare integer keeps the old meaning: Chaikin passes."""
    if spec is None:
        return 'none', {}
    if isinstance(spec, int) or str(spec).isdigit():
        n = int(spec)
        return ('chaikin', dict(iters=n)) if n > 0 else ('none', {})
    method, _, rest = str(spec).partition(':')
    params = {}
    for kv in filter(None, rest.split(',')):
        k, _, v = kv.partition('=')
        if not v:                       # positional single param
            k, v = {'chaikin': 'iters', 'spline': 'lam', 'gaussian': 'sigma',
                    'dp': 'tol', 'savgol': 'window',
                    'laplacian': 'iters'}[method], k
        params[k] = int(v) if v.lstrip('-').isdigit() else float(v)
    return method, params


def stitch(pieces, tol, consensus, smooth_spec, bridge_gap=0.0,
           bridge_angle=30.0):
    live = [p for p in pieces if not p.get('frozen')]
    frozen = [p for p in pieces if p.get('frozen')]
    merged = ps.merge_grouped(live, key=lambda p: p['cls'], tol=tol,
                              consensus=consensus, bridge_gap=bridge_gap,
                              bridge_angle=bridge_angle)
    method, params = parse_smooth(smooth_spec)
    if method != 'none':
        for m in merged:
            m['points'] = ps.smooth(m['points'], method, **params)
    for p in frozen:
        q = ps.Piece(p)
        q['members'] = [p]
        q['n_members'] = 1
        merged.append(q)
    return merged


def reclip(merged, tokens, origins, radius, num_pts, extend='members'):
    """Merged world lines -> per-tile local vectors.

    ``extend`` says which tiles receive a piece of a line:

    * ``'members'`` (default): only the tiles whose own predictions went
      into it. A line nobody joined goes back to its one tile; a joined
      line is re-cut into each contributing tile. Per-tile counts stay at
      the head's 50, so an AP delta isolates the geometry change.
    * ``'all'``: every tile the line crosses, including tiles that predicted
      nothing there (a neighbour's line extended into the overlap strip).
      Measured on the 2-class run this takes 190k predictions to 337k
      per-tile vectors -- mostly stubs of unjoined lines duplicated into
      the neighbour's overlap zone, i.e. new false positives.
    """
    boxes = {t: (origins[t][0] - radius, origins[t][1] - radius,
                 origins[t][0] + radius, origins[t][1] + radius)
             for t in tokens if t in origins}
    # spatial prefilter: which tiles can a line touch
    tok_list = list(boxes)
    B = np.array([boxes[t] for t in tok_list]) if tok_list else np.zeros((0, 4))
    out = {t: [] for t in tokens}
    for m in merged:
        pts = m['points']
        lo, hi = pts.min(0), pts.max(0)
        cand = np.flatnonzero((B[:, 0] <= hi[0]) & (B[:, 2] >= lo[0])
                              & (B[:, 1] <= hi[1]) & (B[:, 3] >= lo[1]))
        member_score = {mem['source']: float(mem['score'])
                        for mem in m.get('members', [m])}
        for k in cand:
            t = tok_list[k]
            if extend == 'members' and t not in member_score:
                continue
            box = boxes[t]
            o = origins[t]
            for sub in ps.clip_to_box([pts], box, min_len=0.0):
                local = ps.resample_polyline(sub, num_pts) - o[:2]
                out[t].append(dict(
                    pts=local.tolist(), pts_num=num_pts,
                    cls_name=m['cls'], type=int(m.get('type', 0)),
                    confidence_level=member_score.get(t, float(m['score']))))
    return out


def clip_to_tile(blob, radius, num_pts):
    """Predictions in a sample's own frame -> clipped to +-radius."""
    box = (-radius, -radius, radius, radius)
    out = {}
    for entry in blob['results']:
        vecs = []
        for v in entry['vectors']:
            pts = np.asarray(v['pts'], dtype=np.float64)
            if pts.ndim != 2 or len(pts) < 2:
                continue
            for sub in ps.clip_to_box([pts[:, :2]], box, min_len=0.0):
                vecs.append(dict(v, pts=ps.resample_polyline(sub, num_pts).tolist(),
                                 pts_num=num_pts))
        out[entry['sample_token']] = vecs
    return out


def write_results(path, meta, per_token, order):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    results = [dict(sample_token=t, vectors=per_token.get(t, []))
               for t in order]
    with open(path, 'w') as f:
        json.dump(dict(meta=meta, results=results), f)


# ---------------------------------------------------------------------------
def evaluate(results_path, gt_pkl):
    """Global per-class chamfer AP via dataset_viewer's numpy eval."""
    sys.path.insert(0, HERE)
    import dataset_viewer as dv  # noqa: E402  (flask/matplotlib on host)
    # the viewer fills these in its own main(); outside it they are absent
    for k in ('results_cache', 'gt_cache', 'gt_classes_cache'):
        dv.STATE.setdefault(k, {})
    preds = dv.load_results(results_path)
    gts = dv.load_gt_pkl(gt_pkl)
    classes = dv.gt_classes(gt_pkl)
    rows = []
    for tok, gt in gts.items():
        p = preds.get(tok, [])
        rows.append(dv.eval_tile(p, gt, 0.0, classes=classes))
    glob = dv.global_ap(rows)
    per_cls = {}
    for c in range(len(classes)):
        vals = []
        for thr in dv.CHAMFER_THRESHOLDS:
            parts = [r['by_cls'][c] for r in rows if c in r['by_cls']]
            n_gts = sum(q['n_gt'] for q in parts)
            tp = np.concatenate([q['tp_thr'][thr][0] for q in parts]) if parts else np.zeros(0)
            fp = np.concatenate([q['tp_thr'][thr][1] for q in parts]) if parts else np.zeros(0)
            sc = np.concatenate([q['scores'] for q in parts]) if parts else np.zeros(0)
            vals.append(dv.ap_from_tpfp(tp, fp, sc, n_gts))
        per_cls[classes[c]] = vals
    return glob, per_cls, sum(r['n_pred'] for r in rows)


def fmt_eval(name, glob, per_cls, n_pred):
    thr = (0.5, 1.0, 1.5)
    lines = [f'{name}: mAP {glob["mean"]:.4f}  '
             + '  '.join(f'AP@{t}={glob[t]:.4f}' for t in thr)
             + f'  ({n_pred} predictions)']
    for c, vals in per_cls.items():
        v = [x if x is not None else float("nan") for x in vals]
        lines.append(f'    {c:>10}: mean {np.mean(v):.4f}  '
                     + '  '.join(f'{t}={x:.4f}' for t, x in zip(thr, v)))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('results', help='carlamap_results.json')
    ap.add_argument('--gt', required=True,
                    help='converter pkl: tile origins (and eval GT)')
    ap.add_argument('-o', '--out', required=True, help='output results json')
    ap.add_argument('--tile-radius', type=float, default=None,
                    help='tile half-size for re-clipping; default: the pkl')
    ap.add_argument('--merge-tol', type=float, default=1.0,
                    help='m; two tiles\' copies of one line may differ by '
                         'this much and still be joined (default 1.0)')
    ap.add_argument('--bridge-gap', type=float, default=0.0, metavar='M',
                    help='also join two pieces whose ends face each other '
                         'across a gap of up to M metres with agreeing '
                         'headings (for crops that do not overlap; 0 = off)')
    ap.add_argument('--bridge-angle', type=float, default=30.0,
                    help='max heading disagreement for a bridge (degrees)')
    ap.add_argument('--consensus', action='store_true',
                    help='blend the overlap zone by confidence')
    ap.add_argument('--smooth', default='0', metavar='METHOD[:PARAMS]',
                    help='smoother for every merged line: none, chaikin[:N], '
                         'savgol[:window=7,order=3], spline[:lam=4], '
                         'gaussian[:sigma=1.5], laplacian[:iters=20,step=0.5,'
                         'fidelity=0.2], dp[:tol=0.3]. A bare integer is '
                         'Chaikin passes (0 = none). See '
                         'polyline_stitch.SMOOTHERS.')
    ap.add_argument('--score-thresh', type=float, default=0.0,
                    help='predictions below this confidence are passed '
                         'through untouched and never merged (default 0: '
                         'everything merges)')
    ap.add_argument('--nms-tol', type=float, default=None, metavar='M',
                    help='within-tile duplicate suppression: drop a '
                         'prediction whose mean chamfer distance to a '
                         'higher-scoring kept one is under M metres '
                         '(applied to predictions above --score-thresh; '
                         'the rest are frozen anyway)')
    ap.add_argument('--drop-frozen', action='store_true',
                    help='discard predictions below --score-thresh instead '
                         'of passing them through (a "best polylines only" '
                         'output; do not use for AP, which needs the tail)')
    ap.add_argument('--num-pts', type=int, default=20,
                    help='vertices per output polyline (the head\'s 20)')
    ap.add_argument('--clip-to-tile', type=float, default=None, metavar='R',
                    help='neighbourhood mode: clip each sample\'s '
                         'predictions to the centre tile +-R and stop')
    ap.add_argument('--extend', choices=('members', 'all'), default='members',
                    help='which tiles get a piece of a merged line: only '
                         'those that contributed to it (default) or every '
                         'tile it crosses')
    ap.add_argument('--world-out', default=None,
                    help='also write the merged world-frame lines (json)')
    ap.add_argument('--eval', action='store_true',
                    help='score input and output against --gt')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)

    if os.path.abspath(args.out) == os.path.abspath(args.results) and not args.force:
        ap.error('output would overwrite the input; pass --force')

    blob, origins, pkl_radius, classes = load_pkl(args.gt)
    res = load_results(args.results)
    order = [e['sample_token'] for e in res['results']]
    radius = args.tile_radius or pkl_radius
    if radius is None:
        ap.error('the pkl records no tile_radius; pass --tile-radius')
    t0 = time.time()

    if args.clip_to_tile is not None:
        per_token = clip_to_tile(res, args.clip_to_tile, args.num_pts)
        n_in = sum(len(e['vectors']) for e in res['results'])
        n_out = sum(len(v) for v in per_token.values())
        print(f'[clip] {len(order)} samples: {n_in} -> {n_out} vectors '
              f'inside +-{args.clip_to_tile} m ({time.time() - t0:.1f}s)')
    else:
        pieces, tokens, skipped = results_to_pieces(res, origins,
                                                    args.score_thresh)
        if skipped:
            print(f'[warn] {skipped} tiles in the results are not in the '
                  'pkl and were dropped')
        if args.drop_frozen:
            pieces = [p for p in pieces if not p.get('frozen')]
        if args.nms_tol is not None:
            live = suppress_duplicates(
                [p for p in pieces if not p.get('frozen')], args.nms_tol)
            n_before = sum(1 for p in pieces if not p.get('frozen'))
            pieces = live + [p for p in pieces if p.get('frozen')]
            print(f'[nms] {n_before} -> {len(live)} confident predictions '
                  f'after within-tile duplicate suppression at '
                  f'{args.nms_tol} m')
        merged = stitch(pieces, args.merge_tol, args.consensus, args.smooth,
                        args.bridge_gap, args.bridge_angle)
        n_multi = sum(1 for m in merged if m['n_members'] > 1)
        n_frozen = sum(1 for p in pieces if p.get('frozen'))
        print(f'[stitch] {len(pieces)} predictions over {len(tokens)} tiles '
              f'({n_frozen} below --score-thresh {args.score_thresh}, left '
              'untouched) '
              f'-> {len(merged)} world lines, {n_multi} of them joined '
              f'across tiles (tol {args.merge_tol} m, consensus='
              f'{args.consensus}, smooth={args.smooth}, bridge_gap={args.bridge_gap}) '
              f'({time.time() - t0:.1f}s)')
        per_token = reclip(merged, tokens, origins, radius, args.num_pts,
                           args.extend)
        n_out = sum(len(v) for v in per_token.values())
        print(f'[reclip] {n_out} per-tile vectors written at +-{radius} m')
        if args.world_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.world_out)),
                        exist_ok=True)
            with open(args.world_out, 'w') as f:
                json.dump([dict(pts=m['points'].tolist(), cls_name=m['cls'],
                                score=m['score'], n_members=m['n_members'],
                                members=[x['source'] for x in m['members']])
                           for m in merged], f)
            print(f'[world] {len(merged)} lines -> {args.world_out}')

    write_results(args.out, res.get('meta'), per_token, order)
    print(f'[out] {args.out}')

    if args.eval:
        b = evaluate(args.results, args.gt)
        a = evaluate(args.out, args.gt)
        print(fmt_eval('before', *b))
        print(fmt_eval('after ', *a))
        print(f'delta mAP: {a[0]["mean"] - b[0]["mean"]:+.4f}')


if __name__ == '__main__':
    main()
