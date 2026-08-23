#!/usr/bin/env python3
"""Rewrite a carlamap_results.json with each polyline's vertices reordered.

WHY THIS EXISTS
---------------
The HM configs supervise geometry with `loss_mode='emd'` and set
`loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.0)`. `exact_emd_loss` is a
balanced assignment over point *sets*, so it is permutation-invariant, and
with the direction loss zeroed **nothing in the objective constrains vertex
order**. The head duly emits vertices that lie on the right line but walk
back and forth along it: measured over one HM run's 94,875 predictions, the
path traced in index order is 3.35x longer than the same points traversed in
order, and 99% of vertices turn by more than 90 degrees (baseline L1+dir:
1.00x and 3%).

The damage is to anything that reads the polyline as an ordered curve:
rendering, arc-length resampling, heading/curvature, or feeding it to a
planner. This script repairs the ordering after the fact, without retraining.

It is a post-hoc fix, not a substitute for the real one. Giving `loss_dir` a
non-zero weight (or adding a small per-index L1 alongside the EMD) makes the
model produce ordered vertices in the first place. Use this to make existing
checkpoints' output legible.

REORDERING LOWERS chamfer AP, AND THAT IS NOT A BUG
---------------------------------------------------
Measured on the 3795-tile HM run against
`carla_map_infos_test_30m_tc_2cls.pkl` (pooled over classes):

    original     AP@0.5=0.20985  AP@1.0=0.42186  AP@1.5=0.55333  mAP=0.39501
    reordered    AP@0.5=0.19097  AP@1.0=0.36655  AP@1.5=0.50938  mAP=0.35563

It is tempting to expect no change -- the vertex *set* is identical and only
its permutation moves -- but the chamfer eval is NOT set-based. It resamples
each polyline to 100 points **by arc length along the stored order**, so the
order decides where those samples land.

The mechanism, measured over 4,025 predictions (perpendicular RMS to each
line's own fitted axis):

    raw vertices              1.0920 m
    resampled ZIGZAG          0.9064 m   <- pulled toward the axis
    resampled ORDERED         1.0861 m   <- faithful to the vertices

The HM vertex cloud is noisy (about 0.6-1.1 m of spread about the fitted
line). A zigzag's long chords cut back and forth *through the middle* of that
cloud, and since those chords dominate the arc length they carry most of the
100 samples -- so resampling a zigzag incidentally **denoises** it toward a
central axis. The ordered path instead hops between neighbouring vertices and
traces the noise faithfully. Chamfer rewards the smoother locus, so the
scrambled version scores about 0.04 mAP higher.

That is an artefact of the metric, not evidence the zigzag is better geometry.
Note the crude principal-axis sort scores 0.35655, essentially the same as the
proper shortest-path ordering's 0.35563 -- the loss is systematic to ordering
itself, not to how well it is done. **Do not use a reordered file to compare
against published AP numbers**, and do not read the drop as damage.

If you want ordered vertices *and* the smoothing, that is a smoothing step,
not an ordering one -- fit or filter the vertices after reordering, and say so.

WHAT IT DOES NOT CHANGE
-----------------------
The vertex set per polyline is bit-identical (a permutation, verified), and
every other field -- scores, classes, `pts_num`, the `meta` block, tile order
-- is copied through verbatim.

METHODS
-------
`path` (default)
    Order the vertices as the shortest open path visiting all of them --
    an open travelling-salesman path, solved by nearest-neighbour from each
    principal-axis extreme followed by 2-opt to a local optimum. This is the
    right general answer: it recovers the traversal order of curved and even
    hairpin polylines, which a projection cannot.
`pca`
    Sort by position along the first principal axis. Cheap and exact for a
    straight line, but it flattens any curve that doubles back -- a U-shaped
    polyline comes out as a zigzag again. Kept for comparison only.
`none`
    Measure and report, write nothing new. Use with --report to characterise
    a run before deciding whether it needs this at all.

DIRECTION
---------
The traversal direction of a polyline is arbitrary here: the loss that
produced these never fixed one, and chamfer scoring is direction-agnostic.
Rather than leave it to solver happenstance, each line is oriented
deterministically (first endpoint lexicographically smallest in x then y) so
that re-running the script on the same input is reproducible.

Plain json + numpy, so this runs on the host -- no torch, no mmdet3d, no
container -- like `dataset_viewer.py`.

Example
-------
    python3 tools/maptrv2/reorder_results.py \
        <work_dir>/results/pts_bbox/carlamap_results.json \
        -o <work_dir>/results_reordered/pts_bbox/carlamap_results.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# ordering
# ---------------------------------------------------------------------------

def _pairwise(pts):
    d = pts[:, None, :] - pts[None, :, :]
    return np.sqrt((d * d).sum(-1))


def _path_length(dist, order):
    return dist[order[:-1], order[1:]].sum()


def _nearest_neighbour(dist, start):
    n = dist.shape[0]
    unvisited = np.ones(n, dtype=bool)
    order = np.empty(n, dtype=int)
    cur = start
    for i in range(n):
        order[i] = cur
        unvisited[cur] = False
        if i == n - 1:
            break
        # argmin over the remaining points; masked rather than deleted so the
        # indices stay aligned with `dist`.
        masked = np.where(unvisited, dist[cur], np.inf)
        cur = int(masked.argmin())
    return order


def _two_opt(dist, order, max_passes=50):
    """2-opt on an OPEN path: reverse order[i:j+1] and keep it if shorter.

    The deltas for every (i, j) are computed as one vectorised matrix rather
    than in a double loop -- with 20 vertices that is 190 candidate moves per
    pass, and this runs over ~10^5 polylines.

    Reversing a segment changes at most the two edges at its boundary. The
    prefix case (i == 0) and suffix case (j == n-1) each have only one such
    edge; the missing one is treated as zero cost on both sides of the
    comparison, which is what makes reversing a prefix or suffix a legal move
    here and not on a closed tour.
    """
    n = len(order)
    if n < 4:
        return order
    order = order.copy()
    i_idx, j_idx = np.triu_indices(n, k=1)
    prev_ok = i_idx > 0
    next_ok = j_idx < n - 1
    prev_i = np.maximum(i_idx - 1, 0)
    next_j = np.minimum(j_idx + 1, n - 1)
    for _ in range(max_passes):
        p = order
        # edge (i-1, i) before the reversal becomes (i-1, j) after it;
        # edge (j, j+1) becomes (i, j+1).
        a, b = p[prev_i], p[next_j]
        old = (np.where(prev_ok, dist[a, p[i_idx]], 0.0)
               + np.where(next_ok, dist[p[j_idx], b], 0.0))
        new = (np.where(prev_ok, dist[a, p[j_idx]], 0.0)
               + np.where(next_ok, dist[p[i_idx], b], 0.0))
        delta = new - old
        k = int(delta.argmin())
        if delta[k] >= -1e-12:
            break
        i, j = int(i_idx[k]), int(j_idx[k])
        order[i:j + 1] = order[i:j + 1][::-1]
    return order


def order_path(pts):
    """Shortest open path through every vertex (NN + 2-opt, local optimum)."""
    n = len(pts)
    if n < 3:
        return np.arange(n)
    dist = _pairwise(pts)
    # Seed from the two extremes along the principal axis. A true endpoint is
    # almost always one of them, and seeding from a real endpoint is what
    # keeps nearest-neighbour from stranding a point at the far end. Trying
    # all n starts costs n times as much for no measurable gain after 2-opt.
    centred = pts - pts.mean(0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    t = centred @ axis
    best, best_len = None, np.inf
    for start in {int(t.argmin()), int(t.argmax())}:
        cand = _two_opt(dist, _nearest_neighbour(dist, start))
        length = _path_length(dist, cand)
        if length < best_len:
            best, best_len = cand, length
    return best


def order_pca(pts):
    """Sort along the first principal axis."""
    if len(pts) < 3:
        return np.arange(len(pts))
    centred = pts - pts.mean(0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    return np.argsort(centred @ axis)


ORDERERS = {'path': order_path, 'pca': order_pca}


def orient(pts):
    """Flip so the first endpoint is lexicographically smallest.

    Direction is not determined by anything upstream (see the module
    docstring), so it is pinned here purely for reproducibility.
    """
    if len(pts) < 2:
        return pts
    head, tail = pts[0], pts[-1]
    if (tail[0], tail[1]) < (head[0], head[1]):
        return pts[::-1]
    return pts


# ---------------------------------------------------------------------------
# the zigzag measurement, so the script can show its own effect
# ---------------------------------------------------------------------------

def zigzag_stats(lines):
    """Index-order path length vs the shortest ordering, and turn angles.

    The ratio is against each line's own shortest open path, so 1.00 means
    "already ordered" regardless of the line's shape -- unlike a comparison
    against a projection, which would penalise a genuinely curved line.
    """
    ratios, turns = [], []
    for pts in lines:
        if len(pts) < 3:
            continue
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        length = seg.sum()
        if length < 1e-9:
            continue
        dist = _pairwise(pts)
        shortest = _path_length(dist, order_path(pts))
        ratios.append(length / max(shortest, 1e-9))
        d = np.diff(pts, axis=0)
        norm = np.linalg.norm(d, axis=1, keepdims=True)
        d = d / np.clip(norm, 1e-9, None)
        cos = np.clip((d[:-1] * d[1:]).sum(1), -1.0, 1.0)
        turns.append(np.degrees(np.arccos(cos)).mean())
    if not ratios:
        return None
    ratios, turns = np.array(ratios), np.array(turns)
    return dict(n=len(ratios),
                ratio_med=float(np.median(ratios)),
                ratio_p90=float(np.percentile(ratios, 90)),
                turn_mean=float(turns.mean()),
                turn_over_90=float((turns > 90).mean()))


def fmt_stats(s):
    if s is None:
        return '(no polylines with >= 3 points)'
    return (f'n={s["n"]:6d}  path/shortest med={s["ratio_med"]:5.2f} '
            f'p90={s["ratio_p90"]:5.2f}  mean turn={s["turn_mean"]:5.1f} deg  '
            f'turn>90 deg={s["turn_over_90"]:.2f}')


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Reorder the vertices of every polyline in a '
                    'carlamap_results.json.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument('results', help='input carlamap_results.json')
    ap.add_argument('-o', '--out',
                    help='output json (default: alongside the input, '
                         '<name>_reordered.json). Parent dirs are created.')
    ap.add_argument('-m', '--method', default='path',
                    choices=['path', 'pca', 'none'],
                    help='ordering method (default: path)')
    ap.add_argument('--no-orient', action='store_true',
                    help='leave traversal direction to the solver instead of '
                         'pinning it deterministically')
    ap.add_argument('--report', action='store_true',
                    help='measure the zigzag before and after and print it. '
                         'Costs a second ordering solve per polyline.')
    ap.add_argument('--sample', type=int, default=0, metavar='N',
                    help='with --report, measure on the first N tiles only '
                         '(0 = all). The rewrite still covers every tile.')
    ap.add_argument('--force', action='store_true',
                    help='allow overwriting the input file')
    args = ap.parse_args(argv)

    out = args.out or (os.path.splitext(args.results)[0] + '_reordered.json')
    if args.method != 'none':
        if os.path.abspath(out) == os.path.abspath(args.results) and not args.force:
            ap.error('refusing to overwrite the input; pass --force or -o')

    with open(args.results) as f:
        blob = json.load(f)
    results = blob.get('results')
    if not isinstance(results, list):
        ap.error(f'{args.results}: no "results" list -- is this a '
                 f'carlamap_results.json?')

    def tile_lines(tiles):
        for tile in tiles:
            for vec in tile.get('vectors', []):
                pts = np.asarray(vec['pts'], dtype=float)
                if pts.ndim == 2 and len(pts) >= 3:
                    yield pts[:, :2]

    sample = results[:args.sample] if args.sample else results
    if args.report:
        t0 = time.time()
        print(f'before   {fmt_stats(zigzag_stats(tile_lines(sample)))}'
              f'   ({time.time() - t0:.1f}s)')

    if args.method == 'none':
        print('method=none: nothing written.')
        return 0

    orderer = ORDERERS[args.method]
    n_vec = n_moved = 0
    t0 = time.time()
    for i, tile in enumerate(results):
        for vec in tile.get('vectors', []):
            pts = np.asarray(vec['pts'], dtype=float)
            if pts.ndim != 2 or len(pts) < 2:
                continue
            n_vec += 1
            new = pts[orderer(pts[:, :2])]
            if not args.no_orient:
                new = orient(new)
            if not np.array_equal(new, pts):
                n_moved += 1
            # Only the row order changes, so `pts_num` and every sibling
            # field remain correct untouched.
            vec['pts'] = new.tolist()
        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{len(results)} tiles  ({time.time() - t0:.0f}s)',
                  file=sys.stderr)

    if args.report:
        print(f'after    {fmt_stats(zigzag_stats(tile_lines(sample)))}')

    parent = os.path.dirname(os.path.abspath(out))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(blob, f)
    print(f'{n_vec} polylines ({n_moved} reordered, '
          f'{n_vec - n_moved} already in order) in {len(results)} tiles '
          f'-> {out}  [{time.time() - t0:.0f}s]')
    return 0


if __name__ == '__main__':
    sys.exit(main())
