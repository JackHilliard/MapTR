"""Merging, blending, smoothing and re-clipping of tile-clipped polylines.

Pure numpy, no torch / mmdet / shapely -- deliberately, so the same code runs
inside the training container (``CustomCarlaNeighbourhoodDataset`` stitches
its GT with it) AND on the host (``tools/maptrv2/stitch_results.py`` merges
predictions with it, like ``dataset_viewer.py`` and ``reorder_results.py``
run without the container). The host tool loads this file by path because
``projects.mmdet3d_plugin`` cannot be imported without torch.

The problem it solves: the CARLA export cuts the world into 30 m tiles on a
24 m stride (``overlap`` 0.2), so every road line that crosses a tile edge
exists as several *pieces*, one per tile, each clipped to its own box and
overlapping its neighbours' pieces by up to 6 m. That is true of the
reference-line GT (the json for each tile holds its own clipped copy of the
same OpenDRIVE ``road_id``) and of the predictions (each tile's model run
sees the line to its own edge and no further). Putting tiles back together
-- for a multi-tile training sample, or to smooth a per-tile prediction with
what its neighbours saw -- means recognising those pieces as one line and
joining them.

``merge_pieces`` is the one algorithm, used in two regimes:

* **GT** (exact): pieces are clipped copies of one master polyline, so
  their shared vertices coincide to float precision and a tile-edge vertex
  lies exactly on the other piece's segment. Keyed by ``(class, road_id)``
  with ``tol`` 5 cm, the join is exact -- no heuristics decide which lines
  are the same, the export's own ids do.
* **predictions** (fuzzy): keyed by class only, ``tol`` ~1 m, and never two
  pieces from the same tile (the model already decided those were distinct
  instances). With ``consensus=True`` the overlap zone is blended: the
  master's vertices are pulled toward their projection on the other piece,
  weighted by confidence, which is the "amend each tile from its
  neighbours" idea in its simplest form.
"""

from __future__ import annotations

from typing import Callable, Dict, Hashable, List, Sequence

import numpy as np

__all__ = [
    'Piece', 'polyline_distance', 'polyline_length', 'resample_polyline',
    'merge_pieces', 'merge_grouped', 'chaikin', 'savgol', 'smoothing_spline',
    'gaussian_smooth', 'laplacian_smooth', 'douglas_peucker', 'smooth',
    'SMOOTHERS', 'clip_polyline_to_box',
    'clip_to_box', 'shift_polylines',
]


class Piece(dict):
    """A polyline fragment: ``points`` (N, 2|3), plus whatever the caller
    wants to carry along (``key``, ``score``, ``source``, ``road_id``...).

    A dict so it round-trips through json/pickle without a class; the
    attribute access is sugar."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def polyline_length(pts) -> float:
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1).sum())


def _cumlen(poly):
    seg = np.linalg.norm(np.diff(poly[:, :2], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)]), seg


def polyline_distance(pts, poly):
    """Distance from each point to a polyline, plus where it lands.

    Returns ``(dist, s, proj)``: ``dist`` (N,) is the Euclidean xy distance
    to the nearest segment, ``s`` (N,) the arc-length position of that
    nearest point along ``poly`` (so the caller can tell which way a piece
    runs relative to the polyline), and ``proj`` (N, 2) the nearest point
    itself. A one-vertex ``poly`` degenerates to a point. xy only: z is
    real terrain on the newer exports and would make a 2D overlap test
    fail on a ramp.
    """
    pts = np.asarray(pts, dtype=np.float64)[:, :2]
    poly = np.asarray(poly, dtype=np.float64)[:, :2]
    if len(poly) == 1:
        d = np.linalg.norm(pts - poly[0], axis=1)
        return d, np.zeros(len(pts)), np.repeat(poly[:1], len(pts), 0)
    a, b = poly[:-1], poly[1:]
    ab = b - a
    l2 = (ab ** 2).sum(1)
    ap = pts[:, None, :] - a[None]
    t = (ap * ab[None]).sum(-1) / np.maximum(l2[None], 1e-12)
    t = np.clip(t, 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    d = np.linalg.norm(pts[:, None, :] - proj, axis=-1)
    k = d.argmin(1)
    rows = np.arange(len(pts))
    cum, seg = _cumlen(poly)
    s = cum[k] + t[rows, k] * seg[k]
    return d[rows, k], s, proj[rows, k]


def resample_polyline(pts, n):
    """``n`` equally spaced points along the arc length (linear between
    vertices, i.e. what shapely's ``interpolate`` does and what the chamfer
    eval applies to both sides)."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 1:
        return np.repeat(pts, n, axis=0)
    cum, seg = _cumlen(pts)
    total = cum[-1]
    if total <= 0:
        return np.repeat(pts[:1], n, axis=0)
    want = np.linspace(0.0, total, n)
    idx = np.clip(np.searchsorted(cum, want, side='right') - 1, 0,
                  len(seg) - 1)
    frac = (want - cum[idx]) / np.maximum(seg[idx], 1e-12)
    return pts[idx] + frac[:, None] * (pts[idx + 1] - pts[idx])


def chaikin(pts, iters=2):
    """Chaikin corner cutting; endpoints are pinned so a stitched line does
    not shrink away from the tile edge it was clipped at."""
    pts = np.asarray(pts, dtype=np.float64)
    for _ in range(iters):
        if len(pts) < 3:
            return pts
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        mid = np.empty((2 * (len(pts) - 1), pts.shape[1]))
        mid[0::2], mid[1::2] = q, r
        pts = np.concatenate([pts[:1], mid, pts[-1:]], axis=0)
    return pts


def shift_polylines(polys, shift):
    shift = np.asarray(shift, dtype=np.float64)
    out = []
    for p in polys:
        p = np.asarray(p, dtype=np.float64).copy()
        k = min(p.shape[1], len(shift))
        p[:, :k] += shift[:k]
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# clipping (Liang-Barsky), moved here from carla50m_crop_dataset.py -- it was
# vendored there from Pointcept's grid_tile_export.py so crop GT is clipped
# by EXACTLY the tiler's logic (boundary vertex placed on the edge, runs of
# <2 vertices dropped). Same code, importable without torch now.
# ---------------------------------------------------------------------------
def _clip_segment_to_box(p0, p1, box):
    """Returns (q0, q1, t0, t1) clipped to box, or None."""
    xmin, ymin, xmax, ymax = box
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin),
                 (dy, ymax - y0)):
        if abs(p) < 1e-12:
            if q < 0:
                return None
        else:
            r = q / p
            if p < 0:
                if r > t1:
                    return None
                if r > t0:
                    t0 = r
            else:
                if r < t0:
                    return None
                if r < t1:
                    t1 = r
    q0 = p0 + t0 * (p1 - p0)
    q1 = p0 + t1 * (p1 - p0)
    return q0, q1, t0, t1


def clip_polyline_to_box(points, curved, box) -> List[dict]:
    """Clip a polyline to a box -> continuous sub-polylines reaching the
    edge. ``curved`` is a per-vertex flag carried along (pass None to skip);
    ``box`` is ``(xmin, ymin, xmax, ymax)``. Extra columns (z) are clipped
    linearly with xy."""
    points = np.asarray(points, dtype=np.float64)
    if curved is None:
        curved = np.zeros(len(points), dtype=bool)
    out: List[dict] = []
    cur_pts: List[np.ndarray] = []
    cur_cur: List[bool] = []

    def flush():
        if len(cur_pts) >= 2:
            out.append({
                'points': np.asarray(cur_pts, dtype=np.float32),
                'curved': np.asarray(cur_cur, dtype=bool)
            })
        cur_pts.clear()
        cur_cur.clear()

    for i in range(len(points) - 1):
        res = _clip_segment_to_box(points[i], points[i + 1], box)
        if res is None:
            flush()
            continue
        q0, q1, t0, t1 = res
        c_i = bool(curved[i])
        if not cur_pts:
            cur_pts.append(q0)
            cur_cur.append(c_i)
        elif np.linalg.norm(cur_pts[-1][:2] - q0[:2]) > 1e-4:
            flush()
            cur_pts.append(q0)
            cur_cur.append(c_i)
        cur_pts.append(q1)
        cur_cur.append(bool(curved[i + 1]) if t1 >= 1.0 - 1e-9 else c_i)
        if t1 < 1.0 - 1e-9:
            flush()
    flush()
    return out


def clip_to_box(polys, box, min_len=0.0):
    """Clip many polylines; drop sub-polylines shorter than ``min_len``."""
    out = []
    for p in polys:
        for sub in clip_polyline_to_box(p, None, box):
            if polyline_length(sub['points']) >= min_len:
                out.append(sub['points'])
    return out


# ---------------------------------------------------------------------------
# merging
# ---------------------------------------------------------------------------
def _runs(mask):
    """Length of the leading and trailing False runs of a bool mask, and
    whether everything in between is True."""
    n = len(mask)
    lead = 0
    while lead < n and not mask[lead]:
        lead += 1
    trail = 0
    while trail < n - lead and not mask[n - 1 - trail]:
        trail += 1
    middle_ok = bool(mask[lead:n - trail].all()) if lead < n - trail else True
    return lead, trail, middle_ok


def _orient(piece, master, tol):
    """Flip ``piece`` so it runs in ``master``'s direction, judged by the
    arc-length positions of its covered vertices. Returns the (possibly
    flipped) points and the coverage mask, or None if nothing is covered."""
    d, s, _ = polyline_distance(piece, master)
    cov = d <= tol
    if not cov.any():
        return None
    idx = np.flatnonzero(cov)
    flip = False
    if len(idx) >= 2:
        trend = s[idx[-1]] - s[idx[0]]
        if trend < 0:
            flip = True
        elif trend == 0:
            # all covered vertices project to one spot (a corner touch):
            # decide by which end of the piece is free, as below
            flip = _flip_single(idx, s, len(piece), master)
    else:
        flip = _flip_single(idx, s, len(piece), master)
    if flip:
        piece = piece[::-1]
        cov = cov[::-1]
    return piece, cov


def _flip_single(idx, s, n, master):
    """One covered vertex: if it sits nearer the master's END the piece will
    be appended, so the covered vertex must be the piece's FIRST vertex
    (uncovered run trailing); nearer the START it is prepended, so the
    covered vertex must be its LAST."""
    total = polyline_length(master)
    near_end = s[idx[0]] > total / 2.0
    pos = idx[len(idx) // 2]
    return (pos > n / 2.0) if near_end else (pos < n / 2.0)


def _blend(master, piece, tol, w_master, w_piece):
    """Pull master vertices inside the overlap toward the piece."""
    d, _s, proj = polyline_distance(master, piece)
    inside = d <= tol
    if not inside.any():
        return master
    master = master.copy()
    wsum = w_master + w_piece
    master[inside, :2] = (w_master * master[inside, :2]
                          + w_piece * proj[inside]) / wsum
    return master


def _try_join(master, piece, tol, consensus, w_master, w_piece):
    """Join ``piece`` onto ``master`` if it is the same line.

    The piece may extend the master at its start, its end, both, or not at
    all (a duplicate, absorbed). Anything else -- a piece that leaves the
    master's locus in the middle -- is a different line and is refused.
    """
    o = _orient(piece, master, tol)
    if o is None:
        return None
    piece, cov = o
    lead, trail, middle_ok = _runs(cov)
    if not middle_ok:
        return None
    # An extension must leave the master at its END (or enter at its
    # START), not fork off partway along it. "Near" is judged in the piece's
    # own (median) vertex spacing: a sparse GT piece (vertices ~10 m apart)
    # may have its last covered vertex well before the master's end and
    # still be a true continuation, whereas a densely sampled prediction
    # that leaves the locus a metre before the end is a fork. Median rather
    # than max so one long jump segment cannot widen its own allowance.
    n = len(piece)
    _d, s, _p = polyline_distance(piece, master)
    idx = np.flatnonzero(cov)
    seg = np.linalg.norm(np.diff(piece[:, :2], axis=0), axis=1)
    slack = max(2.0 * tol, float(np.median(seg)) if len(seg) else 0.0) + tol
    total = polyline_length(master)
    if trail and s[idx[-1]] < total - slack:
        return None
    if lead and s[idx[0]] > slack:
        return None
    # ...and continuously: the first free vertex must sit within the same
    # slack of the master's endpoint, so a piece that jumps sideways off the
    # end is not stitched on either.
    if trail and np.linalg.norm(piece[n - trail, :2] - master[-1, :2]) > slack:
        return None
    if lead and np.linalg.norm(piece[lead - 1, :2] - master[0, :2]) > slack:
        return None
    if consensus:
        master = _blend(master, piece, tol, w_master, w_piece)
    parts = []
    if lead:
        parts.append(piece[:lead])
    parts.append(master)
    if trail:
        parts.append(piece[n - trail:])
    return np.concatenate(parts, axis=0)


def _bbox(pts):
    return np.array([pts[:, 0].min(), pts[:, 1].min(),
                     pts[:, 0].max(), pts[:, 1].max()])


def merge_pieces(pieces: Sequence[Piece], tol: float, consensus=False,
                 min_len: float = 0.0) -> List[Piece]:
    """Merge fragments of the same line into one polyline each.

    ``pieces`` must already be one group (same class, and for GT the same
    ``road_id`` where that means something -- the export writes a real
    OpenDRIVE id on driving lines but the literal ``'curb'`` on every curb):
    this function only asks whether two fragments overlap or abut
    geometrically within ``tol``. Each piece carries ``points``; an optional
    ``score`` weights the consensus blend and becomes the merged line's
    score (max over members); an optional ``source`` forbids merging two
    pieces of the same source -- two instances the export (or the model)
    emitted for ONE tile stay two instances, so only tile-edge joins happen.

    Greedy: the longest piece seeds a master and absorbs every piece that
    joins it, repeatedly, until none does; leftovers seed further masters.
    Pieces that never overlap anything come out unchanged. The merged
    ``members`` list records what went in.

    Candidates are prefiltered by bounding box (expanded by ``tol``): a
    town-wide group -- one road id spans hundreds of tiles, and all curbs
    are one group -- is otherwise quadratic in pieces that are kilometres
    apart.
    """
    n = len(pieces)
    if n == 0:
        return []
    pts_list = [np.asarray(p['points'], dtype=np.float64) for p in pieces]
    boxes = np.stack([_bbox(p) for p in pts_list])
    lengths = np.array([polyline_length(p) for p in pts_list])
    alive = np.ones(n, dtype=bool)
    out = []
    for i in np.argsort(-lengths):
        if not alive[i]:
            continue
        alive[i] = False
        seed = pieces[i]
        master = pts_list[i]
        mbox = boxes[i].copy()
        members = [seed]
        src0 = seed.get('source')
        sources = {src0} if src0 is not None else set()
        score = float(seed.get('score', 1.0))
        changed = True
        while changed and alive.any():
            changed = False
            cand = np.flatnonzero(
                alive & (boxes[:, 0] <= mbox[2] + tol)
                & (boxes[:, 2] >= mbox[0] - tol)
                & (boxes[:, 1] <= mbox[3] + tol)
                & (boxes[:, 3] >= mbox[1] - tol))
            for j in cand:
                piece = pieces[j]
                src = piece.get('source')
                if src is not None and src in sources:
                    continue
                joined = _try_join(master, pts_list[j], tol, consensus,
                                   score, float(piece.get('score', 1.0)))
                if joined is None:
                    continue
                master = joined
                mbox = _bbox(master)
                members.append(piece)
                if src is not None:
                    sources.add(src)
                score = max(score, float(piece.get('score', 1.0)))
                alive[j] = False
                changed = True
        if polyline_length(master) < min_len:
            continue
        merged = Piece(seed)
        merged['points'] = master
        merged['score'] = score
        merged['members'] = members
        merged['n_members'] = len(members)
        out.append(merged)
    return out


def merge_grouped(pieces: Sequence[Piece],
                  key: Callable[[Piece], Hashable], tol: float,
                  consensus=False, min_len: float = 0.0) -> List[Piece]:
    """``merge_pieces`` within each ``key`` group, concatenated."""
    groups: Dict[Hashable, List[Piece]] = {}
    for p in pieces:
        groups.setdefault(key(p), []).append(p)
    out: List[Piece] = []
    for k in groups:
        out.extend(merge_pieces(groups[k], tol, consensus, min_len))
    return out


# ---------------------------------------------------------------------------
# smoothing
#
# Every smoother here (except Chaikin, which subdivides, and Douglas-Peucker,
# which only removes vertices) first resamples the line to evenly spaced
# vertices -- kernel and penalty methods assume that -- and pins both
# endpoints, so a stitched line never shrinks away from the tile edge it was
# clipped at. All numpy: the host has no guarantee of scipy and this file
# must import in both places.
# ---------------------------------------------------------------------------
def _even(pts, spacing):
    pts = np.asarray(pts, dtype=np.float64)
    n = max(3, int(round(polyline_length(pts) / spacing)) + 1)
    return resample_polyline(pts, n)


def _pin(out, pts):
    out[0], out[-1] = pts[0], pts[-1]
    return out


def savgol(pts, window=7, order=3, spacing=0.5):
    """Savitzky-Golay: least-squares local polynomial of ``order`` over a
    sliding ``window`` (odd) of evenly spaced vertices. Removes jitter while
    keeping curvature. Near the ends the window is shifted inward and the
    polynomial evaluated at the vertex's own offset; endpoints pinned."""
    pts = _even(pts, spacing)
    n = len(pts)
    window = min(window if window % 2 else window + 1, n if n % 2 else n - 1)
    if window < order + 2:
        return pts
    h = window // 2
    x = np.arange(-h, h + 1, dtype=np.float64)
    pinv = np.linalg.pinv(np.vander(x, order + 1, increasing=True))
    out = np.empty_like(pts)
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        if hi - lo < window:
            lo, hi = (0, window) if i < h else (n - window, n)
        t = np.array([i - lo - h], dtype=np.float64)
        w = np.vander(t, order + 1, increasing=True)[0] @ pinv
        out[i] = w @ pts[lo:hi]
    return _pin(out, pts)


def smoothing_spline(pts, lam=4.0, spacing=0.5):
    """Discrete cubic smoothing spline (Whittaker smoother): minimise
    ||z - p||^2 + lam * ||D2 z||^2 per coordinate over evenly spaced
    vertices, with the endpoints held by a large data weight. ``lam`` in
    vertex units^4: larger is smoother. Dense solve; lines here have tens of
    vertices."""
    pts = _even(pts, spacing)
    n = len(pts)
    if n < 4:
        return pts
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i:i + 3] = (1.0, -2.0, 1.0)
    W = np.ones(n)
    W[0] = W[-1] = 1e6
    A = np.diag(W) + lam * (D.T @ D)
    z = np.linalg.solve(A, W[:, None] * pts)
    return _pin(z, pts)


def gaussian_smooth(pts, sigma=1.5, spacing=0.5):
    """Gaussian kernel over the vertex index of an evenly spaced line
    (``sigma`` in vertices), renormalised at the ends, endpoints pinned."""
    pts = _even(pts, spacing)
    n = len(pts)
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    out = np.empty_like(pts)
    for i in range(n):
        lo, hi = max(0, i - r), min(n, i + r + 1)
        w = k[lo - i + r:hi - i + r]
        out[i] = (w[:, None] * pts[lo:hi]).sum(0) / w.sum()
    return _pin(out, pts)


def laplacian_smooth(pts, iters=20, step=0.5, fidelity=0.2, spacing=0.5):
    """Iterative Laplacian smoothing with a spring back to the original
    vertex: z <- z + step * (L z) - fidelity * (z - p). The fidelity term
    stops the inward drift that plain Laplacian (and Chaikin) shows on
    arcs."""
    p = _even(pts, spacing)
    z = p.copy()
    for _ in range(iters):
        lap = np.zeros_like(z)
        lap[1:-1] = 0.5 * (z[:-2] + z[2:]) - z[1:-1]
        z = z + step * lap - fidelity * (z - p)
        z[0], z[-1] = p[0], p[-1]
    return z


def douglas_peucker(pts, tol=0.3):
    """Ramer-Douglas-Peucker simplification: drop vertices within ``tol`` of
    the chord between kept ones. Removes jitter, keeps real corners sharp."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        d, _, _ = polyline_distance(pts[a + 1:b], pts[[a, b]])
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[a + 1 + k] = True
            stack.append((a, a + 1 + k))
            stack.append((a + 1 + k, b))
    return pts[keep]


SMOOTHERS = {
    'none': lambda p, **kw: np.asarray(p, dtype=np.float64),
    'chaikin': lambda p, iters=1, **kw: chaikin(p, iters),
    'savgol': lambda p, window=7, order=3, **kw: savgol(p, window, order),
    'spline': lambda p, lam=4.0, **kw: smoothing_spline(p, lam),
    'gaussian': lambda p, sigma=1.5, **kw: gaussian_smooth(p, sigma),
    'laplacian': lambda p, iters=20, step=0.5, fidelity=0.2, **kw:
        laplacian_smooth(p, iters, step, fidelity),
    'dp': lambda p, tol=0.3, **kw: douglas_peucker(p, tol),
}


def smooth(pts, method='chaikin', **params):
    """Dispatch by name; see ``SMOOTHERS`` for the parameter each takes."""
    if method not in SMOOTHERS:
        raise ValueError(f'unknown smoother {method!r}; '
                         f'one of {sorted(SMOOTHERS)}')
    return SMOOTHERS[method](pts, **params)
