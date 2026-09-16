"""Tests for map_utils/polyline_stitch.py -- host-side, numpy only.

    python3 tools/maptrv2/tests/test_polyline_stitch.py [--real <export>/test]

Synthetic cases cover both regimes (exact GT joins; fuzzy prediction joins
with forks, jumps, parallel lines and same-source pieces refused). With
``--real`` it also merges every tile's GT of a 30 m export by
(class, road_id) and checks the merge stays within tolerance of its pieces
(on ../carla_test: 13277 pieces -> 4020 lines in ~6 s, max residual 3.9 cm).
"""
import argparse, collections, importlib.util, json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
spec = importlib.util.spec_from_file_location(
    'polyline_stitch',
    os.path.join(REPO, 'projects/mmdet3d_plugin/datasets/map_utils/polyline_stitch.py'))
ps = importlib.util.module_from_spec(spec); spec.loader.exec_module(ps)


def synthetic():
    def line(x0, x1, n, y=0.0, yfun=None):
        x = np.linspace(x0, x1, n)
        y = np.full(n, y) if yfun is None else yfun(x)
        return np.stack([x, y], 1)

    # two clipped copies of one straight line, overlapping 6 m
    a = line(-15, 9, 25); b = line(3, 27, 25)
    m = ps.merge_pieces([ps.Piece(points=a), ps.Piece(points=b)], tol=0.05)
    assert len(m) == 1 and abs(ps.polyline_length(m[0]['points']) - 42) < 1e-6, m
    # reversed direction still joins
    m = ps.merge_pieces([ps.Piece(points=a), ps.Piece(points=b[::-1])], tol=0.05)
    assert len(m) == 1 and abs(ps.polyline_length(m[0]['points']) - 42) < 1e-6
    # abutting (touching endpoints, no overlap)
    m = ps.merge_pieces([ps.Piece(points=line(-15, 0, 10)), ps.Piece(points=line(0, 15, 10))], tol=0.05)
    assert len(m) == 1 and abs(ps.polyline_length(m[0]['points']) - 30) < 1e-6
    # disjoint pieces stay apart
    m = ps.merge_pieces([ps.Piece(points=line(-15, -5, 10)), ps.Piece(points=line(5, 15, 10))], tol=0.05)
    assert len(m) == 2
    # parallel lines 3.5 m apart never merge at 1 m tol
    m = ps.merge_pieces([ps.Piece(points=line(-15, 9, 25)), ps.Piece(points=line(3, 27, 25, y=3.5))], tol=1.0)
    assert len(m) == 2
    # a fork: shares the locus then leaves it before the master's end -> refused
    c = np.concatenate([line(3, 5, 3), line(5.5, 27, 30, yfun=lambda x: (x - 5) * 0.5)])
    m = ps.merge_pieces([ps.Piece(points=a), ps.Piece(points=c)], tol=0.05)
    assert len(m) == 2, m
    # a sideways jump off the end -> refused
    c = np.concatenate([line(3, 9, 7), line(10, 20, 11, y=5.0)])  # shorter than a, so a stays master
    m = ps.merge_pieces([ps.Piece(points=a), ps.Piece(points=c)], tol=0.05)
    assert len(m) == 2, m
    # same-source pieces never merge
    m = ps.merge_pieces([ps.Piece(points=a, source='t'), ps.Piece(points=b, source='t')], tol=0.05)
    assert len(m) == 2
    # three pieces of a curve (arc) in random order
    th = np.linspace(0, np.pi, 91); arc = np.stack([20*np.cos(th), 20*np.sin(th)], 1)
    parts = [arc[0:35], arc[30:65], arc[60:91]]
    m = ps.merge_pieces([ps.Piece(points=parts[1]), ps.Piece(points=parts[2][::-1]), ps.Piece(points=parts[0])], tol=0.05)
    assert len(m) == 1, len(m)
    assert abs(ps.polyline_length(m[0]['points']) - ps.polyline_length(arc)) < 1e-6
    # consensus: noisy duplicate pulls master halfway (equal weights)
    n1 = line(-15, 9, 25); n2 = line(3, 27, 25, y=0.4)
    m = ps.merge_pieces([ps.Piece(points=n1, score=1.0, source='a'), ps.Piece(points=n2, score=1.0, source='b')], tol=1.0, consensus=True)
    pts = m[0]['points']; inside = (pts[:, 0] > 3.5) & (pts[:, 0] < 8.5)
    assert np.allclose(pts[inside, 1], 0.2, atol=1e-6), pts[inside]
    # chaikin keeps endpoints
    ch = ps.chaikin(np.array([[0, 0], [1, 1], [2, 0.]]), 2)
    assert np.allclose(ch[0], [0, 0]) and np.allclose(ch[-1], [2, 0])
    # clip
    sub = ps.clip_to_box([line(-40, 40, 81)], (-15, -15, 15, 15))
    assert len(sub) == 1 and abs(sub[0][0, 0] + 15) < 1e-6 and abs(sub[0][-1, 0] - 15) < 1e-6
    print('synthetic OK')


def real(root):
    d = os.path.join(root, 'reference_curb_driving_lines')
    man = json.load(open(os.path.join(root, 'manifest.json')))
    pieces = []
    t0 = time.time()
    for t in man['tiles']:
        j = json.load(open(os.path.join(d, f"{t['name']}_reference_lines.json")))
        for pl in j['polylines']:
            p = np.asarray(pl['points'], dtype=np.float64)
            if len(p) < 2:
                continue
            pieces.append(ps.Piece(points=p, key=(pl['class'], pl['road_id']), source=t['name']))
    print(f'{len(pieces)} pieces loaded in {time.time()-t0:.1f}s')
    t0 = time.time()
    merged = ps.merge_grouped(pieces, key=lambda p: p['key'], tol=0.05)
    print(f'{len(merged)} merged lines in {time.time()-t0:.1f}s')
    hist = collections.Counter(m['n_members'] for m in merged)
    print('members per merged line', sorted(hist.items()))
    # exactness: every original vertex within 1e-3 of its merged line, and
    # every merged vertex within 1e-3 of some member piece
    worst_a = worst_b = 0.0
    for m in merged:
        mp = m['points']
        for mem in m['members']:
            dd, _, _ = ps.polyline_distance(mem['points'], mp)
            worst_a = max(worst_a, dd.max())
        dmin = np.full(len(mp), np.inf)
        for mem in m['members']:
            dd, _, _ = ps.polyline_distance(mp, mem['points'])
            dmin = np.minimum(dmin, dd)
        worst_b = max(worst_b, dmin.max())
    print(f'max vertex->merged {worst_a:.2e} m, max merged->pieces {worst_b:.2e} m')
    assert worst_a <= 0.05 and worst_b < 1e-3
    # how many (class, road_id) groups still hold >1 line (disjoint pieces)?
    per_key = collections.Counter(m['key'] for m in merged)
    multi = [k for k, v in per_key.items() if v > 1]
    print('groups with >1 merged line:', len(multi), 'of', len(per_key))
    gaps = []
    for k in multi:
        lines = [m['points'] for m in merged if m['key'] == k]
        best = np.inf
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                dd, _, _ = ps.polyline_distance(lines[i], lines[j]); best = min(best, dd.min())
        gaps.append(best)
    gaps = np.array(gaps)
    print('min gap between lines of one group: <0.05 m:', int((gaps < 0.05).sum()), ' 0.05-2 m:', int(((gaps >= 0.05) & (gaps < 2)).sum()), ' >=2 m:', int((gaps >= 2).sum()))
    touch = [k for k, g in zip(multi, gaps) if g < 0.05][:5]
    print('examples touching-but-unmerged:', touch)
    print('real GT OK')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', default=None, help='<export>/<split> directory')
    a = ap.parse_args()
    synthetic()
    if a.real:
        real(a.real)
