"""Tests for text-polyline (Carla_PC_LLM `eval_*.json`) results in the viewer.

Same conventions as test_viewer_views_tags.py: the real Flask app through
its test client, HTML parsed with html.parser -- never curl on a URL built
here (the `&gt=` entity trap in CLAUDE.md). Fixture: a 2-tile synthetic
export with the 50 m layout (tile_radius 25, class_lookup driving/curb) and a
work dir holding an eval_all.json in the LLM's own schema.
"""
import json, os, sys, tempfile
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import dataset_viewer as V

fails = []
def ck(name, cond, extra=''):
    print(('PASS ' if cond else 'FAIL ') + name + (('  ' + str(extra)) if extra else ''))
    if not cond:
        fails.append(name)


class Grab(HTMLParser):
    def __init__(self):
        super().__init__()
        self.selects, self._cur, self.imgs, self.title, self._it = {}, None, [], '', False
        self.text = []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'select':
            self._cur = a.get('name'); self.selects[self._cur] = []
        elif tag == 'option' and self._cur:
            self.selects[self._cur].append((a.get('value'), 'selected' in a))
        elif tag == 'img':
            self.imgs.append(a.get('src'))
        elif tag == 'title':
            self._it = True
    def handle_endtag(self, tag):
        if tag == 'select': self._cur = None
        elif tag == 'title': self._it = False
    def handle_data(self, d):
        if self._it: self.title += d
        self.text.append(d)


# --------------------------------------------------------------- fixtures
DATA = tempfile.mkdtemp(prefix='viewer_text_')
DS = os.path.join(DATA, 'test')
WD = tempfile.mkdtemp(prefix='viewer_text_wd_')
os.makedirs(os.path.join(DS, 'blocks'))
os.makedirs(os.path.join(DS, 'reference_lines'))
R = 25.0
CENTRES = {'tile_00000': [1000.0, 2000.0], 'tile_00001': [1050.0, 2000.0]}
rng = np.random.RandomState(0)
manifest = {'split': 'test', 'tile_radius': R, 'tile_side': 2 * R,
            'class_lookup': {'0': 'driving', '1': 'curb'}, 'tiles': []}
# GT, world frame: one driving line and one curb line per tile
GT = {
    'tile_00000': [('driving', [[-20.0, 0.0], [20.0, 0.0]]),
                   ('curb', [[-20.0, 10.0], [20.0, 10.0]])],
    'tile_00001': [('driving', [[0.0, -20.0], [0.0, 20.0]])],
}
for nm, c in CENTRES.items():
    n = 3000
    xy = rng.uniform(-R, R, size=(n, 2))
    z = rng.uniform(-1, 1, size=(n, 1))
    feat = np.hstack([xy, z, rng.uniform(0, 1, size=(n, 3))]).astype(np.float32)
    off = np.array([c[0] + 3.0, c[1] - 2.0, 7.0], dtype=np.float32)
    np.savez(os.path.join(DS, 'blocks', f'{nm}.npz'),
             features=feat, labels=np.zeros(n, dtype=np.int32) - 1,
             points=(feat[:, :3] + off), offset=off,
             tile_center=np.array([c[0], c[1], 7.0], dtype=np.float32),
             tile_radius=np.float32(R))
    pl = {'tile_center': [c[0], c[1], 7.0], 'tile_radius': R,
          'classes': {'0': 'driving', '1': 'curb'},
          'polylines': [{'class_id': 0 if cls == 'driving' else 1, 'class': cls,
                         'type': 'straight', 'is_arc': False, 'road_id': 'r',
                         'points': [[c[0] + x, c[1] + y, 7.0] for x, y in pts]}
                        for cls, pts in GT[nm]]}
    with open(os.path.join(DS, 'reference_lines', f'{nm}_reference_lines.json'), 'w') as f:
        json.dump(pl, f)
    manifest['tiles'].append({'name': nm, 'n_points': n, 'center': c,
                              'n_polylines': len(GT[nm])})
with open(os.path.join(DS, 'manifest.json'), 'w') as f:
    json.dump(manifest, f)


def q(m):
    """The LLM's quantise(): floor((m + 25) / 0.1), clipped to [0, 499]."""
    return int(np.clip(np.floor((m + R) / 0.1), 0, 499))


def line(cls, pts):
    return f'{cls}: ' + ' '.join(f'{q(x)},{q(y)}' for x, y in pts)


# tile 0: driving exact, curb 0.7 m off (inside 1.0 m, outside 0.5 m), plus
# a bogus extra driving line far away and one unparsable line.
# tile 1: exact, plus a vertex outside the grid (the LLM can emit any int).
EVAL = {
    'metrics': {'driving': {'mAP': 0.5, 'AP@1.0': 0.5, 'n_pred': 3, 'n_gt': 2,
                            'chamfer': 0.3},
                'curb': {'mAP': 0.25, 'n_pred': 1, 'n_gt': 1},
                'overall': {'mAP': 0.375}, 'n_tiles': 2, 'unparsable_lines': 1,
                'subset': 'all', 'step': 'final'},
    'predictions': [
        {'name': 'tile_00000', 'pred': '\n'.join([
            line('driving', [[-20, 0], [20, 0]]),
            line('curb', [[-20, 10.7], [20, 10.7]]),
            line('driving', [[-20, -20], [20, -20]]),
            'driving: 0,477 4x,273 92,0'])},
        {'name': 'tile_00001', 'pred': line('driving', [[0, -20], [0, 20]])
                                       + '\ndriving: 600,-150 700,-150'},
    ],
}
with open(os.path.join(WD, 'eval_all.json'), 'w') as f:
    json.dump(EVAL, f)
# a sibling that must NOT be picked up as predictions
with open(os.path.join(WD, 'run.json'), 'w') as f:
    json.dump({'llm': 'x'}, f)
with open(os.path.join(WD, 'val_history.json'), 'w') as f:
    json.dump([{'step': 1, 'val_loss': 0.5}], f)

V.STATE.update({
    'data_root': DATA, 'max_points': 150000, 'work_dir': WD,
    'results_cache': {}, 'results_meta': {}, 'text_grid': (25.0, 0.1),
    'gt_cache': {}, 'shape_cache': {}, 'frame_cache': {},
    'frame': 'tile_center', 'gt_json': None, 'deep': {},
    'scan_grid': (0.1, 0.1, 0.4), 'scan_workers': 2, 'scan_stride': 1,
    'pc_range_z': (-72.0, 96.0), 'num_pts_per_vec': 20,
    'cache_dir': tempfile.mkdtemp(prefix='viewer_cache_'),
    'tags_file': None, 'density_max': None, 'density_ceiling_cache': None,
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
app = V.app.test_client()
os.chdir(tempfile.mkdtemp(prefix='viewer_text_cwd_'))   # no data/carla here

# --------------------------------------------------------------- parser
vecs, bad, out = V.parse_pred_text('driving: 0,0 499,499\ncurb: 250,250 251,251')
ck('two lines parse', len(vecs) == 2 and bad == 0 and out == 0)
ck('cell 0 -> -24.95 m (cell centre)', np.allclose(vecs[0][0][0], [-24.95, -24.95]))
ck('cell 499 -> +24.95 m', np.allclose(vecs[0][0][1], [24.95, 24.95]))
ck('cell 250 -> +0.05 m', np.allclose(vecs[1][0][0], [0.05, 0.05], atol=1e-5))
ck('class names kept', [c for _p, c in vecs] == ['driving', 'curb'])
vecs, bad, out = V.parse_pred_text('none')
ck('"none" is zero lines, not a bad line', vecs == [] and bad == 0)
vecs, bad, out = V.parse_pred_text('driving: 0,477 4x,273 92,0\ndriving: 1,1')
ck('a stray non-digit rejects the whole line; <2 vertices too', vecs == [] and bad == 2)
vecs, bad, out = V.parse_pred_text('driving: 600,-150 700,-150')
ck('out-of-grid vertices counted, kept, decoded', len(vecs) == 1 and out == 2
   and np.allclose(vecs[0][0][0], [35.05, -39.95]))
vecs, bad, out = V.parse_pred_text('road: 1,1 2,2', classes=('driving', 'curb'))
ck('unknown class is a bad line when a taxonomy is given', vecs == [] and bad == 1)
vecs, _b, _o = V.parse_pred_text('driving: 0,0 100,0', extent=15.0, res=0.1)
ck('--text-grid extent honoured', np.allclose(vecs[0][0][0], [-14.95, -14.95]))

# --------------------------------------------------------------- discovery / loading
found = V.discover_results(WD)
ck('eval_all.json discovered, run.json / val_history.json not',
   list(found) == ['eval_all.json'], list(found))
path = found['eval_all.json']
preds = V.load_results(path)
meta = V.results_meta(path)
ck('schema detected as text, score-less', meta['kind'] == 'text' and meta['has_scores'] is False)
ck('unparsable count matches the file\'s own', meta['n_bad'] == 1 == EVAL['metrics']['unparsable_lines']
   and meta['bad_by_token'] == {'tile_00000': 1})
ck('out-of-grid vertices counted', meta['n_out_of_grid'] == 2)
ck('stored metrics carried', meta['metrics']['overall']['mAP'] == 0.375)
ck('per-tile prediction lists', len(preds['tile_00000']) == 3 and len(preds['tile_00001']) == 2)
ck('every prediction scores 1.0', all(s == 1.0 for v in preds.values() for _p, s, _c in v))
ck('carlamap schema still loads', (lambda: (
    json.dump({'results': [{'sample_token': 'tile_00000', 'vectors': [
        {'pts': [[0, 0], [1, 1]], 'confidence_level': 0.4, 'cls_name': 'driving'}]}]},
              open(os.path.join(WD, 'sub_results.json'), 'w')),
    V.load_results(os.path.join(WD, 'sub_results.json')),
    V.results_meta(os.path.join(WD, 'sub_results.json')))[-1])() == {'kind': 'carlamap', 'has_scores': True})

# --------------------------------------------------------------- GT from reference_lines
gts = V.discover_gt(WD)
ref = os.path.abspath(os.path.join(DS, 'reference_lines'))
ck('reference_lines dir offered as GT', ref in gts, gts)
ck('its frame is tile_center', V.pkl_gt_frame(ref) == 'tile_center')
ck('its classes follow class_lookup id order', V.gt_classes(ref) == ('driving', 'curb'))
g = V.load_gt(ref)
ck('one GT entry per tile', sorted(g) == ['tile_00000', 'tile_00001'])
ck('GT centred on the tile (world - centre)',
   np.allclose(g['tile_00000'][0][0], [[-20, 0], [20, 0]]) and g['tile_00000'][0][1] == 0
   and g['tile_00000'][1][1] == 1)
ck('best_gt_for picks it by token overlap', V.best_gt_for(path, gts) == ref)

# --------------------------------------------------------------- eval with no scores
ck('tied scores -> AP = precision x recall',
   abs(V.ap_from_tpfp(np.array([1., 0, 1]), np.array([0., 1, 0]), np.ones(3), 4) - (2 / 4) * (2 / 3)) < 1e-12)
ck('distinct scores keep the ranked AP',
   abs(V.ap_from_tpfp(np.array([1., 0]), np.array([0., 1]), np.array([0.9, 0.2]), 1) - 1.0) < 1e-12)
rows = V.evaluate_run(path, ref, 0.0)
r0 = next(r for r in rows if r['name'] == 'tile_00000')
r1 = next(r for r in rows if r['name'] == 'tile_00001')
ck('per-class split engaged', V.REval['per_class'] is True)
ck('tile 0: 3 preds (bad line dropped), 2 GT', r0['n_pred'] == 3 and r0['n_gt'] == 2)
# driving: 2 preds (one exact, one bogus) vs 1 GT -> P 1/2, R 1 -> 0.5 at every thr
# curb: 0.7 m off -> 0 at 0.5 m, 1 at 1.0 / 1.5 m
ck('tile 0 AP@0.5 = mean(0.5, 0)', abs(r0['ap_thr'][0.5] - 0.25) < 1e-9, r0['ap_thr'])
ck('tile 0 AP@1.0 = mean(0.5, 1)', abs(r0['ap_thr'][1.0] - 0.75) < 1e-9, r0['ap_thr'])
ck('tile 0 count error +1, TP 2 @ 1.0 m', r0['count_error'] == 1 and r0['n_tp'] == 2)
ck('tile 1: exact line + out-of-grid line -> AP 0.5, count error +1',
   abs(r1['ap'] - 0.5) < 1e-9 and r1['count_error'] == 1, (r1['ap'], r1['count_error']))
ga = V.REval['global_ap']
# global @1.0: driving pooled P=2/4 R=2/2 -> 0.5 ; curb P=R=1 -> 1 ; mean 0.75
ck('global AP@1.0 pools per class then averages', abs(ga[1.0] - 0.75) < 1e-9, ga)

# --------------------------------------------------------------- results page
# sub_results.json (written above) is newer, so name the file
r = app.get('/?tab=results&count=2&results=eval_all.json')
ck('results page 200', r.status_code == 200)
gr = Grab(); gr.feed(r.get_data(as_text=True))
txt = ' '.join(gr.text)
ck('title', gr.title == 'CARLA training results', gr.title)
ck('results select lists both files, eval_all.json selected',
   sorted(v for v, _s in gr.selects['results']) == ['eval_all.json', 'sub_results.json']
   and ('eval_all.json', True) in gr.selects['results'], gr.selects['results'])
ck('GT select offers and selects the reference dir', (ref, True) in gr.selects['gt'])
ck('top_n defaults to all for score-less predictions', ('all', True) in gr.selects['top_n'])
ck('text-results note present', 'Text-polyline' in txt)
ck('unparsable line reported', '1 unparsable line' in txt)
ck('out-of-grid vertices reported', 'outside the' in txt and '500-cell' in txt)
ck('stored metrics table present', 'As recorded in the file' in txt and '0.3750' in txt)
ck('fixed-num_vec count warning suppressed', 'Count metrics are not informative' not in txt)
ck('no frame mismatch', 'Frame mismatch' not in txt)
srcs = [s for s in gr.imgs if s.startswith('/tile.png')]
ck('tiles drawn', len(srcs) >= 2)
qs = parse_qs(urlparse(srcs[0]).query)
ck('tile img carries frame + gt, no top_n', qs.get('frame') == ['tile_center'] and qs.get('gt') == [ref]
   and 'top_n' not in qs, {k: v for k, v in qs.items() if k in ('frame', 'gt', 'top_n')})
r2 = app.get(srcs[0])
ck('tile png renders', r2.status_code == 200 and r2.data[:4] == b'\x89PNG')

# a carlamap file keeps the old default
r = app.get('/?tab=results&count=1&results=sub_results.json')
gr2 = Grab(); gr2.feed(r.get_data(as_text=True))
ck('scored results keep top_n=gt default', ('gt', True) in gr2.selects['top_n'])
ck('no text note on a carlamap file', 'Text-polyline' not in ' '.join(gr2.text))

# --------------------------------------------------------------- browse tab / legend
r = app.get('/?count=2&polylines=1&results=eval_all.json')
ck('browse page 200 with predictions', r.status_code == 200 and 'no preds for this' not in r.get_data(as_text=True))
items = V.tile_metrics('test/tile_00000', path)
labels = [lab for _s, lab, _v in items]
ck('caption says "predictions", not a threshold', 'predictions' in labels and
   not any(l.startswith('predictions ≥') for l in labels), labels)
# legend text: render and check the Line2D labels through the renderer's
# own path -- the legend labels are the only place the threshold shows
import matplotlib
fig_labels = []
_orig = V.Line2D
class _L(_orig):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); fig_labels.append(k.get('label'))
V.Line2D = _L
try:
    V.render_tile('tile_00000', mode='points', show_polylines=True,
                  point_size=1.5, max_points=150000, ds='test',
                  results_path=path, score_thresh=0.3, frame='tile_center')
finally:
    V.Line2D = _orig
pl = [l for l in fig_labels if l and l.startswith('pred')]
ck('legend omits "@ score" for score-less predictions',
   pl and all('score' not in l for l in pl), pl)

print()
print(f'{len(fails)} failure(s)' if fails else 'ALL PASS')
sys.exit(1 if fails else 0)
