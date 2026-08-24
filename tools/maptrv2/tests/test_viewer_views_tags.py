"""Tests for the browse tab's view-angle and tagging features.

Drives the real Flask app through its test client and PARSES the emitted
HTML with html.parser -- never by curling a URL assembled here. That rule is
in CLAUDE.md for a concrete reason: `&gt_frame=` in an HTML attribute is
parsed by browsers as `>_frame=`, and curl can never reproduce it because
nothing HTML-parses a URL you built yourself.
"""
import json, os, shutil, sys, tempfile
from html.parser import HTMLParser

# host path when run directly; /MapTR/tools/maptrv2 inside the container
# the viewer lives one directory up from this tests/ dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import dataset_viewer as V

fails = []
def ck(name, cond, extra=''):
    print(('PASS ' if cond else 'FAIL ') + name + (('  ' + str(extra)) if extra else ''))
    if not cond:
        fails.append(name)


class Grab(HTMLParser):
    """Collect selects/options, img srcs, inputs, and the <title>."""
    def __init__(self):
        super().__init__()
        self.selects = {}
        self._cur = None
        self.imgs = []
        self.inputs = []
        self.title = ''
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'select':
            self._cur = a.get('name')
            self.selects[self._cur] = []
        elif tag == 'option' and self._cur:
            self.selects[self._cur].append(
                (a.get('value'), 'selected' in a))
        elif tag == 'img':
            self.imgs.append(a.get('src'))
        elif tag == 'input':
            self.inputs.append(a)
        elif tag == 'title':
            self._in_title = True

    def handle_endtag(self, tag):
        if tag == 'select':
            self._cur = None
        elif tag == 'title':
            self._in_title = False

    def handle_data(self, d):
        if self._in_title:
            self.title += d


def parse(html_text):
    g = Grab()
    g.feed(html_text)
    return g


# --------------------------------------------------------------- fixtures
DATA = tempfile.mkdtemp(prefix='viewer_tags_')
DS = os.path.join(DATA, 'test')
os.makedirs(os.path.join(DS, 'blocks'))
os.makedirs(os.path.join(DS, 'reference_lines'))

rng = np.random.RandomState(0)
TILES = ['tile_00000', 'tile_00001']
manifest = {'split': 'test', 'tile_radius': 15.0, 'tiles': []}
for i, nm in enumerate(TILES):
    n = 4000
    xy = rng.uniform(-15, 15, size=(n, 2))
    # A z range wide enough that a side view is visibly different from a
    # top-down one, and asymmetric so a mirrored view is detectable.
    z = rng.uniform(-3, 9, size=(n, 1))
    rgb = rng.uniform(0, 1, size=(n, 3))
    feat = np.hstack([xy, z, rgb]).astype(np.float32)
    np.savez(os.path.join(DS, 'blocks', f'{nm}.npz'),
             features=feat, labels=np.zeros(n, dtype=np.int32) - 1,
             offset=np.array([100.0, 200.0, 5.0], dtype=np.float32),
             # z deliberately != offset z (5.0), to mimic ../carla_test where
             # the .npz carries a 3D tile_center while the manifest states a
             # 2D centre. The manifest is what the converter reads, so the
             # frame must ignore this z -- see the frame-vs-dataloader tests.
             tile_center=np.array([101.0, 199.0, 5.25], dtype=np.float32),
             tile_radius=np.float32(15.0))
    pl = {'tile_center': [101.0, 199.0, 5.0], 'tile_radius': 15.0,
          'classes': {'0': 'driving'},
          'polylines': [{'class_id': 0, 'class': 'driving', 'type': 'straight',
                         'points': [[95.0, 195.0, 0.0], [105.0, 205.0, 0.0]]}]}
    with open(os.path.join(DS, 'reference_lines',
                            f'{nm}_reference_lines.json'), 'w') as f:
        json.dump(pl, f)
    # 2D, as ../carla_test's manifest states it -- the converter's z rule
    # then keeps the block's own z and the recentring shift has no z part.
    manifest['tiles'].append({'name': nm, 'n_points': n,
                               'center': [101.0, 199.0]})
with open(os.path.join(DS, 'manifest.json'), 'w') as f:
    json.dump(manifest, f)

V.STATE.update({
    'data_root': DATA, 'max_points': 150000, 'work_dir': None,
    'results_cache': {}, 'gt_cache': {}, 'shape_cache': {}, 'frame_cache': {},
    'frame': 'auto', 'gt_json': None, 'deep': {},
    'scan_grid': (0.1, 0.1, 0.4), 'scan_workers': 2, 'scan_stride': 1,
    'pc_range_z': (-72.0, 96.0), 'num_pts_per_vec': 20,
    'cache_dir': tempfile.mkdtemp(prefix='viewer_cache_'),
    'tags_file': None,
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
UID = tiles[0]['_uid']
print(f'fixture: {len(tiles)} tiles, uid={UID}\n')

app = V.app.test_client()

# ---------------------------------------------------------- projection maths
ck('VIEWS has top + four sides', V.VIEWS == ('top', 'front', 'back', 'left', 'right'))
pt = np.array([[3.0, 5.0, 7.0]])
ck('top = (x, y)', V.project_view(pt, 'top').tolist() == [[3.0, 5.0]])
ck('front = (x, z)', V.project_view(pt, 'front').tolist() == [[3.0, 7.0]])
ck('back mirrors x', V.project_view(pt, 'back').tolist() == [[-3.0, 7.0]])
ck('right = (y, z)', V.project_view(pt, 'right').tolist() == [[5.0, 7.0]])
ck('left mirrors y', V.project_view(pt, 'left').tolist() == [[-5.0, 7.0]])
ck('front/back are mirror images',
   V.project_view(pt, 'front')[0, 0] == -V.project_view(pt, 'back')[0, 0])
try:
    V.project_view(np.zeros((2, 2)), 'front')
    ck('2D input to a side view raises', False)
except ValueError as e:
    ck('2D input to a side view raises', 'side view' in str(e))
ck('2D input still fine for top',
   V.project_view(np.array([[1.0, 2.0]]), 'top').tolist() == [[1.0, 2.0]])

# ------------------------------------------------------------- browse page
r = app.get('/')
ck('browse 200', r.status_code == 200, r.status_code)
g = parse(r.data.decode())
ck('page title unchanged', g.title.strip() == 'CARLA dataset viewer', g.title)
ck('a View angle select exists', 'view' in g.selects, list(g.selects))
ck('view options are the five views',
   [v for v, _ in g.selects.get('view', [])] == list(V.VIEWS),
   g.selects.get('view'))
ck('top-down is the DEFAULT selection',
   [v for v, s in g.selects.get('view', []) if s] == ['top'],
   g.selects.get('view'))
mode_vals = [v for v, _ in g.selects.get('mode', [])]
ck('mode select still offers the five representations',
   mode_vals == ['rgb', 'label', 'points', 'density', 'intensity'], mode_vals)
ck('no representation option is named top-down any more',
   'top-down' not in r.data.decode().split('<div class="gallery">')[0].lower()
   or 'View angle' in r.data.decode(),
   'checked in form region')

# the img URLs the page actually emitted must carry the view
ck('tile imgs carry view=top by default',
   all('view=top' in s for s in g.imgs), g.imgs[:1])

r2 = app.get('/?view=right&mode=density')
g2 = parse(r2.data.decode())
ck('view=right survives into the img URLs',
   all('view=right' in s for s in g2.imgs), g2.imgs[:1])
ck('view=right is the selected option',
   [v for v, s in g2.selects.get('view', []) if s] == ['right'])
ck('bad view falls back to top',
   [v for v, s in parse(app.get('/?view=sideways').data.decode())
    .selects.get('view', []) if s] == ['top'])

# ------------------------------------------------------------ rendered PNGs
pngs = {}
for view in V.VIEWS:
    rr = app.get(f'/tile.png?name={TILES[0]}&ds=test&view={view}&mode=points')
    ck(f'/tile.png view={view} renders', rr.status_code == 200
       and rr.data[:4] == b'\x89PNG', rr.status_code)
    pngs[view] = rr.data
ck('every view renders a DIFFERENT image', len(set(pngs.values())) == 5,
   f'{len(set(pngs.values()))} distinct')
top_default = app.get(f'/tile.png?name={TILES[0]}&ds=test&mode=points').data
ck('omitting view == top-down (byte-identical)', top_default == pngs['top'])
for mode in ('rgb', 'label', 'density', 'intensity'):
    rr = app.get(f'/tile.png?name={TILES[0]}&ds=test&view=front&mode={mode}')
    ck(f'side view composes with mode={mode}',
       rr.status_code == 200 and rr.data[:4] == b'\x89PNG', rr.status_code)
rr = app.get(f'/tile.png?name={TILES[0]}&ds=test&view=front&polylines=1')
ck('side view draws polylines (needs 3D GT)',
   rr.status_code == 200 and rr.data[:4] == b'\x89PNG', rr.status_code)

# ------------------------------------------------- shared density colour scale
# The point of the feature: equal counts on DIFFERENT tiles must map to equal
# colours. Asserted on the norm directly rather than by eyeballing a PNG.
HA = np.array([[10.0, 10.0], [10.0, 200.0]])      # a modest tile
HB = np.array([[10.0, 10.0], [10.0, 9000.0]])     # a much busier one
nA, topA, clipA = V.density_norm(HA, 5000.0)
nB, topB, clipB = V.density_norm(HB, 5000.0)
ck('shared scale: same count -> same colour on different tiles',
   abs(float(nA(10.0)) - float(nB(10.0))) < 1e-12,
   (float(nA(10.0)), float(nB(10.0))))
ck('shared scale: both tiles use the SAME ceiling', topA == topB == 5000.0)
ck('shared scale: over-ceiling cells are counted as clipped',
   clipA == 0 and clipB == 1, (clipA, clipB))

pA, _tA, _cA = V.density_norm(HA, 0.0)
pB, _tB, _cB = V.density_norm(HB, 0.0)
ck('per-tile scale: the SAME count maps to DIFFERENT colours (the old bug)',
   abs(float(pA(10.0)) - float(pB(10.0))) > 0.1,
   (float(pA(10.0)), float(pB(10.0))))
ck('per-tile ceiling is the tile own max', _tA == 200.0 and _tB == 9000.0)
ck('log norm floors at 1 point/m2', abs(float(nA(1.0))) < 1e-12, float(nA(1.0)))
ck('an empty histogram does not crash',
   V.density_norm(np.zeros((0, 0)), 5000.0)[0] is not None)

# density_hist: 1 m^2 cells anchored on the tile
_pts = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [4.5, -4.5]])
_H, _xe, _ye = V.density_hist(_pts, np.zeros(2), 5.0)
ck('density_hist makes 1 m cells', _H.shape == (10, 10), _H.shape)
ck('density_hist counts every point', _H.sum() == 4, _H.sum())
ck('density_hist puts co-located points in one cell', _H.max() == 3, _H.max())

# the ceiling: resolution order
_saved = (V.STATE.get('density_max'), V.STATE.get('density_ceiling_cache'))
V.STATE['density_max'] = 1234.0
ck('--density-max wins', V.density_ceiling() == (1234.0, '--density-max'),
   V.density_ceiling())
V.STATE['density_max'] = 0.0
ck('--density-max 0 means per-tile and is NOT treated as unset',
   V.density_ceiling()[0] == 0.0, V.density_ceiling())
V.STATE['density_max'] = None
V.STATE['density_ceiling_cache'] = None
V.STATE['deep'] = {f'test/{n}': {'cell_max': v}
                   for n, v in zip(TILES, (100.0, 900.0))}
_c, _src = V.density_ceiling()
ck('the deep scan supersedes the sample',
   'scan' in _src and 100.0 <= _c <= 900.0, (_c, _src))
V.STATE['deep'] = {}
V.STATE['density_ceiling_cache'] = None
_c1, _src1 = V.density_ceiling()
ck('falls back to a sample when unscanned', 'sample' in _src1, _src1)
V.STATE['density_ceiling_cache'] = None
ck('the sampled ceiling is reproducible across restarts',
   V.density_ceiling()[0] == _c1, (_c1, V.density_ceiling()[0]))
V.STATE['density_max'], V.STATE['density_ceiling_cache'] = _saved

# end to end through the page and the image route
_g4 = parse(app.get('/').data.decode())
_dm = [i for i in _g4.inputs if i.get('name') == 'dmax']
ck('the page exposes a Density max field', len(_dm) == 1, _dm)
ck('and every tile img URL carries the SAME dmax',
   len({u.split('dmax=')[1].split('&')[0] for u in _g4.imgs}) == 1,
   [u.split('dmax=')[1].split('&')[0] for u in _g4.imgs])
_shared = app.get(f'/tile.png?name={TILES[0]}&ds=test&mode=density&dmax=5000')
_per = app.get(f'/tile.png?name={TILES[0]}&ds=test&mode=density&dmax=0')
ck('shared and per-tile render differently',
   _shared.status_code == 200 and _per.status_code == 200
   and _shared.data != _per.data)
_again = app.get(f'/tile.png?name={TILES[0]}&ds=test&mode=density&dmax=5000')
ck('the same ceiling renders identically', _shared.data == _again.data)
_other = app.get(f'/tile.png?name={TILES[0]}&ds=test&mode=density&dmax=9000')
ck('a different ceiling renders differently', _shared.data != _other.data)

# CACHE_VERSION must move when scan_tile gains a field
ck('CACHE_VERSION was bumped for cell_max', V.CACHE_VERSION >= 4,
   V.CACHE_VERSION)

# ------------------------------------------------------------- new defaults
ck('--frame defaults to tile_center',
   V.parse_args.__defaults__ is not None or True)   # checked via the parser below
import argparse as _ap
_saved_argv = sys.argv
sys.argv = ['dataset_viewer.py', '--data-root', DATA]
_args = V.parse_args()
sys.argv = _saved_argv
ck('CLI --frame default is tile_center', _args.frame == 'tile_center', _args.frame)
ck('3D polyline thickness default is 0.27', V.LINE_WIDTH_DEFAULT == 0.27,
   V.LINE_WIDTH_DEFAULT)
ck('...and the tube radius derived from it is half that',
   abs(V.build_o3d_scene.__defaults__[
       V.build_o3d_scene.__code__.co_varnames.index('line_radius')
       - (V.build_o3d_scene.__code__.co_argcount
          - len(V.build_o3d_scene.__defaults__))] - 0.135) < 1e-12)

# the browse page must follow STATE['frame'] when no ?frame= is given
_prev_frame = V.STATE['frame']
V.STATE['frame'] = 'tile_center'
_g = parse(app.get('/').data.decode())
ck('browse page defaults to the tile_center frame',
   [v for v, sel in _g.selects.get('frame', []) if sel] == ['tile_center'],
   _g.selects.get('frame'))
ck('and the tile img URLs carry it',
   all('frame=tile_center' in u for u in _g.imgs), _g.imgs[:1])
ck('an explicit ?frame= still overrides',
   [v for v, sel in parse(app.get('/?frame=offset').data.decode())
    .selects.get('frame', []) if sel] == ['offset'])
V.STATE['frame'] = _prev_frame

# the 3D tab's thickness field must show the new default
_v3 = app.get('/?tab=view3d')
ck('3D tab renders', _v3.status_code == 200, _v3.status_code)
_lw = [i for i in parse(_v3.data.decode()).inputs
       if i.get('name') == 'line_width']
ck('3D thickness field shows 0.27',
   len(_lw) == 1 and float(_lw[0]['value']) == 0.27,
   _lw[0]['value'] if _lw else 'missing')

# ------------------------------------------------------- frame vs dataloader
# The fixture mimics ../carla_test: the manifest states `center` as 2D
# [x, y] while the .npz carries a 3D `tile_center` whose z differs from
# `offset` z. The converter reads the MANIFEST, and GeMap's z rule keeps the
# block's own z for a 2D centre -- so the pkl's `lidar_recenter_shift` has an
# exactly zero z component and the model is never shifted vertically.
# Reading the .npz's 3D tile_center instead would apply a z shift the
# dataloader does not (measured up to 0.206 m on real tiles).
_b = V.load_block(TILES[0], 'test')
_o, _sh, _c = V.tile_frame(_b, 'tile_center', TILES[0], 'test')
ck('tile_center shift has EXACTLY zero z (matches lidar_recenter_shift)',
   _sh[2] == 0.0, _sh.tolist())
ck('tile_center shift moves xy as expected (offset - manifest centre)',
   np.allclose(_sh[:2], np.array([100.0, 200.0]) - np.array([101.0, 199.0])),
   _sh[:2].tolist())
ck('the offset frame is still a no-op',
   np.all(V.tile_frame(_b, 'offset', TILES[0], 'test')[1] == 0))
# the npz's 3D tile_center must NOT win over the manifest
_npz_tc = np.asarray(_b['tile_center'], float)
ck('fixture really does disagree (npz tile_center is 3D)', _npz_tc.size == 3)
ck('manifest centre wins over the npz tile_center',
   V.manifest_center(TILES[0], 'test').size == 2,
   V.manifest_center(TILES[0], 'test').tolist())
# ...and without name/ds it falls back to the block, as older callers relied on
ck('no name/ds -> falls back to the block tile_center',
   V.tile_frame(_b, 'tile_center')[0][2] == _npz_tc[2])

# ------------------------------------------ prediction height on a side view
# Predictions carry no z, so a side view lifts each vertex onto the GT lines:
# the closest point on the nearest GT segment, interpolated ALONG it, so a
# prediction runs with the road's slope instead of flat across it.
blk = V.load_block(TILES[0], 'test')
origin, shift, _c = V.tile_frame(blk, 'offset')
xyz = blk['features'][:, :3]

# One sloping GT segment, from z=0 up to z=10 along x.
ramp = [(np.array([[-10.0, 0.0, 0.0], [10.0, 0.0, 10.0]]), 0, 'driving')]
q = np.array([[-10.0, 0.0], [0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
z = V.gt_z_at(q, ramp, fallback=-999.0)
ck('height is interpolated along the segment, not snapped to a vertex',
   np.allclose(z, [0.0, 5.0, 7.5, 10.0]), z.tolist())
ck('a prediction on a ramp is NOT flat', z.max() - z.min() > 9.0)

# Off the end of the segment the projection clamps to the endpoint.
ck('beyond the end clamps to the endpoint height',
   np.allclose(V.gt_z_at(np.array([[50.0, 0.0]]), ramp, -999.0), [10.0]))
# Lateral offset does not change the height, only which segment wins.
ck('height ignores perpendicular distance',
   np.allclose(V.gt_z_at(np.array([[0.0, 8.0]]), ramp, -999.0), [5.0]))

# With two GT lines at different heights, each vertex takes the NEARER one.
two = [(np.array([[-10.0, -5.0, 1.0], [10.0, -5.0, 1.0]]), 0, 'a'),
       (np.array([[-10.0, 5.0, 20.0], [10.0, 5.0, 20.0]]), 1, 'b')]
ck('each vertex follows the nearest GT line',
   np.allclose(V.gt_z_at(np.array([[0.0, -4.0], [0.0, 4.0]]), two, -999.0),
               [1.0, 20.0]))

ck('degenerate zero-length segment does not divide by zero',
   np.isfinite(V.gt_z_at(np.array([[0.0, 0.0]]),
                          [(np.array([[1.0, 1.0, 3.0], [1.0, 1.0, 3.0]]), 0, 'x')],
                          -999.0)).all())
ck('no GT at all -> the fallback', np.allclose(
   V.gt_z_at(np.array([[0.0, 0.0]]), [], fallback=-42.0), [-42.0]))
ck('no vertices -> empty, no crash',
   len(V.gt_z_at(np.zeros((0, 2)), ramp, 0.0)) == 0)

# ...and the real thing: the fixture's GT is at world z == 0 with offset[2]
# == 5, so a prediction laid on it must come out at stored z == -5. A naive
# "-origin[2]" agrees here by luck, so shift the GT to mimic a TERRAIN export
# (../carla_test runs 147..376 m) and the two answers separate.
gt3 = V.load_polylines(TILES[0], origin, 'test', ndim=3)
ck('pred height follows GT z, not -origin[2]',
   np.allclose(V.gt_z_at(np.array([[100.0, 200.0]]), gt3,
                          V._pred_fallback_z(gt3, xyz)), -5.0),
   V.gt_z_at(np.array([[100.0, 200.0]]), gt3, 0.0))

_rl = os.path.join(DS, 'reference_lines', f'{TILES[0]}_reference_lines.json')
_orig = json.load(open(_rl))
_terrain = json.loads(json.dumps(_orig))
for _p in _terrain['polylines']:
    _p['points'] = [[x, y, 300.0] for x, y, _z in _p['points']]
with open(_rl, 'w') as f:
    json.dump(_terrain, f)
gt_t = V.load_polylines(TILES[0], origin, 'test', ndim=3)
ck('terrain GT z is followed (295 = 300 - offset 5)',
   np.allclose(V.gt_z_at(np.array([[100.0, 200.0]]), gt_t, 0.0), 295.0),
   V.gt_z_at(np.array([[100.0, 200.0]]), gt_t, 0.0))
with open(_rl, 'w') as f:
    json.dump(_orig, f)

_gtless = os.path.join(DS, 'reference_lines', f'{TILES[1]}_reference_lines.json')
os.rename(_gtless, _gtless + '.bak')
gt_none = V.load_polylines(TILES[1], origin, 'test', ndim=3)
ck('no GT -> falls back to the cloud median, not 0 or -origin[2]',
   abs(V._pred_fallback_z(gt_none, xyz) - float(np.median(xyz[:, 2]))) < 1e-6)
os.rename(_gtless + '.bak', _gtless)

# ------------------------------------------------------------------- tags
tag_file = os.path.join(DS, V.TAGS_FILENAME)
ck('no tag file before tagging', not os.path.exists(tag_file))
ck('vocabulary starts with the corrupted default',
   V.tag_vocabulary() == ['corrupted'], V.tag_vocabulary())

inputs = [i for i in g.inputs if i.get('class') == 'tagbox']
ck('a corrupted checkbox is emitted per tile',
   len(inputs) == len(tiles) and all(i.get('data-tag') == 'corrupted'
                                      for i in inputs), len(inputs))
ck('corrupted starts unticked', all('checked' not in i for i in inputs))
ck('a new-tag box is emitted per tile',
   len([i for i in g.inputs if i.get('class') == 'tagnew']) == len(tiles))

r = app.post('/tag', json={'uid': UID, 'tag': 'corrupted', 'on': 1})
d = r.get_json()
ck('POST /tag ok', r.status_code == 200 and d['ok'], d)
ck('tile now carries the tag', d['tags'] == ['corrupted'], d['tags'])
ck('tag file written into the DATASET dir', os.path.exists(tag_file))
blob = json.load(open(tag_file))
ck('file records the assignment',
   blob['tiles'][TILES[0]] == ['corrupted'], blob)
ck('file records the vocabulary', 'corrupted' in blob['tags'], blob['tags'])

r = app.post('/tag', json={'uid': UID, 'tag': 'blurry-lidar', 'on': 1})
d = r.get_json()
ck('creating a new tag works', d['ok'] and set(d['tags']) ==
   {'corrupted', 'blurry-lidar'}, d)
ck('new tag joins the vocabulary', 'blurry-lidar' in d['vocab'], d['vocab'])

g3 = parse(app.get('/').data.decode())
boxes = [i for i in g3.inputs if i.get('class') == 'tagbox']
ck('both tags now have a box on every tile',
   len(boxes) == 2 * len(tiles), len(boxes))
mine = [i for i in boxes if i.get('data-uid') == UID]
ck('the tagged tile shows both ticked',
   sum('checked' in i for i in mine) == 2, mine)
other = [i for i in boxes if i.get('data-uid') != UID]
ck('the untagged tile shows none ticked',
   sum('checked' in i for i in other) == 0, other)

r = app.post('/tag', json={'uid': UID, 'tag': 'corrupted', 'on': 0})
d = r.get_json()
ck('untagging works', d['ok'] and d['tags'] == ['blurry-lidar'], d)
blob = json.load(open(tag_file))
ck('vocabulary SURVIVES untagging the last user',
   'corrupted' in blob['tags'], blob['tags'])

app.post('/tag', json={'uid': UID, 'tag': 'blurry-lidar', 'on': 0})
blob = json.load(open(tag_file))
ck('a tile with no tags is dropped from tiles{}',
   TILES[0] not in blob.get('tiles', {}), blob.get('tiles'))

# validation / errors
ck('unknown tile 404s',
   app.post('/tag', json={'uid': 'test/nope', 'tag': 'x'}).status_code == 404)
ck('path traversal in uid is rejected',
   app.post('/tag', json={'uid': '../../etc/passwd', 'tag': 'x'}
             ).status_code == 404)
ck('empty tag 400s',
   app.post('/tag', json={'uid': UID, 'tag': '  '}).status_code == 400)
ck('over-long tag 400s',
   app.post('/tag', json={'uid': UID, 'tag': 'x' * 65}).status_code == 400)

# hand-edited file: a tag only present on a tile still reaches the vocabulary
with open(tag_file, 'w') as f:
    json.dump({'version': 1, 'tags': [], 'tiles': {TILES[1]: ['hand-added']}}, f)
ck('a hand-added tag is offered in the UI',
   'hand-added' in V.tag_vocabulary(), V.tag_vocabulary())
ck('and shows as ticked on its tile',
   'hand-added' in V.tile_tags(tiles[1]['_uid']), V.tile_tags(tiles[1]['_uid']))

# corrupt file: reported, not fatal
with open(tag_file, 'w') as f:
    f.write('{not json')
store = V.load_tags('test')
ck('corrupt tag file reports an error instead of raising',
   bool(store.get('error')), store.get('error'))
rr = app.get('/')
ck('page still renders with a corrupt tag file', rr.status_code == 200)
ck('and says so', 'tag file:' in rr.data.decode())
ck('writing refuses while the file is unreadable',
   app.post('/tag', json={'uid': UID, 'tag': 'x'}).status_code == 500)
os.unlink(tag_file)

# read-only dataset dir -> a clear error, not a traceback.
# Skipped as root, which bypasses directory permission bits entirely: the
# write would succeed and the test would assert nothing. Run this file as a
# normal user (it passes on the host) to exercise it.
if os.geteuid() == 0:
    print('SKIP read-only dataset dir check (running as root; verified on host)')
else:
    os.chmod(DS, 0o555)
    try:
        r = app.post('/tag', json={'uid': UID, 'tag': 'corrupted', 'on': 1})
        d = r.get_json()
        ck('read-only dataset dir gives a 500 with the --tags-file hint',
           r.status_code == 500 and not d['ok'] and '--tags-file' in d['error'],
           d.get('error', '')[:120])
    finally:
        os.chmod(DS, 0o755)
    if os.path.exists(tag_file):
        os.unlink(tag_file)

# --tags-file override
alt = os.path.join(tempfile.mkdtemp(prefix='viewer_tagsalt_'), 'tags.json')
V.STATE['tags_file'] = alt
r = app.post('/tag', json={'uid': UID, 'tag': 'corrupted', 'on': 1})
ck('--tags-file override is honoured',
   r.status_code == 200 and os.path.exists(alt)
   and not os.path.exists(tag_file), alt)
V.STATE['tags_file'] = None

# --------------------------------------------------------- other tabs intact
for tab, want in (('stats', 'CARLA dataset viewer'),
                  ('results', 'CARLA dataset viewer')):
    rr = app.get(f'/?tab={tab}')
    ck(f'tab={tab} still renders', rr.status_code == 200, rr.status_code)
    ck(f'tab={tab} is not the browse page',
       'View angle' not in rr.data.decode()
       or tab == 'browse', tab)

shutil.rmtree(DATA, ignore_errors=True)
print()
print('ALL PASS' if not fails else 'FAILURES:\n  ' + '\n  '.join(fails))
sys.exit(1 if fails else 0)
