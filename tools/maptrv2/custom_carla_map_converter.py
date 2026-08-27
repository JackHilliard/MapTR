"""Convert the CARLA road-polyline tile dataset into MapTR's map-annotation
pkl format.

Unlike the nuScenes/AV2 converters, CARLA tiles are static square patches
(not per-timestamp driving-log frames), so there is no ego pose/SE3 transform
and no need to clip polylines against a moving patch -- this script only
needs to shift each tile's world-frame reference-line polylines into the
same frame as its point cloud, by subtracting the block's own `offset` (the
frame `LoadCarlaPointsFromFile` reads, and NOT the same as `tile_center` --
see the comment on the np.load call below).

Nothing here assumes a particular tile size. The exporter has produced at
least 25m tiles (`tile_radius` 12.5) and 60m ones (`tile_radius` 30.0), and
both layouts it writes are accepted::

    <data_root>/<split>/manifest.json            # split export
    <data_root>/<split>/blocks/<tile>.npz
    <data_root>/<split>/reference_lines/<tile>_reference_lines.json

    <data_root>/grid_manifest.json               # single-town grid export
    <data_root>/blocks/<tile>.npz                # (no <split> level, and
    <data_root>/reference_lines/<tile>_...json   #  tile names carry no town)

An export may ship SEVERAL polyline directories holding different GT sets --
the 30m test export has an unclassified `reference_lines/` next to a
`reference_curb_driving_lines/` that carries `class_id`/`class` and states
its own `classes` lookup per tile. `--reference-dir` picks one; by default
`reference_lines/` wins, except that --classes/--map-classes falls back to a
sibling that actually has a taxonomy. The choice is always printed, and the
one used is recorded in the pkl's `tile_geometry.reference_dir`.

The class lookup is likewise read from the manifest's `class_lookup` where
there is one, and from the polyline json's own `classes` key where there is
not.

Where a directory holds both files, `manifest.json` wins and
`grid_manifest.json` only backfills keys it lacks -- the same rule
`tools/maptrv2/dataset_viewer.py` uses. `manifest.json` is written by a
later packaging step and is the *curated* view: in the Town10HD grid export
it lists 30 tiles where grid_manifest.json lists 33, and the 3 it omits
contain only sidewalk_edge polylines and no driving centerline at all
(`gt_source: driving_lanes`). Its dataset-level *counts* are still not
trustworthy -- the same file reports `n_tiles: 4103` while listing 30 -- so
everything here is derived from the tile entries, never from those counts.

Note the reference-lines json on disk is not filtered to match: it still
holds every class, so a class-carrying export converts all of them unless
--classes or --map-classes says otherwise.

By default every kept polyline becomes one `divider` instance, i.e. a
single-class map. ``--map-classes name=export_class[,export_class...]``
instead emits one annotation key per named map class, in the order given --
which is the label order, and must equal the training config's
``map_classes``::

    --map-classes driving=driving_centerline curb=curb

Each tile's LiDAR block is also scanned for points that would actually
survive the training pipeline's filters (``z <= --z-max`` and inside
``--lidar-point-cloud-range``). Tiles with fewer than ``--min-lidar-points``
such points would voxelize to *zero* voxels and crash ``extract_lidar_feat``,
so they are dropped from the pkl and listed in a sidecar report. Every kept
sample records its own count so the dataset can re-check cheaply. The
default range is derived from the manifest's own tile size rather than
hardcoded, so the scan describes the tiles being converted.

Usage::

    python tools/maptrv2/custom_carla_map_converter.py \\
        --data-root /path/to/carla --out-dir data/carla/ --split test
"""

import argparse
import json
import os

import mmcv
import numpy as np

# z half of the range the training config's LiDAR branch uses
# (`lidar_point_cloud_range` in projects/configs/maptrv2/
# maptrv2_carla_r50_24ep_lidar.py) and `z_max` from
# projects/configs/carla/carlasim_map.py. The xy half is NOT a constant:
# it follows the tile size read from the manifest (see default_pc_range),
# because a range narrower than the tile silently crops it and a wider one
# just wastes BEV cells. The whole range is recorded into the pkl so the
# dataset can warn if config and pkl ever drift apart.
DEFAULT_Z_RANGE = (-72.0, 96.0)
DEFAULT_Z_MAX = 96.0
# Only used when a manifest carries no tile geometry at all; matches the
# original 25m export this converter was written against.
FALLBACK_TILE_RADIUS = 12.5
# Slack allowed before a polyline is reported as leaving its own tile. The
# exporter samples arcs at a fixed arc_gap, so a vertex can land slightly
# past the boundary without anything being wrong.
BOUNDS_MARGIN = 1.0

MANIFEST_NAMES = ('manifest.json', 'grid_manifest.json')


def parse_args():
    parser = argparse.ArgumentParser(
        description='CARLA map data converter arg parser')
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='root of the CARLA tile dataset (contains <split>/manifest.json, '
        'or a manifest.json/grid_manifest.json directly)')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='data/carla/',
        help='output directory for the generated pkl')
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='split subdirectory under --data-root to convert; also used as '
        'the label in the output filename. If no such subdirectory exists, '
        '--data-root itself is read and this is only the output label')
    parser.add_argument(
        '--classes',
        type=str,
        nargs='+',
        default=None,
        metavar='CLASS',
        help="optional subset of polyline classes to keep as divider "
        "instances, as class_lookup ids or names (e.g. `0` or "
        "`driving_centerline`). Default: keep every polyline. Only exports "
        "whose polylines carry a class can be filtered")
    parser.add_argument(
        '--map-classes',
        type=str,
        nargs='+',
        default=None,
        metavar='NAME=CLASS[,CLASS...]',
        help='emit SEVERAL map classes instead of one `divider`, as an '
        'ordered list of `name=export_class[,export_class]` groups (e.g. '
        '`--map-classes driving=driving_centerline curb=curb`). A bare '
        '`name` is shorthand for `name=name`. The names, in this order, are '
        "the config's `map_classes`, so the pkl and the config must agree. "
        'Mutually exclusive with --classes, which only ever produces one '
        'class')
    parser.add_argument(
        '--lane-types',
        type=str,
        nargs='+',
        default=None,
        metavar='CLASS',
        help='deprecated alias for --classes (it never matched anything: it '
        "compared lane_type_lookup ids against each polyline's `type`, which "
        "holds a geometry kind such as 'arc'/'straight')")
    parser.add_argument(
        '--lidar-point-cloud-range',
        type=float,
        nargs=6,
        default=None,
        metavar=('X_MIN', 'Y_MIN', 'Z_MIN', 'X_MAX', 'Y_MAX', 'Z_MAX'),
        help='range the LiDAR voxelizer will use; points outside it are '
        'dropped before voxelization, so a tile with none inside produces '
        'zero voxels. Default: +/-tile_radius in xy (read from the manifest) '
        f'and {DEFAULT_Z_RANGE[0]}..{DEFAULT_Z_RANGE[1]} in z')
    parser.add_argument(
        '--reference-dir',
        type=str,
        default=None,
        metavar='DIR',
        help='name of the per-tile polyline directory under the tile dir '
        "(default: 'reference_lines', or a sibling that actually carries a "
        'class taxonomy when --classes/--map-classes asks for one -- some '
        'exports keep the classified polylines in a separate directory such '
        'as reference_curb_driving_lines/). The resolved directory is '
        'always printed')
    parser.add_argument(
        '--out-tag',
        type=str,
        default=None,
        metavar='TAG',
        help='suffix for the output filenames, giving '
        '`carla_map_infos_<split>_<TAG>.pkl`. Use it to keep several '
        'datasets (tile sizes, GT frames, class taxonomies) in ONE --out-dir '
        'without silently overwriting each other, since the filename is '
        'otherwise a function of --split alone')
    parser.add_argument(
        '--z-max',
        type=float,
        default=DEFAULT_Z_MAX,
        help="LoadCarlaPointsFromFile's early z filter (default: matches "
        'the training config)')
    parser.add_argument(
        '--min-lidar-points',
        type=int,
        default=1,
        help='drop tiles with fewer than this many in-range LiDAR points; '
        '1 drops only genuinely zero-voxel tiles (default: 1)')
    parser.add_argument(
        '--gt-frame',
        type=str,
        default='tile_center',
        choices=['offset', 'tile_center'],
        help="origin every tile's polylines AND point cloud are expressed "
        "relative to. 'tile_center' (default, the project-wide convention "
        "and GeMap's) is the tile's nominal geometric centre, which "
        'requires the loader to shift the points by `offset - tile_center` '
        'at load time (LoadCarlaPointsFromFile(recenter=True), wired via '
        "the pkl's per-sample `lidar_recenter_shift`) -- every current "
        "training config sets that. 'offset' is the block's own `offset` "
        'array, the frame the raw `features` are already stored in; it '
        'crops tiles against the origin-centred lidar_point_cloud_range '
        '(|offset - tile_center| reaches 117 m on some exports), so use it '
        'only to reproduce old offset-frame runs')
    parser.add_argument(
        '--no-lidar-check',
        action='store_true',
        help='skip the per-tile LiDAR scan entirely (faster, but no tile is '
        'dropped and no point counts are recorded)')
    return parser.parse_args()


def load_manifest(data_root, split):
    """Locate and load the manifest describing one set of tiles.

    Accepts either export layout: ``<data_root>/<split>/`` (the split
    export) or ``<data_root>`` itself (a grid export, which has no split
    level). Returns ``(tile_dir, manifest)``; `tile_dir` is where `blocks/`
    and `reference_lines/` live.
    """
    for candidate in ([os.path.join(data_root, split)] if split else
                      []) + [data_root]:
        present = [
            n for n in MANIFEST_NAMES
            if os.path.isfile(os.path.join(candidate, n))
        ]
        if not present:
            continue
        manifest = {}
        # reversed so the preferred name (first in MANIFEST_NAMES) is applied
        # last and its keys win, while the other only backfills. A null value
        # never overwrites a real one -- grid_manifest.json spells absent
        # options as `null` rather than omitting them.
        for name in reversed(present):
            with open(os.path.join(candidate, name), encoding='utf-8') as f:
                loaded = json.load(f)
            manifest.update({
                k: v
                for k, v in loaded.items()
                if v is not None or k not in manifest
            })
        if not manifest.get('tiles'):
            raise ValueError(
                f'{candidate}: manifest lists no tiles')
        return candidate, manifest
    raise FileNotFoundError(
        f'no {" or ".join(MANIFEST_NAMES)} under {data_root!r}'
        f'{f" or {os.path.join(data_root, split)!r}" if split else ""}')


def _first_ref_json(tile_dir, ref_dir, tiles):
    """Parse the first tile's polyline json under ``ref_dir``, or None."""
    for tile in tiles:
        path = os.path.join(tile_dir, ref_dir,
                            f'{tile["name"]}_reference_lines.json')
        if os.path.isfile(path):
            with open(path, encoding='utf-8') as f:
                return json.load(f)
    return None


def reference_dir_candidates(tile_dir):
    """Directories that could hold per-tile polyline jsons, preferred first.

    ``reference_lines`` is the name every export has used for the unclassified
    set, so it stays first; any other ``reference*`` directory is a variant
    (this dataset ships ``reference_curb_driving_lines`` and
    ``reference_driving_curb_crosswalk``).

    The match is deliberately just the ``reference`` prefix and NOT
    ``reference*lines*``: the crosswalk-carrying directory added on
    2026-08-23 is named ``reference_driving_curb_crosswalk``, with no
    "lines" in it, and the narrower pattern made it invisible -- so
    ``--map-classes ... crosswalk`` fell back to the two-class directory and
    died with "unknown class 'crosswalk'" while the data sat right there.
    """
    try:
        entries = sorted(os.listdir(tile_dir))
    except OSError:
        return []
    found = [
        d for d in entries
        if d.startswith('reference')
        and os.path.isdir(os.path.join(tile_dir, d))
    ]
    found.sort(key=lambda d: (d != 'reference_lines', d))
    return found


def required_class_tokens(classes, map_classes):
    """The export-class names/ids a run needs a directory's taxonomy to hold.

    ``--classes`` entries are ids or names directly. ``--map-classes`` entries
    are ``name=member,member`` or a bare ``name`` (which means ``name=name``),
    so it is the MEMBERS that must exist in the export, never the label name.
    Returned as raw strings; a candidate satisfies a token if the token is one
    of its class names or one of its ids.
    """
    tokens = set()
    for entry in classes or []:
        tokens.add(str(entry))
    for entry in map_classes or []:
        entry = str(entry)
        name, sep, members = entry.partition('=')
        if sep and members.strip():
            tokens.update(m.strip() for m in members.split(',') if m.strip())
        else:
            tokens.add(name.strip())
    return {t for t in tokens if t}


def _lookup_satisfies(lookup, tokens):
    if not lookup:
        return False
    names = set(lookup.values())
    ids = set(lookup)
    return all(t in names or t in ids for t in tokens)


def resolve_reference_dir(tile_dir, manifest, requested, need_classes,
                          required_tokens=None):
    """Pick the polyline directory to convert, and its class lookup.

    Returns ``(ref_dir, class_lookup)``. The lookup comes from the manifest
    where an export states one, and otherwise from the polyline json's own
    ``classes`` key -- this dataset's manifest has no ``class_lookup`` at all,
    and declares ``{"0": "driving", "1": "curb"}`` per tile instead.

    An export can ship BOTH an unclassified ``reference_lines`` and a
    classified sibling, and those are different GT sets, not two spellings of
    one. So the choice is never silent: an explicit --reference-dir is obeyed,
    and the automatic fallback (only ever taken when --classes/--map-classes
    needs a taxonomy that the default directory lacks) prints what it picked
    and why.
    """
    tiles = manifest.get('tiles') or []
    candidates = reference_dir_candidates(tile_dir)
    if requested:
        path = os.path.join(tile_dir, requested)
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f'{path} does not exist; this export has: '
                f'{", ".join(candidates) or "no reference*lines* directory"}')
        candidates = [requested]
    if not candidates:
        raise FileNotFoundError(
            f'no reference*lines* directory under {tile_dir}')

    def lookup_for(ref_dir):
        stated = manifest.get('class_lookup')
        if stated:
            return {str(k): v for k, v in stated.items()}
        ref = _first_ref_json(tile_dir, ref_dir, tiles)
        classes = (ref or {}).get('classes') or {}
        return {str(k): v for k, v in classes.items()}

    chosen = candidates[0]
    lookup = lookup_for(chosen)
    if need_classes and not _lookup_satisfies(lookup, required_tokens or set()):
        # Only now is it worth looking past the default directory. An export
        # can ship SEVERAL classified siblings holding different taxonomies
        # (`../carla_test` has a two-class one and a three-class one), so the
        # test is not "has any taxonomy" but "has the classes this run asked
        # for" -- otherwise the first classified directory wins and a request
        # naming a class only the second one carries fails with `unknown
        # class`, which reads as missing data rather than a wrong directory.
        for alt in candidates[1:]:
            alt_lookup = lookup_for(alt)
            if not alt_lookup:
                continue
            if required_tokens and not _lookup_satisfies(alt_lookup,
                                                         required_tokens):
                continue
            stated = ', '.join(
                f'{k}={v}' for k, v in sorted(alt_lookup.items()))
            why = ('carries no class taxonomy' if not lookup
                   else 'does not carry '
                        + ', '.join(sorted(
                            t for t in (required_tokens or set())
                            if not _lookup_satisfies(lookup, {t}))))
            print(f'  [ref] {chosen}/ {why}; using '
                  f'{alt}/ instead, which declares {stated}')
            return alt, alt_lookup
        # Nothing carries everything that was asked for, so this run is going
        # to die in resolve_classes() no matter which directory is picked.
        # Hand it the RICHEST taxonomy available anyway: the choice cannot
        # change the outcome, only the error message, and "unknown class
        # 'median'; this export has: 0=driving, 1=curb, 2=crosswalk" points at
        # the real problem where the unclassified directory's "this export has
        # no class_lookup" would send the reader hunting for missing data.
        classified = [(alt, lookup_for(alt)) for alt in candidates[1:]]
        classified = [(a, l) for a, l in classified if l]
        if classified:
            alt, alt_lookup = max(classified, key=lambda kv: len(kv[1]))
            stated = ', '.join(f'{k}={v}' for k, v in sorted(alt_lookup.items()))
            print(f'  [ref] no directory carries every requested class; '
                  f'reporting against {alt}/, which declares {stated}')
            return alt, alt_lookup
    print(f'  [ref] reading polylines from {chosen}/'
          + (f' (classes: {", ".join(sorted(lookup.values()))})'
             if lookup else ' (no class taxonomy)'))
    return chosen, lookup


def manifest_tile_radius(manifest):
    """Half a tile's side, in metres, from whichever key the export used.

    `tile_radius` (25m export) and `tile_side` (some grid manifests) are
    dataset-level; failing both, the widest per-tile `bounds` gives the same
    answer for a uniform grid and an upper bound otherwise, which is the
    safe direction for sizing a point-cloud range. Returns None only when an
    export states its geometry nowhere -- the per-tile bounds check can
    still fall back to each reference-lines json, so that is not fatal.
    """
    if manifest.get('tile_radius') is not None:
        return float(manifest['tile_radius'])
    if manifest.get('tile_side') is not None:
        return float(manifest['tile_side']) / 2.0
    radii = []
    for tile in manifest.get('tiles', []):
        bounds = tile.get('bounds')
        if bounds is None:
            continue
        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.size == 4:
            radii.append(np.max(bounds[2:] - bounds[:2]) / 2.0)
        elif bounds.size == 6:
            radii.append(np.max(bounds[3:5] - bounds[:2]) / 2.0)
    return float(max(radii)) if radii else None


def tile_footprint(tile, ref, tile_radius):
    """World-frame xy footprint of one tile, as ``(lo, hi)`` arrays.

    Prefers the tile's own `bounds` (exact, and the only source that would
    describe a non-square tile), then its centre plus the export's tile
    radius. The reference-lines json repeats all three keys, so it backs up
    a manifest whose tile entries are sparse. Returns None when nothing
    supplies a footprint, which only disables the bounds warning below.
    """
    bounds = tile.get('bounds') or ref.get('tile_bounds')
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.size == 4:  # [x_min, y_min, x_max, y_max]
            return bounds[:2], bounds[2:]
        if bounds.size == 6:  # [x_min, y_min, z_min, x_max, y_max, z_max]
            return bounds[:2], bounds[3:5]

    center = tile.get('center') or ref.get('tile_center')
    radius = ref.get('tile_radius')
    radius = float(radius) if radius is not None else tile_radius
    if center is None or radius is None:
        return None
    center = np.asarray(center, dtype=np.float64)[:2]
    return center - radius, center + radius


def default_pc_range(tile_radius):
    """The range a training config for this tile size would use: square in
    xy, matching the tile, with the config's z span."""
    radius = tile_radius if tile_radius is not None else FALLBACK_TILE_RADIUS
    z_min, z_max = DEFAULT_Z_RANGE
    return [-radius, -radius, z_min, radius, radius, z_max]


def resolve_classes(selected, manifest):
    """Map ``--classes`` entries (ids or names) onto class_lookup ids.

    Returns a set of int ids, or None for "keep everything". Raises if the
    export has no class taxonomy or a name/id is not in it -- silently
    keeping nothing is how the old --lane-types flag behaved, and it
    produced an empty pkl with no indication why.
    """
    if not selected:
        return None
    lookup = manifest.get('class_lookup') or {}
    if not lookup:
        raise ValueError(
            'this export has no `class_lookup`, so its polylines carry no '
            'class and cannot be filtered or split; re-run without '
            '--classes/--map-classes (every polyline then becomes one '
            '`divider` instance)')
    by_name = {name: int(cid) for cid, name in lookup.items()}
    out = set()
    for entry in selected:
        entry = str(entry)
        if entry in by_name:
            out.add(by_name[entry])
        elif entry.lstrip('-').isdigit() and str(int(entry)) in lookup:
            out.add(int(entry))
        else:
            known = ', '.join(
                f'{cid}={name}' for cid, name in sorted(lookup.items()))
            raise ValueError(
                f'unknown class {entry!r}; this export has: {known}')
    return out


def resolve_map_classes(selected, manifest):
    """Parse ``--map-classes`` into ordered ``[(name, {class ids}), ...]``.

    Each entry is ``name=export_class[,export_class...]``; a bare ``name``
    means ``name=name``. The order is the label order the model trains
    against (the dataset builds its CLASS2LABEL from the config's
    ``map_classes`` list, in order), so it is preserved exactly as given.

    A class appearing in two groups is an error rather than a duplication:
    the same polyline would become two GT instances at the same coordinates
    with different labels, which no matching can resolve.
    """
    if not selected:
        return None
    groups = []
    seen_names = set()
    owner = {}
    for entry in selected:
        name, _, members = entry.partition('=')
        name = name.strip()
        if not name:
            raise ValueError(f'--map-classes entry {entry!r} has no name')
        if name in seen_names:
            raise ValueError(f'--map-classes names must be unique; {name!r} '
                             'appears twice')
        seen_names.add(name)
        member_list = [m.strip() for m in members.split(',') if m.strip()]
        ids = resolve_classes(member_list or [name], manifest)
        for cid in sorted(ids):
            if cid in owner:
                raise ValueError(
                    f'export class id {cid} is listed under both '
                    f'{owner[cid]!r} and {name!r}; a polyline can only carry '
                    'one label')
            owner[cid] = name
        groups.append((name, ids))
    return groups


def polyline_class_id(poly):
    """The polyline's class_lookup id, or None on a class-free export.

    Deliberately does not fall back to `type`: that key exists in every
    export and holds a geometry kind ('arc'/'straight'), not a class.
    """
    if poly.get('class_id') is not None:
        return int(poly['class_id'])
    return None


def tile_center_origin(tile, offset):
    """The 3-vector to subtract from a tile's world-frame polylines under
    ``--gt-frame tile_center``.

    xy is the tile centre: ``center`` where the export states one, else the
    midpoint of ``bounds`` (identical on every tile of the 25m export, which
    states both).

    z is the awkward part, because not every export gives its tile centres a
    z -- some write ``center`` as ``[x, y]``. Where a z is stated it is used;
    where it is not, it falls back to the block's own ``offset``, making the
    z component of the recentring shift exactly zero. That is the only choice
    that keeps polylines and points in ONE frame: the shift is rigid and
    applied to both, so alignment holds either way, and a 2D manifest simply
    leaves z in the offset frame. Nothing downstream minds -- ``code_size=2``
    means z never reaches a regression target.

    This mirrors GeMap's ``tile_origin()`` exactly, which is the point: the
    two converters must agree, or a checkpoint trained under one is scored
    against GT built by the other. Verified on the 259-tile test split --
    every annotation array bit-identical between the two, max |difference|
    0.0.
    """
    center = tile.get('center')
    if center is None:
        bounds = tile.get('bounds')
        if bounds is None:
            raise ValueError(
                f'tile {tile.get("name")!r} states neither `center` nor '
                '`bounds`, so --gt-frame tile_center cannot place it')
        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.size == 4:            # [x_min, y_min, x_max, y_max]
            center = (bounds[:2] + bounds[2:]) / 2.0
        elif bounds.size == 6:          # [x_min, y_min, z_min, x_max, ...]
            center = (bounds[:3] + bounds[3:]) / 2.0
        else:
            raise ValueError(
                f'tile {tile.get("name")!r}: cannot read a centre from '
                f'`bounds` of length {bounds.size}')
    center = np.asarray(center, dtype=np.float32).ravel()
    if center.size == 2:
        return np.array([center[0], center[1], offset[2]], dtype=np.float32)
    if center.size == 3:
        return center.astype(np.float32)
    raise ValueError(
        f'tile {tile.get("name")!r}: `center` has {center.size} components, '
        'expected 2 or 3')


def count_points_in_range(lidar_path, pc_range, z_max, shift=None):
    """Count a block's points that would survive the training pipeline.

    Mirrors what the model actually sees: ``LoadCarlaPointsFromFile``'s
    ``z <= z_max`` filter, then the LiDAR ``Voxelization``'s own
    range filtering. ``GridSamplePoints`` sits between the two but only
    *clamps* grid coordinates -- it never drops points and never moves
    them -- so it cannot change this count.

    Returns:
        tuple[int, int]: raw point count, and count surviving both filters.
    """
    with np.load(lidar_path) as block:
        xyz = np.asarray(block['features'][:, :3], dtype=np.float32)
    n_raw = int(xyz.shape[0])
    if z_max is not None:
        xyz = xyz[xyz[:, 2] <= z_max]
    if shift is not None:
        # Applied after the z_max cut, matching LoadCarlaPointsFromFile's
        # order: that filter is defined on the stored (offset-frame) z.
        xyz = xyz + np.asarray(shift, dtype=np.float32)
    lo = np.asarray(pc_range[:3], dtype=np.float32)
    hi = np.asarray(pc_range[3:], dtype=np.float32)
    # `>= lo` / `< hi` mirrors hard_voxelize's own
    # `floor((p - lo) / voxel) in [0, grid)` convention rather than
    # PointsRangeFilter.in_range_3d's strict-both-sides test. The two differ
    # only for points exactly on a boundary plane, which can never flip a
    # zero/non-zero verdict in practice.
    n_in_range = int(np.all((xyz >= lo) & (xyz < hi), axis=1).sum())
    return n_raw, n_in_range


def convert_carla_tiles(data_root,
                        split,
                        classes=None,
                        map_classes=None,
                        pc_range=None,
                        z_max=DEFAULT_Z_MAX,
                        min_lidar_points=1,
                        lidar_check=True,
                        gt_frame='offset',
                        reference_dir=None):
    assert gt_frame in ('offset', 'tile_center')
    tile_dir, manifest = load_manifest(data_root, split)
    ref_dir, class_lookup = resolve_reference_dir(
        tile_dir, manifest, reference_dir,
        need_classes=bool(classes or map_classes),
        required_tokens=required_class_tokens(classes, map_classes))
    # The lookup may have come from the polyline jsons rather than the
    # manifest; resolve_classes() only ever reads it off the manifest, so put
    # it there. A copy -- the caller's dict is not ours to edit.
    manifest = dict(manifest, class_lookup=class_lookup)
    tile_radius = manifest_tile_radius(manifest)
    if classes and map_classes:
        raise ValueError('--classes and --map-classes both filter the same '
                         'polylines; pass only one')
    keep_classes = resolve_classes(classes, manifest)
    # One annotation key per map class, in the order the config lists them.
    # The default reproduces the historical single-class output exactly:
    # every kept polyline becomes a `divider` instance.
    groups = resolve_map_classes(map_classes, manifest) or [('divider',
                                                            keep_classes)]
    class_names = [name for name, _ in groups]
    if pc_range is None:
        pc_range = default_pc_range(tile_radius)
        if tile_radius is None:
            print(f'[warn] this export states no tile size; assuming '
                  f'tile_radius={FALLBACK_TILE_RADIUS}m for the point-cloud '
                  'range. Pass --lidar-point-cloud-range if that is wrong')
    print(f'{tile_dir}: {len(manifest["tiles"])} tiles, tile_radius='
          f'{tile_radius if tile_radius is not None else "unknown"}, '
          f'pc_range={[round(v, 2) for v in pc_range]}')

    samples = []
    dropped = []
    out_of_bounds = []
    coverage = []
    total_instances = 0
    class_conflicts = []
    per_class_instances = {name: 0 for name in class_names}
    prog_bar = mmcv.ProgressBar(len(manifest['tiles']))
    for idx, tile in enumerate(manifest['tiles']):
        prog_bar.update()
        name = tile['name']

        lidar_path = os.path.abspath(
            os.path.join(tile_dir, 'blocks', f'{name}.npz'))
        if not os.path.isfile(lidar_path):
            raise FileNotFoundError(lidar_path)

        # The polylines below are in WORLD coordinates and must be shifted
        # into the same frame as the LiDAR points the model actually sees.
        # That frame is the block's `offset`, NOT its `tile_center`:
        # LoadCarlaPointsFromFile reads `features[:, 0:3]`, and
        # `points - offset == features[:, :3]` holds exactly, while
        # tile_center differs from offset by a mean of ~2.4m (max >7m)
        # across the 25m train split -- and by up to ~17m on the 60m grid
        # export, where the displacement scales with tile size. Measured
        # against real driving-surface returns (label == 0), `- offset`
        # puts polylines a median 0.038m from the road vs 0.388m for
        # `- tile_center`. Using tile_center here (as this converter
        # originally did) misaligns every GT polyline against its own point
        # cloud, which matters a lot given chamfer eval thresholds of
        # 0.5/1.0/1.5m.
        #
        # np.load is lazy, so this reads only the small `offset` array --
        # it does not pull the full point cloud into memory. (The separate
        # count_points_in_range() call below does read the full `features`
        # array when --no-lidar-check is off; that is the expensive part of
        # this loop, ~14ms for a 110K-point tile.)
        with np.load(lidar_path) as block:
            block_offset = np.asarray(block['offset'], dtype=np.float32)

        # --gt-frame tile_center reproduces GeMap's convention: everything --
        # polylines AND the point cloud -- is expressed relative to the tile's
        # nominal geometric centre instead of the block's `offset`. The two
        # differ by 1-2m on the 25m export and up to ~17m on the 60m grid one,
        # so this is not a cosmetic relabelling: the stored `features` are in
        # the offset frame, so the loader has to add `shift = offset - origin`
        # to every point for GT and points to stay aligned. That shift is
        # recorded per sample below and applied by
        # LoadCarlaPointsFromFile(recenter=True).
        if gt_frame == 'tile_center':
            origin = tile_center_origin(tile, block_offset)
        else:
            origin = block_offset
        shift = (block_offset - origin).astype(np.float32)

        n_raw = n_in_range = None
        if lidar_check:
            # Counted in the *same* frame the model will see, i.e. after the
            # recentring shift -- otherwise a tile_center run would measure a
            # square range against a cloud displaced from it, which is exactly
            # the coverage problem this frame choice is meant to remove.
            n_raw, n_in_range = count_points_in_range(
                lidar_path, pc_range, z_max, shift=shift)
            if n_in_range < min_lidar_points:
                # This tile would voxelize to zero (or near-zero) voxels and
                # crash extract_lidar_feat mid-run. Drop it before its
                # polylines are counted, so the reported instance totals
                # describe what training will actually see.
                dropped.append(
                    dict(
                        name=name,
                        town=tile.get('town'),
                        n_points=n_raw,
                        n_points_in_range=n_in_range))
                continue
            if n_raw:
                coverage.append((n_in_range / n_raw, name))

        ref_path = os.path.join(tile_dir, ref_dir,
                                f'{name}_reference_lines.json')
        with open(ref_path, encoding='utf-8') as f:
            ref = json.load(f)

        # Every tile of this dataset restates the taxonomy, so check it
        # rather than trusting the first file: a tile whose ids mean
        # something else would be relabelled silently, and the class ids are
        # what --map-classes matches on.
        stated = ref.get('classes')
        if stated and {str(k): v for k, v in stated.items()} != class_lookup:
            class_conflicts.append(dict(name=name, classes=stated))

        annotation = {c: [] for c in class_names}
        for poly in ref['polylines']:
            cid = polyline_class_id(poly)
            # Groups are disjoint by construction, so this appends at most
            # once; the class ids are resolved before the tile loop, so the
            # points array is only built for a polyline that is kept.
            # (`cls_name`, not `name` -- that is the TILE's name, and a
            # loop variable leaks past its loop in Python.)
            for cls_name, keep in groups:
                if keep is not None and cid not in keep:
                    continue
                pts = np.array(poly['points'], dtype=np.float32)
                if pts.shape[0] < 2:
                    break
                annotation[cls_name].append(pts - origin)
                per_class_instances[cls_name] += 1
                total_instances += 1
        # Flat view of every instance in this tile, for the bounds check
        # below -- which is about tile geometry, not about class.
        divider = [p for c in class_names for p in annotation[c]]

        # Sanity check only (tiles are asserted to already be the patch, so
        # no clipping is applied) -- collect, don't crash, if this fires.
        #
        # Compared against the tile's own world-frame footprint shifted into
        # the offset frame, NOT against +/-tile_radius around zero: `origin`
        # is the block's `offset`, which is not the tile centre, so in this
        # frame the tile sits at `footprint - origin`. Testing |xy| against a
        # bare radius therefore fired on most tiles of the 25m export (where
        # the displacement is ~1-2m) and on essentially all of the 60m one
        # (up to ~17m) -- pure noise that buried any real out-of-bounds tile.
        footprint = tile_footprint(tile, ref, tile_radius)
        if footprint is not None and divider:
            lo, hi = (footprint[0] - origin[:2] - BOUNDS_MARGIN,
                      footprint[1] - origin[:2] + BOUNDS_MARGIN)
            pts = np.concatenate([p[:, :2] for p in divider])
            overshoot = float(
                np.max(np.maximum(lo - pts, pts - hi), initial=0.0))
            if overshoot > 0:
                out_of_bounds.append(dict(name=name, overshoot=overshoot))

        samples.append(
            dict(
                lidar_path=lidar_path,
                sample_idx=name,
                token=name,
                timestamp=idx,
                town=tile.get('town'),
                tile_center=tile['center'],
                # the origin the annotation below is actually relative to --
                # recorded explicitly so the frame isn't ambiguous when
                # reading the pkl back (tile_center above is NOT it)
                annotation_origin=origin.tolist(),
                # which of the two origins the one above is
                gt_frame=gt_frame,
                # the block's own frame, kept regardless of gt_frame so the
                # shift can be re-derived or undone later
                lidar_offset=block_offset.tolist(),
                # `offset - annotation_origin`: what LoadCarlaPointsFromFile
                # must add to the stored features to land in the GT frame.
                # All-zero when gt_frame == 'offset'.
                lidar_recenter_shift=shift.tolist(),
                tile_bounds=tile.get('bounds'),
                # None when --no-lidar-check was passed; the dataset treats
                # a missing count as "unknown" and keeps the sample.
                num_lidar_points=n_raw,
                num_lidar_points_in_range=n_in_range,
                annotation=annotation,
            ))

    n = max(len(samples), 1)
    print()
    if lidar_check:
        print(f'{split}: {len(samples)} tiles kept, {len(dropped)} dropped '
              f'(<{min_lidar_points} in-range LiDAR point'
              f'{"" if min_lidar_points == 1 else "s"}), {total_instances} '
              f'map instances ({total_instances / n:.1f} per tile)')
        report_class_counts(per_class_instances, n)
        for entry in dropped[:20]:
            print(f'  [drop] {entry["name"]}: {entry["n_points"]} raw points, '
                  f'{entry["n_points_in_range"]} in range')
        if len(dropped) > 20:
            # A long list usually means the range is wrong, not the data --
            # the sidecar json below has the rest.
            print(f'  [drop] ... and {len(dropped) - 20} more')
    else:
        print(f'{split}: {len(samples)} tiles, {total_instances} map '
              f'instances ({total_instances / n:.1f} per tile) '
              '[LiDAR check skipped]')
        report_class_counts(per_class_instances, n)
    report_coverage(coverage, pc_range)
    report_out_of_bounds(out_of_bounds)
    report_class_conflicts(class_conflicts, class_lookup)
    return samples, dropped, dict(
        tile_dir=tile_dir,
        reference_dir=ref_dir,
        tile_radius=tile_radius,
        tile_side=manifest.get('tile_side'),
        pc_range=list(pc_range),
        gt_frame=gt_frame,
        class_lookup=manifest.get('class_lookup') or {},
        classes_kept=sorted(keep_classes) if keep_classes else None,
        map_classes=class_names,
        class_groups={
            name: (sorted(keep) if keep is not None else None)
            for name, keep in groups
        },
        n_out_of_bounds=len(out_of_bounds))


def report_class_counts(per_class_instances, n_tiles):
    """Per-map-class instance totals.

    Printed even for the single-class default, so a run's output always
    states which annotation keys the pkl actually carries -- a config whose
    `map_classes` disagrees with them trains against silently empty GT for
    the missing ones.
    """
    for name, count in per_class_instances.items():
        print(f'  [class] {name}: {count} instances '
              f'({count / max(n_tiles, 1):.1f} per tile)')
        if count == 0:
            print(f'  [class] {name} is EMPTY -- check the export carries '
                  'that class')


def report_coverage(coverage, pc_range):
    """How much of each tile the configured range actually keeps.

    A range narrower than the tile crops it, and because the frame's origin
    is the block `offset` rather than the tile centre, a square range around
    zero is offset from the tile by an amount that grows with tile size --
    so this is the check that catches a range copied from a smaller export.
    """
    if not coverage:
        return
    fracs = np.array([c for c, _ in coverage])
    worst_frac, worst_name = min(coverage)
    poor = int((fracs < 0.9).sum())
    print(f'  [range] median {np.median(fracs):.1%} of raw points fall inside '
          f'{[round(v, 2) for v in pc_range]}; worst tile {worst_name} '
          f'{worst_frac:.1%}')
    if poor:
        print(f'  [range] {poor}/{len(fracs)} tiles keep <90% of their points '
              '-- check the range covers the tile in the `offset` frame')


def report_class_conflicts(class_conflicts, class_lookup):
    """Tiles whose own class taxonomy disagrees with the one being used.

    Not fatal -- the conversion has already happened by the time this prints
    -- but it means some tiles' `class_id`s were matched against the wrong
    names, so the resulting labels cannot be trusted.
    """
    if not class_conflicts:
        return
    print(f'  [warn] {len(class_conflicts)} tile(s) declare a DIFFERENT class '
          f'lookup than the {class_lookup} used for --map-classes/--classes; '
          'their labels are not trustworthy:')
    for entry in class_conflicts[:10]:
        print(f'  [warn]   {entry["name"]}: {entry["classes"]}')
    if len(class_conflicts) > 10:
        print(f'  [warn]   ... and {len(class_conflicts) - 10} more')


def report_out_of_bounds(out_of_bounds):
    if not out_of_bounds:
        return
    print(f'  [warn] {len(out_of_bounds)} tile(s) have polyline points '
          f'outside their own tile bounds (+{BOUNDS_MARGIN}m margin):')
    for entry in sorted(
            out_of_bounds, key=lambda e: -e['overshoot'])[:10]:
        print(f'  [warn]   {entry["name"]}: {entry["overshoot"]:.2f}m past '
              'the boundary')
    if len(out_of_bounds) > 10:
        print(f'  [warn]   ... and {len(out_of_bounds) - 10} more')


def main():
    args = parse_args()
    lidar_check = not args.no_lidar_check
    classes = args.classes
    if args.lane_types:
        print('[warn] --lane-types is deprecated and never matched anything; '
              'treating it as --classes (matched against class_lookup)')
        classes = (classes or []) + args.lane_types
    samples, dropped, meta = convert_carla_tiles(
        args.data_root,
        args.split,
        classes,
        map_classes=args.map_classes,
        pc_range=args.lidar_point_cloud_range,
        z_max=args.z_max,
        min_lidar_points=args.min_lidar_points,
        lidar_check=lidar_check,
        gt_frame=args.gt_frame,
        reference_dir=args.reference_dir)
    # Derived from the manifest's tile size inside convert_carla_tiles when
    # not given on the CLI, so read it back rather than re-deriving it here.
    pc_range = meta['pc_range']
    mmcv.mkdir_or_exist(args.out_dir)
    tag = f'_{args.out_tag}' if args.out_tag else ''
    out_path = os.path.join(args.out_dir,
                            f'carla_map_infos_{args.split}{tag}.pkl')
    mmcv.dump(
        dict(
            samples=samples,
            split=args.split,
            data_root=args.data_root,
            # The tile geometry these samples were built from, so a later
            # reader can tell a 25m export from a 60m one without reopening
            # the source manifest.
            tile_geometry=dict(
                tile_dir=meta['tile_dir'],
                # which polyline directory the annotations came from -- an
                # export can ship several, holding DIFFERENT GT sets
                reference_dir=meta['reference_dir'],
                tile_radius=meta['tile_radius'],
                tile_side=meta['tile_side']),
            # The origin every sample's annotation (and, via
            # lidar_recenter_shift, its point cloud) is relative to.
            gt_frame=meta['gt_frame'],
            class_lookup=meta['class_lookup'],
            classes_kept=meta['classes_kept'],
            # The annotation keys every sample carries, in label order.
            # CustomCarlaLocalMapDataset checks its own map_classes against
            # this, so a config/pkl mismatch is reported rather than
            # silently training on empty GT.
            map_classes=meta['map_classes'],
            class_groups=meta['class_groups'],
            # Records the geometry the per-sample counts were measured
            # against, so CustomCarlaLocalMapDataset can warn if the config
            # it is running under no longer matches.
            lidar_check=dict(
                enabled=lidar_check,
                point_cloud_range=list(pc_range),
                z_max=args.z_max,
                min_lidar_points=args.min_lidar_points),
            dropped_tiles=dropped),
        out_path)
    print(f'Saved {out_path}')

    if lidar_check:
        # Sidecar report, so a cluster run can be inspected without
        # unpickling the (large) infos file.
        report_path = os.path.join(
            args.out_dir, f'carla_map_infos_{args.split}{tag}_dropped.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(
                dict(
                    split=args.split,
                    data_root=args.data_root,
                    tile_radius=meta['tile_radius'],
                    point_cloud_range=list(pc_range),
                    z_max=args.z_max,
                    min_lidar_points=args.min_lidar_points,
                    n_kept=len(samples),
                    n_dropped=len(dropped),
                    dropped=dropped),
                f,
                indent=2)
        print(f'Saved {report_path}')


if __name__ == '__main__':
    main()
