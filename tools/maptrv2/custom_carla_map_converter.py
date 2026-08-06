"""Convert the CARLA road-polyline tile dataset into MapTR's map-annotation
pkl format.

Unlike the nuScenes/AV2 converters, CARLA tiles are static 25m x 25m patches
(not per-timestamp driving-log frames), so there is no ego pose/SE3 transform
and no need to clip polylines against a moving patch -- each tile's
reference-line polylines are already tile-local; this script only needs to
subtract the tile center to get tile-relative coordinates (matching the
`offset` convention `LoadCarlaPointsFromFile` already uses for point clouds).

Expected input layout (see projects/mmdet3d_plugin/datasets/carla_utils.py)::

    <data_root>/<split>/manifest.json
    <data_root>/<split>/blocks/<tile_name>.npz
    <data_root>/<split>/reference_lines/<tile_name>_reference_lines.json

Each tile's LiDAR block is also scanned for points that would actually
survive the training pipeline's filters (``z <= --z-max`` and inside
``--lidar-point-cloud-range``). Tiles with fewer than ``--min-lidar-points``
such points would voxelize to *zero* voxels and crash ``extract_lidar_feat``,
so they are dropped from the pkl and listed in a sidecar report. Every kept
sample records its own count so the dataset can re-check cheaply.

Usage::

    python tools/maptrv2/custom_carla_map_converter.py \\
        --data-root /path/to/carla --out-dir data/carla/ --split test
"""

import argparse
import json
import os

import mmcv
import numpy as np

# Must match the training config's LiDAR branch geometry:
# `lidar_point_cloud_range` in projects/configs/maptrv2/
# maptrv2_carla_r50_24ep_lidar.py and `z_max` in
# projects/configs/carla/carlasim_map.py. Recorded into the pkl so the
# dataset can warn if the two ever drift apart.
DEFAULT_LIDAR_PC_RANGE = [-12.5, -12.5, -72.0, 12.5, 12.5, 96.0]
DEFAULT_Z_MAX = 96.0


def parse_args():
    parser = argparse.ArgumentParser(
        description='CARLA map data converter arg parser')
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='root of the CARLA tile dataset (contains <split>/manifest.json)')
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
        'the label in the output filename')
    parser.add_argument(
        '--lane-types',
        type=int,
        nargs='+',
        default=None,
        help='optional subset of lane_type_lookup ids to keep as divider '
        'instances (default: keep all types)')
    parser.add_argument(
        '--lidar-point-cloud-range',
        type=float,
        nargs=6,
        default=DEFAULT_LIDAR_PC_RANGE,
        metavar=('X_MIN', 'Y_MIN', 'Z_MIN', 'X_MAX', 'Y_MAX', 'Z_MAX'),
        help='range the LiDAR voxelizer will use; points outside it are '
        'dropped before voxelization, so a tile with none inside produces '
        'zero voxels (default: matches the training config)')
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
        '--no-lidar-check',
        action='store_true',
        help='skip the per-tile LiDAR scan entirely (faster, but no tile is '
        'dropped and no point counts are recorded)')
    return parser.parse_args()


def count_points_in_range(lidar_path, pc_range, z_max):
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
                        lane_types=None,
                        pc_range=None,
                        z_max=DEFAULT_Z_MAX,
                        min_lidar_points=1,
                        lidar_check=True):
    split_dir = os.path.join(data_root, split)
    with open(os.path.join(split_dir, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    tile_radius = float(manifest.get('tile_radius', 12.5))
    if pc_range is None:
        pc_range = DEFAULT_LIDAR_PC_RANGE

    samples = []
    dropped = []
    total_instances = 0
    prog_bar = mmcv.ProgressBar(len(manifest['tiles']))
    for idx, tile in enumerate(manifest['tiles']):
        prog_bar.update()
        name = tile['name']

        lidar_path = os.path.abspath(
            os.path.join(split_dir, 'blocks', f'{name}.npz'))
        if not os.path.isfile(lidar_path):
            raise FileNotFoundError(lidar_path)

        # The polylines below are in WORLD coordinates and must be shifted
        # into the same frame as the LiDAR points the model actually sees.
        # That frame is the block's `offset`, NOT its `tile_center`:
        # LoadCarlaPointsFromFile reads `features[:, 0:3]`, and
        # `points - offset == features[:, :3]` holds exactly, while
        # tile_center differs from offset by a mean of ~2.4m (max >7m)
        # across the train split. Measured against real driving-surface
        # returns (label == 0), `- offset` puts polylines a median 0.038m
        # from the road vs 0.388m for `- tile_center`. Using tile_center
        # here (as this converter originally did) misaligns every GT
        # polyline against its own point cloud, which matters a lot given
        # chamfer eval thresholds of 0.5/1.0/1.5m.
        #
        # np.load is lazy, so this reads only the small `offset` array --
        # it does not pull the full point cloud into memory. (The separate
        # count_points_in_range() call below does read the full `features`
        # array when --no-lidar-check is off; that is the expensive part of
        # this loop, ~14ms for a 110K-point tile.)
        with np.load(lidar_path) as block:
            origin = np.asarray(block['offset'], dtype=np.float32)

        n_raw = n_in_range = None
        if lidar_check:
            n_raw, n_in_range = count_points_in_range(lidar_path, pc_range,
                                                      z_max)
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

        ref_path = os.path.join(split_dir, 'reference_lines',
                                f'{name}_reference_lines.json')
        with open(ref_path, encoding='utf-8') as f:
            ref = json.load(f)

        divider = []
        for poly in ref['polylines']:
            if lane_types is not None and poly['type'] not in lane_types:
                continue
            pts = np.array(poly['points'], dtype=np.float32)
            if pts.shape[0] < 2:
                continue
            divider.append(pts - origin)
            total_instances += 1

        # Sanity check only (tiles are asserted to already be the patch, so
        # no clipping is applied) -- warn, don't crash, if this ever fires.
        margin = 1.0
        for pts in divider:
            if np.any(np.abs(pts[:, :2]) > tile_radius + margin):
                print(f'[warn] {name}: polyline point(s) outside tile '
                      f'radius ({tile_radius}+{margin}m margin), max abs xy '
                      f'= {np.abs(pts[:, :2]).max():.2f}')
                break

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
                tile_bounds=tile.get('bounds'),
                # None when --no-lidar-check was passed; the dataset treats
                # a missing count as "unknown" and keeps the sample.
                num_lidar_points=n_raw,
                num_lidar_points_in_range=n_in_range,
                annotation=dict(divider=divider),
            ))

    n = max(len(samples), 1)
    print()
    if lidar_check:
        print(f'{split}: {len(samples)} tiles kept, {len(dropped)} dropped '
              f'(<{min_lidar_points} in-range LiDAR point'
              f'{"" if min_lidar_points == 1 else "s"}), {total_instances} '
              f'divider instances ({total_instances / n:.1f} per tile)')
        for entry in dropped[:20]:
            print(f'  [drop] {entry["name"]}: {entry["n_points"]} raw points, '
                  f'{entry["n_points_in_range"]} in range')
        if len(dropped) > 20:
            # A long list usually means the range is wrong, not the data --
            # the sidecar json below has the rest.
            print(f'  [drop] ... and {len(dropped) - 20} more')
    else:
        print(f'{split}: {len(samples)} tiles, {total_instances} divider '
              f'instances ({total_instances / n:.1f} per tile) '
              '[LiDAR check skipped]')
    return samples, dropped


def main():
    args = parse_args()
    lidar_check = not args.no_lidar_check
    samples, dropped = convert_carla_tiles(
        args.data_root,
        args.split,
        args.lane_types,
        pc_range=args.lidar_point_cloud_range,
        z_max=args.z_max,
        min_lidar_points=args.min_lidar_points,
        lidar_check=lidar_check)
    mmcv.mkdir_or_exist(args.out_dir)
    out_path = os.path.join(args.out_dir, f'carla_map_infos_{args.split}.pkl')
    mmcv.dump(
        dict(
            samples=samples,
            split=args.split,
            data_root=args.data_root,
            # Records the geometry the per-sample counts were measured
            # against, so CustomCarlaLocalMapDataset can warn if the config
            # it is running under no longer matches.
            lidar_check=dict(
                enabled=lidar_check,
                point_cloud_range=list(args.lidar_point_cloud_range),
                z_max=args.z_max,
                min_lidar_points=args.min_lidar_points),
            dropped_tiles=dropped),
        out_path)
    print(f'Saved {out_path}')

    if lidar_check:
        # Sidecar report, so a cluster run can be inspected without
        # unpickling the (large) infos file.
        report_path = os.path.join(
            args.out_dir, f'carla_map_infos_{args.split}_dropped.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(
                dict(
                    split=args.split,
                    data_root=args.data_root,
                    point_cloud_range=list(args.lidar_point_cloud_range),
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
