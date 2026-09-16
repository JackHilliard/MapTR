"""3x3 tile neighbourhoods (78 m box), two classes, `emdv2` loss.

Multi-tile inference feasibility: each sample is a 30 m tile PLUS every
tile whose footprint overlaps it (up to 8 on the export's 24 m stride), the
neighbours' point clouds shifted into the centre tile's frame and their GT
stitched by `road_id` (`CustomCarlaNeighbourhoodDataset`). The model is the
unchanged LiDAR-only MapTRv2 seeing a bigger tile: this file is the
`..._30m_2cls_EMDV2.py` baseline with every tile-size knob grown from 30 m
to 78 m the same way the 30 m config grew them from 25 m. No crop or
rotation augmentation -- the neighbourhood is assembled as exported.

What "grown" means, knob by knob (all marked [3x3] below):

* `point_cloud_range` / `lidar_point_cloud_range` xy: +-15 -> +-39
  (30 + 2 x 24 stride = 78 m; the dataset asserts its
  `neighbourhood_radius` equals this half-extent).
* `bev_h_`/`bev_w_`: 120 -> 312, holding the BEV cell at 0.25 m. That is
  6.76x the BEV tokens (97,344 vs 14,400) into the decoder's deformable
  cross-attention and the seg head -- the first thing to trade away if it
  OOMs (a 0.5 m cell is 156).
* `sparse_shape`: 301 -> 781 in x/y at the unchanged 0.1 m voxel; z
  untouched (421). `lidar_bev_proj.in_channels` stays 3200 -- it is a
  function of z only (gotcha #4; measured, see CLAUDE.md).
* `max_voxels`: scaled with area (x6.76, rounded) so a dense
  neighbourhood is not silently truncated.
* coder `post_center_range` and every `pc_range` (coder, assigner,
  encoder, `loss_pts`) follow the range.
* `loss_pts.loss_weight`: the geometry loss sees coordinates normalised by
  the range, so a metre of error is 78/30 = 2.6x cheaper here. Scaled by
  that ratio (7.07 -> 18.38), per the tile-size parameterisation the HM
  configs established (CLAUDE.md, "polyline geometry losses are tile-size
  parameterised"; degree-1 losses scale linearly). Note the base
  `PtsL1Loss` was NOT rescaled between 25 m and 30 m, so this is a choice;
  it keeps the physical cost of an error equal to the single-tile run's.
* `transformer.lidar_proj_before_interp=True` (new flag, transformer.py):
  the stock LiDAR path bicubic-interpolates the sparse encoder's
  3200-channel map to (bev_h, bev_w) BEFORE the 3200->256 projection. At
  312x312 that intermediate alone is 1.25 GB per sample (measured: the
  forward OOMs exactly there on an 8 GB card), against 184 MB at 120x120.
  Projecting first at the encoder's native 98x98 and interpolating the
  256-channel result is 12.5x smaller. It is a different model (conv and
  interpolation do not commute), so it is opt-in and the 30 m configs keep
  the original order.
* queries: `num_vec_one2one` stays 50 (`max_num` 50 with it, gotcha #15).
  Stitched GT per neighbourhood averages 5.6 lines and maxes 33 on
  `../carla_test` (single tiles: 3.5), so 50 covers it; the one2many
  branch repeats every GT `k_one2many`=6 times into 300 slots, which caps
  GT at 50 per sample.

Memory, measured on the busiest neighbourhood (8 neighbours, 732k points
after the box cut) on the local 3070: `extract_lidar_feat` peaks at
1.2 GB and a full no-grad `forward_test` at 1.3 GB (5.8 s, first call).
The 30 m config trains at ~3.3 GB at batch 1; a training step here has
NOT been run (the GPU was occupied), so budget for the H100 and measure
before choosing `samples_per_gpu`. The sparse encoder scales with
occupied voxels (~3-4 tiles' worth on average, not 9), the BEV-side
modules with 6.76x the tokens.

Eval note: `evaluation` scores the stitched neighbourhood GT, so
`CarlaMap_chamfer/*` here is a different metric from the single-tile
config's (each overlap-zone line is scored once per neighbourhood it lies
in). For a like-for-like number, run `tools/test.py --format-only`, then
`tools/maptrv2/stitch_results.py <results> --gt <2cls pkl> --clip-to-tile
15 --eval`, which clips every prediction back to its centre tile and scores
it on the per-tile pkl the baseline is scored on.
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_30m_2cls_EMDV2.py']

# [3x3] neighbourhood geometry
tile_side = 30.0
tile_stride = 24.0                      # manifest: tile_side 30, overlap 0.2
neighbourhood_radius = tile_side / 2.0 + tile_stride   # 39.0
_r = neighbourhood_radius

point_cloud_range = [-_r, -_r, -30.0, _r, _r, 20.0]
lidar_point_cloud_range = [-_r, -_r, -72.0, _r, _r, 96.0]
lidar_voxel_size = [0.1, 0.1, 0.4]
dataset_pc_range = [-_r, -_r, -2.0, _r, _r, 24.0]
voxel_size = [0.15, 0.15, 20.0]

map_classes = ['driving', 'curb']
num_map_classes = len(map_classes)
fixed_ptsnum_per_gt_line = 20
fixed_ptsnum_per_pred_line = 20
eval_use_same_gt_sample_num_flag = True

# [3x3] 0.25 m BEV cells, as at 25 m and 30 m
bev_h_ = int(round(2 * _r / 0.25))      # 312
bev_w_ = bev_h_
_dim_ = 256
_pos_dim_ = _dim_ // 2

# [3x3] geometry-loss weight follows the range (see docstring)
loss_pts_weight = 7.07 * (2 * _r) / tile_side

data_root = 'data/carla/'
ann_file_train = data_root + 'carla_map_infos_train_30m_tc_2cls.pkl'
ann_file_val = data_root + 'carla_map_infos_test_30m_tc_2cls.pkl'
ann_file_test = data_root + 'carla_map_infos_test_30m_tc_2cls.pkl'
# Own GT json: the stitched neighbourhood GT is a different GT set from the
# per-tile one, and _format_gt() never regenerates an existing file.
map_ann_file = data_root + 'carla_map_gt_3x3_tc_2cls.json'

aux_seg_cfg = dict(
    use_aux_seg=True,
    bev_seg=True,
    pv_seg=False,
    seg_classes=1,
    feat_down_sample=32,
    pv_thickness=1,
)

model = dict(
    lidar_encoder=dict(
        voxelize=dict(
            point_cloud_range=lidar_point_cloud_range,
            # [3x3] area x6.76 over the 30 m config's [90000, 120000]
            max_voxels=[600000, 800000]),
        backbone=dict(
            # [3x3] 78 m / 0.1 m = 780 -> 781 (same +1 convention); z as is
            sparse_shape=[781, 781, 421]),
    ),
    pts_bbox_head=dict(
        bev_h=bev_h_,
        bev_w=bev_w_,
        transformer=dict(
            lidar_proj_before_interp=True,   # [3x3] see docstring
            encoder=dict(
                pc_range=point_cloud_range,
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='GeometrySptialCrossAttention',
                            pc_range=point_cloud_range,
                            attention=dict(
                                type='GeometryKernelAttention',
                                embed_dims=_dim_,
                                num_heads=4,
                                dilation=1,
                                kernel_size=(3, 5),
                                num_levels=1),
                            embed_dims=_dim_,
                        )
                    ]))),
        bbox_coder=dict(
            # [3x3] range plus the same 2 m margin
            post_center_range=[-_r - 2, -_r - 2, -_r - 2, -_r - 2,
                               _r + 2, _r + 2, _r + 2, _r + 2],
            pc_range=point_cloud_range),
        positional_encoding=dict(
            row_num_embed=bev_h_,
            col_num_embed=bev_w_),
        loss_pts=dict(
            loss_weight=loss_pts_weight,
            pc_range=point_cloud_range)),
    train_cfg=dict(pts=dict(
        point_cloud_range=point_cloud_range,
        assigner=dict(pc_range=point_cloud_range))))

train_pipeline = [
    dict(
        type='LoadCarlaNeighbourhoodPoints',
        coord_type='LIDAR',
        load_dim=3,
        use_dim=3,
        z_max=96.0,
        recenter=True),
    dict(
        type='GridSamplePoints',
        grid_size=lidar_voxel_size,
        point_cloud_range=lidar_point_cloud_range),
    dict(
        type='DefaultFormatBundle3D',
        with_gt=False,
        with_label=False,
        class_names=map_classes),
    dict(type='CustomCollect3D', keys=['points'])
]

test_pipeline = [
    dict(
        type='LoadCarlaNeighbourhoodPoints',
        coord_type='LIDAR',
        load_dim=3,
        use_dim=3,
        z_max=96.0,
        recenter=True),
    dict(
        type='GridSamplePoints',
        grid_size=lidar_voxel_size,
        point_cloud_range=lidar_point_cloud_range),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1, 1),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                with_gt=False,
                with_label=False,
                class_names=map_classes),
            dict(type='CustomCollect3D', keys=['points'])
        ])
]

_nbhd = dict(
    type='CustomCarlaNeighbourhoodDataset',
    neighbourhood_radius=neighbourhood_radius,
    neighbour_max_offset=tile_side,   # footprints overlap -> neighbours
    gt_merge_tol=0.05,
    min_polyline_len=1.0,
    pc_range=dataset_pc_range,
    bev_size=(bev_h_, bev_w_),
    lidar_pc_range=lidar_point_cloud_range,
    map_classes=map_classes,
    data_root=data_root,
)

data = dict(
    train=dict(_nbhd, ann_file=ann_file_train, pipeline=train_pipeline,
               aux_seg=aux_seg_cfg),
    val=dict(_nbhd, ann_file=ann_file_val, map_ann_file=map_ann_file,
             pipeline=test_pipeline),
    test=dict(_nbhd, ann_file=ann_file_test, map_ann_file=map_ann_file,
              pipeline=test_pipeline),
)

evaluation = dict(interval=2, pipeline=test_pipeline, metric='chamfer',
                  save_best='CarlaMap_chamfer/mAP', rule='greater')
