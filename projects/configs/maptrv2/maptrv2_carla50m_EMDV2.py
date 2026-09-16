# 50 m CARLA tiles -> rotated 30 m crops (train-time augmentation).
# AUTO-DERIVED from maptrv2_carla_ORIG.py (+ emdv2 loss) by tools/carla50m/make_configs.py.
# Only the dataset/pipeline blocks change; loss/model/schedule are ORIG's.
#
# Augmentation (train only): uniform yaw about the tile centre + <=3.79 m centre
# shift, 30 m square crop, sigma=0.02 m Gaussian point noise. GT is rotated and
# clipped to the SAME box with the tiler's own clip_polyline_to_box.
#
# val/test are DETERMINISTIC (no yaw/shift/noise): MapTRv2 caches GT to
# map_ann_file once, so a randomly augmented val split would score every epoch
# against GT that does not match the crops the model was actually fed.
#
# Loss: the order-preserving monotone-OT fix (emdv2), proven at parity with
# ordered L1 on the 30 m benchmark (mAP 0.7474 vs 0.7485).
checkpoint_config = dict(interval=2, max_keep_ckpts=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/MapTR/work_dirs/maptrv2-carla_dataset_30m_dense_aligned-V1'
load_from = None
resume_from = None
workflow = [('train', 1)]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
dataset_type = 'CustomCarla50mCropDataset'
data_root = '/data_carla50'
point_cloud_range = [-15.0, -15.0, -30.0, 15.0, 15.0, 20.0]
map_classes = ['divider', 'boundary']
load_dim = 4
use_dim = 4
z_max = 96.0
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
train_pipeline = [
    dict(type='LoadCarla50mCrop', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(
        type='GridSamplePoints',
        grid_size=[0.1, 0.1, 0.4],
        point_cloud_range=[-15.0, -15.0, -72.0, 15.0, 15.0, 96.0]),
    dict(
        type='DefaultFormatBundle3D',
        with_gt=False,
        with_label=False,
        class_names=['divider', 'boundary']),
    dict(type='CustomCollect3D', keys=['points'])
]
test_pipeline = [
    dict(type='LoadCarla50mCrop', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(
        type='GridSamplePoints',
        grid_size=[0.1, 0.1, 0.4],
        point_cloud_range=[-15.0, -15.0, -72.0, 15.0, 15.0, 96.0]),
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
                class_names=['divider', 'boundary']),
            dict(type='CustomCollect3D', keys=['points'])
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=6,
    train=dict(
        type='CustomCarla50mCropDataset',
        data_root='/data_carla50',
        ann_file='/data_carla50/train/manifest.json',
        pipeline=[
            dict(
                type='LoadCarla50mCrop',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4),
            dict(
                type='GridSamplePoints',
                grid_size=[0.1, 0.1, 0.4],
                point_cloud_range=[-15.0, -15.0, -72.0, 15.0, 15.0, 96.0]),
            dict(
                type='DefaultFormatBundle3D',
                with_gt=False,
                with_label=False,
                class_names=['divider', 'boundary']),
            dict(type='CustomCollect3D', keys=['points'])
        ],
        classes=[],
        map_classes=['divider', 'boundary'],
        pc_range=[-15.0, -15.0, -2.0, 15.0, 15.0, 24.0],
        modality=dict(use_lidar=True, use_camera=False),
        box_type_3d='LiDAR',
        filter_empty_gt=True,
        test_mode=False,
        fixed_ptsnum_per_line=20,
        eval_use_same_gt_sample_num_flag=True,
        bev_size=(120, 120),
        min_lidar_points=1,
        lidar_pc_range=[-15.0, -15.0, -72.0, 15.0, 15.0, 96.0],
        aux_seg=dict(
            use_aux_seg=True,
            bev_seg=True,
            pv_seg=False,
            seg_classes=2,
            feat_down_sample=32,
            pv_thickness=1),
        split='train',
        rotate=True,
        translate=True,
        noise_std=0.02,
        require_gt=True,
        gt_tries=32,
        min_polyline_len=2.0,
        gt_classes=('driving', 'curb'),
        crop_size=30.0),
    val=dict(
        type='CustomCarla50mCropDataset',
        data_root='/data_carla50',
        ann_file='/data_carla50/test/manifest.json',
        map_ann_file='/MapTR2/data/carla50m_map_gt_2cls.json',
        pipeline=[
            dict(
                type='LoadCarla50mCrop',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4),
            dict(
                type='GridSamplePoints',
                grid_size=[0.1, 0.1, 0.4],
                point_cloud_range=[-15.0, -15.0, -72.0, 15.0, 15.0, 96.0]),
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
                        class_names=['divider', 'boundary']),
                    dict(type='CustomCollect3D', keys=['points'])
                ])
        ],
        classes=[],
        map_classes=['divider', 'boundary'],
        pc_range=[-15.0, -15.0, -2.0, 15.0, 15.0, 24.0],
        modality=dict(use_lidar=True, use_camera=False),
        box_type_3d='LiDAR',
        test_mode=True,
        fixed_ptsnum_per_line=20,
        eval_use_same_gt_sample_num_flag=True,
        bev_size=(120, 120),
        min_lidar_points=1,
        lidar_pc_range=[-15.0, -15.0, -72.0, 15.0, 15.0, 96.0],
        split='test',
        rotate=False,
        translate=True,
        noise_std=0.0,
        require_gt=True,
        gt_tries=32,
        crop_seed=0,
        min_polyline_len=2.0,
        gt_classes=('driving', 'curb'),
        crop_size=30.0),
    test=dict(
        type='CustomCarla50mCropDataset',
        data_root='/data_carla50',
        ann_file='/data_carla50/test/manifest.json',
        map_ann_file='/MapTR2/data/carla50m_map_gt_2cls.json',
        pipeline=[
            dict(
                type='LoadCarla50mCrop',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4),
            dict(
                type='GridSamplePoints',
                grid_size=[0.1, 0.1, 0.4],
                point_cloud_range=[-15.0, -15.0, -72.0, 15.0, 15.0, 96.0]),
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
                        class_names=['divider', 'boundary']),
                    dict(type='CustomCollect3D', keys=['points'])
                ])
        ],
        classes=[],
        map_classes=['divider', 'boundary'],
        pc_range=[-15.0, -15.0, -2.0, 15.0, 15.0, 24.0],
        modality=dict(use_lidar=True, use_camera=False),
        box_type_3d='LiDAR',
        test_mode=True,
        fixed_ptsnum_per_line=20,
        eval_use_same_gt_sample_num_flag=True,
        bev_size=(120, 120),
        min_lidar_points=1,
        lidar_pc_range=[-15.0, -15.0, -72.0, 15.0, 15.0, 96.0],
        split='test',
        rotate=False,
        translate=True,
        noise_std=0.0,
        require_gt=True,
        gt_tries=32,
        crop_seed=0,
        min_polyline_len=2.0,
        gt_classes=('driving', 'curb'),
        crop_size=30.0),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
voxel_size = [0.15, 0.15, 20.0]
lidar_point_cloud_range = [-15.0, -15.0, -72.0, 15.0, 15.0, 96.0]
lidar_voxel_size = [0.1, 0.1, 0.4]
num_vec = 50
fixed_ptsnum_per_gt_line = 20
fixed_ptsnum_per_pred_line = 20
eval_use_same_gt_sample_num_flag = True
num_map_classes = 2
dataset_pc_range = [-15.0, -15.0, -2.0, 15.0, 15.0, 24.0]
_dim_ = 256
_pos_dim_ = 128
_ffn_dim_ = 512
_num_levels_ = 1
bev_h_ = 120
bev_w_ = 120
queue_length = 1
aux_seg_cfg = dict(
    use_aux_seg=True,
    bev_seg=True,
    pv_seg=False,
    seg_classes=2,
    feat_down_sample=32,
    pv_thickness=1)
model = dict(
    type='MapTRv2',
    use_grid_mask=True,
    video_test_mode=False,
    modality='lidar',
    lidar_encoder=dict(
        voxelize=dict(
            max_num_points=10,
            point_cloud_range=[-15.0, -15.0, -72.0, 15.0, 15.0, 96.0],
            voxel_size=[0.1, 0.1, 0.4],
            max_voxels=[90000, 120000]),
        backbone=dict(
            type='SparseEncoder',
            in_channels=4,
            sparse_shape=[301, 301, 421],
            output_channels=128,
            order=('conv', 'norm', 'act'),
            encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128),
                              (128, 128)),
            encoder_paddings=([0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]], [0, 0]),
            block_type='basicblock')),
    pts_bbox_head=dict(
        type='MapTRv2Head',
        bev_h=120,
        bev_w=120,
        num_query=900,
        num_vec_one2one=50,
        num_vec_one2many=300,
        k_one2many=6,
        num_pts_per_vec=20,
        num_pts_per_gt_vec=20,
        dir_interval=1,
        query_embed_type='instance_pts',
        transform_method='minmax',
        gt_shift_pts_pattern='v2',
        num_classes=2,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=2,
        code_weights=[1.0, 1.0, 1.0, 1.0],
        aux_seg=dict(
            use_aux_seg=True,
            bev_seg=True,
            pv_seg=False,
            seg_classes=2,
            feat_down_sample=32,
            pv_thickness=1),
        transformer=dict(
            type='MapTRPerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=256,
            modality='lidar',
            lidar_bev_proj=dict(
                type='ConvFuser', in_channels=[3200], out_channels=256),
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=1,
                pc_range=[-15.0, -15.0, -30.0, 15.0, 15.0, 20.0],
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=256,
                            num_levels=1),
                        dict(
                            type='GeometrySptialCrossAttention',
                            pc_range=[-15.0, -15.0, -30.0, 15.0, 15.0, 20.0],
                            attention=dict(
                                type='GeometryKernelAttention',
                                embed_dims=256,
                                num_heads=4,
                                dilation=1,
                                kernel_size=(3, 5),
                                num_levels=1),
                            embed_dims=256)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='MapTRDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DecoupledDetrTransformerDecoderLayer',
                    num_vec=50,
                    num_pts_per_vec=20,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=256,
                            num_levels=1)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'self_attn', 'norm',
                                     'cross_attn', 'norm', 'ffn', 'norm')))),
        bbox_coder=dict(
            type='MapTRNMSFreeCoder',
            post_center_range=[
                -17.0, -17.0, -17.0, -17.0, 17.0, 17.0, 17.0, 17.0
            ],
            pc_range=[-15.0, -15.0, -30.0, 15.0, 15.0, 20.0],
            max_num=50,
            voxel_size=[0.15, 0.15, 20.0],
            num_classes=2),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=128,
            row_num_embed=120,
            col_num_embed=120),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_pts=dict(
            type='PolylineGeomLoss',
            mode='emdv2',
            loss_weight=7.07,
            emdv2_pointwise_reduction='sum',
            pc_range=[-15.0, -15.0, -30.0, 15.0, 15.0, 20.0]),
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
        loss_seg=dict(type='SimpleLoss', pos_weight=4.0, loss_weight=1.0),
        loss_pv_seg=dict(type='SimpleLoss', pos_weight=1.0, loss_weight=2.0)),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=[0.15, 0.15, 20.0],
            point_cloud_range=[-15.0, -15.0, -30.0, 15.0, 15.0, 20.0],
            out_size_factor=4,
            assigner=dict(
                type='MapTRAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(
                    type='BBoxL1Cost', weight=0.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
                pts_cost=dict(type='OrderedPtsL1Cost', weight=5),
                pc_range=[-15.0, -15.0, -30.0, 15.0, 15.0, 20.0]))))
file_client_args = dict(backend='disk')
optimizer = dict(type='AdamW', lr=0.0006, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
total_epochs = 30
evaluation = dict(
    interval=2,
    pipeline=[
        dict(
            type='LoadCarlaPointsFromFile',
            coord_type='LIDAR',
            load_dim=4,
            use_dim=4,
            z_max=96.0,
            recenter=True),
        dict(
            type='GridSamplePoints',
            grid_size=[0.1, 0.1, 0.4],
            point_cloud_range=[-15.0, -15.0, -72.0, 15.0, 15.0, 96.0]),
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
                    class_names=['divider', 'boundary']),
                dict(type='CustomCollect3D', keys=['points'])
            ])
    ],
    metric='chamfer',
    save_best='CarlaMap_chamfer/mAP',
    rule='greater')
runner = dict(type='EpochBasedRunner', max_epochs=30)
fp16 = dict(loss_scale=512.0)
find_unused_parameters = True
gpu_ids = range(0, 1)
