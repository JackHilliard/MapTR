_base_ = [
    '../_base_/default_runtime.py'
]
#
# CARLA simulator LiDAR-only data config.
#
# This is a *dataloader* config: it wires up `CarlaSegDataset` +
# `LoadCarlaPointsFromFile` to load LiDAR point clouds (xyz + strength) from the
# CARLA road-block `.npz` files. There is no model/task head yet, so this config
# is meant for exercising the data path, e.g.:
#
#   python tools/misc/browse_dataset.py \
#       projects/configs/carla/carlasim_lidar.py --output-dir vis_carla
#
# or programmatically:
#
#   from mmcv import Config
#   from mmdet3d.datasets import build_dataset
#   cfg = Config.fromfile('projects/configs/carla/carlasim_lidar.py')
#   ds = build_dataset(cfg.data.train)
#   sample = ds[0]                 # {'img_metas': ..., 'points': DataContainer}
#
# Load the plugin so the custom dataset + transform get registered.
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

dataset_type = 'CarlaSegDataset'
data_root = 'data/carla/'

# CARLA road-block layout selector: <data_root>/<split>/road_blocks_<tile_size>/blocks/*.npz
tile_size = 15

# LiDAR points are [x, y, z, strength]; the z<=15.0 filter matches the source
# Pointcept dataset.
load_dim = 4
use_dim = 4
z_max = 15.0

input_modality = dict(use_lidar=True, use_camera=False)

# No task classes while the dataset is LiDAR-only.
class_names = []

train_pipeline = [
    dict(
        type='LoadCarlaPointsFromFile',
        coord_type='LIDAR',
        load_dim=load_dim,
        use_dim=use_dim,
        z_max=z_max),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_gt=False,
        with_label=False),
    dict(type='CustomCollect3D', keys=['points'])
]

test_pipeline = [
    dict(
        type='LoadCarlaPointsFromFile',
        coord_type='LIDAR',
        load_dim=load_dim,
        use_dim=use_dim,
        z_max=z_max),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_gt=False,
        with_label=False),
    dict(type='CustomCollect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        split='train',
        tile_size=tile_size,
        pipeline=train_pipeline,
        modality=input_modality,
        classes=class_names,
        box_type_3d='LiDAR',
        test_mode=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',
        tile_size=tile_size,
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        box_type_3d='LiDAR',
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',
        tile_size=tile_size,
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        box_type_3d='LiDAR',
        test_mode=True),
)
