"""CARLA Simulator LiDAR dataset.

Converted from the Pointcept ``CarlaSegDataset`` to the mmdet3d /
mmdetection3d dataset API used by MapTR.

For now this dataset only loads LiDAR point clouds (xyz + strength) from the
per-block ``.npz`` files. The semantic / instance labels that live in those
blocks are ignored until a task head is wired up; loading is handled by the
``LoadCarlaPointsFromFile`` pipeline transform.

Expected directory layout (mirrors the Pointcept ``get_data_list``)::

    <data_root>/<split>/road_blocks_<tile_size>/blocks/*.npz
"""

import glob
import os
from collections.abc import Sequence

import numpy as np
from mmcv.utils import print_log
from mmdet.datasets import DATASETS
from mmdet3d.datasets.custom_3d import Custom3DDataset


@DATASETS.register_module()
class CarlaSegDataset(Custom3DDataset):
    """CARLA simulator point-cloud dataset (LiDAR-only for now).

    Args:
        data_root (str): Root directory of the CARLA road-block dataset.
        pipeline (list[dict]): Data processing pipeline.
        split (str | Sequence[str]): Split name(s) under ``data_root`` to
            gather ``.npz`` blocks from. Defaults to ``'train'``.
        tile_size (int): Road-block tile size, selecting the
            ``road_blocks_<tile_size>`` sub-directory. Defaults to 15.
        classes (tuple[str] | None): Kept for API compatibility; unused while
            the dataset is LiDAR-only. Defaults to None.
        modality (dict | None): Input modality. Defaults to LiDAR-only.
        box_type_3d (str): 3D box type. Defaults to ``'LiDAR'``.
        filter_empty_gt (bool): Kept for API compatibility. Defaults to False
            (there are no GT boxes yet).
        test_mode (bool): Whether the dataset is used for testing.
    """

    CLASSES = None

    def __init__(self,
                 data_root,
                 pipeline=None,
                 split='train',
                 tile_size=15,
                 ann_file=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=False,
                 test_mode=False,
                 **kwargs):
        # These must be set before super().__init__ since load_annotations()
        # (called from the base __init__) relies on them.
        self.split = split
        self.tile_size = tile_size
        if modality is None:
            modality = dict(use_lidar=True, use_camera=False)
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

    def load_annotations(self, ann_file):
        """Build the sample list by globbing the CARLA block layout.

        The ``ann_file`` argument is ignored; samples are discovered from the
        directory structure instead (matching the Pointcept dataset).
        """
        if isinstance(self.split, str):
            split_list = [self.split]
        elif isinstance(self.split, Sequence):
            split_list = self.split
        else:
            raise NotImplementedError(
                f'Unsupported split type: {type(self.split)}')

        data_infos = []
        for split in split_list:
            pattern = os.path.join(self.data_root, split,
                                   f'road_blocks_{self.tile_size}', 'blocks',
                                   '*.npz')
            for bf in sorted(glob.glob(pattern)):
                data_infos.append(
                    dict(
                        pts_path=os.path.relpath(bf, self.data_root),
                        name=os.path.splitext(os.path.basename(bf))[0],
                        split=split))
        return data_infos

    def get_data_info(self, index):
        """Return the input dict consumed by the pipeline.

        LiDAR-only: no ``ann_info`` is attached yet.
        """
        info = self.data_infos[index]
        pts_filename = os.path.join(self.data_root, info['pts_path'])
        return dict(
            pts_filename=pts_filename,
            sample_idx=info['name'],
            file_name=pts_filename)

    def prepare_train_data(self, index):
        """Training data preparation.

        Overridden to skip the base class' GT-emptiness filtering, which
        assumes ``gt_labels_3d`` is produced by the pipeline. With LiDAR-only
        loading there are no labels, so preparation matches the test path.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        return self.pipeline(input_dict)

    def evaluate(self, results, logger=None, **kwargs):
        """Placeholder evaluation.

        LiDAR-only loading has no associated task metric yet, so this is a
        no-op that returns an empty result dict.
        """
        print_log(
            'CarlaSegDataset.evaluate is a no-op (LiDAR-only loading; no task '
            'metric defined yet).',
            logger=logger)
        return {}
