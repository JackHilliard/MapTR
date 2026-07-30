"""Render predicted-vs-GT CARLA divider polylines in bird's-eye-view.

CARLA-specific, LiDAR-only counterpart to tools/maptrv2/av2_vis_pred.py --
that script's BEV plotting is entangled with AV2's multi-camera image
projection (irrelevant here, CARLA has no camera branch) and its own
multi-class (divider/ped_crossing/boundary/centerline) bookkeeping
(irrelevant here, CARLA is divider-only). This keeps just the BEV plot,
adapted to this dataset's ±12.5m square tile and single class, and writes
one combined GT+prediction PNG per sample (not separate GT_MAP/PRED_MAP
files) for easier side-by-side viewing.

GT is read directly from the dataset's own stored annotations
(`dataset.data_infos[i]['annotation']['divider']`) rather than through the
model-input pipeline, since CustomCarlaLocalMapDataset's test-mode path
(unlike its train-mode path) never attaches gt_bboxes_3d/gt_labels_3d --
there's nothing to denormalize or transform, the converter's coordinates
are already tile-relative metric xy, matching the model's predicted xy.

Usage (inside the container, from /MapTR):
  python3 tools/maptrv2/carla_bev_vis.py <config> <checkpoint> \
      --work-dir <dir> --num-samples 20 --score-thresh 0.3

Writes PNGs to <work_dir>/vis/<sample_idx>.png.
"""
import argparse
import os.path as osp

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from PIL import Image

import projects.mmdet3d_plugin  # noqa: registers custom modules (MapTRv2, etc.)
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from projects.mmdet3d_plugin.datasets.builder import build_dataloader


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--work-dir', required=True,
                    help='PNGs are written to <work-dir>/vis/')
    p.add_argument('--num-samples', type=int, default=20)
    p.add_argument('--score-thresh', type=float, default=0.3)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    vis_dir = osp.join(args.work_dir, 'vis')
    mmcv.mkdir_or_exist(osp.abspath(vis_dir))

    car_img_path = osp.join(osp.dirname(osp.dirname(osp.dirname(
        osp.abspath(__file__)))), 'figs', 'car.png')
    car_img = Image.open(car_img_path) if osp.exists(car_img_path) else None

    pc_range = cfg.point_cloud_range
    xlim = (pc_range[0], pc_range[3])
    ylim = (pc_range[1], pc_range[4])

    n = min(args.num_samples, len(dataset))
    print(f'Rendering {n} of {len(dataset)} samples to {vis_dir}')
    prog_bar = mmcv.ProgressBar(n)

    for i, data in enumerate(data_loader):
        if i >= n:
            break

        img_metas = data['img_metas'][0].data[0]
        sample_idx = img_metas[0].get('sample_idx')
        if sample_idx is None:
            pts_filename = img_metas[0].get('pts_filename', f'sample_{i:05d}')
            sample_idx = osp.splitext(osp.basename(pts_filename))[0]

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        pred = result[0]['pts_bbox']

        gt_polylines = dataset.data_infos[i]['annotation'].get('divider', [])

        keep = pred['scores_3d'] > args.score_thresh
        pred_pts = pred['pts_3d'][keep]

        plt.figure(figsize=(5, 5))
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.axis('off')

        for pts in gt_polylines:
            pts = np.asarray(pts)
            plt.plot(pts[:, 0], pts[:, 1], color='black', linewidth=1.5,
                      alpha=0.8, zorder=1, label='_nolegend_')

        for pts in pred_pts:
            pts = pts.numpy() if torch.is_tensor(pts) else np.asarray(pts)
            plt.plot(pts[:, 0], pts[:, 1], color='orange', linewidth=1.5,
                      alpha=0.9, zorder=2, linestyle='--',
                      label='_nolegend_')

        # cheap legend: two dummy lines, since per-polyline labels above are
        # suppressed to avoid one legend entry per instance
        plt.plot([], [], color='black', linewidth=1.5, label='GT')
        plt.plot([], [], color='orange', linewidth=1.5, linestyle='--',
                  label='pred')
        plt.legend(loc='upper right', fontsize=6, frameon=False)

        if car_img is not None:
            plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2], zorder=3)

        out_path = osp.join(vis_dir, f'{sample_idx}.png')
        plt.savefig(out_path, bbox_inches='tight', format='png', dpi=150)
        plt.close()

        prog_bar.update()

    print(f'\nDone. {n} images written to {vis_dir}')


if __name__ == '__main__':
    main()
