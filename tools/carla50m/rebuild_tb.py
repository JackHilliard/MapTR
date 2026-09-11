"""Rebuild TB scalars from mmcv's log.json.

Needed because I deleted c50m_ORIG's event file while trying to clean up a
backfill file. mmcv writes log.json independently of TensorBoard, so the full
history survives there. Writes a NEW event file alongside the live one --
TensorBoard merges scalars across files in the same run directory. Never
deletes anything.
"""
import sys, json, glob, os
from torch.utils.tensorboard import SummaryWriter

RUN = sys.argv[1]
ITERS_PER_EPOCH = int(sys.argv[2]) if len(sys.argv) > 2 else 932
d = f'/MapTR2/work_dirs/{RUN}'
logs = sorted(glob.glob(f'{d}/*.log.json'))
w = SummaryWriter(f'{d}/tf_logs')
ntr = nval = 0
for L in logs:
    for ln in open(L, errors='ignore'):
        try:
            e = json.loads(ln)
        except Exception:
            continue
        mode = e.get('mode')
        if mode == 'train':
            step = (int(e['epoch']) - 1) * ITERS_PER_EPOCH + int(e['iter'])
            for k, v in e.items():
                if k in ('mode', 'epoch', 'iter', 'time', 'data_time', 'memory'):
                    continue
                if isinstance(v, (int, float)):
                    tag = 'learning_rate' if k == 'lr' else (
                          'momentum' if k == 'momentum' else f'train/{k}')
                    w.add_scalar(tag, v, step)
            ntr += 1
        elif mode == 'val':
            step = int(e['epoch']) * ITERS_PER_EPOCH + 1
            for k, v in e.items():
                if k in ('mode', 'epoch', 'iter') or not isinstance(v, (int, float)):
                    continue
                w.add_scalar(f'val/{k}', v, step)
            nval += 1
w.flush(); w.close()
print(f'{RUN}: rebuilt {ntr} train points, {nval} val points from {len(logs)} log.json')
