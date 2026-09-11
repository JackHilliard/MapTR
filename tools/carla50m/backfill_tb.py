"""Write precision + chamfer for the LATEST eval into each run's TB log.

Only the latest eval can be backfilled: MapTRv2 overwrites carlamap_results.json
each time, so earlier epochs' predictions no longer exist.
"""
import sys, json, glob, re, warnings
sys.path.insert(0,'/MapTR2/projects/mmdet3d_plugin/datasets'); warnings.filterwarnings('ignore')
from carla50m_metrics import precision_and_chamfer
from torch.utils.tensorboard import SummaryWriter
CLS=['divider','boundary']
gts=json.load(open('/MapTR2/data/carla50m_map_gt_2cls.json'))['GTs']
for run in ('c50m_ORIG','c50m_EMDV2'):
    fs=glob.glob(f'/MapTR2/work_dirs/{run}/*/pts_bbox/carlamap_results.json')
    if not fs: print(f'{run}: no results'); continue
    # Align to the step the EXISTING AP tags use, so precision/chamfer plot on
    # the same x-axis. (val entries' "iter" is the val-sample count, 2170 --
    # NOT the global training step; using it put the points at x=2170.)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    # Search ALL event files for the mAP tag -- the training writer and any
    # previous backfill writer each own a separate file, and the newest is not
    # necessarily the one holding mAP.
    step=None; ep=0
    for f in sorted(glob.glob(f'/MapTR2/work_dirs/{run}/tf_logs/events.out.tfevents*')):
        ea=EventAccumulator(f, size_guidance={'scalars':0}); ea.Reload()
        ap=[t for t in ea.Tags()['scalars'] if t.endswith('/mAP')]
        if ap:
            sc=ea.Scalars(ap[0]); step=sc[-1].step; ep=len(sc)*2
    if step is None: print(f'{run}: no mAP tag to align to'); continue
    res=json.load(open(fs[0]))['results']
    out=precision_and_chamfer(res,gts,CLS)
    w=SummaryWriter(f'/MapTR2/work_dirs/{run}/tf_logs')
    for k,v in out.items():
        w.add_scalar(f'val/CarlaMap_chamfer/{k}', v, step)
    w.flush(); w.close()
    print(f'{run}: wrote {len(out)} scalars at step {step} (epoch {ep})  '
          f'mean_chamfer={out.get("mean_chamfer",float("nan")):.4f}')
