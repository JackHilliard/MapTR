import sys, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for run in ('c50m_ORIG','c50m_EMDV2'):
    fs=sorted(glob.glob(f"/MapTR2/work_dirs/{run}/tf_logs/events.out.tfevents*"))
    if not fs: print(f"{run}: none"); continue
    ea=EventAccumulator(fs[-1], size_guidance={'scalars':0}); ea.Reload()
    tags=ea.Tags()['scalars']
    print(f"\n{run}: {len(tags)} scalar tags, {len(fs)} file(s)")
    for t in sorted(tags):
        v=ea.Scalars(t)
        print(f"   {t:<46} n={len(v):<5} last={v[-1].value:.4f} step={v[-1].step}")
