import sys, json, time, warnings
sys.path.insert(0,'/MapTR2/projects/mmdet3d_plugin/datasets'); warnings.filterwarnings('ignore')
from carla50m_metrics import precision_and_chamfer
CLS=['divider','boundary']
for run, known in (('c50m_ORIG', {'divider':0.7688,'boundary':0.8599}),
                   ('c50m_EMDV2', None)):
    import glob
    fs=glob.glob(f'/MapTR2/work_dirs/{run}/*/pts_bbox/carlamap_results.json')
    if not fs: print(f'{run}: no results'); continue
    res=json.load(open(fs[0]))['results']
    gt=json.load(open('/MapTR2/data/carla50m_map_gt_2cls.json'))['GTs']
    t=time.time(); out=precision_and_chamfer(res, gt, CLS); el=time.time()-t
    print(f"\n=== {run}  ({el:.0f}s, {len(res)} tiles) ===")
    print(f"  {'class':<10} {'thr':>4} {'precision':>10} {'recall':>8} {'F1':>8} {'APcheck':>9}")
    for n in CLS:
        for thr in (0.5,1.0,1.5):
            print(f"  {n:<10} {thr:>4} {out.get(f'{n}_precision_thr_{thr}',0):>10.4f} "
                  f"{out.get(f'{n}_recall_thr_{thr}',0):>8.4f} {out.get(f'{n}_F1_thr_{thr}',0):>8.4f} "
                  f"{out.get(f'{n}_APcheck_thr_{thr}',0):>9.4f}")
        ap=sum(out.get(f'{n}_APcheck_thr_{t}',0) for t in (0.5,1.0,1.5))/3
        line=f"  {n:<10} mean AP {ap:.4f}"
        if known: line += f"   evaluator logged {known[n]:.4f}   diff {abs(ap-known[n]):.4f}"
        print(line)
        print(f"  {n:<10} chamfer mean {out.get(f'{n}_chamfer',float('nan')):.4f} m  "
              f"median {out.get(f'{n}_chamfer_median',float('nan')):.4f} m")
    print(f"  mean_chamfer {out.get('mean_chamfer',float('nan')):.4f} m")
