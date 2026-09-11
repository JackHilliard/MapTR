"""Per-CLASS precision / recall / F1 (not just AP) at every chamfer threshold.

MapTRv2's evaluator logs per-class AP only; eval_map computes the full per-class
PR curve and throws the precision away. This recovers it.

Usage: class_precision.py <results.json> [<results.json> ...]
       results.json = the carlamap_results.json written by tools/test.py
"""
import sys, types, importlib, json, warnings, numpy as np
warnings.filterwarnings('ignore')
MJ='/MapTR2'; sys.path.insert(0, MJ)
for n,p in {"projects":MJ+"/projects","projects.mmdet3d_plugin":MJ+"/projects/mmdet3d_plugin",
 "projects.mmdet3d_plugin.datasets":MJ+"/projects/mmdet3d_plugin/datasets",
 "projects.mmdet3d_plugin.datasets.map_utils":MJ+"/projects/mmdet3d_plugin/datasets/map_utils"}.items():
    if n not in sys.modules:
        m=types.ModuleType(n); m.__path__=[p]; sys.modules[n]=m
mean_ap=importlib.import_module("projects.mmdet3d_plugin.datasets.map_utils.mean_ap")

GT   = sys.argv[2] if len(sys.argv)>2 and sys.argv[2].endswith('gt.json') else None
CLS  = ['divider','boundary']
THR  = [0.5, 1.0, 1.5]
PC   = [-15.0,-15.0,-2.0,15.0,15.0,24.0]

def f1(p,r): return 2*p*r/np.maximum(p+r,1e-12)

def run(res_path, gt_path):
    res = json.load(open(res_path))['results']
    anns = json.load(open(gt_path))['GTs']
    cg, ct = mean_ap.format_res_gt_by_classes(
        res_path, res, anns, cls_names=CLS, num_pred_pts_per_instance=100,
        eval_use_same_gt_sample_num_flag=True, pc_range=PC, code_size=2, nproc=12)
    out={}
    for thr in THR:
        _, cls_res = mean_ap.eval_map(
            res, anns, cg, ct, threshold=thr, cls_names=CLS, logger='silent',
            num_pred_pts_per_instance=100, pc_range=PC, metric='chamfer',
            code_size=2, nproc=12)
        for i, name in enumerate(CLS):
            r = cls_res[i]
            rec = np.asarray(r['recall']).ravel()
            pre = np.asarray(r['precision']).ravel()
            if rec.size == 0:
                out[(name,thr)] = dict(ap=float(r['ap']), ngt=int(r['num_gts']),
                                       ndet=int(r['num_dets']), pk=0., rk=0., fk=0., prec_at_full=0.)
                continue
            F = f1(pre, rec); k = int(np.argmax(F))
            out[(name,thr)] = dict(ap=float(r['ap']), ngt=int(r['num_gts']),
                                   ndet=int(r['num_dets']),
                                   pk=float(pre[k]), rk=float(rec[k]), fk=float(F[k]),
                                   prec_at_full=float(pre[-1]), rec_at_full=float(rec[-1]))
    return out

def show(tag, out):
    print(f"\n{'='*82}\n{tag}\n{'='*82}")
    print(f"  {'class':<10} {'thr':>4} | {'AP':>7} | {'precision':>9} {'recall':>7} {'F1':>7}  <- at max-F1")
    print("  " + "-"*74)
    for name in CLS:
        for thr in THR:
            d = out[(name,thr)]
            print(f"  {name:<10} {thr:>4} | {d['ap']:>7.4f} | {d['pk']:>9.4f} {d['rk']:>7.4f} {d['fk']:>7.4f}")
        aps=[out[(name,t)]['ap'] for t in THR]
        print(f"  {name:<10} {'mean':>4} | {np.mean(aps):>7.4f} |   GT {out[(name,THR[0])]['ngt']}  "
              f"det {out[(name,THR[0])]['ndet']}")
        print("  " + "-"*74)
    mAP = np.mean([out[(n,t)]['ap'] for n in CLS for t in THR])
    print(f"  {'mAP (mean over classes and thresholds)':<48} {mAP:>7.4f}")

if __name__ == '__main__':
    args=[a for a in sys.argv[1:] if not a.endswith('gt.json')]
    gt=[a for a in sys.argv[1:] if a.endswith('gt.json')]
    gt = gt[0] if gt else '/MapTR2/data/carla50m_map_gt_2cls.json'
    for a in args:
        show(a.split('/')[-3] if '/' in a else a, run(a, gt))
