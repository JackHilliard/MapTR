"""Score a Pointcept polyline dump with MapTRv2's *actual* chamfer-AP protocol.

Loads MapTRv2's real eval code (map_utils.mean_ap.eval_map + custom_tpfp_gen +
custom_polyline_score) and runs it on predictions + GT dumped by
tools/dump_polyline_preds.py. Reports AP@{0.5,1.0,1.5} and mAP exactly as the
CARLA offline-map dataset does (eval_use_same_gt_sample_num_flag=True => both pred
and GT resampled to 100 pts; symmetric chamfer; VOC area AP).

Run inside the MapTR container (maptrv2.sif), which has mmdet/shapely/scipy.
"""
import argparse
import os
import pickle
import sys
import types
import importlib

import numpy as np

MJ = "/MapTR2"
MAP_UTILS_DIR = os.path.join(
    MJ, "projects/mmdet3d_plugin/datasets/map_utils")


def load_maptr_eval():
    """Import the real map_utils.mean_ap without triggering the heavy plugin __init__."""
    if MJ not in sys.path:
        sys.path.insert(0, MJ)
    chain = {
        "projects": os.path.join(MJ, "projects"),
        "projects.mmdet3d_plugin": os.path.join(MJ, "projects/mmdet3d_plugin"),
        "projects.mmdet3d_plugin.datasets": os.path.join(MJ, "projects/mmdet3d_plugin/datasets"),
        "projects.mmdet3d_plugin.datasets.map_utils": MAP_UTILS_DIR,
    }
    for name, path in chain.items():
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [path]
            sys.modules[name] = m
    return importlib.import_module(
        "projects.mmdet3d_plugin.datasets.map_utils.mean_ap")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="preds_gt pkl from dump_polyline_preds.py")
    ap.add_argument("--workdir", default="/tmp/polyline_eval")
    ap.add_argument("--nproc", type=int, default=12)
    ap.add_argument("--cls-name", default="lane")
    args = ap.parse_args()

    mean_ap_mod = load_maptr_eval()
    eval_map = mean_ap_mod.eval_map
    format_res_gt_by_classes = mean_ap_mod.format_res_gt_by_classes

    with open(args.dump, "rb") as f:
        dump = pickle.load(f)
    samples = dump["samples"]
    print(f"Loaded {len(samples)} tiles from {args.dump}")

    CLS = args.cls_name
    pc_range = [-15.0, -15.0, -30.0, 15.0, 15.0, 20.0]
    code_size = 2

    gen_results, annotations = [], []
    n_pred = n_gt = 0
    for s in samples:
        pvecs = []
        for pts, score, lbl in s["pred"]:
            pvecs.append(dict(pts=np.asarray(pts, dtype=np.float64),
                              pts_num=len(pts), cls_name=CLS, type=0,
                              confidence_level=float(score)))
        gvecs = []
        for pts in s["gt_pts"]:
            gvecs.append(dict(pts=np.asarray(pts, dtype=np.float64),
                              pts_num=len(pts), cls_name=CLS, type=0))
        gen_results.append(dict(sample_token=s["sample_token"], vectors=pvecs))
        annotations.append(dict(sample_token=s["sample_token"], vectors=gvecs))
        n_pred += len(pvecs)
        n_gt += len(gvecs)
    print(f"  preds={n_pred}  gts={n_gt}")

    os.makedirs(args.workdir, exist_ok=True)
    result_path = os.path.join(args.workdir, "results.json")

    cls_gens, cls_gts = format_res_gt_by_classes(
        result_path, gen_results, annotations,
        cls_names=[CLS], num_pred_pts_per_instance=100,
        eval_use_same_gt_sample_num_flag=True,
        pc_range=pc_range, code_size=code_size, nproc=args.nproc)

    thresholds = [0.5, 1.0, 1.5]
    aps = {}
    for thr in thresholds:
        _, cls_ap = eval_map(
            gen_results, annotations, cls_gens, cls_gts,
            threshold=thr, cls_names=[CLS], logger=None,
            num_pred_pts_per_instance=100, pc_range=pc_range,
            metric="chamfer", code_size=code_size, nproc=args.nproc)
        aps[thr] = float(cls_ap[0]["ap"])
        print(f"  AP@{thr} = {aps[thr]:.4f}")

    mAP = float(np.mean(list(aps.values())))
    print("\n================ MapTRv2-protocol chamfer AP ================")
    print(f"  AP@0.5 = {aps[0.5]:.4f}")
    print(f"  AP@1.0 = {aps[1.0]:.4f}")
    print(f"  AP@1.5 = {aps[1.5]:.4f}   <-- headline (MapTRv2 EMD=0.741 / L1=0.652)")
    print(f"  mAP    = {mAP:.4f}   (mean over {{0.5,1.0,1.5}})")
    print("=============================================================")


if __name__ == "__main__":
    main()
