"""Round 2: pc_range/normalisation consistency, BEV seg-mask alignment,
and val-GT vs cached map_ann_file agreement."""
import sys, json, warnings, numpy as np, torch
sys.path.insert(0,'/MapTR2'); warnings.filterwarnings('ignore')
from mmcv import Config
from mmdet3d.datasets import build_dataset
import projects.mmdet3d_plugin  # noqa: F401
cfg=Config.fromfile('/MapTR2/projects/configs/maptrv2/maptrv2_carla50m_ORIG.py')

print("="*80); print("4. pc_range CONSISTENCY (crop box vs every consumer)".center(80)); print("="*80)
head=cfg.model['pts_bbox_head']
print(f"  dataset  pc_range : {cfg.data.train['pc_range']}")
print(f"  head     pc_range : {head.get('pc_range','<none>')}")
print(f"  loss_pts pc_range : {head['loss_pts'].get('pc_range','<n/a for PtsL1Loss>')}")
print(f"  GridSample range  : {[s.get('point_cloud_range') for s in cfg.data.train['pipeline'] if s['type']=='GridSamplePoints']}")
print(f"  transform/bev     : {cfg.get('point_cloud_range')}")
xy=lambda r:(r[0],r[1],r[3],r[4])
rs={'dataset':cfg.data.train['pc_range'], 'head':head.get('pc_range'), 'global':cfg.get('point_cloud_range')}
rs={k:v for k,v in rs.items() if v}
base=xy(list(rs.values())[0])
ok=all(xy(v)==base for v in rs.values())
print(f"  xy extent identical across all of them: {ok}  ({base})")
print(f"  VERDICT: {'CONSISTENT -- GT normalises against the same box it was cropped to' if ok else 'MISMATCH'}")

ds=build_dataset(cfg.data.train)
print("\n"+"="*80); print("5. NORMALISED GT LANDS IN [0,1]".center(80)); print("="*80)
from projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head import normalize_2d_pts
pr=head.get('pc_range') or cfg.get('point_cloud_range')
print(f"  gt_shift_pts_pattern={head.get('gt_shift_pts_pattern')}  z_cfg={head.get('z_cfg')}")
print("  NOTE: the LOSS target is shift_fixed_num_sampled_points_v2 (2 coords), not the")
print("        3-coord fixed_num_sampled_points -- checking the tensor the loss actually sees.")
lo=hi=None; n=0; shp=set()
for i in (5,11,23,37,52):
    ex=ds[i]
    if ex is None: continue
    G=ex['gt_bboxes_3d'].data.shift_fixed_num_sampled_points_v2
    shp.add(tuple(G.shape[1:]))
    N=normalize_2d_pts(G, pr)
    lo = N.min().item() if lo is None else min(lo, N.min().item())
    hi = N.max().item() if hi is None else max(hi, N.max().item())
    n+=1
print(f"  loss-target shape per sample (shifts, pts, coords): {shp}")
print(f"  normalize_2d_pts over {n} samples -> min {lo:.5f}  max {hi:.5f}")
print(f"  VERDICT: {'IN RANGE' if (lo>=-1e-4 and hi<=1+1e-4) else 'OUT OF RANGE -- normalisation box wrong'}")

print("\n"+"="*80); print("6. BEV SEG MASK IS DRAWN IN THE SAME FRAME AS GT".center(80)); print("="*80)
hits=[]
for i in (5,11,23,37,52):
    ex=ds[i]
    if ex is None or 'gt_seg_mask' not in ex: continue
    M=ex['gt_seg_mask'].data.numpy()
    G=ex['gt_bboxes_3d'].data.fixed_num_sampled_points.numpy()[:,:,:2]
    m=M[0] if M.ndim==3 else M
    H,W=m.shape
    # GT xy (+-15) -> mask pixel; mask should be lit where GT runs
    px=((G[:,:,0].ravel()+15.0)/30.0*(W-1)).round().astype(int).clip(0,W-1)
    py=((G[:,:,1].ravel()+15.0)/30.0*(H-1)).round().astype(int).clip(0,H-1)
    # dilate by 1 px to tolerate rasterisation thickness
    hit=0
    for a,b in zip(py,px):
        if m[max(0,a-1):a+2, max(0,b-1):b+2].any(): hit+=1
    hits.append(hit/len(px))
    if len(hits)==1: print(f"  mask shape {m.shape}  lit pixels {int(m.sum())}")
print(f"  fraction of GT vertices landing on lit mask pixels: "
      f"{np.mean(hits)*100:.1f}%  (per-sample {[f'{h*100:.0f}%' for h in hits]})")
print(f"  VERDICT: {'SEG MASK MATCHES GT FRAME' if np.mean(hits)>0.9 else 'SEG MASK IN A DIFFERENT FRAME'}")

print("\n"+"="*80); print("7. VAL GT == CACHED map_ann_file (what AP is scored against)".center(80)); print("="*80)
dv=build_dataset(cfg.data.val)
cache=json.load(open('/MapTR2/data/carla50m_map_gt.json'))['GTs']
by={g['sample_token']:g for g in cache}
print(f"  cache tiles {len(cache)}   val dataset {len(dv)}")
errs=[]; miss=0
for i in range(40):
    tok=dv.data_infos[i]['sample_idx']
    if tok not in by: miss+=1; continue
    live=dv.vectormap_pipeline({}, dv.data_infos[i])['gt_bboxes_3d'].data.instance_list
    cv=by[tok]['vectors']
    if len(live)!=len(cv): errs.append(9.99); continue
    for L,C in zip(live,cv):
        a=np.array(list(L.coords))[:,:2]; b=np.array(C['pts'])[:,:2]
        errs.append(np.abs(a-b).max() if a.shape==b.shape else 9.99)
print(f"  tiles missing from cache: {miss}/40")
print(f"  max |live GT - cached GT| over {len(errs)} polylines: {max(errs):.8f} m")
print(f"  VERDICT: {'CACHE MATCHES what the model is scored against' if max(errs)<1e-5 else 'CACHE DIVERGES'}")
