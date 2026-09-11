"""Pre-build the cached eval GT so concurrent runs don't race on it."""
import sys, os, warnings
sys.path.insert(0,'/MapTR2'); warnings.filterwarnings('ignore')
from mmcv import Config
from mmdet3d.datasets import build_dataset
import projects.mmdet3d_plugin  # registers CustomCarla50mCropDataset  # noqa: F401
cfg = Config.fromfile('/MapTR2/projects/configs/maptrv2/maptrv2_carla50m_ORIG.py')
ds = build_dataset(cfg.data.val)
print(f"val dataset: {len(ds)} samples; map_ann_file={ds.map_ann_file}")
if os.path.exists(ds.map_ann_file):
    print("  already exists, nothing to do")
else:
    ds._format_gt()
    import json
    g=json.load(open(ds.map_ann_file))
    n=len(g['GTs']); tot=sum(len(x['vectors']) for x in g['GTs'])
    print(f"  built: {n} tiles, {tot} GT polylines, {tot/n:.2f}/tile")
