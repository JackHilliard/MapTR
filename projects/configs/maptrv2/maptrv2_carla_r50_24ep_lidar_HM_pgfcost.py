_base_ = [
    './maptrv2_carla_r50_24ep_lidar_HM.py',
]
#
# HM (polyline geometry loss) with the FAST matching cost, plus a knob for
# the direction loss.
#
# Two changes from maptrv2_carla_r50_24ep_lidar_HM.py, and nothing else --
# this file deliberately inherits rather than duplicating, so it tracks any
# future change to the HM config automatically. It differs from the 30m
# sibling in kind: that one duplicates because it changes GEOMETRY, this one
# inherits because it changes SETTINGS.
#
#   1. cost_mode 'emd' -> 'pgf'      (see the SUPERSEDED note below)
#   2. loss_dir weight 0.0 -> tunable (EMD supervises no ordering at all)
#
# SUPERSEDED, read this first: change (1) was worth ~20x when this file was
# written, because cost_mode='emd' ran at 10-17 s/iter. PolylineGeomCost's
# `dedup_gt_slices` fix has since brought that to 0.6-0.9 s/iter -- about the
# same as 'pgf'. So (1) is now close to a wash on speed, and the honest
# reason to prefer 'pgf' is gone: with EMD affordable on both sides, matching
# and supervision measuring the SAME geometry is the better default, and that
# is what the plain HM config now gives you.
#
# What remains genuinely useful here is (2), loss_dir. That is orthogonal to
# the cost mode and applies just as well on top of the HM config -- consider
# running the sweep below with --cfg-options against that config instead:
#   python tools/train.py projects/configs/maptrv2/maptrv2_carla_r50_24ep_lidar_HM.py \
#       --cfg-options model.pts_bbox_head.loss_dir.loss_weight=0.05
# Keep this file for the A/B on matching geometry (pgf-matched vs
# emd-matched, everything else equal), which is still a real experiment.
#
# `plugin`/`plugin_dir` must be repeated: mmcv executes each config file in
# isolation before _base_ merging, so tools/train.py reads them from THIS
# file's namespace, not the base's.
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# --- 1. why the matching cost moves to PGF ----------------------------------
# The HM config mirrors the Pointcept source faithfully, and that source uses
# EMD for BOTH the loss and the matching cost. Faithful, but the two sides
# have wildly different cost profiles here, per the HM config's own measured
# table (1 epoch, CARLA test subset, single RTX 3070):
#
#     loss_mode   cost_mode   s/iter
#     emd         pgf         ~0.6
#     emd         emd         10-17     <- what the HM config ships
#
# ~20x, and the reason is structural rather than incidental. The LOSS runs on
# matched pairs only (~50/image). The COST runs on every (query x gt x order)
# combination, which MapTRv2's one2many branch (300 queries x k_one2many=6)
# makes ~18,000 pairs per assigner call -- each one a separate scipy
# linear_sum_assignment on CPU, inside the training step. PGF is fully
# vectorised (_pgf_cost_vectorised), so it has no such loop.
#
# The trade this accepts: matching and supervision now measure geometry
# DIFFERENTLY (PGF picks the assignment, EMD grades it). That is a real
# inconsistency, but a mild one -- the assigner only has to identify WHICH gt
# a query belongs to, a decision dominated by gross position, while the loss
# has to grade HOW WRONG it is, which is where EMD's sampling-invariance
# actually pays. Spending 20x of the training budget to refine a decision
# that PGF and EMD almost always agree on is the worse half of the trade.
#
# NOTE, and this is easy to miss: PolylineGeomCost median-normalises its cost
# matrix in 'cwot'/'emd' modes but NOT in 'pgf' (polyline_loss.py:621). So
# `weight` is inert there and live here -- switching mode can silently
# rebalance geometry against cls_cost(weight=1.0) in the assigner.
#
# Measured on 600 real CARLA GT polylines from carla_map_infos_test.pkl,
# resampled to 20 pts and normalised as the head normalises them, over 30
# assigner-shaped draws: the median cost-matrix entry at weight=1.0 is
#     emd (median-normalised)  1.0000
#     pgf (raw)                1.0465     -> ratio 0.956
# i.e. PGF's raw magnitude on this data already sits within ~5% of what
# median-normalisation was producing. So `weight` is left at the inherited
# cost_pts_weight and the matching balance carries over unchanged.
#
# That near-unity is a property of THIS dataset's geometry, not a guarantee.
# Re-measure if the tile size, fixed_ptsnum_per_line or the class taxonomy
# changes; a ratio far from 1.0 means correcting `weight` by it, otherwise
# the assigner quietly starts preferring class confidence over geometry (or
# the reverse).
cost_mode = 'pgf'

# --- 2. the direction loss --------------------------------------------------
# EMD treats a polyline as an unordered point SET. It is invariant to any
# permutation of the points, so a prediction that visits the correct 20
# locations in scrambled order scores exactly zero. Nothing else in the HM
# config compensates: it sets loss_dir to 0.0.
#
# That 0.0 mirrors DEAD CODE rather than a considered choice. In the source,
# `pair_loss_mode == 'emd'` takes the first branch of `_loss_after_match`, so
# the branch that would have applied their loss_dir_weight=0.5 is unreachable.
# 0.0 reproduces their effective behaviour, but it is not evidence that
# direction supervision is unhelpful -- they never ran with it on.
#
# The base non-HM config uses 0.005 (PtsL1Loss(5.0) + PtsDirCosLoss(0.005)),
# where index-wise L1 already constrains ordering implicitly and loss_dir is
# only a tiebreaker. With EMD there is no implicit constraint left, so 0.005
# is likely far too weak -- see the suggested sweep below.
#
# PtsDirCosLoss is CosineEmbeddingLoss over consecutive-point difference
# vectors, so each SEGMENT contributes 1-cos in [0, 2]. Two consequences:
#   * it is bounded, so a large weight cannot blow the loss up the way an
#     unbounded geometry term could; and
#   * cosine is scale-free, so unlike loss_pts it is NOT tile-dependent and
#     needs no tile_scale correction. It is stated as a plain constant here
#     deliberately -- do NOT "fix" it by multiplying it by tile_scale. (It is
#     computed on DENORMALISED metre coordinates in loss_single, which is
#     irrelevant for the same reason.)
#
# --- picking the weight: the [0, 2] bound is misleading ---------------------
# The reduction sums over the 19 segments of a polyline and divides by
# num_total_pos (the MATCHED-PAIR count), not by the segment count -- so the
# reported loss_dir is per-polyline, in [0, 38], not per-segment in [0, 2].
# Assuming the latter under-weights it by ~19x.
#
# Measured from a real run of THIS config on the 4103-tile train split
# (RTX 3070, samples_per_gpu=1, ~iteration 900 of epoch 1):
#     loss_pts  ~0.62   (raw ~0.31 at loss_weight 2.0)
#     loss_dir  ~0.55   (raw ~11.0 at loss_weight 0.05)
# so raw loss_dir is ~35x raw loss_pts, and the contribution ratio is
#     loss_dir : loss_pts  ~=  17.7 * loss_dir_weight
#
# A suggested sweep, log-spaced, with the ratio each value buys:
#     0.0     control -- reproduces the HM config exactly. Run this.
#     0.005   ~1:11   the base non-HM config's value. A tiebreaker, which is
#                     all it needs to be there (index-wise PtsL1Loss already
#                     constrains ordering); probably too weak with EMD.
#     0.02    ~1:3    direction as a clear secondary signal.
#     0.05    ~1:1    parity. The default here.
#     0.15    ~3:1    direction-dominant; tests whether ordering is the
#                     binding constraint rather than position.
#     0.5     ~9:1    almost certainly too much -- included to bracket the
#                     top so the curve has a falling side.
# If the budget is 3 runs, take 0.0 / 0.05 / 0.15: 0.005 is close enough to
# 0.0 in effect that it rarely separates from the control.
#
# Two caveats on those ratios. They drift during training -- 1-cos falls as
# the model converges, so a weight tuned at epoch 1 grows relatively weaker
# later. And they are the same for every decoder layer and the one2many
# branch, since both losses are applied identically across all of them, so
# the ratio is what matters and the layer count divides out.
#
# Judge the sweep on mAP AND by eye. The failure EMD permits -- correct point
# positions in scrambled order -- barely moves a chamfer-based mAP, because
# chamfer is itself order-insensitive. It is glaring in
# tools/maptrv2/dataset_viewer.py's overlay, where a scrambled polyline draws
# as a zig-zag through the right corridor. If mAP is flat across the sweep,
# that is a reason to look at the renders, not to conclude the weight does
# nothing.
#
# Exposed as a module-level name so a sweep is a --cfg-options override:
#   --cfg-options model.pts_bbox_head.loss_dir.loss_weight=0.15
loss_dir_weight = 0.05

model = dict(
    pts_bbox_head=dict(
        # Re-enabled. See the sweep note above; 0.05 is a starting point, not
        # a tuned value.
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=loss_dir_weight),
    ),
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                pts_cost=dict(
                    # Only `mode` changes. weight/pgf_w*/pc_range and the
                    # derived cwot_* values all merge in from the HM config,
                    # which keeps them tile-parameterised.
                    mode=cost_mode,
                ),
            ),
        ),
    ),
)
