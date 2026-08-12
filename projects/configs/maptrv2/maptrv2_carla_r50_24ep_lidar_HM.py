_base_ = [
    './maptrv2_carla_r50_24ep_lidar.py',
]
#
# MapTRv2 LiDAR-only (CARLA) with the alternative polyline geometry loss
# adapted from the Pointcept-side PolylineSetLoss -- see
# projects/mmdet3d_plugin/maptr/losses/polyline_loss.py for what was and
# wasn't carried over.
#
# This inherits the entire lidar-only config and overrides exactly two
# things: the point-regression loss (`loss_pts`) and the assigner's
# point-matching cost (`pts_cost`). Everything else -- backbone, BEV path,
# GridSamplePoints, pipelines, optimiser, schedule -- is unchanged, so any
# accuracy difference is attributable to the loss/matching swap alone.
#
# `plugin`/`plugin_dir` must be repeated: mmcv executes each config file in
# isolation before _base_ merging, so the plugin loader in tools/train.py
# reads them from THIS file's namespace, not the base's.
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# --- which geometry to use --------------------------------------------------
# 'pgf'  : position + velocity + acceleration + coverage. Cheap; fully
#          vectorised on the matching side. Closest to the existing L1+dir.
# 'cwot' : debiased Sinkhorn curve-Wasserstein + transport-coupled direction.
#          Permutation/reparameterisation-invariant, no nearest-neighbour
#          collapse -- the source file's flagship. Most expensive.
# 'emd'  : exact per-pair Hungarian 1-Wasserstein over points.
#
# Measured, 1 epoch on the CARLA test subset, single RTX 3070, everything
# else identical so these differ only by mode:
#
#     loss_mode   cost_mode   s/iter    note
#     pgf         pgf         ~0.7      baseline
#     emd         pgf         ~0.6      EMD is ~free as a loss
#     emd         emd         0.6-0.9   was 10-17 before the dedup fix
#     cwot        pgf         10-17     ~20x slower
#     cwot        cwot        --        never completed 10 iters in 52 min
#
# The pattern WAS that the matching cost is the bottleneck: it is evaluated
# for every (query x gt x order) combination, and MapTRv2's one2many branch
# (300 queries x k_one2many=6) makes that ~18,000 pairs per assigner call vs
# ~500 for one2one, each a CPU Hungarian solve.
#
# For 'emd' that is now largely fixed in PolylineGeomCost itself
# (`dedup_gt_slices`, on by default), which costs each DISTINCT (gt, order)
# slice once. Two exact redundancies make that ~19x: 17 of the 19 order
# slots are padding for an open polyline, and EMD -- being a balanced
# assignment over point SETS -- cannot distinguish the 2 real orders
# (forward/flipped are permutations of each other). Measured 9.6x on the
# cost alone and 10-17 -> 0.6-0.9 s/iter end to end, i.e. cost_mode='emd'
# is now roughly as cheap as 'pgf'.
#
# 'cwot' does NOT get the same relief: its direction term reads tangents, so
# it is blind only to exact duplicates (the padding), not to permutations,
# and still solves 2 orders per gt. It remains the expensive option.
#
# All three loss_mode values train stably (loss decreasing, no NaN).
#
# --- Settings mirrored from the Pointcept PTv3 config -----------------------
# (polylinecarla25mv2.py). That config sets:
#     use_emd_cost=True, pair_loss_mode='emd'   -> emd for BOTH loss and cost
#     loss_polyline_weight=2.0                  -> loss_pts weight
#     match_class_weight=1.0                    -> class weight in matching
#     loss_ce_weight=1.0, no_polyline_weight=1.0
#     num_queries=25, num_points_per_polyline=40
#     optimizer AdamW lr=1e-4, weight_decay=0.01
#
# Two of its settings are DEAD CODE in the source and so are not carried:
#   * loss_dir_weight=0.5 -- `pair_loss_mode == 'emd'` takes the first branch
#     of `_loss_after_match`, so the `elif maptrv2_style` branch that would
#     apply the direction loss never runs. Their total is CE + 2.0*EMD only.
#     Mirrored here by setting loss_dir weight to 0.0 (see below).
#   * cost_polyline=5.0 / match_l1_weight -- when `use_emd_cost=True` the
#     matcher uses `cost = cost_emd + match_class_weight*cost_class`, so the
#     L1 geometry weight is unused. Mirrored by pts_cost weight=1.0.
loss_mode = 'emd'
cost_mode = 'emd'

# --- TILE SIZE --------------------------------------------------------------
# The dataset's tile size is not a constant: the 25 m tiles this config was
# tuned against and the 60 m `grid_tiles` export (tile_radius=30) both exist,
# and a future export could use something else again. It matters here because
# both loss_pts and pts_cost operate on NORMALISED coordinates (~[0,1] over
# pc_range), not metres -- `normalize_2d_pts` divides by the pc_range extent,
# so a physical error of d metres reaches the loss as `d / tile_size`.
#
# EMD and PGF are homogeneous of degree 1 in the coordinates they are handed,
# so they simply rescale by 1/tile_size. That has two consequences, both keyed
# off `tile_size` below rather than hardcoded:
#   * absolute quantities (`cwot_eps`) denote a different physical distance
#     on a different tile, so they are rescaled from a reference value; and
#   * the geometry-vs-classification balance shifts, so `loss_weight` is
#     scaled to stay referenced to metres.
#
# CW-OT is the exception and needs more than one common factor, because its
# two terms have DIFFERENT homogeneity degrees: the position term transports
# SQUARED distances (`cwot_squared`, degree 2 -- so it rescales by
# 1/tile_size**2, not 1/tile_size), while the direction term is built from
# unit-normalised tangents and is scale-free (degree 0). A single weight that
# fixes one necessarily breaks the other's balance against it, so the two are
# corrected separately further down.
#
# `point_cloud_range` is restated here because mmcv evaluates each config file
# in isolation -- the same reason `fixed_ptsnum_per_line` is repeated below.
# KEEP IT EQUAL TO THE BASE'S. Changing the tile size means editing the base
# config (maptrv2_carla_r50_24ep_lidar.py): bev_h_/bev_w_, the coder, the
# assigner's own pc_range and the LiDAR geometry are all bound in the base's
# namespace and do NOT follow a change made here. This value only tells the
# loss what geometry it is working in.
point_cloud_range = [-12.5, -12.5, -30.0, 12.5, 12.5, 20.0]

tile_size_x = point_cloud_range[3] - point_cloud_range[0]   # metres
tile_size_y = point_cloud_range[4] - point_cloud_range[1]   # metres
# The longest in-plane extent -- what normalisation maps to 1.0, hence the
# unit every quantity below is implicitly expressed in.
tile_size = max(tile_size_x, tile_size_y)

# The tile size the mirrored Pointcept numbers further down were tuned at.
# `tile_scale` is what carries them to any other tile size.
ref_tile_size = 25.0
tile_scale = tile_size / ref_tile_size

# Keep the loss referenced to METRES rather than to fractions of a tile: with
# it True, a 1 m error costs the same on a 60 m tile as on a 25 m one, and the
# geometry:classification ratio is preserved. Set False to keep the loss in
# tile fractions instead -- i.e. to let a bigger tile tolerate proportionally
# bigger absolute error, which is the behaviour you get from hardcoding 2.0.
scale_loss_weight_with_tile = True
# The power of tile_scale that loss_weight itself contributes. The CW-OT term
# weights below are stated relative to this, so turning the flag off stays
# self-consistent instead of silently double-counting.
loss_weight_tile_power = 1 if scale_loss_weight_with_tile else 0
loss_pts_weight = 2.0 * tile_scale ** loss_weight_tile_power

# Same treatment for the matching cost's weight. Note this is inert in the
# 'cwot'/'emd' modes, whose costs are median-normalised inside
# PolylineGeomCost -- that divides any overall scale (the tile's included)
# straight back out. Kept uniform anyway so the two sides stay in step and
# the value is still right if `normalize_median` is ever turned off.
cost_pts_weight = 1.0 * tile_scale

# CW-OT's two terms, per the degree argument above. `loss_weight`/`weight`
# already supply some power of tile_scale, so each term is only topped up
# from there to its own degree: position to 2 (or 1 if cwot_squared is turned
# off), direction back down to 0 (it is scale-free). Both reduce to the
# original 1.0 / 0.5 at the reference tile, whatever the flags say.
# `cwot_squared` is stated explicitly rather than left to the class defaults,
# because these expressions are only correct if the two agree -- and the two
# classes ship DIFFERENT defaults (loss True, cost False).
loss_cwot_squared = True
cost_cwot_squared = False
loss_cwot_degree = 2 if loss_cwot_squared else 1
cost_cwot_degree = 2 if cost_cwot_squared else 1
loss_cwot_w_pos = 1.0 * tile_scale ** (loss_cwot_degree - loss_weight_tile_power)
loss_cwot_w_dir = 0.5 * tile_scale ** (-loss_weight_tile_power)
# cost_pts_weight always contributes tile_scale**1.
cost_cwot_w_pos = 1.0 * tile_scale ** (cost_cwot_degree - 1)
cost_cwot_w_dir = 0.5 / tile_scale

# The Sinkhorn blur radius needs the same degree treatment, and it is easy to
# get wrong: `cwot_eps` regularises the entropic solver against the TRANSPORT
# COST (`log_K = -cost / eps`), so it carries that cost's units -- squared
# distance while cwot_squared is on, plain distance otherwise. Dividing it by
# tile_size as if it were always a length would leave `cost / eps` -- the only
# thing Sinkhorn actually sees -- drifting with tile size, silently changing
# how sharp the coupling is on a geometry that did not change.
#
# So it is scaled by tile_scale**degree from the value tuned at the reference
# tile, rather than expressed in metres: 0.05 is not a metre-calibrated number
# to begin with (at degree 2 it corresponds to a blur of sqrt(0.05)*25 ~ 5.6 m,
# not 1.25 m). Both sides reduce to exactly 0.05 at the reference tile.
cwot_eps_ref = 0.05          # normalised units, tuned at ref_tile_size
loss_cwot_eps = cwot_eps_ref / tile_scale ** loss_cwot_degree
cost_cwot_eps = cwot_eps_ref / tile_scale ** cost_cwot_degree

# NOT tile-dependent, and deliberately left as plain constants:
#   * pgf_w0/w1/w2/wc -- all four PGF terms are degree-1 homogeneous, so a
#     tile change rescales them identically and their *ratios* are invariant.
#     `loss_pts_weight` already carries the common factor.
#   * cwot_arc_marginals -- the marginals are normalised to sum to 1, so they
#     are scale-free whether they are arc-length or uniform.
#   * cwot_iters, num_pts_per_vec, num_vec_* -- counts, not distances.
#
# `pc_range` is also passed to both classes below. That is a *separate*
# tile concern from the scalars above: normalisation divides x and y by their
# own extents independently, so a NON-SQUARE pc_range stretches the space
# every cdist here measures in, over-weighting error along the shorter axis.
# No config scalar can express that, so the classes undo it themselves. It is
# an exact no-op for CARLA's square tiles -- it only starts mattering if a
# future export is rectangular.
# --- geometry / query settings from the Pointcept config --------------------
# num_queries=25 there ("CARLA: max 20 GT polylines/tile (mean 3.9) -> fewer
# FP slots => higher precision"). MapTRv2's one2one branch is the equivalent
# slot count.
num_vec_one2one = 25
# Pointcept has no one2many branch, so there is nothing to mirror this
# against -- but it must still be scaled DOWN alongside one2one to keep the
# base config's convention (base: 50 one2one x k_one2many=6 = 300).
num_vec_one2many = num_vec_one2one * 6

# NOT mirrored: the Pointcept config uses num_points_per_polyline=40; this
# stays at MapTRv2's inherited 20. Deliberate -- 40 is expensive here in a
# way it is not there, for two compounding reasons:
#   * decoder query tokens are num_vec x num_pts_per_vec and self-attention
#     is O(n^2), and
#   * MapTR's `num_orders` axis itself scales with points-per-line
#     (shift_num = pts_num - 1), so 20 -> 40 doubles BOTH the order count
#     and the per-pair point count, growing the assigner's cost tensor ~8x.
# Pointcept pays neither: 25 queries, no one2many arm, no order axis. Both
# effects OOM'd an 8 GB card here. 20 also keeps this config directly
# comparable with the non-HM baseline, which is the point of the file.
fixed_ptsnum_per_gt_line = 20
fixed_ptsnum_per_pred_line = 20

# NOTE on the Pointcept comparison, which normalises differently again:
# coord_scale=12.5 maps a 25 m tile to [-1, 1], i.e. 1 unit = 12.5 m, whereas
# MapTRv2 maps it to [0, 1], i.e. 1 unit = tile_size. So at the reference tile
# an identical physical error yields a HALF-SIZED number here. `loss_weight`
# takes their nominal 2.0 (via `loss_pts_weight` above); if you want to match
# their effective gradient magnitude instead, double the 2.0 there to 4.0 --
# `tile_scale` is orthogonal to that choice and applies either way.
model = dict(
    pts_bbox_head=dict(
        num_vec_one2one=num_vec_one2one,
        num_vec_one2many=num_vec_one2many,
        num_pts_per_vec=fixed_ptsnum_per_pred_line,
        num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
        loss_pts=dict(
            _delete_=True,
            type='PolylineGeomLoss',
            mode=loss_mode,
            # = loss_polyline_weight (2.0), carried to this tile size
            loss_weight=loss_pts_weight,
            # Tile geometry: undoes per-axis normalisation anisotropy.
            # No-op while the tile is square.
            pc_range=point_cloud_range,
            # PGF term weights (unused while mode='emd'). Tile-invariant
            # ratios -- the common scale factor is in loss_weight.
            pgf_w0=0.2,
            pgf_w1=1.0,
            pgf_w2=0.5,
            pgf_wc=0.5,
            # CW-OT knobs (unused while mode='emd'). All three tile-dependent
            # values are derived above -- do not hardcode them back.
            cwot_eps=loss_cwot_eps,
            cwot_iters=50,
            cwot_squared=loss_cwot_squared,
            cwot_w_pos=loss_cwot_w_pos,
            cwot_w_dir=loss_cwot_w_dir,
        ),
        # 0.0 to mirror the source: with pair_loss_mode='emd' its direction
        # term is unreachable, so their geometry supervision is EMD alone.
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.0),
        # = loss_ce_weight (1.0). MUST equal assigner.cls_cost.weight below:
        # mmdet's DETRHead.__init__ asserts they are identical ("The
        # classification weight for loss and matcher should be exactly the
        # same"). Convenient here -- the source sets loss_ce_weight and
        # match_class_weight both to 1.0, so one value satisfies both.
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        transformer=dict(
            decoder=dict(
                transformerlayers=dict(
                    num_vec=num_vec_one2one,
                    num_pts_per_vec=fixed_ptsnum_per_pred_line))),
    ),
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                # match_class_weight=1.0 there; the class term is the only
                # other cost with a non-zero weight once geometry is EMD.
                cls_cost=dict(type='FocalLossCost', weight=1.0),
                pts_cost=dict(
                    _delete_=True,
                    type='PolylineGeomCost',
                    mode=cost_mode,
                    # 1.0, not 5.0: with use_emd_cost=True the source adds the
                    # median-normalised EMD cost unweighted (cost_polyline is
                    # dead), so the geometry:class ratio there is 1:1.
                    # Tile-scaled only in 'pgf' mode -- see cost_pts_weight.
                    weight=cost_pts_weight,
                    # Must match loss_pts's, or matching and supervision
                    # would measure distance in two different geometries.
                    pc_range=point_cloud_range,
                    pgf_w0=0.2,
                    pgf_w1=1.0,
                    pgf_w2=0.5,
                    pgf_wc=0.5,
                    cwot_eps=cost_cwot_eps,
                    cwot_iters=25,
                    cwot_squared=cost_cwot_squared,
                    cwot_w_pos=cost_cwot_w_pos,
                    cwot_w_dir=cost_cwot_w_dir,
                ),
            ),
        ),
    ),
)

# --- data / optimiser settings from the Pointcept config --------------------
# batch_size=4 (total), num_worker=8.
# fixed_ptsnum_per_line MUST be repeated per split: mmcv evaluates each config
# file in isolation, so the base's `data.train.fixed_ptsnum_per_line` is
# already bound to the base's 20 and does NOT track the module-level
# `fixed_ptsnum_per_gt_line = 40` set above. Leaving it would feed the head
# 20-point GT while it expects 40 -- a shape mismatch in loss_single.
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line),
    val=dict(fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line),
    test=dict(fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line),
)
# AdamW lr=1e-4, weight_decay=0.01 (inherited base uses lr=6e-4)
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)

# NOT mirrored, and why:
#   * epoch=200 / OneCycleLR -- this file stays at the inherited 24 epochs and
#     CosineAnnealing, both to match its name and to keep it comparable with
#     the non-HM config. Change total_epochs/lr_config if you want their
#     schedule too.
#   * aux_layer_weight=0.0 (their aux decoder losses OFF, "made it ~3x
#     slower") -- NOT expressible here. MapTRv2Head.loss() unconditionally
#     computes losses on every decoder layer and sums them; there is no
#     weight knob. Disabling it would mean editing the head, which is beyond
#     "only where absolutely necessary". Expect this config to be slower per
#     iteration than their setup for that reason.
#   * no_polyline_weight=1.0 / loss_ce_weight=1.0 -- their CE uses an explicit
#     background-class weight; MapTRv2 uses FocalLoss, which handles the
#     foreground/background imbalance by a different mechanism. loss_cls is
#     left at its inherited FocalLoss(loss_weight=2.0).
