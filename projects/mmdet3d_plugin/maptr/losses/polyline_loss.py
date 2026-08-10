"""Alternative polyline geometry losses, adapted from a Pointcept-side
`polyline_loss.py` (`PolylineSetLoss`).

--- What was adapted, and why only this much ---

The source is a *complete* set-prediction loss: it owns a Hungarian matcher,
a cross-entropy classification term, and aux-decoder-layer handling, and is
called as `forward_batch(pred_polylines, pred_logits, gt_polylines,
gt_types, gt_offset, ...)`.

MapTRv2 already does every one of those itself:
  * matching            -> `MapTRAssigner` (projects/.../assigners/)
  * classification      -> `loss_cls` (FocalLoss), built in `MapTRv2Head`
  * aux decoder layers  -> `MapTRv2Head.loss()` loops decoder outputs
  * one2many branch     -> also `MapTRv2Head.loss()`

So porting `PolylineSetLoss` wholesale would mean replacing MapTRv2's entire
head-loss pipeline. Instead only the *geometry primitives* are carried over
(verbatim where possible), exposed through the two interfaces MapTR already
has slots for:

  `PolylineGeomLoss`  -- a `loss_pts` drop-in.  Signature matches
                         `PtsL1Loss`: (pred, target, weight, avg_factor).
                         Operates on ALREADY-MATCHED, ALREADY-NORMALISED
                         point sets, so no matching happens here.
  `PolylineGeomCost`  -- a `pts_cost` drop-in for `MapTRAssigner`.
                         Signature matches `OrderedPtsL1Cost`:
                         (pts_pred[Q,P,C], gt_pts[G,O,P,C]) -> [Q, G*O].

Both take `mode` in {'cwot', 'pgf', 'emd'} selecting which primitive to use.

--- Two source behaviours deliberately dropped ---

1. Flip alignment (`align_gt_flip`, `flip_aware_poly_cost`). MapTR encodes
   polyline direction/permutation equivalence *in the data*: GT arrives as
   `(num_gts, num_orders, num_pts, coords)` and `MapTRAssigner` takes
   `min` over the order axis, then the loss is given the winning order
   only. Re-applying a flip search here would double up on that and could
   pick a different order than the one matching was resolved against.
2. The CE / no-object weighting. That is `loss_cls`'s job in this repo.

--- Coordinate spaces (easy to get wrong) ---

`loss_pts` is called on *normalised* points (roughly [0,1] over
`pc_range`), not metres -- see `MapTRv2Head.loss_single`, which normalises
targets and only denormalises separately for `loss_dir`. `pts_cost` is
likewise called on normalised coordinates. So the `eps` of the Sinkhorn
solver and any distance-like quantity here are in *normalised* units; an
`eps` sensible in metres would be far too large. Defaults below are set for
normalised space.

--- Tile size (dataset geometry) ---

That normalisation is what couples these losses to the dataset's tile size,
which is NOT a constant for CARLA: the 25 m tiles and the 60 m `grid_tiles`
export both exist. Two consequences, handled in two different places:

1. *Scale.* `normalize_2d_pts` divides by the pc_range extent, so a fixed
   physical error of d metres arrives here as `d / tile_size`. That
   silently changes the geometry:classification balance, and makes any
   absolute quantity (`cwot_eps`) mean a different number of metres.
   Handled **in the config**, by deriving `loss_weight` and `cwot_eps`
   from a `tile_size` variable rather than hardcoding them; see
   `maptrv2_carla_r50_24ep_lidar_HM.py`.

   Mind the homogeneity degrees when doing that, because they are not all
   the same: `exact_emd_loss` and every `pgf_*` term are degree 1, but
   `curve_ot_loss` transports SQUARED distances by default
   (`squared=True`), making its position term degree 2, while its
   direction term is built from unit tangents and is degree 0 (scale-free).
   So CW-OT cannot be carried across tile sizes by one common factor --
   `cwot_w_pos` and `cwot_w_dir` need correcting relative to each other.

2. *Anisotropy.* `normalize_2d_pts` divides x and y by their own extents
   independently. On a non-square pc_range (nuScenes' 30x60 m, say) the
   normalised space is therefore *stretched*, and every Euclidean distance
   taken here -- `torch.cdist` in EMD, CW-OT and PGF's coverage term -- is
   measured in that stretched space, over-weighting error along the shorter
   axis. No config scalar can fix this, so it is handled here: pass
   `pc_range` to either class and coordinates are rescaled back to
   metre-proportional units before any geometry is computed. It is an exact
   no-op for a square pc_range (CARLA's), so it changes nothing measured so
   far; it only matters if the tile stops being square.
"""
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from mmdet.core.bbox.match_costs.builder import MATCH_COST
from mmdet.models.builder import LOSSES

# ---------------------------------------------------------------------------
# CW-OT (Curve-Wasserstein) primitives -- carried over verbatim from the
# source file apart from being dimension-agnostic (works for 2D or 3D pts).
# ---------------------------------------------------------------------------


def cwot_arc_length_marginal(p, eps=1e-6):
    """Point mass proportional to adjacent segment length (normalized to 1)."""
    n = p.shape[0]
    if n == 1:
        return p.new_ones(1)
    seg = (p[1:] - p[:-1]).norm(dim=-1)
    mass = p.new_zeros(n)
    mass[:-1] = mass[:-1] + 0.5 * seg
    mass[1:] = mass[1:] + 0.5 * seg
    total = mass.sum()
    if float(total.detach()) <= eps:
        return p.new_full((n,), 1.0 / n)
    return mass / total


def cwot_sinkhorn_plan(cost, a, b, eps, iters=50):
    """Balanced entropic OT coupling via log-domain Sinkhorn (differentiable)."""
    eps = max(eps, 1e-6)
    log_a = torch.log(a + 1e-30)
    log_b = torch.log(b + 1e-30)
    log_K = -cost / eps
    f = torch.zeros_like(log_a)
    g = torch.zeros_like(log_b)
    for _ in range(iters):
        f = log_a - torch.logsumexp(log_K + g[None, :], dim=1)
        g = log_b - torch.logsumexp(log_K + f[:, None], dim=0)
    T = torch.exp(log_K + f[:, None] + g[None, :])
    return torch.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)


def _cwot_ot_cost(p, g, a, b, eps, iters, squared):
    d = torch.cdist(p, g, p=2)
    cost = d * d if squared else d
    T = cwot_sinkhorn_plan(cost, a, b, eps, iters)
    return (T * cost).sum()


def _cwot_segment_coupling(T):
    if T.shape[0] < 2 or T.shape[1] < 2:
        return T.new_zeros((max(T.shape[0] - 1, 0), max(T.shape[1] - 1, 0)))
    row_avg = 0.5 * (T[:-1, :] + T[1:, :])
    return 0.5 * (row_avg[:, :-1] + row_avg[:, 1:])


def _cwot_direction_loss(p, g, a, b, eps, iters):
    if p.shape[0] < 2 or g.shape[0] < 2:
        return p.sum() * 0.0
    d = torch.cdist(p, g, p=2)
    T = cwot_sinkhorn_plan(d * d, a, b, eps, iters)
    T_seg = _cwot_segment_coupling(T)
    tp = F.normalize(p[1:] - p[:-1], dim=-1, eps=1e-6)
    tg = F.normalize(g[1:] - g[:-1], dim=-1, eps=1e-6)
    cos = (tp @ tg.t()).clamp(-1.0, 1.0)
    mass = T_seg.sum().clamp(min=1e-8)
    return (T_seg * (1.0 - cos)).sum() / mass


def curve_ot_loss(p, g, *, eps=0.05, iters=50, w_pos=1.0, w_dir=0.5,
                   debiased=True, arc_marginals=True, squared=True):
    """Curve-Wasserstein: debiased Sinkhorn divergence + coupled direction."""
    if p.numel() == 0 or g.numel() == 0:
        return p.sum() * 0.0
    if arc_marginals:
        a = cwot_arc_length_marginal(p)
        b = cwot_arc_length_marginal(g)
    else:
        a = p.new_full((p.shape[0],), 1.0 / p.shape[0])
        b = g.new_full((g.shape[0],), 1.0 / g.shape[0])
    ot_pg = _cwot_ot_cost(p, g, a, b, eps, iters, squared)
    if debiased:
        ot_pp = _cwot_ot_cost(p, p, a, a, eps, iters, squared)
        ot_gg = _cwot_ot_cost(g, g, b, b, eps, iters, squared)
        pos = (ot_pg - 0.5 * ot_pp - 0.5 * ot_gg).clamp(min=0.0)
    else:
        pos = ot_pg.clamp(min=0.0)
    total = w_pos * pos
    if w_dir > 0.0:
        total = total + w_dir * _cwot_direction_loss(p, g, a, b, eps, iters)
    return torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# PGF (Polyline Geometric Fidelity) primitives -- verbatim from the source.
#   L_0  position, L_1 velocity, L_2 acceleration, L_cov coverage
# ---------------------------------------------------------------------------


def asym_chamfer_gt_to_pred(pred, gt):
    """Mean over GT points of distance to nearest predicted point."""
    if pred.numel() == 0 or gt.numel() == 0:
        return pred.sum() * 0.0
    d = torch.cdist(pred, gt, p=2)
    return d.min(dim=0).values.mean()


def pgf_position_loss(pred, gt):
    """L_0: per-index L1 on raw point coordinates."""
    return F.l1_loss(pred, gt)


def pgf_velocity_loss(pred, gt):
    """L_1: per-index L1 on tangent vectors; subsumes direction + seg length."""
    if pred.shape[0] < 2:
        return pred.sum() * 0.0
    return F.l1_loss(pred[1:] - pred[:-1], gt[1:] - gt[:-1])


def pgf_acceleration_loss(pred, gt):
    """L_2: per-index L1 on second differences (anti-zigzag)."""
    if pred.shape[0] < 3:
        return pred.sum() * 0.0
    ap = pred[2:] - 2.0 * pred[1:-1] + pred[:-2]
    ag = gt[2:] - 2.0 * gt[1:-1] + gt[:-2]
    return F.l1_loss(ap, ag)


def pgf_coverage_loss(pred, gt):
    """L_cov: asym Chamfer (GT->Pred) + |total length difference|."""
    if pred.numel() == 0 or gt.numel() == 0:
        return pred.sum() * 0.0
    cov = asym_chamfer_gt_to_pred(pred, gt)
    if pred.shape[0] >= 2 and gt.shape[0] >= 2:
        len_p = (pred[1:] - pred[:-1]).norm(dim=-1).sum()
        len_g = (gt[1:] - gt[:-1]).norm(dim=-1).sum()
        len_diff = (len_p - len_g).abs()
    else:
        len_diff = pred.sum() * 0.0
    return cov + len_diff


def pgf_loss(pred, gt, *, w0=0.2, w1=1.0, w2=0.5, wc=0.5):
    """Weighted sum of the four PGF terms for one matched pair."""
    total = pred.sum() * 0.0
    if w0 > 0:
        total = total + w0 * pgf_position_loss(pred, gt)
    if w1 > 0:
        total = total + w1 * pgf_velocity_loss(pred, gt)
    if w2 > 0:
        total = total + w2 * pgf_acceleration_loss(pred, gt)
    if wc > 0:
        total = total + wc * pgf_coverage_loss(pred, gt)
    return torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Exact EMD primitive
# ---------------------------------------------------------------------------


def exact_emd_loss(pred, gt):
    """Exact 1-Wasserstein (mean matched Euclidean displacement) for one pair.

    Hungarian assignment is solved with no-grad on the point distance matrix;
    gradients flow through the selected distances into ``pred``.
    """
    if pred.numel() == 0 or gt.numel() == 0:
        return pred.sum() * 0.0
    if pred.shape[0] != gt.shape[0]:
        raise ValueError(
            f'exact_emd_loss requires equal point counts, got '
            f'{pred.shape[0]} vs {gt.shape[0]}')
    cost = torch.cdist(pred, gt, p=2)
    with torch.no_grad():
        ri, ci = linear_sum_assignment(cost.detach().cpu().numpy())
    ri_t = torch.as_tensor(ri, device=pred.device, dtype=torch.long)
    ci_t = torch.as_tensor(ci, device=pred.device, dtype=torch.long)
    return torch.nan_to_num(cost[ri_t, ci_t].mean(),
                             nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Tile-size handling -- see the "Tile size" section of the module docstring
# ---------------------------------------------------------------------------


def tile_size_from_pc_range(pc_range):
    """Tile size in metres: the larger of the two in-plane pc_range extents.

    This is the unit that `normalize_2d_pts` maps to 1.0 along the longest
    axis, i.e. the scale that any absolute quantity in this file (notably
    `cwot_eps`) is implicitly expressed in.
    """
    return max(pc_range[3] - pc_range[0], pc_range[4] - pc_range[1])


def axis_scale_from_pc_range(pc_range):
    """Per-axis multipliers restoring isotropy to normalised coordinates.

    `normalize_2d_pts`/`normalize_3d_pts` divide each axis by its *own*
    pc_range extent, so one normalised unit means a different number of
    metres per axis whenever those extents differ. Multiplying by
    `extent_axis / tile_size` undoes exactly that, leaving coordinates
    proportional to metres with the longest in-plane axis still spanning
    [0, 1] -- so a square pc_range yields all-ones and this is a no-op,
    while distances on a non-square one stop being skewed.

    Returns None when no rescale is needed, so the caller can skip the work.
    """
    if pc_range is None:
        return None
    if len(pc_range) != 6:
        raise ValueError(
            f'pc_range must have 6 elements [x0,y0,z0,x1,y1,z1], got '
            f'{len(pc_range)}')
    tile = tile_size_from_pc_range(pc_range)
    if tile <= 0:
        raise ValueError(f'pc_range has non-positive extent: {pc_range}')
    scale = [(pc_range[3] - pc_range[0]) / tile,
             (pc_range[4] - pc_range[1]) / tile,
             (pc_range[5] - pc_range[2]) / tile]
    if all(abs(s - 1.0) < 1e-9 for s in scale):
        return None
    return scale


def _apply_axis_scale(pts, scale):
    """Multiply the last (coordinate) axis of `pts` by `scale`.

    Skipped entirely when the relevant axes are already unit, so the common
    square-tile 2D case costs nothing. The z entry only ever applies when
    points carry a third coordinate (`z_cfg.gt_z_flag`), which no CARLA
    config currently sets.
    """
    if scale is None:
        return pts
    used = scale[:pts.shape[-1]]
    if all(abs(s - 1.0) < 1e-9 for s in used):
        return pts
    return pts * pts.new_tensor(used)


# ---------------------------------------------------------------------------
# mmdet-registered wrappers
# ---------------------------------------------------------------------------

_MODES = ('cwot', 'pgf', 'emd')


@LOSSES.register_module()
class PolylineGeomLoss(nn.Module):
    """Drop-in replacement for MapTRv2's ``loss_pts``.

    Args:
        mode (str): one of 'cwot', 'pgf', 'emd'.
        loss_weight (float): scales the final loss, as in ``PtsL1Loss``.
        cwot_* : CW-OT knobs (see :func:`curve_ot_loss`). ``eps`` is in
            NORMALISED units, not metres, and it regularises against the
            transport cost -- so it carries that cost's units: SQUARED
            normalised distance while ``cwot_squared`` is set, plain
            normalised distance otherwise. Derive it from the config's
            ``tile_size`` (at the matching degree) rather than hardcoding,
            or the coupling changes on a differently-sized tile.
        pgf_w0/w1/w2/wc : PGF term weights (see :func:`pgf_loss`).
        pc_range (list[float] | None): the head's ``pc_range``. Optional,
            and only used to undo the anisotropy that per-axis normalisation
            introduces on a non-square tile -- see the module docstring.
            A no-op for a square pc_range; ``None`` disables it entirely.

    Inputs follow the ``PtsL1Loss`` contract exactly:
        pred   (N, P, C) normalised, already matched to target
        target (N, P, C) normalised, already in the assigner-chosen order
        weight (N, P, C) per-point weights; used here only to drop rows whose
               weight is entirely zero (the padded/unmatched entries)
        avg_factor (int) usually ``num_total_pos``
    """

    def __init__(self,
                 mode='pgf',
                 loss_weight=1.0,
                 cwot_eps=0.05,
                 cwot_iters=50,
                 cwot_w_pos=1.0,
                 cwot_w_dir=0.5,
                 cwot_debiased=True,
                 cwot_arc_marginals=True,
                 cwot_squared=True,
                 pgf_w0=0.2,
                 pgf_w1=1.0,
                 pgf_w2=0.5,
                 pgf_wc=0.5,
                 pc_range=None):
        super().__init__()
        if mode not in _MODES:
            raise ValueError(f'mode must be one of {_MODES}, got {mode!r}')
        self.mode = mode
        self.loss_weight = loss_weight
        self.cwot_kwargs = dict(eps=cwot_eps, iters=cwot_iters,
                                 w_pos=cwot_w_pos, w_dir=cwot_w_dir,
                                 debiased=cwot_debiased,
                                 arc_marginals=cwot_arc_marginals,
                                 squared=cwot_squared)
        self.pgf_kwargs = dict(w0=pgf_w0, w1=pgf_w1, w2=pgf_w2, wc=pgf_wc)
        self.pc_range = pc_range
        self.tile_size = (None if pc_range is None
                          else tile_size_from_pc_range(pc_range))
        self.axis_scale = axis_scale_from_pc_range(pc_range)

    def _pair_loss(self, p, g):
        if self.mode == 'cwot':
            return curve_ot_loss(p, g, **self.cwot_kwargs)
        if self.mode == 'pgf':
            return pgf_loss(p, g, **self.pgf_kwargs)
        return exact_emd_loss(p, g)

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        if pred.numel() == 0:
            return pred.sum() * self.loss_weight

        # Isotropy restore (no-op on a square tile). A constant per-axis
        # multiplier, so gradients still flow to `pred` unchanged up to that
        # constant.
        pred = _apply_axis_scale(pred, self.axis_scale)
        target = _apply_axis_scale(target, self.axis_scale)

        if weight is not None:
            # weight is (N, P, C); a fully-zero row is a padded/unmatched
            # entry. Reduce to a per-pair validity mask rather than trying to
            # apply per-coordinate weights -- these losses are scalar per
            # pair and have no meaningful per-coordinate decomposition.
            valid = weight.reshape(weight.shape[0], -1).abs().sum(-1) > 0
        else:
            valid = torch.ones(pred.shape[0], dtype=torch.bool,
                                device=pred.device)

        total = pred.sum() * 0.0
        n = 0
        for i in range(pred.shape[0]):
            if not bool(valid[i]):
                continue
            total = total + self._pair_loss(pred[i], target[i])
            n += 1

        if avg_factor is None:
            avg_factor = max(n, 1)
        loss = total / max(avg_factor, 1)
        return self.loss_weight * torch.nan_to_num(
            loss, nan=0.0, posinf=0.0, neginf=0.0)


@MATCH_COST.register_module()
class PolylineGeomCost(object):
    """Drop-in replacement for ``OrderedPtsL1Cost`` in ``MapTRAssigner``.

    Matches that class's contract exactly:
        pts_pred  (num_query, num_pts, num_coords)   normalised
        gt_pts    (num_gts, num_orders, num_pts, C)  normalised
        returns   (num_query, num_gts * num_orders)
    The assigner reshapes to (Q, G, O) and takes ``min`` over orders, which
    is how polyline direction equivalence is handled -- so this cost must NOT
    do its own flip search.

    PERFORMANCE -- read before setting mode to 'cwot' or 'emd' here.
    Unlike the loss (which only touches *matched* pairs, a handful per
    sample), this cost is evaluated for every (query x gt x order)
    combination. MapTRv2's one2many branch dominates that count:
    num_vec_one2many=300 queries against k_one2many=6 replicated GT means
    ~18,000 pairs per assigner call versus ~500 for one2one -- 36x.

    Measured on the CARLA lidar-only config, 1 epoch, single RTX 3070
    (identical settings otherwise, so these differ only by mode):

        loss=pgf   cost=pgf    ~0.7 s/iter    baseline
        loss=emd   cost=pgf    ~0.6 s/iter    EMD is ~free as a loss
        loss=emd   cost=emd    10-17 s/iter   ~20x slower
        loss=cwot  cost=pgf    10-17 s/iter   ~20x slower
        loss=cwot  cost=cwot   did not complete 10 iters in 52 min

    'cwot' as a cost works out to ~11M tiny Sinkhorn iterations per sample
    in a Python double loop -- kernel-launch bound, not compute bound, and
    effectively unusable. Keep mode='pgf' here (fully vectorised) and put
    the expensive geometry in the loss instead. A batched Sinkhorn over all
    pairs would be the fix if cost-side CW-OT is ever actually needed.

    TILE SIZE. `pc_range` has the same meaning and effect as in
    `PolylineGeomLoss` (see the module docstring) and should be given the
    same value -- otherwise matching and supervision measure distance in
    two different geometries. `weight` is tile-dependent in mode='pgf'
    only: 'cwot'/'emd' costs are median-normalised below, which removes
    their scale (and hence the tile's) entirely.
    """

    def __init__(self,
                 mode='pgf',
                 weight=1.0,
                 cwot_eps=0.05,
                 cwot_iters=25,
                 cwot_w_pos=1.0,
                 cwot_w_dir=0.5,
                 cwot_debiased=True,
                 cwot_arc_marginals=False,
                 cwot_squared=False,
                 pgf_w0=0.2,
                 pgf_w1=1.0,
                 pgf_w2=0.5,
                 pgf_wc=0.5,
                 normalize_median=True,
                 max_cost_elems=64_000_000,
                 pc_range=None):
        if mode not in _MODES:
            raise ValueError(f'mode must be one of {_MODES}, got {mode!r}')
        if mode in ('cwot', 'emd'):
            # Loud on purpose: measured ~20x slower ('emd') to effectively
            # hung ('cwot') on this repo's one2many query count. Easy to
            # mistake for a stall rather than a config choice.
            warnings.warn(
                f"PolylineGeomCost(mode='{mode}') evaluates a per-pair solve "
                f'for every query x gt x order combination (~18k pairs per '
                f'assigner call with MapTRv2 one2many). Measured ~20x slower '
                f"than mode='pgf' for 'emd'; 'cwot' did not complete 10 "
                f'iterations in 52 minutes. Prefer mode=\'pgf\' for the '
                f'matching cost and put expensive geometry in loss_pts.',
                RuntimeWarning)
        self.mode = mode
        self.weight = weight
        self.cwot_kwargs = dict(eps=cwot_eps, iters=cwot_iters,
                                 w_pos=cwot_w_pos, w_dir=cwot_w_dir,
                                 debiased=cwot_debiased,
                                 arc_marginals=cwot_arc_marginals,
                                 squared=cwot_squared)
        self.pgf_kwargs = dict(w0=pgf_w0, w1=pgf_w1, w2=pgf_w2, wc=pgf_wc)
        # The source normalises cwot/emd costs by their median before adding
        # the class term, so the geometry term's scale doesn't swamp it.
        # Kept, since MapTR likewise sums cls/reg/iou/pts costs unweighted
        # beyond their configured weights.
        self.normalize_median = normalize_median
        # Peak-element budget for the chunked 'emd' cdist (64M float32 ~
        # 256 MB). Lower it if the assigner still OOMs on a small card.
        self.max_cost_elems = max_cost_elems
        # Tile geometry -- see the module docstring. Only the anisotropy
        # correction lives here; the *scale* half is the config's job, via
        # `weight` and `cwot_eps`.
        self.pc_range = pc_range
        self.tile_size = (None if pc_range is None
                          else tile_size_from_pc_range(pc_range))
        self.axis_scale = axis_scale_from_pc_range(pc_range)

    def _pgf_cost_vectorised(self, pts_pred, gt_flat):
        """PGF cost for all (query, gt*order) pairs without Python loops."""
        Q, P, C = pts_pred.shape
        M = gt_flat.shape[0]
        p = pts_pred[:, None]                      # (Q, 1, P, C)
        g = gt_flat[None]                          # (1, M, P, C)
        cost = pts_pred.new_zeros((Q, M))
        w = self.pgf_kwargs
        if w['w0'] > 0:
            cost = cost + w['w0'] * (p - g).abs().mean(dim=(-1, -2))
        if w['w1'] > 0 and P >= 2:
            dp = p[..., 1:, :] - p[..., :-1, :]
            dg = g[..., 1:, :] - g[..., :-1, :]
            cost = cost + w['w1'] * (dp - dg).abs().mean(dim=(-1, -2))
        if w['w2'] > 0 and P >= 3:
            ap = p[..., 2:, :] - 2.0 * p[..., 1:-1, :] + p[..., :-2, :]
            ag = g[..., 2:, :] - 2.0 * g[..., 1:-1, :] + g[..., :-2, :]
            cost = cost + w['w2'] * (ap - ag).abs().mean(dim=(-1, -2))
        if w['wc'] > 0:
            d = torch.cdist(pts_pred.reshape(Q * P, C),
                             gt_flat.reshape(M * P, C), p=2)
            d = d.reshape(Q, P, M, P).permute(0, 2, 1, 3)   # (Q, M, P, P)
            cov = d.min(dim=2).values.mean(dim=-1)          # GT->pred
            if P >= 2:
                len_p = (pts_pred[:, 1:] - pts_pred[:, :-1]).norm(
                    dim=-1).sum(-1)
                len_g = (gt_flat[:, 1:] - gt_flat[:, :-1]).norm(
                    dim=-1).sum(-1)
                cov = cov + (len_p[:, None] - len_g[None, :]).abs()
            cost = cost + w['wc'] * cov
        return cost

    @torch.no_grad()
    def __call__(self, pts_pred, gt_pts):
        num_gts, num_orders, num_pts, num_coords = gt_pts.shape
        gt_flat = gt_pts.reshape(num_gts * num_orders, num_pts, num_coords)
        Q = pts_pred.shape[0]
        M = gt_flat.shape[0]
        if Q == 0 or M == 0:
            return pts_pred.new_zeros((Q, M))

        # Isotropy restore (no-op on a square tile), applied before any
        # distance is taken. Must match the loss's, or matching and
        # supervision would disagree about what "close" means.
        pts_pred = _apply_axis_scale(pts_pred, self.axis_scale)
        gt_flat = _apply_axis_scale(gt_flat, self.axis_scale)

        if self.mode == 'pgf':
            cost = self._pgf_cost_vectorised(pts_pred, gt_flat)
        elif self.mode == 'emd':
            # Batched cdist -> single GPU->CPU transfer -> exact per-pair
            # Hungarian on CPU, as in the source. CHUNKED over queries: the
            # full (Q, M, P, P) tensor is far larger here than in the source,
            # because M carries MapTR's order axis and `num_orders` itself
            # scales with points-per-line (shift_num = pts_num - 1). Going
            # 20 -> 40 pts/line therefore grows this ~8x (P^2 x O), and the
            # unchunked version asked for 2.3 GiB in one allocation and OOM'd
            # an 8 GB card. Chunking is numerically identical -- same cdist,
            # same assignments -- just bounded in peak memory.
            elems_per_query = M * num_pts * num_pts
            chunk = max(1, int(self.max_cost_elems // max(elems_per_query, 1)))
            cost_np = np.zeros((Q, M), dtype=np.float32)
            for lo in range(0, Q, chunk):
                hi = min(lo + chunk, Q)
                q = hi - lo
                d_np = torch.cdist(
                    pts_pred[lo:hi, None].expand(q, M, num_pts, num_coords),
                    gt_flat[None].expand(q, M, num_pts, num_coords),
                    p=2).cpu().numpy()
                for i in range(q):
                    for j in range(M):
                        ri, ci = linear_sum_assignment(d_np[i, j])
                        cost_np[lo + i, j] = d_np[i, j][ri, ci].mean()
            cost = pts_pred.new_tensor(cost_np)
        else:  # cwot
            cost = pts_pred.new_zeros((Q, M))
            for i in range(Q):
                for j in range(M):
                    cost[i, j] = curve_ot_loss(pts_pred[i], gt_flat[j],
                                                **self.cwot_kwargs)

        if self.normalize_median and self.mode in ('cwot', 'emd'):
            finite = cost[torch.isfinite(cost)]
            if finite.numel() > 0:
                cost = cost / finite.median().clamp(min=1e-6)
        cost = torch.nan_to_num(cost, nan=0.0, posinf=0.0, neginf=0.0)
        return cost * self.weight
