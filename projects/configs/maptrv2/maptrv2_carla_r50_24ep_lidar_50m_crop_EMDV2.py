"""50 m rotated-crop benchmark, monotone-OT (`emdv2`) geometry loss.

Differs from `maptrv2_carla_r50_24ep_lidar_50m_crop.py` in exactly ONE line:
`loss_pts` becomes `PolylineGeomLoss(mode='emdv2')`, so any result difference
between the two is the transport, not reweighting. The matching cost stays
`OrderedPtsL1Cost` on purpose -- it is already order-sensitive, and the DTW
cost over every (query x gt x order) triple measured 2.4x the L1 baseline's
iteration time.

`emdv2` (`polyline_loss.py`) is an order-preserving, many-to-many monotone
(DTW-style) transport plan. It exists because plain `mode='emd'` lost to
`PtsL1Loss` on the 30 m data (AP@0.5 0.318 vs 0.637) for two measured reasons:

1. SCALE. `PtsL1Loss` sums P*C = 40 elementwise terms per matched pair;
   `PolylineGeomLoss` returned one mean per pair over the same avg_factor --
   a 58.9x weaker gradient at every error scale. `emdv2_pointwise_reduction=
   'sum'` restores the degree (mean-along-path * P); the weight below is
   5.0 * sqrt(C=2) = 7.07 because this term sums P L2 norms where L1 sums
   P*C absolute components.
2. ORDER. A free Hungarian assignment is permutation-invariant, and the HM
   configs also zero `loss_dir`, so nothing constrained vertex order while
   the evaluator resamples predictions *along stored order*. A monotone plan
   enforces order but still lets the curve reparameterise, which rigid
   index-to-index L1 cannot.

On the 30 m pkl data this reached parity with L1 (mAP 0.7474 vs 0.7485). On
the 50 m crops, 30 epochs, identical data/schedule/seed:

                  mAP     driving AP   curb AP
    ordered L1    83.2      79.2        87.2
    monotone OT   85.3      78.7        91.8

The gain is the curb class (28% tighter matched chamfer, 0.174 vs 0.243 m).
The lr here is the L1 config's 6e-4, never tuned for this loss; see
`..._50m_crop_EMDV2_LR3e-4.py` for the tuned recipe (88.4 mAP).

Those numbers were measured with 4-channel (colour-carrying) points; this
tree is colour-free (see the parent), so re-measure before quoting them
against new runs.
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_50m_crop.py']

# Must equal the base's point_cloud_range (restated -- config isolation).
# Passed so the loss undoes normalize_2d_pts's per-axis scaling; a no-op on
# this square range, kept for the day a rectangular export appears.
point_cloud_range = [-15.0, -15.0, -30.0, 15.0, 15.0, 20.0]

# 5.0 (PtsL1Loss) * sqrt(C=2): PtsL1Loss sums P*C elementwise |d| terms,
# this sums P L2 norms, so parity needs sqrt(C).
loss_pts_weight = 7.07

model = dict(
    pts_bbox_head=dict(
        loss_pts=dict(
            _delete_=True,
            type='PolylineGeomLoss',
            mode='emdv2',
            loss_weight=loss_pts_weight,
            emdv2_pointwise_reduction='sum',
            # uniform_spacing_loss weight; 0.05 was tried as EMDV2S on the
            # 30 m data and stayed at parity, so it is off here.
            emdv2_spacing_w=0.0,
            pc_range=point_cloud_range)))
