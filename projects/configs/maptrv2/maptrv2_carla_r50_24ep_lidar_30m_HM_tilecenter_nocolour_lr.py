"""30m tile-centred HM, colour-free, with the learning rate DERIVED from the
batch size instead of hardcoded.

Changes exactly two numbers against its parent -- `optimizer.lr` and
`lr_config.warmup_iters` -- and derives both from `samples_per_gpu` x
`num_gpus`, which is stated here so the two cannot drift apart. Everything
else (model, losses, assigner, pipelines, ann_file, map_ann_file) is
inherited untouched, so the GT is identical and checkpoints ARE comparable
with the parent's.

--- Why this exists ---

The parent chain sets `AdamW lr=1e-4` to mirror the Pointcept source the
polyline geometry loss was ported from. That number has an UNSTATED reference
batch, which is the dangerous kind: an LR is only meaningful paired with the
batch it was tuned at, and nothing in the config chain records what that was.

The repo's own MapTRv2 configs do record theirs. Every one pairs `lr=6e-4`
with `samples_per_gpu=4`, and README.md:58 states the experiments ran on 8
GPUs -- so the tuned setting is 6e-4 at an EFFECTIVE batch of 32. That is a
known-good anchor, and it is what this config scales from.

Note the trap that makes writing it down worthwhile: `samples_per_gpu` is the
PER-GPU batch, not the total. Reading it as the total over-scales by the GPU
count -- the error CLAUDE.md records as producing 1.3e-2 where ~5e-4 was
wanted, which diverges immediately.

--- Which scaling rule ---

`sqrt` (`lr proportional to sqrt(batch)`), not linear. The linear rule (Goyal
et al. 2017) was derived for SGD+momentum on ImageNet and was never validated
for AdamW, which this repo uses; the sqrt variant is the one usually argued
for adaptive optimizers and stays closer to the anchor. The two agree closely
near the reference batch and only diverge far from it, so the choice matters
little at batch 4-8 and a lot at batch 64.

Both are heuristics. Set `lr_rule = 'linear'` below to compare, or override
the result entirely with `--cfg-options optimizer.lr=...`, which merges after
this file and wins.

--- What this does NOT settle ---

Whether 6e-4@32 transfers to the HM objective at all. The parent's loss is
`PolylineGeomLoss` mode 'emd', not the baseline's `PtsL1Loss`, and a different
loss surface can want a different step size regardless of batch. This config
picks the defensible default and makes the assumption legible; it is not a
measurement. If a run diverges, that is information -- drop `lr_reference`,
not the rule.
"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_30m_HM_tilecenter_nocolour.py']

# ---------------------------------------------------------------- batch size
# mmcv evaluates each config file in isolation, so this is restated rather
# than read back from the parent. It is ALSO set on `data` below, so the batch
# the LR is derived from is by construction the batch that trains -- editing
# this one line moves both.
samples_per_gpu = 4
# Not expressible in the config otherwise: nothing in mmcv records how many
# GPUs a run will use, and `tools/train.py --autoscale-lr` guesses wrong for
# this purpose (it multiplies by len(gpu_ids)/8, silently dividing the LR by 8
# on a single GPU). State it here and do NOT pass --autoscale-lr.
num_gpus = 1
effective_batch = samples_per_gpu * num_gpus

# ------------------------------------------------------------- learning rate
# The anchor: every MapTRv2 config in this repo pairs lr=6e-4 with
# samples_per_gpu=4, and README.md:58 states 8 GPUs -> effective batch 32.
lr_reference = 6e-4
lr_reference_batch = 32

lr_rule = 'sqrt'          # 'sqrt' (recommended, AdamW) or 'linear'
lr_power = 0.5 if lr_rule == 'sqrt' else 1.0
lr = lr_reference * (effective_batch / lr_reference_batch) ** lr_power

# weight_decay stays at the parent's 0.01 -- decoupled in AdamW, so it does
# not track the LR and has no batch-size argument attached to it.
optimizer = dict(type='AdamW', lr=lr, weight_decay=0.01)

# ------------------------------------------------------------------- warm-up
# `warmup_iters` is counted in ITERATIONS while `total_epochs` is counted in
# epochs, so it is implicitly batch-size dependent: a larger batch means fewer
# iters/epoch, so a fixed 500 eats a growing fraction of the run. Upstream
# nuScenes (~28k samples at batch 32, ~880 iters/epoch) spent ~0.6 of an epoch
# warming up; that fraction is what is preserved here.
#
# `n_train_tiles` CANNOT be derived -- it is a property of the pkl, which mmcv
# never opens. It must be updated when the train split is generated. The value
# below is `../carla_test`'s tile count, used as a stand-in because the
# `30m_tc` train pkl this config's parent names does not exist yet; if the
# real split differs materially, so does the right warm-up.
n_train_tiles = 3795
warmup_epochs = 0.6
warmup_iters = max(1, int(warmup_epochs * n_train_tiles / effective_batch))

# Restated in full: mmcv merges dicts recursively, so naming only
# `warmup_iters` would work, but the policy/min_lr_ratio are spelled out
# because `min_lr_ratio` is RELATIVE to the base LR and therefore already
# tracks the value derived above -- worth being able to see that at a glance.
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=warmup_iters,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

# ---------------------------------------------------------------------- data
# Only the batch size is set; ann_file, map_ann_file, pipelines, pc_range,
# map_classes and the rest merge through from the parent untouched.
data = dict(samples_per_gpu=samples_per_gpu)
