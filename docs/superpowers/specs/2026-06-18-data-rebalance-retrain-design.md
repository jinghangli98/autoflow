# Data-Rebalance Retrain — Design

Date: 2026-06-18
Status: proposed

## Goal / hypothesis under test

The current multi-anatomy flow-matching model
(`checkpoints/flow_matching_3d_brain_knee_prostate_final_061726.pt`) shows weak
restoration on knee/prostate and on anisotropic artifacts. The hypothesis being
tested by **this run in isolation**:

> knee/prostate are under-served by data exposure, and feeding the model more
> knee/prostate (relative to brain) will lift their restoration quality.

This is deliberately a **single-variable experiment**: the *only* thing that
changes versus the prior run is the per-anatomy sampling mix. Loss, sampling
steps, augmentation, architecture, and checkpoint-selection logic are left
untouched so the result is attributable to data balance alone.

## Evidence motivating the design

Per-anatomy training pairs (from `dataset.build_samples`, train split):

| anatomy  | total pairs | artifact2raw | artifact2denoised | raw2denoised |
|----------|-------------|--------------|-------------------|--------------|
| brain    | 104,825     | 50,278       | 50,278            | 4,269        |
| knee     | 88,150      | 42,312       | 42,312            | 3,526        |
| prostate | 30,275      | 14,532       | 14,532            | 1,211        |

Current sampler uses `--samples_per_contrast 0`, which **balances to the
smallest pool** → every epoch draws 30,275 from each anatomy (brain:knee:prostate
= 1:1:1). So knee/prostate are *not* starved relative to brain today; to favor
them we must weight them *above* brain.

Eval (`eval_test_whole_shard*.csv`, gs=1.0 model vs degraded-input baseline)
shows the dominant failure is **anisotropic across all anatomies** (brain
+1.3 dB, knee +1.0 dB, prostate +0.05 dB PSNR). Because brain — the data-richest
anatomy, and aniso is its most-represented artifact (phase/read/par × 3 levels)
— also fails aniso, the aniso failure is believed to be **structural** (MSE
velocity loss → mean/blur; 2 sampling steps), not an exposure problem.

**Expected outcome of this run:** knee/prostate absolute PSNR/SSIM rise; aniso
stays roughly flat. A flat-aniso result confirms the structural diagnosis and
motivates the follow-up (loss + sampling + selection) prong.

## Scope

In scope:
- Per-anatomy sampling quotas, strongly weighted toward knee/prostate.

Explicitly out of scope (deferred to a later run to keep this test clean):
- High-frequency / x1-reconstruction loss term.
- Increasing validation/inference sampling steps.
- Wiring in `build_augmentation`.
- Artifact-aware checkpoint selection.
- Architecture / cross-attention changes.

## Design

### Per-anatomy quotas

Target per-epoch mix (strong up-weight):

| anatomy  | per-epoch samples | vs pool          |
|----------|-------------------|------------------|
| brain    | 20,000            | 0.19× (subset)   |
| knee     | 50,000            | 0.57× (subset)   |
| prostate | 50,000            | 1.65× (oversample w/ replacement) |

`BalancedDistributedSampler` already oversamples-with-replacement when the
requested count exceeds a pool, so prostate at 50k is handled. Brain/knee draw a
fresh random subset each epoch (reseeded by `seed + epoch`).

### Code changes

1. **`dataset.py` — `BalancedDistributedSampler`**: accept `samples_per_contrast`
   as either an `int` (current behavior, applied to every contrast) **or** a
   `dict[str, int]` mapping anatomy → per-epoch count. Per-anatomy lookup in
   `__iter__`; `num_samples`/`total_size` computed from the summed quotas.

2. **`dataset.py` — `getloader_3d_patches`**: pass a dict through unchanged when
   given one; `use_balanced` stays true when `samples_per_contrast is not None`.

3. **`train_flow.py` — `parse_args`**: let `--samples_per_contrast` take
   `nargs="+"`. One value → scalar (back-compat). N values matching the N
   `--contrast` entries (same order) → build the
   `{contrast: count}` dict before constructing the loader.

4. **`train_flow.sh`**: set
   `--contrast brain knee prostate` and
   `--samples_per_contrast 20000 50000 50000`.
   Warm-start `--checkpoint_path` unchanged (continue from the current
   multi-anatomy final checkpoint).

Everything else in `train_flow.sh` (epochs=100, batch=8, fp16, compile,
num_sampling_steps=2, cfg_dropout_prob=0.1, size=192) stays identical to the
prior run.

## Validation & evaluation

- Training-time validation/checkpoint selection logic is unchanged (pooled
  raw/denoised score at 2 steps) — same as the prior run, so checkpoint
  selection is comparable.
- Final assessment re-runs `evaluate_test_whole.sh` (unchanged: 2 steps, euler,
  gs 1.0/0.0, axial) on the new checkpoint, then compares per-(anatomy, artifact)
  PSNR/SSIM/LPIPS deltas against the current `eval_test_whole_shard*.csv`.

## Success criteria

- **Primary (hypothesis):** knee and prostate absolute PSNR/SSIM on the
  fully_sampled and denoised tasks improve over the current checkpoint
  (target: prostate denoised PSNR meaningfully above ~24–26; knee/prostate
  undersampling & denoised SSIM up).
- **Diagnostic:** record whether aniso ΔPSNR moves. Flat aniso → proceed to the
  structural prong; improved aniso → revisit the structural diagnosis.

## Risks

- **Prostate overfit:** 50k/epoch from a 30k pool (1.65× oversample) over 100
  epochs ≈ 165 repeats per pair, and prostate has only one acquisition
  (`tse_ax_3T`). Watch for prostate train/val divergence. If it overfits, the
  cleanest follow-up knob is wiring in `build_augmentation` for prostate (was
  intentionally left out here to keep the test single-variable).
- **Brain regression:** brain drops to 0.19× per-epoch exposure; brain metrics
  may dip. Acceptable for the test, but track it.
