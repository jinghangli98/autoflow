# Per-(anatomy × artifact) validation metrics → W&B

**Date:** 2026-06-22
**Branch:** sigma
**Files touched:** `dataset.py`, `train_flow.py`

## Problem

The model restores undersampled brain patches well but struggles with
anisotropic (aniso) artifacts. The current validation only bins metrics by
`(guidance_scale, target_type=raw/denoised)`, so there is no per-artifact or
per-anatomy signal to confirm or track this over training. We want a per-epoch
W&B breakdown that isolates which `(anatomy, artifact_family)` combinations lag.

## Goal

Each validation epoch, log SSIM / PSNR / LPIPS and a sample count for every
`(anatomy × artifact_family)` group, at guidance scales **cfg=0 and cfg=1**,
pooled over `target_type`. Guarantee **≥100 validation images per group**
(where that many exist) so the per-group metrics are stable. Checkpoint
selection logic is left unchanged.

## Definitions

**Artifact families** (derived from the artifact sibling dir suffix, matched by
`_ARTIFACT_SUFFIX_RE`):

| Suffix pattern        | Family         |
|-----------------------|----------------|
| `_R<n>`               | `undersampled` |
| `_SPIKE_R<n>`         | `spike`        |
| `_ANISO_{phase,read,par}<n>` | `aniso` |
| (no artifact — `raw2denoised` task) | `clean` |

**Anatomy:** `brain` / `knee` / `prostate` — already known at `build_samples`
time (it is the function's `anatomy` argument).

**Enumerated group set:** `{brain, knee, prostate} × {undersampled, spike,
aniso, clean}` (12 groups). Only groups actually present in the val set are
logged; absent groups are skipped.

## Design

### 1. `dataset.py`

- Add a helper `_artifact_family(suffix: str) -> str` that maps an artifact
  suffix to `undersampled` / `spike` / `aniso`.
- In `build_samples`, add two fields to **every** sample dict:
  - `anatomy`: the function's `anatomy` argument.
  - `artifact`: `_artifact_family(rest)` for the artifact→raw and
    artifact→denoised tasks (where `rest` is the suffix of the artifact
    sibling dir), and `"clean"` for the raw→denoised task.
- `PatchPairDataset.__getitem__` returns a **6-tuple**:
  `(condition, target, prompt, target_type, anatomy, artifact)`.
  The default collate converts the two new strings into per-sample lists,
  exactly as it already does for `prompt` and `target_type` — no custom
  collate is required.

### 2. `train_flow.py`

**Constants (module level):**
```python
BREAKDOWN_SCALES = (0.0, 1.0)   # guidance scales to log the breakdown at
MIN_PER_GROUP = 100             # min val images per (anatomy, artifact) group
ARTIFACT_FAMILIES = ("undersampled", "spike", "aniso", "clean")
```

**Unpack-site updates:**
- Train step (currently `for step, (condition, target, prompts, _target_types)
  in progress_bar:`): ignore the two new fields with `*_`.
- Val loop (`for i, (condition, target, prompts, target_types) in
  enumerate(val_loader):`): capture `anatomies, artifacts` as the 5th/6th
  unpacked values.

**Per-group target setup (before the val loop):**
```python
from collections import Counter
group_total = Counter(
    (s["anatomy"], s["artifact"]) for s in val_loader.dataset.samples
)
present_groups = sorted(group_total)               # deterministic order
group_target = {g: min(MIN_PER_GROUP, group_total[g]) for g in present_groups}
```
`val_loader.dataset.samples` is the full, unsharded list under DDP, so totals
are global.

**Accumulators:** in addition to the existing per-`(scale, target_type)` sums,
add breakdown sums keyed by `(scale, anatomy, artifact)` for
`scale in BREAKDOWN_SCALES`:
```python
bd_ssim  = {(sc, a, art): 0.0 for sc in BREAKDOWN_SCALES for (a, art) in present_groups}
bd_psnr  = {...}
bd_lpips = {...}
bd_n     = {...}
```
and a per-group image counter `group_seen = {g: 0 for g in present_groups}`
(image count, independent of scale).

**In the batch loop:** the existing code already computes central-slice SSIM /
PSNR / LPIPS per `(scale, target_type)` over the indices `idx` of each
target_type. For the breakdown, additionally accumulate per sample into its
`(scale, anatomy, artifact)` group. Because the metrics function operates on a
group of slices at once, accumulation is done per artifact/anatomy sub-group of
the batch (group the batch indices by `(anatomy, artifact)` and call
`evaluate_image_quality` per sub-group, weighting by sub-group size), for each
`scale in BREAKDOWN_SCALES`. Increment `group_seen[(anatomy, artifact)]` by the
batch's per-group image counts once per batch (not per scale).

**DDP-safe stopping (replaces the flat `max_val_images_per_rank` cap):**
After each batch:
```python
counts = torch.tensor([group_seen[g] for g in present_groups],
                      dtype=torch.float64, device=device)
if args.distributed:
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)   # global per-group counts
done = all(counts[k].item() >= group_target[g]
          for k, g in enumerate(present_groups))
if done:
    break
```
All ranks derive an identical `done` from the reduced vector, so they break on
the same iteration → no collective-mismatch deadlock with the post-loop
`all_reduce`. If a group can never reach its target on some shard, `done` stays
False and the loop ends naturally at loader exhaustion (still deadlock-free,
since exhaustion is symmetric across equal-length shards). The non-distributed
path checks local counts directly.

**DDP reduction of breakdown sums:** extend the existing flatten →
`all_reduce(SUM)` → unflatten block to also carry `bd_ssim/bd_psnr/bd_lpips/
bd_n` over the fixed, identically-ordered `present_groups` list (all ranks
agree on the key order, a prerequisite for a flat tensor reduce).

**Averaging + logging (rank 0):**
```python
for sc in BREAKDOWN_SCALES:
    for (a, art) in present_groups:
        n = bd_n[(sc, a, art)]
        if n <= 0:
            continue
        tag = f"cfg{sc:g}_{a}_{art}"
        log_dict[f"val_ssim_{tag}"]  = bd_ssim[(sc, a, art)]  / n
        log_dict[f"val_psnr_{tag}"]  = bd_psnr[(sc, a, art)]  / n
        log_dict[f"val_lpips_{tag}"] = bd_lpips[(sc, a, art)] / n
        log_dict[f"val_n_{tag}"]     = n
```
Also print a short per-group table on rank 0 alongside the existing validation
print. Groups with zero samples in an epoch are skipped (no key emitted), so
W&B never receives NaNs.

### 3. Untouched

- Checkpoint selection (`selection_score`, `best_score`, combined per-target
  -type score) is unchanged. The breakdown is purely additional diagnostic
  logging.
- The existing per-`(scale, target_type)` metrics and W&B keys remain.

## Consequences / trade-offs

- **Slower validation.** Guaranteeing ≥100 images for each of ~12 groups means
  processing ≥~1200 images vs the old ~1000 cap, and groups are not uniformly
  frequent, so in practice more of the val set is consumed. This is inherent to
  the request and accepted.
- **One extra small `all_reduce` per val batch** (a ~12-element vector) — cheap
  relative to a flow-sampling step.
- **Sparse groups.** A group with fewer than 100 total val samples logs all it
  has (target clamped via `min`). Its curve is noisier; that is expected.

## Test / verification

- Unit-level: `build_samples` now stamps `anatomy`/`artifact` on each sample;
  spot-check that an `_ANISO_*` sibling yields `aniso`, `_R8` yields
  `undersampled`, `_SPIKE_R*` yields `spike`, and `raw2denoised` yields
  `clean`.
- Smoke run: a short single-GPU run (`--distributed` off, tiny
  `--max_epochs`/subsample) should produce `val_*_cfg0_<anat>_<art>` and
  `val_*_cfg1_<anat>_<art>` keys in W&B (or in `log_dict` when `--log` is off)
  and a per-group print, with each present group's `val_n` ≥ its clamped
  target.
- DDP smoke (if feasible): confirm no hang at the post-loop `all_reduce` and
  that global `val_n` per group ≥ target.
