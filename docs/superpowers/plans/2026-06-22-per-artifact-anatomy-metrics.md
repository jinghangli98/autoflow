# Per-(anatomy × artifact) Validation Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log per-epoch validation SSIM/PSNR/LPIPS and sample counts to W&B, broken down by `(anatomy × artifact_family)` at guidance scales cfg=0 and cfg=1, with ≥100 val images guaranteed per group.

**Architecture:** `dataset.py` stamps each sample with its `anatomy` and `artifact` family and returns them in a 6-tuple. A new dependency-light module `metrics_breakdown.py` holds the pure bookkeeping (group targets, done-check, log-dict assembly) so it is unit-testable without GPU/torch. `train_flow.py` imports those helpers, threads the two new fields through the train/val loops, accumulates the breakdown, replaces the flat val cap with a DDP-safe per-group stop, and logs the result.

**Tech Stack:** Python 3, PyTorch + DDP, MONAI, W&B, pytest 8.4.2 (run inside the `vsr` conda env).

## Global Constraints

- Run all commands inside the `vsr` conda env: prefix with `conda run -n vsr` (e.g. `conda run -n vsr pytest ...`).
- Artifact families are exactly: `undersampled` (`_R<n>`), `spike` (`_SPIKE_R<n>`), `aniso` (`_ANISO_{phase,read,par}<n>`), `clean` (raw→denoised task, no artifact).
- Anatomy values are exactly: `brain`, `knee`, `prostate`.
- Breakdown guidance scales: `BREAKDOWN_SCALES = (0.0, 1.0)`. Min images per group: `MIN_PER_GROUP = 100`.
- W&B key format: `val_<metric>_cfg<scale:g>_<anatomy>_<artifact>` where metric ∈ {ssim, psnr, lpips, n}, e.g. `val_lpips_cfg1_brain_aniso`.
- Checkpoint-selection logic (`selection_score`, `best_score`, the per-target-type combined score) MUST remain unchanged.
- Place new tests under `tests/`.

---

### Task 1: Artifact-family classifier in `dataset.py`

**Files:**
- Modify: `dataset.py` (add `_artifact_family` near `_ARTIFACT_SUFFIX_RE`, ~line 57)
- Test: `tests/test_dataset_artifact.py`

**Interfaces:**
- Produces: `_artifact_family(suffix: str) -> str`. Input is the artifact dir suffix (the part after the raw subject name, e.g. `"_R8"`, `"_SPIKE_R4"`, `"_ANISO_phase3.5"`). Returns one of `"undersampled"`, `"spike"`, `"aniso"`, or `"unknown"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dataset_artifact.py`:

```python
import dataset


def test_artifact_family_undersampled():
    assert dataset._artifact_family("_R4") == "undersampled"
    assert dataset._artifact_family("_R8") == "undersampled"


def test_artifact_family_spike_takes_priority_over_r():
    # "_SPIKE_R4" contains "R4" but must classify as spike, not undersampled.
    assert dataset._artifact_family("_SPIKE_R4") == "spike"


def test_artifact_family_aniso_variants():
    assert dataset._artifact_family("_ANISO_phase3.5") == "aniso"
    assert dataset._artifact_family("_ANISO_read4") == "aniso"
    assert dataset._artifact_family("_ANISO_par3") == "aniso"


def test_artifact_family_unknown():
    assert dataset._artifact_family("_WEIRD9") == "unknown"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n vsr pytest tests/test_dataset_artifact.py -v`
Expected: FAIL with `AttributeError: module 'dataset' has no attribute '_artifact_family'`

- [ ] **Step 3: Write minimal implementation**

In `dataset.py`, immediately after the `_ARTIFACT_SUFFIX_RE` definition (the block ending around line 59), add:

```python
def _artifact_family(suffix: str) -> str:
    """Coarse artifact family for a sample, from the artifact dir suffix.

    `suffix` is the part of an artifact sibling dir name after the raw
    subject name, e.g. `_R8`, `_SPIKE_R4`, `_ANISO_phase3.5`. The order of
    checks matters: `_SPIKE_R*` also contains `R*`, so it is tested first.
    """
    if suffix.startswith("_SPIKE_R"):
        return "spike"
    if suffix.startswith("_R"):
        return "undersampled"
    if suffix.startswith("_ANISO_"):
        return "aniso"
    return "unknown"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n vsr pytest tests/test_dataset_artifact.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add dataset.py tests/test_dataset_artifact.py
git commit -m "feat(dataset): add _artifact_family classifier"
```

---

### Task 2: Stamp anatomy/artifact on samples and return them in `__getitem__`

**Files:**
- Modify: `dataset.py` — `build_samples` (~lines 217-303), `PatchPairDataset.__getitem__` (~lines 190-202)
- Test: `tests/test_dataset_samples.py`

**Interfaces:**
- Consumes: `_artifact_family` (Task 1).
- Produces:
  - Every sample dict from `build_samples` now also has keys `"anatomy": str` and `"artifact": str`.
  - `PatchPairDataset.__getitem__` returns a 6-tuple `(condition, target, prompt, target_type, anatomy, artifact)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dataset_samples.py`. It builds a tiny synthetic data tree (empty `.nii.gz` files are fine — `build_samples` only enumerates filenames and reads JSON sidecars, it does not load arrays) and checks the new fields. It also checks the 6-tuple by monkeypatching `_load_patch` so no real NIfTI is read.

```python
import json
import os

import pytest

import dataset


def _write(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _make_subject(acq_dir, name, processing, coords):
    sub = os.path.join(acq_dir, name)
    os.makedirs(sub, exist_ok=True)
    # prompt sidecars only for raw/denoised (have a prompt); artifacts get none.
    if processing is not None:
        with open(os.path.join(sub, f"{name}.json"), "w") as f:
            json.dump({"prompt": f"{processing} prompt", "processing": processing}, f)
    for (x, y, z) in coords:
        _write(os.path.join(sub, f"patch_{x}_{y}_{z}.nii.gz"))


@pytest.fixture
def tiny_root(tmp_path):
    root = str(tmp_path)
    acq = os.path.join(root, "train", "brain", "acq1")
    os.makedirs(acq, exist_ok=True)
    coords = [(0, 0, 0)]
    _make_subject(acq, "subjA", "raw", coords)              # raw GT
    _make_subject(acq, "mdsubjA", "denoised+biascorrected", coords)  # denoised
    _make_subject(acq, "subjA_R8", None, coords)            # undersampled artifact
    _make_subject(acq, "subjA_ANISO_phase3.5", None, coords)  # aniso artifact
    return root


def test_build_samples_stamps_anatomy_and_artifact(tiny_root):
    samples = dataset.build_samples(tiny_root, "train", "brain")
    assert samples, "expected at least one sample"
    for s in samples:
        assert s["anatomy"] == "brain"
        assert "artifact" in s

    families = {(s["task"], s["artifact"]) for s in samples}
    # artifact->raw/denoised from the _R8 sibling are undersampled
    assert ("artifact2raw", "undersampled") in families
    # aniso sibling present
    assert ("artifact2raw", "aniso") in families
    # raw->denoised carries no artifact => clean
    assert ("raw2denoised", "clean") in families


def test_getitem_returns_six_tuple(monkeypatch):
    monkeypatch.setattr(dataset, "_load_patch", lambda path, shape: "TENSOR")
    samples = [{
        "condition_path": "c", "target_path": "t", "prompt": "p",
        "task": "artifact2raw", "anatomy": "knee", "artifact": "undersampled",
    }]
    ds = dataset.PatchPairDataset(samples)
    out = ds[0]
    assert len(out) == 6
    condition, target, prompt, target_type, anatomy, artifact = out
    assert anatomy == "knee"
    assert artifact == "undersampled"
    assert target_type == "raw"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n vsr pytest tests/test_dataset_samples.py -v`
Expected: FAIL — `test_build_samples_stamps_anatomy_and_artifact` fails with `KeyError: 'anatomy'`; `test_getitem_returns_six_tuple` fails with `assert 4 == 6`.

- [ ] **Step 3: Implement — stamp fields in `build_samples`**

In `dataset.py` `build_samples`, set the artifact family once per sibling and add `anatomy`/`artifact` to all three append sites.

After the line `for art in _find_artifact_siblings(subjects, name):` (~line 274) add, as the first line inside the loop body:

```python
                art_family = _artifact_family(art[len(name):])
```

Update the **artifact→raw** append (the `samples.append({...})` with `"task": "artifact2raw"`) to:

```python
                        samples.append({
                            "condition_path": art_path,
                            "target_path": raw_patches[coord],
                            "prompt": raw_prompt,
                            "task": "artifact2raw",
                            "anatomy": anatomy,
                            "artifact": art_family,
                        })
```

Update the **artifact→denoised** append (`"task": "artifact2denoised"`) to:

```python
                        samples.append({
                            "condition_path": art_path,
                            "target_path": den_patches[coord],
                            "prompt": den_prompt,
                            "task": "artifact2denoised",
                            "anatomy": anatomy,
                            "artifact": art_family,
                        })
```

Update the **raw→denoised** append (`"task": "raw2denoised"`) to:

```python
                        samples.append({
                            "condition_path": raw_path,
                            "target_path": den_patches[coord],
                            "prompt": den_prompt,
                            "task": "raw2denoised",
                            "anatomy": anatomy,
                            "artifact": "clean",
                        })
```

- [ ] **Step 4: Implement — return 6-tuple from `__getitem__`**

In `dataset.py` `PatchPairDataset.__getitem__`, change the final return (~line 202) from:

```python
        target_type = "raw" if s["task"] == "artifact2raw" else "denoised"
        return condition, target, s["prompt"], target_type
```

to:

```python
        target_type = "raw" if s["task"] == "artifact2raw" else "denoised"
        return condition, target, s["prompt"], target_type, s["anatomy"], s["artifact"]
```

Also update the class docstring's leading sentence near line 165 from
`"""One sample = (condition patch, target patch, prompt).` to
`"""One sample = (condition, target, prompt, target_type, anatomy, artifact).`

- [ ] **Step 5: Run tests to verify they pass**

Run: `conda run -n vsr pytest tests/test_dataset_samples.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add dataset.py tests/test_dataset_samples.py
git commit -m "feat(dataset): stamp anatomy/artifact and return them per sample"
```

---

### Task 3: Pure breakdown bookkeeping module

**Files:**
- Create: `metrics_breakdown.py`
- Test: `tests/test_metrics_breakdown.py`

**Interfaces:**
- Produces (all torch-free, stdlib only):
  - `BREAKDOWN_SCALES = (0.0, 1.0)`
  - `MIN_PER_GROUP = 100`
  - `ARTIFACT_FAMILIES = ("undersampled", "spike", "aniso", "clean")`
  - `group_targets(group_total: dict, min_per_group: int = MIN_PER_GROUP) -> dict` — maps each `(anatomy, artifact)` key to `min(min_per_group, total)`.
  - `groups_done(global_counts: dict, targets: dict) -> bool` — True iff every group in `targets` has `global_counts.get(g, 0) >= target`.
  - `breakdown_log_dict(bd_ssim, bd_psnr, bd_lpips, bd_n, scales, present_groups) -> dict` — each `bd_*` is a dict keyed by `(scale, anatomy, artifact)`; returns W&B-ready averaged keys, skipping any group with `n <= 0`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_metrics_breakdown.py`:

```python
import metrics_breakdown as mb


def test_group_targets_clamps_to_min():
    total = {("brain", "undersampled"): 500, ("knee", "aniso"): 37}
    targets = mb.group_targets(total, min_per_group=100)
    assert targets[("brain", "undersampled")] == 100
    assert targets[("knee", "aniso")] == 37  # fewer than 100 available


def test_groups_done_true_when_all_met():
    targets = {("brain", "undersampled"): 100, ("knee", "aniso"): 37}
    counts = {("brain", "undersampled"): 104, ("knee", "aniso"): 37}
    assert mb.groups_done(counts, targets) is True


def test_groups_done_false_when_one_short():
    targets = {("brain", "undersampled"): 100, ("knee", "aniso"): 37}
    counts = {("brain", "undersampled"): 104, ("knee", "aniso"): 12}
    assert mb.groups_done(counts, targets) is False


def test_groups_done_missing_group_counts_as_zero():
    targets = {("brain", "undersampled"): 100}
    assert mb.groups_done({}, targets) is False


def test_breakdown_log_dict_averages_and_skips_empty():
    present = [("brain", "undersampled"), ("brain", "aniso")]
    scales = (0.0, 1.0)
    bd_n = {(s, a, art): 0.0 for s in scales for (a, art) in present}
    bd_ssim = dict(bd_n)
    bd_psnr = dict(bd_n)
    bd_lpips = dict(bd_n)
    # populate only (cfg=1, brain, undersampled) with 2 samples
    key = (1.0, "brain", "undersampled")
    bd_n[key] = 2.0
    bd_ssim[key] = 1.6   # avg 0.8
    bd_psnr[key] = 60.0  # avg 30
    bd_lpips[key] = 0.2  # avg 0.1

    out = mb.breakdown_log_dict(bd_ssim, bd_psnr, bd_lpips, bd_n, scales, present)

    assert out["val_ssim_cfg1_brain_undersampled"] == 0.8
    assert out["val_psnr_cfg1_brain_undersampled"] == 30.0
    assert out["val_lpips_cfg1_brain_undersampled"] == 0.1
    assert out["val_n_cfg1_brain_undersampled"] == 2.0
    # empty groups produce no keys
    assert "val_ssim_cfg0_brain_undersampled" not in out
    assert "val_ssim_cfg1_brain_aniso" not in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n vsr pytest tests/test_metrics_breakdown.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'metrics_breakdown'`

- [ ] **Step 3: Write the implementation**

Create `metrics_breakdown.py`:

```python
"""Pure (torch-free) bookkeeping for per-(anatomy x artifact) validation
metrics. Kept dependency-light so it is unit-testable without GPU/torch;
`train_flow.py` does the torch all_reduce and calls these helpers.
"""

BREAKDOWN_SCALES = (0.0, 1.0)   # guidance scales the breakdown is logged at
MIN_PER_GROUP = 100             # min val images per (anatomy, artifact) group
ARTIFACT_FAMILIES = ("undersampled", "spike", "aniso", "clean")


def group_targets(group_total, min_per_group=MIN_PER_GROUP):
    """Per-group image target: min(min_per_group, total available)."""
    return {g: min(min_per_group, n) for g, n in group_total.items()}


def groups_done(global_counts, targets):
    """True iff every group has reached its target image count."""
    return all(global_counts.get(g, 0) >= t for g, t in targets.items())


def breakdown_log_dict(bd_ssim, bd_psnr, bd_lpips, bd_n, scales, present_groups):
    """Sample-weighted averages -> W&B-ready keys, skipping empty groups.

    Each `bd_*` is keyed by `(scale, anatomy, artifact)`. Keys follow
    `val_<metric>_cfg<scale:g>_<anatomy>_<artifact>`.
    """
    out = {}
    for sc in scales:
        for (a, art) in present_groups:
            n = bd_n[(sc, a, art)]
            if n <= 0:
                continue
            tag = f"cfg{sc:g}_{a}_{art}"
            out[f"val_ssim_{tag}"] = bd_ssim[(sc, a, art)] / n
            out[f"val_psnr_{tag}"] = bd_psnr[(sc, a, art)] / n
            out[f"val_lpips_{tag}"] = bd_lpips[(sc, a, art)] / n
            out[f"val_n_{tag}"] = n
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n vsr pytest tests/test_metrics_breakdown.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add metrics_breakdown.py tests/test_metrics_breakdown.py
git commit -m "feat: add pure breakdown-metrics bookkeeping module"
```

---

### Task 4: Wire the breakdown into `train_flow.py`

**Files:**
- Modify: `train_flow.py` — imports (~line 44), train-step unpack (line 572), val loop (lines 640-819)

**Interfaces:**
- Consumes: `metrics_breakdown` (Task 3); the 6-tuple dataset batches (Task 2).
- Produces: extra `val_*_cfg{0,1}_<anatomy>_<artifact>` keys in the per-epoch W&B `log_dict`; a per-group print on rank 0. No new public functions.

This task cannot be exercised by an automated test in this environment (it needs the full model, data, and GPUs). It is verified by a syntax/compile check here, plus a manual smoke run by the user (documented in the final step). Make the edits exactly as below.

- [ ] **Step 1: Add the import**

In `train_flow.py`, next to the other local imports (the `import wandb` line is ~line 44), add:

```python
import metrics_breakdown as mb
```

- [ ] **Step 2: Ignore new fields at the train-step unpack**

Change line 572 from:

```python
        for step, (condition, target, prompts, _target_types) in progress_bar:
```

to:

```python
        for step, (condition, target, prompts, _target_types, *_) in progress_bar:
```

- [ ] **Step 3: Capture new fields at the val-loop unpack and set up targets**

Change the val-loop header (line 660) from:

```python
            for i, (condition, target, prompts, target_types) in enumerate(val_loader):
```

to:

```python
            for i, (condition, target, prompts, target_types, anatomies, artifacts) in enumerate(val_loader):
```

Immediately **before** the `for i, ...` loop (just after `val_images_seen = 0` at ~line 658), insert the per-group target setup and breakdown accumulators:

```python
            from collections import Counter
            val_group_total = Counter(
                (s["anatomy"], s["artifact"]) for s in val_loader.dataset.samples
            )
            present_groups = sorted(val_group_total)            # deterministic order
            group_target = mb.group_targets(dict(val_group_total))
            group_seen = {g: 0 for g in present_groups}
            # Breakdown sample-weighted sums, keyed by (scale, anatomy, artifact),
            # pooled over target_type, only at the cfg=0 / cfg=1 scales.
            bd_scales = [s for s in scales if s in mb.BREAKDOWN_SCALES]
            bd_ssim = {(sc, a, art): 0.0 for sc in bd_scales for (a, art) in present_groups}
            bd_psnr = {(sc, a, art): 0.0 for sc in bd_scales for (a, art) in present_groups}
            bd_lpips = {(sc, a, art): 0.0 for sc in bd_scales for (a, art) in present_groups}
            bd_n = {(sc, a, art): 0.0 for sc in bd_scales for (a, art) in present_groups}
```

- [ ] **Step 4: Accumulate the breakdown inside the scale loop**

The existing inner loop is `for s in scales:` then, after sampling, `for tt, idx in idx_by_tt.items():`. Add a breakdown accumulation block right after that target-type loop closes (i.e. still inside `for s in scales:`, after the `sum_n[s][tt] += n` block that ends ~line 705, before the `if i == 0 and global_rank == 0:` viz capture at line 706). Insert:

```python
                    # Per-(anatomy, artifact) breakdown, pooled over target_type,
                    # only at the cfg=0 / cfg=1 scales. Group this batch's slices
                    # by (anatomy, artifact) and score each sub-group.
                    if s in bd_scales:
                        bd_idx = {}
                        for j in range(condition.shape[0]):
                            g = (anatomies[j], artifacts[j])
                            bd_idx.setdefault(g, []).append(j)
                        for g, gidx in bd_idx.items():
                            if g not in bd_ssim or not gidx:
                                continue
                            gen_g = generated[gidx][..., cz].cpu().numpy()
                            targ_g = target[gidx][..., cz].cpu().numpy()
                            mg = evaluate_image_quality(gen_g, targ_g)
                            pvg = mg["PSNR"] if np.isfinite(mg["PSNR"]) else 100.0
                            ng = len(gidx)
                            bd_ssim[(s, g[0], g[1])] += mg["SSIM"] * ng
                            bd_psnr[(s, g[0], g[1])] += pvg * ng
                            bd_lpips[(s, g[0], g[1])] += mg["LPIPS"] * ng
                            bd_n[(s, g[0], g[1])] += ng
```

- [ ] **Step 5: Increment per-group image counts and replace the stop condition**

The existing code has, after the scale loop and the viz block, `val_images_seen += condition.shape[0]` (~line 709) and later:

```python
                if val_images_seen >= max_val_images_per_rank:
                    break
```

Replace that `if val_images_seen >= max_val_images_per_rank: break` block with a per-group, DDP-safe stop. Replace it with:

```python
                for j in range(condition.shape[0]):
                    g = (anatomies[j], artifacts[j])
                    if g in group_seen:
                        group_seen[g] += 1
                counts = torch.tensor(
                    [float(group_seen[g]) for g in present_groups],
                    dtype=torch.float64, device=device,
                )
                if args.distributed:
                    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                global_counts = {g: counts[k].item() for k, g in enumerate(present_groups)}
                if mb.groups_done(global_counts, group_target):
                    break
```

Leave the `val_images_seen += condition.shape[0]` line in place (it is harmless and still feeds the existing per-target-type accounting); only the `max_val_images_per_rank` break is replaced. The `max_val_images_per_rank = ...` definition line may remain unused; delete it to avoid a dead variable:

Remove the line (~line 657):

```python
            max_val_images_per_rank = max(1, 1000 // max(1, world_size))
```

- [ ] **Step 6: All-reduce the breakdown sums under DDP**

The existing DDP block (lines 729-747) flattens `sum_ssim/sum_psnr/...` into `metrics_tensor`, all-reduces, and unflattens. Add a parallel reduction for the breakdown sums right after that block (after line 747, before the `# Sample-weighted averages` comment at line 749):

```python
            if args.distributed and bd_scales:
                bd_flat = []
                for sc in bd_scales:
                    for (a, art) in present_groups:
                        bd_flat += [bd_ssim[(sc, a, art)], bd_psnr[(sc, a, art)],
                                    bd_lpips[(sc, a, art)], bd_n[(sc, a, art)]]
                bd_tensor = torch.tensor(bd_flat, device=device, dtype=torch.float64)
                dist.all_reduce(bd_tensor, op=dist.ReduceOp.SUM)
                k = 0
                for sc in bd_scales:
                    for (a, art) in present_groups:
                        bd_ssim[(sc, a, art)] = bd_tensor[k].item()
                        bd_psnr[(sc, a, art)] = bd_tensor[k + 1].item()
                        bd_lpips[(sc, a, art)] = bd_tensor[k + 2].item()
                        bd_n[(sc, a, art)] = bd_tensor[k + 3].item()
                        k += 4
```

- [ ] **Step 7: Log breakdown to W&B and print per-group table**

In the W&B block (`if args.log and global_rank == 0:`, lines 802-819), after the existing per-`(scale, target_type)` loop populates `log_dict` and before `wandb.log(log_dict)`, add:

```python
                log_dict.update(
                    mb.breakdown_log_dict(bd_ssim, bd_psnr, bd_lpips, bd_n,
                                          bd_scales, present_groups)
                )
```

Then add a rank-0 print of the breakdown for at-a-glance monitoring. After the existing validation print block (the `if global_rank == 0:` loop ending ~line 780), add:

```python
            if global_rank == 0:
                for sc in bd_scales:
                    for (a, art) in present_groups:
                        n = bd_n[(sc, a, art)]
                        if n <= 0:
                            continue
                        print(f"  breakdown [cfg={sc:g}, {a}/{art}] - "
                              f"SSIM: {bd_ssim[(sc, a, art)] / n:.4f}, "
                              f"PSNR: {bd_psnr[(sc, a, art)] / n:.4f}, "
                              f"LPIPS: {bd_lpips[(sc, a, art)] / n:.4f}, "
                              f"n: {int(n)}")
```

- [ ] **Step 8: Syntax-check the edited file**

Run: `conda run -n vsr python -m py_compile train_flow.py metrics_breakdown.py dataset.py`
Expected: no output, exit code 0 (a syntax error would print a traceback).

- [ ] **Step 9: Re-run the full unit suite to confirm nothing regressed**

Run: `conda run -n vsr pytest tests/ -v`
Expected: all tests from Tasks 1-3 PASS.

- [ ] **Step 10: Commit**

```bash
git add train_flow.py
git commit -m "feat(train_flow): log per-(anatomy x artifact) val metrics to W&B"
```

- [ ] **Step 11: Manual smoke verification (user-run, documented)**

The automated checks above cannot exercise the GPU path. The user should run a short validation and confirm:
1. A single-GPU short run (no `--distributed`, small `--subsample`/few epochs, `--log` on) prints `breakdown [cfg=0, ...]` / `breakdown [cfg=1, ...]` lines and produces `val_*_cfg0_*` / `val_*_cfg1_*` keys in W&B.
2. Each present group's `val_n_cfg{0,1}_<anat>_<art>` is ≥ its clamped target (100, or the group's total if smaller).
3. An 8-GPU `--distributed` run completes a validation epoch without hanging at the post-loop all-reduce, and global `val_n` per group ≥ target.

---

## Self-Review

**Spec coverage:**
- Artifact families (undersampled/spike/aniso/clean) → Task 1 (`_artifact_family`) + Task 2 (`clean` for raw2denoised). ✓
- anatomy/artifact stamped per sample + 6-tuple → Task 2. ✓
- Breakdown at cfg=0/cfg=1, pooled over target_type → Task 4 Step 4 (`bd_scales`, grouping ignores target_type). ✓
- ≥100 images per group, DDP-safe stop → Task 4 Steps 3/5 (`group_targets`, `groups_done`, per-batch all-reduce, identical `done` across ranks). ✓
- DDP reduction of breakdown sums → Task 4 Step 6. ✓
- W&B keys + per-group print, zero-sample groups skipped → Task 3 (`breakdown_log_dict` skips n<=0) + Task 4 Steps 7. ✓
- Checkpoint logic untouched → no task modifies `selection_score`/`best_score`. ✓

**Placeholder scan:** No TBD/TODO; all code shown in full. ✓

**Type consistency:** `group_targets`/`groups_done`/`breakdown_log_dict` signatures and the `(scale, anatomy, artifact)` key shape are identical across Task 3 (definition), Task 3 tests, and Task 4 (call sites). `present_groups` is a sorted list of `(anatomy, artifact)` tuples everywhere. ✓
