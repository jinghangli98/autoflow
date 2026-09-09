"""2D NIfTI slice dataset for the stock 2D SwinIR / Real-ESRGAN baselines.

Built on the whole-volume pipeline in dataset.py (build_samples over
<root>/<anatomy>/<acq>/<subject>/ with no sidecars), so the baselines train
on exactly the data the flow model trains on: same patient-level val split
(`val_fraction` / `split_seed`), same 0.5/99.5 percentile normalization,
same foreground re-draw rule (`fg_fraction`, see VolumePairDataset).

One `__getitem__` visits a (condition, target) volume pair and returns
`slices_per_volume` random 2D crops as a `(K, 1, size, size)` stack -- a
`(size, size, 1)` VolumePairDataset patch with the singleton thin axis
squeezed. Flatten batches with dataset.collate_patches, giving the stock 2D
nets their `(B*K, 1, H, W)` input. dataset.py is imported, never modified.
"""

from dataset import (  # noqa: F401  (re-exported for the trainers)
    BalancedDistributedSampler,
    VolumePairDataset,
    _filter_artifacts,
    _subsample,
    build_samples,
    collate_patches,
)

# Trainer `--target_type` -> dataset `target_type` values kept.
TARGET_TYPES = {
    "raw": {"raw"},             # artifact -> fully sampled
    "denoised": {"denoised"},   # artifact/raw -> denoised + biascorrected
    "md": {"denoised"},         # launcher alias for the same target
}


class SlicePairDataset(VolumePairDataset):
    """One sample = `(K, 1, size, size)` 2D (condition, target) slice stacks.

    A VolumePairDataset with `patch_shape=(size, size, 1)`: crops keep the
    fg re-draw rule and per-plane prompt substitution, and the singleton
    thin axis is squeezed so each patch is a single 2D slice. Returns the
    same 6-tuple as VolumePairDataset (collate with collate_patches).
    """

    def __init__(self, samples, size=192, slices_per_volume=8, **kwargs):
        super().__init__(samples, patch_shape=(size, size, 1),
                         patches_per_volume=slices_per_volume, **kwargs)
        self.size = size

    def __getitem__(self, idx):
        cond, tgt, prompts, tt, anatomy, artifact = super().__getitem__(idx)
        return cond[..., 0], tgt[..., 0], prompts, tt, anatomy, artifact


def filter_target_type(dataset, target_type, rank=0):
    """Keep only samples whose ground truth matches `--target_type`, then
    rebuild the balanced-sampler index maps over the surviving samples."""
    keep_types = TARGET_TYPES[target_type]
    kept = [s for s in dataset.samples if s["target_type"] in keep_types]
    if not kept:
        raise ValueError(
            f"No samples for --target_type {target_type!r} "
            f"(target types present: {sorted({s['target_type'] for s in dataset.samples})})")
    if rank == 0:
        print(f"  target_type={target_type}: kept {len(kept)}/{len(dataset.samples)} pairs")
    dataset.samples = kept
    contrast_indices, group_indices = {}, {}
    for i, s in enumerate(kept):
        contrast_indices.setdefault(s["anatomy"], []).append(i)
        group_indices.setdefault((s["anatomy"], s["artifact"]), []).append(i)
    dataset.contrast_indices = contrast_indices
    dataset.group_indices = group_indices
    return dataset


def get_dataset_2d_slices(data_root, contrast, sample=100.0, size=192,
                          augment=None, artifacts=None, slices_per_volume=8,
                          val_fraction=0.10, split_seed=42,
                          norm_percentiles=(0.5, 99.5), fg_fraction=0.25):
    """Build train + val 2D-slice datasets for one or more anatomies.

    Mirrors dataset.get_dataset_3d_patches (same discovery, split, artifact
    filtering, subsampling and index maps) but yields 2D slice stacks. The
    index maps count volume *pairs*; each pair contributes
    `slices_per_volume` slices per visit, so BalancedDistributedSampler
    quotas are in pairs, exactly like train_flow.
    """
    contrasts = [contrast] if isinstance(contrast, str) else list(contrast)

    train_all, val_all = [], []
    train_ci, val_ci = {}, {}
    for c in contrasts:
        tr = _filter_artifacts(
            build_samples(data_root, c, "train", val_fraction, split_seed), artifacts)
        va = _filter_artifacts(
            build_samples(data_root, c, "val", val_fraction, split_seed), artifacts)
        if not tr:
            raise ValueError(
                f"No (condition, target) volume pairs found for anatomy "
                f"{c!r} in {data_root} (see dataset.build_samples).")
        tr = _subsample(tr, sample, seed=42)

        train_ci[c] = list(range(len(train_all), len(train_all) + len(tr)))
        val_ci[c] = list(range(len(val_all), len(val_all) + len(va)))
        train_all.extend(tr)
        val_all.extend(va)

    def _group_indices(samples):
        groups = {}
        for i, s in enumerate(samples):
            groups.setdefault((s["anatomy"], s["artifact"]), []).append(i)
        return groups

    common = dict(size=size, slices_per_volume=slices_per_volume,
                  augment=augment, norm_percentiles=norm_percentiles,
                  fg_fraction=fg_fraction)
    train_set = SlicePairDataset(train_all, deterministic=False, **common)
    val_set = SlicePairDataset(val_all, deterministic=True, **common)
    train_set.contrast_indices = train_ci
    val_set.contrast_indices = val_ci
    train_set.group_indices = _group_indices(train_all)
    val_set.group_indices = _group_indices(val_all)
    return train_set, val_set
