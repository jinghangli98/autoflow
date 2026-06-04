"""3D NIfTI patch dataset for flow matching.

Pairs undersampled patches (GRAPPA `_R{3..6}` / CS `_CS_R{2,3}`) with their
matching ground-truth patches inside a single contrast (mprage or tse).

Layout assumed:
  <data_root>/<split>/<contrast>/<subject>[_<orient>]/patch_<x>_<y>_<z>.nii.gz
where <subject> is the GT id (no suffix), `<orient>` is one of
{coronal, sagittal} (omitted for axial), and siblings
`<subject>_R<n>[_<orient>]` / `<subject>_CS_R<n>[_<orient>]` contain the same
patch coordinates with the corresponding undersampling.

Patch shapes on disk vary by orientation:
  axial:    (192, 192, 16)
  coronal:  (192, 16, 192)
  sagittal: (16, 192, 192)
All patches are transposed to (192, 192, 16) on load so the model sees a
single canonical layout, then center-cropped to the requested `patch_shape`
(default (192, 192, 16), i.e. no crop). A smaller `patch_shape` such as
(96, 96, 16) yields a genuine sub-volume at the same resolution -- useful for
quick model debugging.
"""

import glob
import json
import math
import os
import random
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

import torchio as tio


_PATCH_RE = re.compile(r"patch_(\d+)_(\d+)_(\d+)\.nii\.gz$")
_UNDERSAMPLED_SUFFIX_RE = re.compile(r"_(?:CS_)?R\d+$")
_ORIENTATIONS = ("coronal", "sagittal")


def _parse_subject(name: str):
    """Split a directory name into (base_subject, orientation).

    `orientation` is one of {"axial", "coronal", "sagittal"}. The orientation
    suffix (when present) comes *after* any undersampling suffix, e.g.
    `1041_R3_coronal` -> base=`1041_R3`, orientation=`coronal`.
    """
    for orient in _ORIENTATIONS:
        suffix = f"_{orient}"
        if name.endswith(suffix):
            return name[: -len(suffix)], orient
    return name, "axial"


def _is_gt_subject(name: str) -> bool:
    """Return True if a subject directory name has no `_R*`/`_CS_R*` suffix."""
    base, _ = _parse_subject(name)
    return _UNDERSAMPLED_SUFFIX_RE.search(base) is None


def _find_undersampled_siblings(contrast_dir: str, gt_name: str):
    """Find sibling undersampled subject dirs for a given GT subject name.

    Siblings share the GT's orientation suffix. For an axial GT `1041`,
    matches `1041_R<n>` / `1041_CS_R<n>`. For `1041_coronal`, matches
    `1041_R<n>_coronal` / `1041_CS_R<n>_coronal`.
    """
    base, orientation = _parse_subject(gt_name)
    orient_suffix = "" if orientation == "axial" else f"_{orientation}"
    pattern = re.compile(
        rf"^{re.escape(base)}_(?:CS_)?R\d+{re.escape(orient_suffix)}$"
    )
    candidates = sorted(
        d for d in os.listdir(contrast_dir)
        if os.path.isdir(os.path.join(contrast_dir, d))
    )
    return [c for c in candidates if pattern.match(c)]


def _index_subject_patches(subject_dir: str):
    """Map (x, y, z) tuple -> absolute patch path for a subject directory."""
    out = {}
    for f in glob.glob(os.path.join(subject_dir, "patch_*.nii.gz")):
        m = _PATCH_RE.search(os.path.basename(f))
        if not m:
            continue
        x, y, z = (int(m.group(i)) for i in (1, 2, 3))
        out[(x, y, z)] = f
    return out


def _center_crop(arr: np.ndarray, shape) -> np.ndarray:
    """Center-crop a 3D array to `shape`.

    No-op along any axis where the target size is >= the current size, so
    `shape=(192, 192, 16)` on an already-(192, 192, 16) array returns it
    unchanged.
    """
    slices = []
    for cur, tgt in zip(arr.shape, shape):
        if tgt >= cur:
            slices.append(slice(None))
        else:
            start = (cur - tgt) // 2
            slices.append(slice(start, start + tgt))
    return arr[tuple(slices)]


def _load_patch(path: str, orientation: str = "axial",
                patch_shape=(196, 196, 16)) -> torch.Tensor:
    """Load a NIfTI patch as float32 tensor (1, *patch_shape).

    On-disk shapes differ by orientation; we transpose so the slab (16) axis
    is always last, then center-crop to `patch_shape`.
    """
    arr = nib.load(path).get_fdata().astype(np.float32)
    if orientation == "coronal":
        # (192, 16, 192) -> (192, 192, 16)
        arr = np.moveaxis(arr, 1, -1)
    elif orientation == "sagittal":
        # (16, 192, 192) -> (192, 192, 16)
        arr = np.moveaxis(arr, 0, -1)
    arr = _center_crop(arr, patch_shape)
    return torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)


def _read_meta(subject_dir: str, subject_name: str):
    """Read sequence params from <base_subject>_params.json.

    For oriented dirs (e.g. `1041_coronal/`), params still live under the
    base id (`1041_params.json`). Returns
    (vx, vy, vz, TR_s, TE_s, TI_s, FlipAngle_deg).
    """
    base, _ = _parse_subject(subject_name)
    params_path = os.path.join(subject_dir, f"{base}_params.json")
    with open(params_path) as f:
        meta = json.load(f)
    vx, vy, vz = meta["voxel_size_mm"]
    return (
        float(vx), float(vy), float(vz),
        float(meta["RepetitionTime"]),
        float(meta["EchoTime"]),
        float(meta["InversionTime"]),
        float(meta["FlipAngle"]),
    )


def encode_context_vec(vx, vy, vz, tr, te, ti, fa):
    """Build the 7-dim context vector with log1p TR/TE/TI and FA/180."""
    return torch.tensor(
        [
            float(vx), float(vy), float(vz),
            math.log1p(float(tr)),
            math.log1p(float(te)),
            math.log1p(float(ti)),
            float(fa) / 180.0,
        ],
        dtype=torch.float32,
    )


def build_augmentation(
    noise_std=(0.0, 0.1),
    ghost_num=(2, 5),
    ghost_intensity=(0.3, 0.7),
    ghost_axes=(0, 1),
    p_noise=0.5,
    p_ghost=0.5,
):
    """Build a TorchIO transform that adds random MR noise + ghosting.

    Operates on a single (C, W, H, D) float tensor (the canonical patch layout
    here is (1, *patch_shape)) and returns one of the same shape. Intended to
    be applied to the *condition* (undersampled input) only, so the GT target
    stays a clean reconstruction goal.

    Args:
        noise_std: std range for `RandomNoise` (sigma ~ U(a, b)). Noise is in
            the same intensity units as the patch, so keep this small relative
            to your data's intensity range.
        ghost_num: range for the number of ghosts (n ~ U(a, b)).
        ghost_intensity: artifact-strength range relative to k-space max
            (s ~ U(a, b)).
        ghost_axes: spatial axis/axes (of W, H, D = 0, 1, 2) along which
            ghosts may appear; one is chosen at random per sample. The slab
            (D=2) axis is excluded by default since ghosting along a 16-slice
            slab is rarely meaningful.
        p_noise, p_ghost: per-sample probability of applying each transform.

    Returns:
        A `torchio.Transform` (Compose). Pass `None` for either probability's
        transform to be skipped by setting the corresponding `p_*` to 0.
    """
    transforms = []
    if p_noise > 0:
        transforms.append(
            tio.RandomNoise(mean=0.0, std=noise_std, p=p_noise)
        )
    if p_ghost > 0:
        transforms.append(
            tio.RandomGhosting(
                num_ghosts=ghost_num,
                axes=ghost_axes,
                intensity=ghost_intensity,
                p=p_ghost,
            )
        )
    return tio.Compose(transforms) if transforms else None


class PatchPairDataset(Dataset):
    """One sample = (undersampled patch, GT patch, context_vec).

    Each volume tensor is shape (1, *patch_shape) (default (1, 192, 192, 16)).
    `context_vec` is a (7,) float32 tensor:
    (vx, vy, vz, log1p(TR), log1p(TE), log1p(TI), FA/180).

    `patch_shape` center-crops both the condition and target after they are
    canonicalized to (192, 192, 16). Both members of a pair share the same
    orientation and crop, so their spatial alignment is preserved.

    `augment` is an optional TorchIO transform (see `build_augmentation`)
    applied to the *condition only*. The GT target is never augmented so it
    remains a clean reconstruction goal.
    """

    def __init__(self, samples, gt_meta_by_subject, patch_shape=(192, 192, 16),
                 augment=None):
        self.samples = samples
        self.gt_meta = gt_meta_by_subject
        self.patch_shape = patch_shape
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        orientation = s.get("orientation", "axial")
        condition = _load_patch(s["condition_path"], orientation, self.patch_shape)
        target = _load_patch(s["target_path"], orientation, self.patch_shape)

        if self.augment is not None:
            condition = self.augment(condition)

        ctx_vec = encode_context_vec(*self.gt_meta[s["gt_subject"]])
        return condition, target, ctx_vec


def build_samples(data_root: str, split: str, contrast: str):
    """Enumerate (undersampled, GT) patch pairs for one split + contrast.

    Returns:
        samples:  list of dicts (see PatchPairDataset.__init__).
        gt_meta:  dict gt_subject -> (vx, vy, vz, TR, TE, TI, FA).
    """
    contrast_dir = os.path.join(data_root, split, contrast)
    if not os.path.isdir(contrast_dir):
        raise FileNotFoundError(f"Contrast directory not found: {contrast_dir}")

    all_subjects = sorted(
        d for d in os.listdir(contrast_dir)
        if os.path.isdir(os.path.join(contrast_dir, d))
    )
    gt_subjects = [s for s in all_subjects if _is_gt_subject(s)]

    samples = []
    gt_meta = {}
    for gt in gt_subjects:
        gt_dir = os.path.join(contrast_dir, gt)
        gt_patches = _index_subject_patches(gt_dir)
        if not gt_patches:
            continue
        try:
            gt_meta[gt] = _read_meta(gt_dir, gt)
        except (FileNotFoundError, KeyError, TypeError):
            # Skip subjects missing required sequence params.
            continue

        _, orientation = _parse_subject(gt)
        siblings = _find_undersampled_siblings(contrast_dir, gt)
        for sib in siblings:
            sib_dir = os.path.join(contrast_dir, sib)
            sib_patches = _index_subject_patches(sib_dir)
            for (x, y, z), us_path in sib_patches.items():
                if (x, y, z) not in gt_patches:
                    continue
                samples.append({
                    "condition_path": us_path,
                    "target_path": gt_patches[(x, y, z)],
                    "gt_subject": gt,
                    "orientation": orientation,
                })

    return samples, gt_meta


def _subsample(samples, percent: float, seed: int = 42):
    """Return a deterministic random subset of `samples` (percent in [0, 100])."""
    if percent >= 100:
        return samples
    rng = random.Random(seed)
    n = max(1, int(len(samples) * percent / 100.0))
    return rng.sample(samples, n)


def get_dataset_3d_patches(data_root: str, contrast, sample: float = 100.0,
                           patch_shape=(192, 192, 16), augment=None):
    """Build train + val datasets for one or more contrasts.

    `contrast` may be a single string or a list/tuple of strings. When multiple
    contrasts are given, samples are concatenated and subject keys are
    namespaced as `<contrast>/<subject>` to avoid cross-contrast collisions.

    `patch_shape` is forwarded to both datasets and center-crops each loaded
    patch (default (192, 192, 16) = no crop). Use e.g. (96, 96, 16) for fast
    debugging.

    `augment` is an optional TorchIO transform (see `build_augmentation`)
    applied to the *condition only*, on both the train and val datasets.

    The returned datasets carry a `contrast_indices` attribute mapping each
    contrast name to the list of dataset indices belonging to that contrast,
    so a balanced sampler can draw evenly across contrasts.

    Train comes from `train/`, validation from `test/`.
    """
    contrasts = [contrast] if isinstance(contrast, str) else list(contrast)
    multi = len(contrasts) > 1

    train_samples_all, val_samples_all = [], []
    train_meta_all = {}
    val_meta_all = {}
    train_contrast_indices, val_contrast_indices = {}, {}

    for c in contrasts:
        tr_samples, tr_meta = build_samples(data_root, "train", c)
        v_samples, v_meta = build_samples(data_root, "test", c)

        tr_samples = _subsample(tr_samples, sample, seed=42)
        v_samples = _subsample(v_samples, min(sample, 5.0), seed=43)

        if multi:
            for s in tr_samples:
                s["gt_subject"] = f"{c}/{s['gt_subject']}"
            for s in v_samples:
                s["gt_subject"] = f"{c}/{s['gt_subject']}"
            tr_meta = {f"{c}/{k}": v for k, v in tr_meta.items()}
            v_meta = {f"{c}/{k}": v for k, v in v_meta.items()}

        train_contrast_indices[c] = list(
            range(len(train_samples_all), len(train_samples_all) + len(tr_samples))
        )
        val_contrast_indices[c] = list(
            range(len(val_samples_all), len(val_samples_all) + len(v_samples))
        )

        train_samples_all.extend(tr_samples)
        val_samples_all.extend(v_samples)
        train_meta_all.update(tr_meta)
        val_meta_all.update(v_meta)

    train_set = PatchPairDataset(train_samples_all, train_meta_all,
                                 patch_shape=patch_shape, augment=augment)
    val_set = PatchPairDataset(val_samples_all, val_meta_all,
                               patch_shape=patch_shape, augment=augment)
    train_set.contrast_indices = train_contrast_indices
    val_set.contrast_indices = val_contrast_indices
    return train_set, val_set


class BalancedDistributedSampler(Sampler):
    """Yields an equal number of samples per contrast each epoch.

    Each epoch reseeds via `seed + epoch`, so the larger contrast cycles
    through different random subsets across epochs. Compatible with DDP via
    rank/num_replicas slicing (set `num_replicas=1, rank=0` for non-DDP).

    If `samples_per_contrast` exceeds a contrast's pool, that contrast is
    oversampled with replacement to reach the target.
    """

    def __init__(self, contrast_indices, samples_per_contrast=0,
                 num_replicas=1, rank=0, shuffle=True, seed=42):
        if not contrast_indices:
            raise ValueError("contrast_indices is empty")
        self.contrast_indices = {k: list(v) for k, v in contrast_indices.items() if v}
        if samples_per_contrast is None or samples_per_contrast <= 0:
            samples_per_contrast = min(len(v) for v in self.contrast_indices.values())
        self.samples_per_contrast = samples_per_contrast
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        n_total = samples_per_contrast * len(self.contrast_indices)
        self.num_samples = math.ceil(n_total / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = []
        for idxs in self.contrast_indices.values():
            n = self.samples_per_contrast
            pool = torch.tensor(idxs)
            if n <= len(idxs):
                perm = torch.randperm(len(idxs), generator=g)[:n]
                chosen = pool[perm].tolist()
            else:
                extra_n = n - len(idxs)
                extra = pool[torch.randint(0, len(idxs), (extra_n,), generator=g)]
                chosen = pool.tolist() + extra.tolist()
            indices.extend(chosen)

        if self.shuffle:
            order = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in order]

        if len(indices) < self.total_size:
            indices += indices[: self.total_size - len(indices)]
        else:
            indices = indices[: self.total_size]

        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def getloader_3d_patches(
    batch_size: int,
    data_root: str,
    contrast,
    sample: float = 100.0,
    num_workers: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    train_shuffle: bool = True,
    samples_per_contrast=None,
    patch_shape=(192, 192, 16),
    augment=None,
    augment_kwargs=None,
):
    """DataLoaders for 3D NIfTI patch reconstruction. DDP-aware.

    `contrast` may be a single string or a list/tuple of strings.

    `patch_shape` center-crops each loaded patch after canonicalization to
    (192, 192, 16). Default (192, 192, 16) = no crop; pass e.g. (96, 96, 16)
    for fast model debugging.

    Augmentation (random MR noise + ghosting) is applied to the *condition
    only*, on both train and val:
      - `augment`: pass a ready-made TorchIO transform to use it directly.
      - `augment_kwargs`: pass a dict of `build_augmentation` kwargs (e.g.
        `{"noise_std": (0, 0.05), "p_ghost": 0.3}`) to build one here.
      - If both are None (default), no augmentation is applied.
    `augment` takes precedence if both are given.

    If `samples_per_contrast` is not None, training uses
    `BalancedDistributedSampler`: each contrast contributes the same number of
    samples per epoch (0 = auto-balance to the smallest contrast). The larger
    contrast(s) cycle through different random subsets across epochs.
    Validation always uses standard sampling.
    """
    if augment is None and augment_kwargs is not None:
        augment = build_augmentation(**augment_kwargs)

    train_set, val_set = get_dataset_3d_patches(
        data_root, contrast, sample, patch_shape, augment=augment
    )

    use_balanced = samples_per_contrast is not None

    if use_balanced:
        train_sampler = BalancedDistributedSampler(
            train_set.contrast_indices,
            samples_per_contrast=samples_per_contrast,
            num_replicas=world_size if distributed else 1,
            rank=rank if distributed else 0,
            shuffle=train_shuffle,
            seed=42,
        )
    elif distributed:
        train_sampler = DistributedSampler(
            train_set, num_replicas=world_size, rank=rank,
            shuffle=train_shuffle, seed=42,
        )
    else:
        train_sampler = None

    if distributed:
        val_sampler = DistributedSampler(
            val_set, num_replicas=world_size, rank=rank,
            shuffle=False, seed=42,
        )
        train_loader = DataLoader(
            train_set, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, sampler=val_sampler,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None) and train_shuffle,
            num_workers=num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

    return train_loader, val_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data_root = "/home/rflab/jil202/grappa-recon/dataset_grappa_nii"
    contrast = "mprage"

    print(f"Building datasets for contrast={contrast} ...")
    train_loader, val_loader = getloader_3d_patches(
        batch_size=1, data_root=data_root, contrast=contrast,
        sample=0.5, num_workers=0, patch_shape=(96, 96, 16),
        augment_kwargs={
            "noise_std": (0.0, 0.05),
            "ghost_num": (2, 5),
            "ghost_intensity": (0.3, 0.7),
            "ghost_axes": (0, 1),
            "p_noise": 0.8,
            "p_ghost": 0.8,
        },
    )
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    for condition, target, ctx_vec in train_loader:
        print(
            f"condition  {tuple(condition.shape)}  range "
            f"[{condition.min():.4f}, {condition.max():.4f}]\n"
            f"target     {tuple(target.shape)}  range "
            f"[{target.min():.4f}, {target.max():.4f}]\n"
            f"ctx_vec    {tuple(ctx_vec.shape)}  values {ctx_vec.tolist()}"
        )
        c = condition[0, 0, :, :, condition.shape[-1] // 2].cpu().numpy()
        t = target[0, 0, :, :, target.shape[-1] // 2].cpu().numpy()
        plt.figure(figsize=(8, 4))
        for i, (img, name) in enumerate([(c, "condition"), (t, "target")]):
            plt.subplot(1, 2, i + 1)
            plt.imshow(img, cmap="gray")
            plt.title(name)
            plt.axis("off")
        plt.tight_layout()
        plt.savefig("dataset_smoketest.png")
        plt.close()
        print("Saved dataset_smoketest.png")
        break