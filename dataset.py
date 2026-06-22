"""3D NIfTI patch dataset for flow matching (multi-task restoration).

Builds (condition, target, prompt) patch pairs for three restoration tasks
inside a single acquisition:

  1. artifact  -> raw (fully sampled, unprocessed) image
  2. artifact  -> denoised + bias-corrected image
  3. raw       -> denoised + bias-corrected image

The *target's* text prompt is used as the cross-attention context, so the
prompt tells the model which output to produce ("Fully sampled ..." vs
"Denoised and biascorrected ...").

Layout assumed:
  <data_root>/<split>/<anatomy>/<acquisition>/<subject>/patch_<x>_<y>_<z>.nii.gz
where:
  * <anatomy>      is one of {brain, knee, prostate}.
  * <acquisition>  is a sequence/orientation/field folder, e.g. `mprage_ax_3T`,
                   `pd_cor_1.5T`, `tse_ax_3T`.
  * <subject>      is the *raw* (fully sampled, unprocessed) id, e.g. `2033AM`.
                   Its prompt sidecar `<subject>.json` has `processing: raw`.
  * `md<subject>`  is the denoised + bias-corrected counterpart; its sidecar
                   `md<subject>.json` has `processing: denoised+biascorrected`.
  * `<subject>_R<n>`, `<subject>_SPIKE_R<n>`,
    `<subject>_ANISO_{phase,read,par}<n>` are the artifact siblings. They carry
    the same patch coordinates but only a `patches_meta.json` (no prompt).

Only the raw and denoised dirs carry a `<dir>.json` prompt sidecar; artifact
dirs do not. Each prompt sidecar stores the ready-made text under the `prompt`
key (no more on-the-fly param-to-text rendering).

All patches are stored canonical `(192, 192, 16)` on disk, so loading is a
plain center-crop to the requested `patch_shape` (default no crop). A smaller
`patch_shape` such as `(96, 96, 16)` yields a genuine sub-volume at the same
resolution -- useful for quick model debugging.
"""

import glob
import json
import math
import os
import random
import re

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

import torchio as tio


_PATCH_RE = re.compile(r"patch_(\d+)_(\d+)_(\d+)\.nii\.gz$")
# Artifact dir suffix appended to a raw subject id, e.g. `_R3`, `_SPIKE_R4`,
# `_ANISO_phase3.5`, `_ANISO_read4`, `_ANISO_par3`.
_ARTIFACT_SUFFIX_RE = re.compile(
    r"_(?:R\d+|SPIKE_R\d+|ANISO_(?:phase|read|par)\d+(?:\.\d+)?)$"
)
_DENOISED_PREFIX = "md"
_ANATOMIES = ("brain", "knee", "prostate")
# Coarse artifact groups used for balanced (anatomy x artifact) validation.
ARTIFACT_GROUPS = ("undersample", "spike", "aniso")


def _artifact_group(suffix: str) -> str:
    """Classify an artifact dir suffix (e.g. `_R3`, `_SPIKE_R2`, `_ANISO_phase4`)
    into one of ARTIFACT_GROUPS; returns 'other' if unrecognized."""
    if suffix.startswith("_SPIKE_R"):
        return "spike"
    if suffix.startswith("_ANISO_"):
        return "aniso"
    if re.match(r"_R\d", suffix):
        return "undersample"
    return "other"


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


def _load_patch(path: str, patch_shape=(192, 192, 16)) -> torch.Tensor:
    """Load a canonical (192, 192, 16) NIfTI patch as float32 (1, *patch_shape)."""
    arr = nib.load(path).get_fdata().astype(np.float32)
    arr = _center_crop(arr, patch_shape)
    return torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)


def _read_prompt(subject_dir: str, subject_name: str):
    """Read (prompt, processing) from a subject's `<subject>.json` sidecar.

    The newer datasets store the ready-made text prompt directly under the
    `prompt` key; `processing` is one of {"raw", "denoised+biascorrected"}.
    """
    with open(os.path.join(subject_dir, f"{subject_name}.json")) as f:
        meta = json.load(f)
    return meta["prompt"], meta.get("processing", "")


def _has_prompt(subject_dir: str, subject_name: str) -> bool:
    return os.path.exists(os.path.join(subject_dir, f"{subject_name}.json"))


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
    be applied to the *condition* (input) only, so the target stays a clean
    reconstruction goal.

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
    """One sample = (condition patch, target patch, prompt).

    Each volume tensor is shape (1, *patch_shape) (default (1, 192, 192, 16)).
    `prompt` is the target's ready-made text description (see the sidecar
    `prompt` key), e.g. "Fully sampled coronal 1.5T knee Proton Density MRI ..."
    or "Denoised and biascorrected ...". Tokenization happens in the training
    script, keeping this module free of any text-encoder dependency.

    `patch_shape` center-crops both the condition and target after they are
    loaded. Both members of a pair share the same patch coordinate so their
    spatial alignment is preserved.

    `augment` is an optional TorchIO transform (see `build_augmentation`)
    applied to the *condition only*. The target is never augmented so it
    remains a clean restoration goal.
    """

    def __init__(self, samples, patch_shape=(192, 192, 16), augment=None):
        self.samples = samples
        self.patch_shape = patch_shape
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        condition = _load_patch(s["condition_path"], self.patch_shape)
        target = _load_patch(s["target_path"], self.patch_shape)

        if self.augment is not None:
            condition = self.augment(condition)

        # target_type names the prompt capability this pair exercises:
        # "raw" = "Fully sampled ..." target, "denoised" = "Denoised and
        # biascorrected ..." target. anatomy/artifact let validation bucket
        # metrics into balanced (anatomy x artifact-group) cells.
        target_type = "raw" if s["task"] == "artifact2raw" else "denoised"
        anatomy = s.get("anatomy", "")
        artifact = s.get("artifact", "none")
        return condition, target, s["prompt"], target_type, anatomy, artifact


def _find_artifact_siblings(subjects, raw_name: str):
    """Artifact dirs of `raw_name`: `<raw_name>_<artifact-suffix>`."""
    out = []
    for d in subjects:
        if not d.startswith(raw_name):
            continue
        rest = d[len(raw_name):]
        if _ARTIFACT_SUFFIX_RE.fullmatch(rest):
            out.append(d)
    return sorted(out)


def build_samples(data_root: str, split: str, anatomy: str):
    """Enumerate (condition, target, prompt) patch pairs for one anatomy.

    Iterates every acquisition folder under `<data_root>/<split>/<anatomy>/`,
    pairing each raw subject's artifact / raw / denoised patches across the
    three restoration tasks (see module docstring).

    Returns:
        samples: list of dicts {condition_path, target_path, prompt, task}.
    """
    anatomy_dir = os.path.join(data_root, split, anatomy)
    if not os.path.isdir(anatomy_dir):
        raise FileNotFoundError(f"Anatomy directory not found: {anatomy_dir}")

    samples = []
    acquisitions = sorted(
        d for d in os.listdir(anatomy_dir)
        if os.path.isdir(os.path.join(anatomy_dir, d))
    )

    for acq in acquisitions:
        acq_dir = os.path.join(anatomy_dir, acq)
        subjects = sorted(
            d for d in os.listdir(acq_dir)
            if os.path.isdir(os.path.join(acq_dir, d))
        )
        subj_set = set(subjects)

        for name in subjects:
            sub_dir = os.path.join(acq_dir, name)
            # Raw GT subjects own the pairing; skip denoised/artifact dirs here.
            if name.startswith(_DENOISED_PREFIX) or not _has_prompt(sub_dir, name):
                continue
            try:
                raw_prompt, processing = _read_prompt(sub_dir, name)
            except (KeyError, json.JSONDecodeError):
                continue
            if processing != "raw":
                continue

            raw_patches = _index_subject_patches(sub_dir)
            if not raw_patches:
                continue

            # Optional denoised + bias-corrected counterpart `md<subject>`.
            den_name = f"{_DENOISED_PREFIX}{name}"
            den_patches, den_prompt = {}, None
            if den_name in subj_set:
                den_dir = os.path.join(acq_dir, den_name)
                if _has_prompt(den_dir, den_name):
                    try:
                        den_prompt, _ = _read_prompt(den_dir, den_name)
                        den_patches = _index_subject_patches(den_dir)
                    except (KeyError, json.JSONDecodeError):
                        den_patches, den_prompt = {}, None

            # Tasks 1 & 2: artifact -> raw, artifact -> denoised.
            for art in _find_artifact_siblings(subjects, name):
                group = _artifact_group(art[len(name):])
                art_patches = _index_subject_patches(os.path.join(acq_dir, art))
                for coord, art_path in art_patches.items():
                    if coord in raw_patches:
                        samples.append({
                            "condition_path": art_path,
                            "target_path": raw_patches[coord],
                            "prompt": raw_prompt,
                            "task": "artifact2raw",
                            "anatomy": anatomy,
                            "artifact": group,
                        })
                    if den_prompt is not None and coord in den_patches:
                        samples.append({
                            "condition_path": art_path,
                            "target_path": den_patches[coord],
                            "prompt": den_prompt,
                            "task": "artifact2denoised",
                            "anatomy": anatomy,
                            "artifact": group,
                        })

            # Task 3: raw -> denoised.
            if den_prompt is not None:
                for coord, raw_path in raw_patches.items():
                    if coord in den_patches:
                        samples.append({
                            "condition_path": raw_path,
                            "target_path": den_patches[coord],
                            "prompt": den_prompt,
                            "task": "raw2denoised",
                            "anatomy": anatomy,
                            "artifact": "none",
                        })

    return samples


def _subsample(samples, percent: float, seed: int = 42):
    """Return a deterministic random subset of `samples` (percent in [0, 100])."""
    if percent >= 100:
        return samples
    rng = random.Random(seed)
    n = max(1, int(len(samples) * percent / 100.0))
    return rng.sample(samples, n)


def build_balanced_val_indices(samples, per_cell, seed=43,
                               groups=ARTIFACT_GROUPS):
    """Indices of `samples` balanced across (anatomy, artifact-group) cells.

    Only artifact samples whose `artifact` is in `groups` are eligible
    (raw->denoised pairs with artifact 'none', and any 'other', are excluded).
    Each (anatomy, group) cell contributes `per_cell` indices, sampled without
    replacement; a cell with fewer than `per_cell` pairs contributes all of
    them. `per_cell` None/<=0 balances every cell to the smallest non-empty
    cell. Both target types (artifact->raw and artifact->denoised) live in a
    cell and are pooled, so they are drawn in roughly equal proportion.
    """
    cells = {}
    for idx, s in enumerate(samples):
        g = s.get("artifact", "none")
        if g not in groups:
            continue
        cells.setdefault((s.get("anatomy", ""), g), []).append(idx)
    if not cells:
        return list(range(len(samples)))
    rng = random.Random(seed)
    if per_cell is None or per_cell <= 0:
        per_cell = min(len(v) for v in cells.values())
    out = []
    for key in sorted(cells):
        pool = cells[key]
        out += rng.sample(pool, per_cell) if per_cell <= len(pool) else list(pool)
    return out


def get_dataset_3d_patches(data_root: str, contrast, sample: float = 100.0,
                           patch_shape=(192, 192, 16), augment=None,
                           val_balanced_per_cell=None, cell_buckets=False):
    """Build train + val datasets for one or more anatomies.

    `val_balanced_per_cell` (if not None) replaces the default 5% val subsample
    with a set balanced across (anatomy x artifact-group) cells: each cell
    contributes this many pairs (0 = balance to the smallest cell). raw->denoised
    pairs are excluded from the balanced val set. See `build_balanced_val_indices`.

    `cell_buckets` controls how the *train* sampler buckets are keyed in the
    returned `train_set.contrast_indices`: when False (default) buckets are
    keyed by anatomy name (legacy); when True they are keyed by
    `(anatomy, artifact-group)` tuples, with the three artifact groups plus a
    `(anatomy, "none")` bucket for raw->denoised pairs, so a per-(anatomy x
    artifact) quota can be applied. Samples whose group is "other" are left
    out of every bucket (never sampled).

    `contrast` is the anatomy group (one of {brain, knee, prostate}) or a
    list/tuple of them. When multiple are given, samples are concatenated.

    `patch_shape` is forwarded to both datasets and center-crops each loaded
    patch (default (192, 192, 16) = no crop). Use e.g. (96, 96, 16) for fast
    debugging.

    `augment` is an optional TorchIO transform (see `build_augmentation`)
    applied to the *condition only*, on both the train and val datasets.

    The returned datasets carry a `contrast_indices` attribute mapping each
    anatomy name to the list of dataset indices belonging to it, so a balanced
    sampler can draw evenly across anatomies.

    Train comes from `train/`, validation from `test/`.
    """
    contrasts = [contrast] if isinstance(contrast, str) else list(contrast)

    train_samples_all, val_samples_all = [], []
    train_contrast_indices, val_contrast_indices = {}, {}

    for c in contrasts:
        tr_samples = build_samples(data_root, "train", c)
        v_samples = build_samples(data_root, "test", c)

        for split_name, split_samples in (("train", tr_samples), ("test", v_samples)):
            if not split_samples:
                raise ValueError(
                    f"No (condition, target) patch pairs found for anatomy "
                    f"'{c}' in {os.path.join(data_root, split_name, c)}. Check "
                    f"that raw subject dirs contain patch_*.nii.gz plus a "
                    f"<subject>.json prompt sidecar, and that artifact "
                    f"(_R*/_SPIKE_R*/_ANISO_*) and md<subject> dirs exist."
                )

        tr_samples = _subsample(tr_samples, sample, seed=42)
        # When the val set is to be cell-balanced, keep the full test pool here
        # and balance across all anatomies after the loop; otherwise take the
        # usual 5% per-anatomy subsample.
        if val_balanced_per_cell is None:
            v_samples = _subsample(v_samples, min(sample, 5.0), seed=43)

        val_contrast_indices[c] = list(
            range(len(val_samples_all), len(val_samples_all) + len(v_samples))
        )

        train_samples_all.extend(tr_samples)
        val_samples_all.extend(v_samples)

    # Train sampler buckets: keyed by anatomy (legacy) or by
    # (anatomy, artifact-group) cell when cell_buckets is set.
    allowed_cell_groups = set(ARTIFACT_GROUPS) | {"none"}
    for i, s in enumerate(train_samples_all):
        if cell_buckets:
            g = s.get("artifact", "none")
            if g not in allowed_cell_groups:
                continue  # 'other' / unrecognized -> never sampled
            key = (s.get("anatomy", ""), g)
        else:
            key = s.get("anatomy", "")
        train_contrast_indices.setdefault(key, []).append(i)

    if val_balanced_per_cell is not None:
        val_idx = build_balanced_val_indices(val_samples_all, val_balanced_per_cell)
        val_samples_all = [val_samples_all[i] for i in val_idx]
        # Rebuild val contrast indices to match the rebalanced selection.
        val_contrast_indices = {}
        for i, s in enumerate(val_samples_all):
            val_contrast_indices.setdefault(s.get("anatomy", ""), []).append(i)

    train_set = PatchPairDataset(train_samples_all,
                                 patch_shape=patch_shape, augment=augment)
    val_set = PatchPairDataset(val_samples_all,
                               patch_shape=patch_shape, augment=augment)
    train_set.contrast_indices = train_contrast_indices
    val_set.contrast_indices = val_contrast_indices
    return train_set, val_set


class BalancedDistributedSampler(Sampler):
    """Yields an equal number of samples per anatomy each epoch.

    Each epoch reseeds via `seed + epoch`, so the larger anatomy cycles
    through different random subsets across epochs. Compatible with DDP via
    rank/num_replicas slicing (set `num_replicas=1, rank=0` for non-DDP).

    If a contrast's quota exceeds its pool, that anatomy is oversampled with
    replacement to reach the target.

    `samples_per_contrast` may be either:
      * an int (or 0/None) applied to every contrast; 0/None auto-balances to
        the smallest pool, or
      * a dict {contrast_name: count} giving a per-anatomy per-epoch quota, e.g.
        {"brain": 20000, "knee": 50000, "prostate": 50000}.
    """

    def __init__(self, contrast_indices, samples_per_contrast=0,
                 num_replicas=1, rank=0, shuffle=True, seed=42):
        if not contrast_indices:
            raise ValueError("contrast_indices is empty")
        self.contrast_indices = {k: list(v) for k, v in contrast_indices.items() if v}

        if isinstance(samples_per_contrast, dict):
            missing = set(self.contrast_indices) - set(samples_per_contrast)
            if missing:
                raise ValueError(
                    f"samples_per_contrast dict missing entries for {sorted(missing)}"
                )
            self.per_contrast = {
                k: int(samples_per_contrast[k]) for k in self.contrast_indices
            }
        else:
            if samples_per_contrast is None or samples_per_contrast <= 0:
                samples_per_contrast = min(len(v) for v in self.contrast_indices.values())
            self.per_contrast = {k: int(samples_per_contrast) for k in self.contrast_indices}

        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        n_total = sum(self.per_contrast.values())
        self.num_samples = math.ceil(n_total / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = []
        for name, idxs in self.contrast_indices.items():
            n = self.per_contrast[name]
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
    val_balanced_per_cell=None,
    cell_buckets=False,
):
    """DataLoaders for 3D NIfTI patch restoration. DDP-aware.

    `contrast` is the anatomy group (one of {brain, knee, prostate}) or a
    list/tuple of them.

    `patch_shape` center-crops each loaded patch (default (192, 192, 16) = no
    crop; pass e.g. (96, 96, 16) for fast model debugging).

    Augmentation (random MR noise + ghosting) is applied to the *condition
    only*, on both train and val:
      - `augment`: pass a ready-made TorchIO transform to use it directly.
      - `augment_kwargs`: pass a dict of `build_augmentation` kwargs (e.g.
        `{"noise_std": (0, 0.05), "p_ghost": 0.3}`) to build one here.
      - If both are None (default), no augmentation is applied.
    `augment` takes precedence if both are given.

    If `samples_per_contrast` is not None, training uses
    `BalancedDistributedSampler`: each bucket contributes the same number of
    samples per epoch (0 = auto-balance to the smallest bucket). The larger
    bucket(s) cycle through different random subsets across epochs. Buckets are
    anatomies by default, or (anatomy x artifact-group) cells when
    `cell_buckets` is set (see `get_dataset_3d_patches`); in cell mode
    `samples_per_contrast` must be a dict keyed by those `(anatomy, group)`
    tuples. Validation always uses standard sampling.

    `val_balanced_per_cell` (if not None) makes the validation set balanced
    across (anatomy x artifact-group) cells -- this many pairs per cell (0 =
    balance to the smallest cell) -- instead of the default 5% subsample, so
    per-epoch validation and checkpoint selection weight every anatomy and
    artifact equally.
    """
    if augment is None and augment_kwargs is not None:
        augment = build_augmentation(**augment_kwargs)

    train_set, val_set = get_dataset_3d_patches(
        data_root, contrast, sample, patch_shape, augment=augment,
        val_balanced_per_cell=val_balanced_per_cell, cell_buckets=cell_buckets,
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
    from collections import Counter

    data_root = "/vast/tibrahim/jil202/data"
    anatomy = "knee"

    print(f"Building datasets for anatomy={anatomy} ...")
    train_set, val_set = get_dataset_3d_patches(
        data_root, anatomy, sample=100.0, patch_shape=(192, 192, 16),
    )
    task_counts = Counter(s["task"] for s in train_set.samples)
    print(f"train pairs: {len(train_set)}, val pairs: {len(val_set)}")
    print(f"train task breakdown: {dict(task_counts)}")

    train_loader, _ = getloader_3d_patches(
        batch_size=1, data_root=data_root, contrast=anatomy,
        sample=1.0, num_workers=0, patch_shape=(96, 96, 16),
    )
    for condition, target, prompts, target_types, anatomies, artifacts in train_loader:
        print(
            f"condition  {tuple(condition.shape)}  range "
            f"[{condition.min():.4f}, {condition.max():.4f}]\n"
            f"target     {tuple(target.shape)}  range "
            f"[{target.min():.4f}, {target.max():.4f}]\n"
            f"target_type {target_types[0]!r}  anatomy {anatomies[0]!r}  "
            f"artifact {artifacts[0]!r}\n"
            f"prompt     {prompts[0]!r}"
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
