"""3D NIfTI patch dataset for flow matching (multi-task restoration).

Builds (condition, target, prompt) patch pairs for three restoration tasks
inside a single acquisition:

  1. artifact  -> raw (fully sampled, unprocessed) image
  2. artifact  -> denoised + bias-corrected image
  3. raw       -> denoised + bias-corrected image

The prompt used as cross-attention context is composed here as
"Input: <input description> <anatomy> MRI. Target: <target description>.",
e.g. "Input: grappa undersampled brain MRI. Target: Fully sampled axial 3T
brain T1-weighted BRAVO MRI of resolution 0.86 x 0.86 x 1.2 mm." The target
half comes verbatim from the target's `<subject>.json` sidecar `prompt` key;
the input half is derived from the artifact family (see `_input_phrase`).

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
dirs do not. Each prompt sidecar stores the ready-made *target* text under
the `prompt` key; the full "Input: ... Target: ..." prompt is assembled in
`build_samples` (see `_input_phrase`).

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


_ARTIFACT_INPUT_PHRASE = {
    "spike": "spiking artifact",
    "undersampled": "grappa undersampled",
    "aniso": "unisotropic undersampled",
}


def _input_phrase(family: str) -> str:
    """Phrase for the "Input: ..." half of a sample's prompt, from its
    `_artifact_family` result. The three ANISO k-space-direction variants
    (phase/read/par) all collapse to the same "unisotropic undersampled"
    phrase."""
    return _ARTIFACT_INPUT_PHRASE.get(family, "artifact")


_DENOISED_PREFIX = "md"


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
    """One sample = (condition, target, prompt, target_type, anatomy, artifact).

    Each volume tensor is shape (1, *patch_shape) (default (1, 192, 192, 16)).
    `prompt` is the composed "Input: ... Target: ..." text description (see
    `build_samples` / `_input_phrase`), e.g. "Input: grappa undersampled knee
    MRI. Target: Fully sampled coronal 1.5T knee Proton Density MRI ..." or
    "Input: fully sampled knee MRI. Target: Denoised and biascorrected ...".
    Tokenization happens in the training script, keeping this module free of
    any text-encoder dependency.

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
        # biascorrected ..." target. Used by validation to score both equally.
        target_type = "raw" if s["task"] == "artifact2raw" else "denoised"
        return condition, target, s["prompt"], target_type, s["anatomy"], s["artifact"]


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
                art_family = _artifact_family(art[len(name):])
                input_phrase = _input_phrase(art_family)
                art_patches = _index_subject_patches(os.path.join(acq_dir, art))
                for coord, art_path in art_patches.items():
                    if coord in raw_patches:
                        samples.append({
                            "condition_path": art_path,
                            "target_path": raw_patches[coord],
                            "prompt": f"Input: {input_phrase} {anatomy} MRI. "
                                      f"Target: {raw_prompt}",
                            "task": "artifact2raw",
                            "anatomy": anatomy,
                            "artifact": art_family,
                        })
                    if den_prompt is not None and coord in den_patches:
                        samples.append({
                            "condition_path": art_path,
                            "target_path": den_patches[coord],
                            "prompt": f"Input: {input_phrase} {anatomy} MRI. "
                                      f"Target: {den_prompt}",
                            "task": "artifact2denoised",
                            "anatomy": anatomy,
                            "artifact": art_family,
                        })

            # Task 3: raw -> denoised.
            if den_prompt is not None:
                for coord, raw_path in raw_patches.items():
                    if coord in den_patches:
                        samples.append({
                            "condition_path": raw_path,
                            "target_path": den_patches[coord],
                            "prompt": f"Input: fully sampled {anatomy} MRI. "
                                      f"Target: {den_prompt}",
                            "task": "raw2denoised",
                            "anatomy": anatomy,
                            "artifact": "clean",
                        })

    return samples


def _subsample(samples, percent: float, seed: int = 42):
    """Return a deterministic random subset of `samples` (percent in [0, 100])."""
    if percent >= 100:
        return samples
    rng = random.Random(seed)
    n = max(1, int(len(samples) * percent / 100.0))
    return rng.sample(samples, n)


def _filter_artifacts(samples, artifacts):
    """Keep samples whose artifact family is selected, plus all clean pairs.

    `artifacts` is an iterable of artifact families to keep (a subset of
    {undersampled, spike, aniso}). The clean `raw->denoised` task (artifact
    == "clean") is always retained regardless of the filter, so a specialized
    model still learns pure denoising alongside artifact removal. Pass
    `artifacts=None` to disable filtering entirely.
    """
    if artifacts is None:
        return samples
    allowed = set(artifacts)
    return [s for s in samples
            if s["artifact"] in allowed or s["artifact"] == "clean"]


def get_dataset_3d_patches(data_root: str, contrast, sample: float = 100.0,
                           patch_shape=(192, 192, 16), augment=None,
                           artifacts=None):
    """Build train + val datasets for one or more anatomies.

    `contrast` is the anatomy group (one of {brain, knee, prostate}) or a
    list/tuple of them. When multiple are given, samples are concatenated.

    `patch_shape` is forwarded to both datasets and center-crops each loaded
    patch (default (192, 192, 16) = no crop). Use e.g. (96, 96, 16) for fast
    debugging.

    `augment` is an optional TorchIO transform (see `build_augmentation`)
    applied to the *condition only*, on both the train and val datasets.

    `artifacts` optionally restricts which artifact families are used (a subset
    of {undersampled, spike, aniso}); the clean `raw->denoised` pairs are always
    kept. `None` (default) keeps every artifact. Applied to both splits, so a
    specialized model is validated on the same artifacts it trains on (+ clean).

    The returned datasets carry two index maps for the balanced sampler:
      * `contrast_indices` -- keyed by anatomy name, for balancing across
        anatomies (the default).
      * `group_indices` -- keyed by the `(anatomy, artifact)` pair, for
        balancing across every anatomy x artifact group (e.g. equal numbers of
        undersampled-brain, undersampled-knee, spike-brain, ...).

    Train comes from `train/`, validation from `test/`.
    """
    contrasts = [contrast] if isinstance(contrast, str) else list(contrast)

    train_samples_all, val_samples_all = [], []
    train_contrast_indices, val_contrast_indices = {}, {}

    for c in contrasts:
        tr_samples = _filter_artifacts(build_samples(data_root, "train", c), artifacts)
        v_samples = _filter_artifacts(build_samples(data_root, "test", c), artifacts)

        for split_name, split_samples in (("train", tr_samples), ("test", v_samples)):
            if not split_samples:
                art_note = (f" with --artifact {sorted(artifacts)}"
                            if artifacts is not None else "")
                raise ValueError(
                    f"No (condition, target) patch pairs found for anatomy "
                    f"'{c}'{art_note} in {os.path.join(data_root, split_name, c)}. "
                    f"Check that raw subject dirs contain patch_*.nii.gz plus a "
                    f"<subject>.json prompt sidecar, and that artifact "
                    f"(_R*/_SPIKE_R*/_ANISO_*) and md<subject> dirs exist. "
                    f"If filtering by artifact, this anatomy may not carry the "
                    f"requested family."
                )

        tr_samples = _subsample(tr_samples, sample, seed=42)
        v_samples = _subsample(v_samples, min(sample, 5.0), seed=43)

        train_contrast_indices[c] = list(
            range(len(train_samples_all), len(train_samples_all) + len(tr_samples))
        )
        val_contrast_indices[c] = list(
            range(len(val_samples_all), len(val_samples_all) + len(v_samples))
        )

        train_samples_all.extend(tr_samples)
        val_samples_all.extend(v_samples)

    def _group_indices(samples):
        """Map (anatomy, artifact) -> list of dataset indices."""
        groups = {}
        for i, s in enumerate(samples):
            groups.setdefault((s["anatomy"], s["artifact"]), []).append(i)
        return groups

    train_set = PatchPairDataset(train_samples_all,
                                 patch_shape=patch_shape, augment=augment)
    val_set = PatchPairDataset(val_samples_all,
                               patch_shape=patch_shape, augment=augment)
    train_set.contrast_indices = train_contrast_indices
    val_set.contrast_indices = val_contrast_indices
    train_set.group_indices = _group_indices(train_samples_all)
    val_set.group_indices = _group_indices(val_samples_all)
    return train_set, val_set


class BalancedDistributedSampler(Sampler):
    """Yields an equal number of samples per group each epoch.

    A "group" is whatever the keys of `contrast_indices` are: anatomy names
    for cross-anatomy balancing, or `(anatomy, artifact)` pairs for
    cross-(anatomy x artifact) balancing. `samples_per_contrast` is the
    per-group quota.

    Each epoch reseeds via `seed + epoch`, so larger groups cycle through
    different random subsets across epochs. Compatible with DDP via
    rank/num_replicas slicing (set `num_replicas=1, rank=0` for non-DDP).

    If `samples_per_contrast` exceeds a group's pool, that group is
    oversampled with replacement to reach the target.
    """

    def __init__(self, contrast_indices, samples_per_contrast=0,
                 num_replicas=1, rank=0, shuffle=True, seed=42):
        if not contrast_indices:
            raise ValueError("contrast_indices is empty")
        self.contrast_indices = {k: list(v) for k, v in contrast_indices.items() if v}
        # `samples_per_contrast` may be a scalar (same quota for every group) or
        # a dict {group_key: quota} for per-group quotas, e.g. an up-weighted
        # aniso group (see `_build_group_quota`). A scalar <= 0 / None
        # auto-balances to the smallest group.
        if isinstance(samples_per_contrast, dict):
            self.quota = {k: int(samples_per_contrast[k])
                          for k in self.contrast_indices}
        else:
            if samples_per_contrast is None or samples_per_contrast <= 0:
                samples_per_contrast = min(len(v) for v in self.contrast_indices.values())
            self.quota = {k: int(samples_per_contrast) for k in self.contrast_indices}
        self.samples_per_contrast = samples_per_contrast
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        n_total = sum(self.quota.values())
        self.num_samples = math.ceil(n_total / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = []
        for key, idxs in self.contrast_indices.items():
            n = self.quota[key]
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


def parse_anatomy_weight(tokens):
    """Parse ['brain=0.1', 'knee=0.45', ...] -> {anatomy: float}.

    Each token is `anatomy=fraction`. Each fraction must be in (0, 1] and the
    set must sum to 1.0 (within 1e-6) -- they are target epoch fractions, not
    relative weights. Returns the dict, or None if `tokens` is empty/None.
    """
    if not tokens:
        return None
    weights = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(
                f"--anatomy_weight expects anatomy=fraction tokens, got {tok!r}"
            )
        anatomy, frac = tok.split("=", 1)
        anatomy = anatomy.strip()
        try:
            frac = float(frac)
        except ValueError:
            raise ValueError(f"anatomy_weight fraction is not a number: {tok!r}")
        if not (0.0 < frac <= 1.0):
            raise ValueError(
                f"anatomy_weight fraction must be in (0, 1], got {frac} for {anatomy!r}"
            )
        if anatomy in weights:
            raise ValueError(f"anatomy_weight has duplicate anatomy {anatomy!r}")
        weights[anatomy] = frac
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"anatomy_weight fractions must sum to 1.0, got {total} ({weights})"
        )
    return weights


def _check_frac(frac, tok):
    if not (0.0 < frac < 1.0):
        raise ValueError(
            f"artifact fraction must be in (0, 1), got {frac} in {tok!r}")
    return frac


def parse_artifact_fraction(tokens):
    """Parse --artifact_fraction tokens into `(family, frac)`.

    `tokens[0]` is the family. Remaining tokens are a bare float (the shared
    default fraction, at most one) and/or `anatomy=fraction` overrides:

        ['aniso', '0.7']                   -> ('aniso', 0.7)            # scalar
        ['aniso', '0.7', 'brain=0.1']      -> ('aniso', {'default':0.7,'brain':0.1})
        ['aniso', 'brain=0.1', 'knee=0.4'] -> ('aniso', {'brain':0.1,'knee':0.4})

    Returns None if `tokens` is empty/None. Anatomies are brain/knee/prostate,
    so the literal key 'default' never collides with a real anatomy.
    """
    if not tokens:
        return None
    family = tokens[0]
    shared = None
    per_anatomy = {}
    for tok in tokens[1:]:
        if "=" in tok:
            anatomy, frac = tok.split("=", 1)
            per_anatomy[anatomy.strip()] = _check_frac(float(frac), tok)
        else:
            if shared is not None:
                raise ValueError(
                    f"--artifact_fraction got two shared fractions; expected one: {tokens}"
                )
            shared = _check_frac(float(tok), tok)
    if not per_anatomy:
        if shared is None:
            raise ValueError(
                f"--artifact_fraction needs a fraction after the family: {tokens}")
        return (family, shared)
    frac = dict(per_anatomy)
    if shared is not None:
        frac["default"] = shared
    return (family, frac)


def _resolve_artifact_frac(artifact_fraction, anatomy):
    """Return `(family, g)` for an anatomy, where `g` is its resolved fraction
    or None (no up-weighting for this anatomy)."""
    if artifact_fraction is None:
        return None, None
    family, frac = artifact_fraction
    if isinstance(frac, dict):
        g = frac.get(anatomy, frac.get("default"))
    else:
        g = frac
    return family, g


def _build_group_quota(group_indices, base, artifact_fraction=None,
                       anatomy_weight=None):
    """Per-group sample quotas for `BalancedDistributedSampler`.

    `group_indices` maps a group key -> dataset indices. Keys are either
    `(anatomy, artifact)` tuples (balance_by='anatomy_artifact') or plain
    anatomy strings (balance_by='anatomy'). Returns `{group_key: int}`.

    Two regimes (see docs/superpowers/specs/2026-06-30-anatomy-artifact-weighting-design.md):

    * `anatomy_weight is None` (legacy): every "other" group keeps `base`; when
      an `artifact_fraction` resolves for an anatomy that has the family group
      plus >=1 other group, the family group is enlarged to
      `base * n_other * g / (1 - g)`.
    * `anatomy_weight` set: epoch size is fixed at `T = base * N` (N = number of
      groups). Anatomy `a` gets budget `T * f_a`, split uniformly across its
      groups or skewed so the family group is `g` of the budget.
    """
    if base is None or base <= 0:
        raise ValueError(
            "anatomy/artifact weighting requires a positive --samples_per_contrast"
        )

    def anatomy_of(key):
        return key[0] if isinstance(key, tuple) else key

    def is_family(key, family):
        return isinstance(key, tuple) and key[1] == family

    by_anatomy = {}
    for key in group_indices:
        by_anatomy.setdefault(anatomy_of(key), []).append(key)

    if anatomy_weight is not None:
        missing = [a for a in by_anatomy if a not in anatomy_weight]
        if missing:
            raise ValueError(
                f"--anatomy_weight must cover every trained anatomy; missing "
                f"{sorted(missing)} (have {sorted(anatomy_weight)})"
            )
        T = base * len(group_indices)

    quota = {}
    for anatomy, keys in by_anatomy.items():
        family, g = _resolve_artifact_frac(artifact_fraction, anatomy)
        fam_keys = [k for k in keys if is_family(k, family)] if family else []
        other_keys = [k for k in keys if k not in fam_keys]
        upweight = bool(fam_keys) and bool(other_keys) and g is not None

        if anatomy_weight is None:
            for k in keys:
                if upweight and k in fam_keys:
                    quota[k] = int(round(base * len(other_keys) * g / (1.0 - g)))
                else:
                    quota[k] = int(base)
        else:
            budget = T * anatomy_weight[anatomy]
            if upweight:
                fam_q = int(round(budget * g))
                other_q = int(round(budget * (1.0 - g) / len(other_keys)))
                for k in keys:
                    quota[k] = fam_q if k in fam_keys else other_q
            else:
                each = int(round(budget / len(keys)))
                for k in keys:
                    quota[k] = each

    nonpos = [k for k, v in quota.items() if v <= 0]
    if nonpos:
        raise ValueError(
            f"computed non-positive sample quota for groups {nonpos}; raise "
            f"--samples_per_contrast or adjust the weights/fractions"
        )
    return quota


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
    balance_by: str = "anatomy",
    patch_shape=(192, 192, 16),
    augment=None,
    augment_kwargs=None,
    artifacts=None,
    artifact_fraction=None,
    anatomy_weight=None,
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
    `BalancedDistributedSampler`: each balancing group contributes the same
    number of samples per epoch (0 = auto-balance to the smallest group).
    Larger groups cycle through different random subsets across epochs.
    Validation always uses standard sampling.

    `balance_by` selects the grouping:
      * "anatomy"          -- equal samples per anatomy (default; mixes the
                              natural artifact distribution within each anatomy).
      * "anatomy_artifact" -- equal samples per `(anatomy, artifact)` group,
                              e.g. 5000 undersampled-brain, 5000 undersampled-
                              knee, 5000 spike-brain, ... each epoch.

    `artifacts` optionally restricts which artifact families are used (a subset
    of {undersampled, spike, aniso}); clean `raw->denoised` pairs are always
    kept. `None` (default) keeps every artifact. See `get_dataset_3d_patches`.

    `artifact_fraction` optionally up-weights one artifact family to a fixed
    fraction of each anatomy's per-epoch samples. Pass `(family, frac)`, e.g.
    `("aniso", 0.7)`: every other group keeps `samples_per_contrast`, and the
    family's group is enlarged so it makes up `frac` of that anatomy's samples
    (see `_build_group_quota`). Requires `balance_by="anatomy_artifact"`
    and a positive `samples_per_contrast`. `None` (default) keeps equal groups.

    `anatomy_weight` optionally re-divides the epoch into fixed per-anatomy
    target fractions, e.g. `{"brain": 0.1, "knee": 0.45, "prostate": 0.45}`.
    The epoch size is held at `samples_per_contrast * <#groups>` and split so
    each anatomy gets its fraction (artifacts uniform within, unless
    `artifact_fraction` also skews them). `None` (default) = equal groups.
    Must cover every trained anatomy and sum to 1.0 (see `_build_group_quota`).
    """
    if balance_by not in ("anatomy", "anatomy_artifact"):
        raise ValueError(
            f"balance_by must be 'anatomy' or 'anatomy_artifact', got {balance_by!r}"
        )
    if augment is None and augment_kwargs is not None:
        augment = build_augmentation(**augment_kwargs)

    train_set, val_set = get_dataset_3d_patches(
        data_root, contrast, sample, patch_shape, augment=augment,
        artifacts=artifacts,
    )

    use_balanced = samples_per_contrast is not None

    if use_balanced:
        balance_indices = (train_set.group_indices
                           if balance_by == "anatomy_artifact"
                           else train_set.contrast_indices)
        spc = samples_per_contrast
        if artifact_fraction is not None or anatomy_weight is not None:
            if artifact_fraction is not None and balance_by != "anatomy_artifact":
                raise ValueError(
                    "artifact_fraction requires balance_by='anatomy_artifact'"
                )
            spc = _build_group_quota(
                balance_indices, samples_per_contrast,
                artifact_fraction=artifact_fraction,
                anatomy_weight=anatomy_weight,
            )
            if (rank if distributed else 0) == 0:
                print(f"[quota] resolved per-group sample quotas: {spc}")
        train_sampler = BalancedDistributedSampler(
            balance_indices,
            samples_per_contrast=spc,
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
    for condition, target, prompts, target_types, *_ in train_loader:
        print(
            f"condition  {tuple(condition.shape)}  range "
            f"[{condition.min():.4f}, {condition.max():.4f}]\n"
            f"target     {tuple(target.shape)}  range "
            f"[{target.min():.4f}, {target.max():.4f}]\n"
            f"target_type {target_types[0]!r}\n"
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
