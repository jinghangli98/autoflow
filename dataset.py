"""Whole-volume NIfTI dataset for flow matching (multi-task restoration).

Builds (condition, target, prompt) volume pairs for three restoration tasks
inside a single acquisition, and serves random thin 3D patches from them:

  1. artifact  -> raw (fully sampled, unprocessed) image
  2. artifact  -> denoised + bias-corrected image
  3. raw       -> denoised + bias-corrected image

Layout assumed (no JSON sidecars, no train/test folders):
  <data_root>/<anatomy>/<acquisition>/<subject>/<file>.nii.gz
where:
  * <anatomy>      is one of {brain, knee, prostate}.
  * <acquisition>  is `<sequence>_<orientation>_<field>`, e.g. `bravo_ax_3T`,
                   `tse_cor_7T`, `pdfs_cor_1.5T`.
  * <subject>      is a subject/session dir. 3T `PRT<id>_<yyyy.mm.dd>` dirs are
                   sessions of patient `PRT<id>` (see `patient_key`).
  * files inside a subject dir are told apart by name (see `classify_file`):
      - `<stem>.nii.gz`                       raw, fully sampled
      - `d<stem>.nii.gz`                      denoised only (indexed, unused)
      - `md<stem>.nii.gz`                     denoised + bias-corrected
      - `<stem>_R<n>`, `<stem>_SPIKE_R<n>`,
        `<stem>_ANISO_{phase,read,par}<n>`    grappa / spike / aniso artifacts
                                              at various severities
      - `r<stem'>_lowres.nii.gz`              real low-resolution acquisition
                                              (bravo/flair 3T); treated as the
                                              `aniso` family
    Not every subject has every artifact or every severity.

The prompt used as cross-attention context is composed here as
"Input: <input description> <anatomy> MRI. Target: <target description>.",
e.g. "Input: grappa undersampled brain MRI. Target: Fully sampled axial 3T
brain T1-weighted BRAVO MRI of resolution 0.86 x 0.86 x 1.2 mm." The target
half comes from the acquisition folder name (sequence, field strength), the
NIfTI voxel size, and the *plane of the extracted patch* -- the orientation
word is decided per patch by its thin axis (sagittal / coronal / axial, via
the NIfTI affine; the folder's `_ax_`/`_cor_` token is not used). Samples
therefore carry a prompt *template* with a `{plane}` slot that
`VolumePairDataset` fills per patch; the input half comes from the artifact
family (`_input_phrase`).

Volumes on disk are *not* intensity-normalized and their scales differ wildly
even within one subject (raw 0-255 or 0-25000, `d*` 0-25000, `md*` 0-255).
Each volume is therefore normalized independently when loaded
(`normalize_volume`): robust percentiles (default 0.5 / 99.5) of that whole
volume map to [0, 1]. Patches are then cropped from the normalized volume,
so a thin patch keeps whatever sub-range of [0, 1] it happens to cover -- it
is *not* re-stretched to span 0..1.

`VolumePairDataset.__getitem__` loads one pair, normalizes both volumes, and
draws `patches_per_volume` random crops of `patch_shape` (default
(96, 96, 7)) at shared coordinates. The thin (7-voxel) axis of each crop is
drawn at random from the sample's `thin_axes`: all three array axes for 3D
acquisitions (axial / sagittal / coronal thin slabs), only the through-plane
axis (largest voxel size) for 2D multi-slice sequences listed in
`SINGLE_PLANE_SEQUENCES` (TSE). Axes along which a 96x96 in-plane crop does
not fit (e.g. 36-slice TSE, 30-slice knee/prostate) are skipped. The thin
axis is moved last, so every patch is `(96, 96, 7)` regardless of plane.
`collate_patches` flattens those so a DataLoader with `batch_size=B` yields
`B * patches_per_volume` patches per step in the usual `(B', 1, X, Y, Z)`
layout.

Train/val: a deterministic patient-level split *within* `data_root`
(`val_fraction`). The held-out test subjects live in a separate root (see
`make_test_split.py`) and are never read here unless that root is passed
explicitly.
"""

import hashlib
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


_NII = ".nii.gz"

# Artifact suffix appended to a raw stem, e.g. `_R3`, `_SPIKE_R4`,
# `_SPIKE_R4.5`, `_ANISO_phase3.5`, `_ANISO_read4`, `_ANISO_par3`.
_ARTIFACT_SUFFIX_RE = re.compile(
    r"_(R\d+|SPIKE_R\d+(?:\.\d+)?|ANISO_(?:phase|read|par)\d+(?:\.\d+)?)$"
)
_LOWRES_RE = re.compile(r"^r(.+)_lowres$")
_SESSION_RE = re.compile(r"^(PRT\d+)_\d{4}\.\d{2}\.\d{2}$")


def _artifact_family(suffix: str) -> str:
    """Coarse artifact family from an artifact suffix (with leading `_`).

    `suffix` is e.g. `_R8`, `_SPIKE_R4`, `_ANISO_phase3.5`. The order of
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
    "aniso": "anisotropic undersampled",
}


def _input_phrase(family: str) -> str:
    """Phrase for the "Input: ..." half of a sample's prompt, from its
    artifact family. The three ANISO k-space-direction variants (phase/read/
    par) and the real low-res acquisitions all collapse to the same
    "anisotropic undersampled" phrase."""
    return _ARTIFACT_INPUT_PHRASE.get(family, "artifact")


# --------------------------------------------------------------------------
# File-name classification
# --------------------------------------------------------------------------

def classify_file(name: str, stems=None):
    """Classify one file name inside a subject dir.

    Returns None for non-NIfTI names, else a dict:
      role      -- "raw" | "denoised" (`d*`) | "md" (`md*`) | "artifact"
      stem      -- the raw stem this file belongs to
      artifact  -- family for role "artifact" (undersampled/spike/aniso), else None
      severity  -- e.g. "R3", "SPIKE_R4", "ANISO_par3.5", "lowres"; else None

    `stems` (optional) is the set of all stems in the same dir. When given,
    the `d`/`md` prefix rules only fire if the prefix-stripped stem exists,
    so a raw stem that happens to start with "d" is not misread.
    """
    if not name.endswith(_NII):
        return None
    stem = name[: -len(_NII)]

    def _info(role, base, artifact=None, severity=None):
        return {"role": role, "stem": base, "artifact": artifact,
                "severity": severity}

    m = _ARTIFACT_SUFFIX_RE.search(stem)
    if m:
        sev = m.group(1)
        return _info("artifact", stem[: m.start()], _artifact_family("_" + sev), sev)
    m = _LOWRES_RE.match(stem)
    if m:
        base = m.group(1) + "_highres"
        if stems is None or base in stems:
            return _info("artifact", base, "aniso", "lowres")
    if stem.startswith("md") and (stems is None or stem[2:] in stems):
        return _info("md", stem[2:])
    if stem.startswith("d") and (stems is None or stem[1:] in stems):
        return _info("denoised", stem[1:])
    return _info("raw", stem)


def patient_key(subject_dir: str) -> str:
    """Collapse `PRT<id>_<yyyy.mm.dd>` sessions to `PRT<id>`; else identity."""
    m = _SESSION_RE.match(subject_dir)
    return m.group(1) if m else subject_dir


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------

# `<sequence>` token of an acquisition folder -> descriptive phrase.
SEQUENCE_DESC = {
    "bravo": "T1-weighted BRAVO",
    "mprage": "T1-weighted MPRAGE",
    "mp2rage": "T1-weighted MP2RAGE",
    "flair": "T2-weighted FLAIR",
    "space": "T2-weighted SPACE",
    "tse": "T2-weighted TSE",
    "pd": "Proton Density",
    "pdfs": "fat-suppressed Proton Density",
    "tof": "Time-of-Flight angiography",
}
# Anatomical plane of a slab whose thin axis runs along a given axis code.
_PLANE_OF_AXCODE = {"R": "sagittal", "L": "sagittal", "A": "coronal",
                    "P": "coronal", "S": "axial", "I": "axial"}
# Fallback when no affine is known: assumes RAS-like storage (x, y, z).
DEFAULT_AXIS_PLANES = ["sagittal", "coronal", "axial"]
TARGET_DESC = {"raw": "Fully sampled", "md": "Denoised and biascorrected"}

# 2D multi-slice sequences: thin patches only along the through-plane axis.
SINGLE_PLANE_SEQUENCES = ("tse",)


def parse_acquisition(acq: str):
    """`bravo_ax_3T` -> ("bravo", "ax", "3T")."""
    parts = acq.split("_")
    if len(parts) != 3:
        raise ValueError(
            f"acquisition folder must be <sequence>_<orientation>_<field>, got {acq!r}")
    return tuple(parts)


def _fmt_mm(z: float) -> str:
    s = f"{float(z):.2f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def axis_planes(affine) -> list:
    """Plane name for a slab thin along each array axis, from the affine:
    RAS storage -> ["sagittal", "coronal", "axial"]."""
    return [_PLANE_OF_AXCODE[c] for c in nib.aff2axcodes(np.asarray(affine))]


def build_target_prompt(kind: str, anatomy: str, acq: str, zooms,
                        plane: str = "{plane}") -> str:
    """Target half of the prompt, e.g.
    "Fully sampled axial 3T brain T1-weighted BRAVO MRI of resolution
    0.86 x 0.86 x 1.2 mm."  `kind` is "raw" or "md". `plane` is the slab
    orientation word; the default leaves a `{plane}` slot to be filled per
    patch with `str.format`. The folder's orientation token is ignored."""
    seq, _orient, field = parse_acquisition(acq)
    res = " x ".join(_fmt_mm(z) for z in zooms[:3])
    return (f"{TARGET_DESC[kind]} {plane} {field} {anatomy} "
            f"{SEQUENCE_DESC.get(seq, seq.upper())} MRI of resolution {res} mm.")


# --------------------------------------------------------------------------
# Pair enumeration + patient split
# --------------------------------------------------------------------------

def _val_patients(patients, val_fraction: float, seed: int):
    """Deterministic patient-level val subset: rank patients by a seeded hash
    and take the first ceil(frac * n), leaving at least one for train."""
    patients = sorted(set(patients))
    if val_fraction <= 0 or len(patients) < 2:
        return set()
    n_val = min(math.ceil(val_fraction * len(patients)), len(patients) - 1)

    def h(p):
        return hashlib.md5(f"{seed}:{p}".encode()).hexdigest()

    return set(sorted(patients, key=h)[:n_val])


def _read_header(path):
    """(zooms, axis_planes) of a NIfTI without loading its data."""
    img = nib.load(path)
    return (tuple(float(z) for z in img.header.get_zooms()[:3]),
            axis_planes(img.affine))


def _drop_shape_mismatches(samples):
    """Remove pairs whose condition/target voxel shapes differ (header check).

    Cheap (one header read per distinct file) and done once at index time so
    a bad file cannot crash a DataLoader worker mid-epoch. Prints one line per
    dropped pair.
    """
    shapes = {}

    def shape_of(path):
        if path not in shapes:
            shapes[path] = tuple(int(n) for n in nib.load(path).shape[:3])
        return shapes[path]

    kept, dropped = [], {}
    for s in samples:
        a, b = shape_of(s["condition_path"]), shape_of(s["target_path"])
        if a == b:
            kept.append(s)
        else:
            key = (os.path.dirname(s["condition_path"]), a, b)
            dropped.setdefault(key, set()).add(os.path.basename(s["condition_path"]))
    for (subject_dir, a, b), files in sorted(dropped.items()):
        print(f"warning: {subject_dir}: skipping {len(files)} condition file(s) "
              f"of shape {a} whose target is {b}: {', '.join(sorted(files))}")
    return kept


def build_samples(data_root: str, anatomy: str, split: str = "train",
                  val_fraction: float = 0.10, split_seed: int = 42):
    """Enumerate (condition, target, prompt) *volume* pairs for one anatomy.

    `split` is "train", "val" or "all". Train/val is a patient-level split
    (`patient_key`) of everything under `<data_root>/<anatomy>/`.

    Pairs whose condition and target differ in voxel shape (header check) are
    skipped with a printed warning (see `_drop_shape_mismatches`).

    Returns a list of dicts with keys: condition_path, target_path, prompt
    (template with a `{plane}` slot), task (artifact2raw | artifact2denoised |
    raw2denoised), target_type (raw | denoised), anatomy, artifact
    (undersampled | spike | aniso | clean), severity, acquisition, subject,
    patient, thin_axes (array axes the thin patch dimension may lie along; see
    `SINGLE_PLANE_SEQUENCES`), axis_planes (plane word per array axis).
    """
    if split not in ("train", "val", "all"):
        raise ValueError(f"split must be train/val/all, got {split!r}")
    anatomy_dir = os.path.join(data_root, anatomy)
    if not os.path.isdir(anatomy_dir):
        raise FileNotFoundError(f"Anatomy directory not found: {anatomy_dir}")

    samples = []
    for acq in sorted(os.listdir(anatomy_dir)):
        acq_dir = os.path.join(anatomy_dir, acq)
        if not os.path.isdir(acq_dir):
            continue
        for subject in sorted(os.listdir(acq_dir)):
            sub_dir = os.path.join(acq_dir, subject)
            if not os.path.isdir(sub_dir):
                continue
            names = sorted(n for n in os.listdir(sub_dir) if n.endswith(_NII))
            stems = {n[: -len(_NII)] for n in names}
            raws, mds, arts = {}, {}, []
            for n in names:
                info = classify_file(n, stems)
                path = os.path.join(sub_dir, n)
                if info["role"] == "raw":
                    raws[info["stem"]] = path
                elif info["role"] == "md":
                    mds[info["stem"]] = path
                elif info["role"] == "artifact":
                    arts.append((info, path))

            patient = patient_key(subject)
            for stem, raw_path in raws.items():
                zooms, planes = _read_header(raw_path)
                raw_prompt = build_target_prompt("raw", anatomy, acq, zooms)
                md_path = mds.get(stem)
                md_prompt = (build_target_prompt("md", anatomy, acq, zooms)
                             if md_path else None)
                if parse_acquisition(acq)[0] in SINGLE_PLANE_SEQUENCES:
                    thin_axes = [int(np.argmax(zooms))]
                else:
                    thin_axes = [0, 1, 2]
                meta = {"anatomy": anatomy, "acquisition": acq,
                        "subject": subject, "patient": patient,
                        "thin_axes": thin_axes, "axis_planes": planes}

                for info, art_path in arts:
                    if info["stem"] != stem:
                        continue
                    phrase = _input_phrase(info["artifact"])
                    samples.append({
                        "condition_path": art_path, "target_path": raw_path,
                        "prompt": f"Input: {phrase} {anatomy} MRI. Target: {raw_prompt}",
                        "task": "artifact2raw", "target_type": "raw",
                        "artifact": info["artifact"], "severity": info["severity"],
                        **meta,
                    })
                    if md_path:
                        samples.append({
                            "condition_path": art_path, "target_path": md_path,
                            "prompt": f"Input: {phrase} {anatomy} MRI. Target: {md_prompt}",
                            "task": "artifact2denoised", "target_type": "denoised",
                            "artifact": info["artifact"], "severity": info["severity"],
                            **meta,
                        })
                if md_path:
                    samples.append({
                        "condition_path": raw_path, "target_path": md_path,
                        "prompt": f"Input: fully sampled {anatomy} MRI. Target: {md_prompt}",
                        "task": "raw2denoised", "target_type": "denoised",
                        "artifact": "clean", "severity": None,
                        **meta,
                    })

    samples = _drop_shape_mismatches(samples)
    if split == "all":
        return samples
    val = _val_patients((s["patient"] for s in samples), val_fraction, split_seed)
    want_val = split == "val"
    return [s for s in samples if (s["patient"] in val) == want_val]


def prompt_for_path(path: str, target: str = "raw"):
    """The loader's prompt (and plane metadata) for one input file, for
    inference. `path` must sit in the training layout
    `<root>/<anatomy>/<acq>/<subject>/<file>.nii.gz`; the file's role and
    artifact family come from its name and its siblings, the resolution and
    plane order from its header. `target` is "raw" (fully sampled) or "md"
    (denoised + bias-corrected).

    Returns a dict: prompt (template with a `{plane}` slot), anatomy,
    acquisition, subject, artifact (undersampled | spike | aniso | clean),
    severity, thin_axes, axis_planes -- the same fields `build_samples` gives
    a training pair.
    """
    if target not in TARGET_DESC:
        raise ValueError(f"target must be one of {tuple(TARGET_DESC)}, got {target!r}")
    path = os.path.abspath(path)
    sub_dir, name = os.path.split(path)
    acq_dir, subject = os.path.split(sub_dir)
    anatomy_dir, acq = os.path.split(acq_dir)
    anatomy = os.path.basename(anatomy_dir)
    if not name.endswith(_NII):
        raise ValueError(f"expected a {_NII} file, got {path}")
    stems = {n[: -len(_NII)] for n in os.listdir(sub_dir) if n.endswith(_NII)}
    info = classify_file(name, stems)
    if info["role"] == "raw":
        if target == "raw":
            raise ValueError(
                f"{name} is already fully sampled (raw); use --target md to "
                f"denoise + bias-correct it")
        phrase, artifact, severity = "fully sampled", "clean", None
    elif info["role"] == "artifact":
        phrase = _input_phrase(info["artifact"])
        artifact, severity = info["artifact"], info["severity"]
    else:
        raise ValueError(
            f"{name} is a {info['role']} output file, not a model input")
    zooms, planes = _read_header(path)
    if parse_acquisition(acq)[0] in SINGLE_PLANE_SEQUENCES:
        thin_axes = [int(np.argmax(zooms))]
    else:
        thin_axes = [0, 1, 2]
    target_prompt = build_target_prompt(target, anatomy, acq, zooms)
    return {"prompt": f"Input: {phrase} {anatomy} MRI. Target: {target_prompt}",
            "anatomy": anatomy, "acquisition": acq, "subject": subject,
            "artifact": artifact, "severity": severity,
            "thin_axes": thin_axes, "axis_planes": planes}


# --------------------------------------------------------------------------
# Volume loading / normalization / cropping
# --------------------------------------------------------------------------

def load_volume(path: str) -> np.ndarray:
    """Load a NIfTI as a float32 array (no scaling, no reorientation)."""
    return np.asanyarray(nib.load(path).dataobj).astype(np.float32)


def volume_window(vol: np.ndarray, percentiles=(0.5, 99.5),
                  max_voxels: int = 2_000_000):
    """`(lo, hi)` intensities that `normalize_volume` maps to 0 and 1 for
    this volume: its `percentiles`, estimated on a fixed random subsample of
    at most `max_voxels` voxels. Falls back to (min, max) for a degenerate
    percentile window. Inference uses it to undo the normalization."""
    vol = np.asarray(vol, dtype=np.float32)
    flat = vol.reshape(-1)
    if flat.size > max_voxels:
        idx = np.random.default_rng(0).integers(0, flat.size, size=max_voxels)
        flat = flat[idx]
    lo, hi = np.percentile(flat, percentiles)
    if hi <= lo:
        lo, hi = float(vol.min()), float(vol.max())
    return float(lo), float(hi)


def normalize_volume(vol: np.ndarray, percentiles=(0.5, 99.5), clip=True,
                     max_voxels: int = 2_000_000) -> np.ndarray:
    """Per-volume robust intensity normalization to [0, 1].

    `lo, hi` are the given percentiles of *this* volume (estimated on a
    fixed random subsample of at most `max_voxels` voxels, so a 512^3 volume
    costs milliseconds rather than seconds), mapped to 0 and 1. With `clip`
    the result is clamped to [0, 1]; the few voxels above `hi` saturate.
    """
    vol = np.asarray(vol, dtype=np.float32)
    lo, hi = volume_window(vol, percentiles, max_voxels)
    if hi <= lo:
        return np.zeros_like(vol)
    out = (vol - lo) / (hi - lo)
    if clip:
        np.clip(out, 0.0, 1.0, out=out)
    return out.astype(np.float32, copy=False)


def _crop(vol, start, shape):
    return vol[tuple(slice(s, s + n) for s, n in zip(start, shape))]


def build_augmentation(
    noise_std=(0.0, 0.1),
    ghost_num=(2, 5),
    ghost_intensity=(0.3, 0.7),
    ghost_axes=(0, 1),
    p_noise=0.5,
    p_ghost=0.5,
):
    """Build a TorchIO transform that adds random MR noise + ghosting.

    Operates on a single (C, W, H, D) float tensor and returns one of the
    same shape. Intended to be applied to the *condition* (input) only, so
    the target stays a clean reconstruction goal. Intensities are in the
    normalized [0, 1] range, so `noise_std` is relative to that.
    """
    transforms = []
    if p_noise > 0:
        transforms.append(tio.RandomNoise(mean=0.0, std=noise_std, p=p_noise))
    if p_ghost > 0:
        transforms.append(
            tio.RandomGhosting(num_ghosts=ghost_num, axes=ghost_axes,
                               intensity=ghost_intensity, p=p_ghost)
        )
    return tio.Compose(transforms) if transforms else None


class VolumePairDataset(Dataset):
    """One item = `patches_per_volume` aligned patches from one volume pair.

    Returns `(condition, target, prompts, target_type, anatomy, artifact)`
    with `condition`/`target` of shape `(K, 1, *patch_shape)` and `prompts`
    a list of K strings -- the sample's prompt template with `{plane}` filled
    by the plane of each patch's thin axis (`axis_planes`). Use
    `collate_patches` to flatten a batch of items into `(B*K, 1, ...)`.

    Both volumes are normalized independently (`normalize_volume`) and then
    cropped at the same coordinates. For each patch the thin axis is drawn
    uniformly from the sample's feasible `thin_axes` (see module docstring)
    and moved last, so the output is always `(K, 1, *patch_shape)`. Crops
    that are mostly air (fewer than `fg_fraction` of target voxels above
    `fg_threshold`) are re-drawn up to `max_tries` times; the best attempt
    is kept otherwise.

    `deterministic=True` (validation) seeds the crop RNG by item index so
    every epoch scores the same patches. `augment` (TorchIO, see
    `build_augmentation`) is applied to each condition patch only.

    After each `__getitem__`, `last_crops` holds `[(thin_axis, start), ...]`
    for the patches just drawn (debug/visualization aid; per worker process).
    """

    def __init__(self, samples, patch_shape=(96, 96, 7), patches_per_volume=4,
                 augment=None, norm_percentiles=(0.5, 99.5),
                 deterministic=False, fg_threshold=0.05, fg_fraction=0.25,
                 max_tries=10, seed=0):
        self.samples = samples
        self.patch_shape = tuple(patch_shape)
        self.patches_per_volume = int(patches_per_volume)
        self.augment = augment
        self.norm_percentiles = norm_percentiles
        self.deterministic = deterministic
        self.fg_threshold = fg_threshold
        self.fg_fraction = fg_fraction
        self.max_tries = max_tries
        self.seed = seed
        self.last_crops = []

    def __len__(self):
        return len(self.samples)

    def _load(self, path):
        return normalize_volume(load_volume(path), self.norm_percentiles)

    def _crop_shape(self, axis):
        """Crop extent in array order with the thin dimension at `axis`."""
        inplane = list(self.patch_shape[:2])
        return tuple(self.patch_shape[2] if a == axis else inplane.pop(0)
                     for a in range(3))

    def _feasible_axes(self, vol_shape, thin_axes):
        return [a for a in thin_axes
                if all(n >= p for n, p in zip(vol_shape, self._crop_shape(a)))]

    def _rng(self, idx):
        return (np.random.default_rng(self.seed + idx) if self.deterministic
                else np.random.default_rng())

    def _draw_axes(self, rng, vol_shape, thin_axes):
        """One thin axis per patch, drawn first so it is reproducible."""
        feasible = self._feasible_axes(vol_shape, thin_axes)
        if not feasible:
            raise ValueError(
                f"no thin axis in {thin_axes} lets a {self.patch_shape} patch "
                f"fit a volume of shape {tuple(vol_shape)}")
        return [feasible[int(rng.integers(0, len(feasible)))]
                for _ in range(self.patches_per_volume)]

    def thin_axes_used(self, idx):
        """Thin axes `self[idx]` will draw (exact only when `deterministic`)."""
        s = self.samples[idx]
        shape = nib.load(s["target_path"]).shape[:3]
        return self._draw_axes(self._rng(idx), shape, s.get("thin_axes", [2]))

    def _draw_start(self, rng, shape, crop_shape):
        return tuple(int(rng.integers(0, n - p + 1))
                     for n, p in zip(shape, crop_shape))

    def __getitem__(self, idx):
        s = self.samples[idx]
        cond = self._load(s["condition_path"])
        tgt = self._load(s["target_path"])
        if cond.shape != tgt.shape:
            raise ValueError(
                f"condition/target shape mismatch {cond.shape} vs {tgt.shape}: "
                f"{s['condition_path']} / {s['target_path']}")
        rng = self._rng(idx)
        axes = self._draw_axes(rng, tgt.shape, s.get("thin_axes", [2]))
        planes = s.get("axis_planes", DEFAULT_AXIS_PLANES)
        prompts = [s["prompt"].format(plane=planes[a]) for a in axes]
        cond_patches, tgt_patches, crops = [], [], []
        for axis in axes:
            crop_shape = self._crop_shape(axis)
            best, best_fg = None, -1.0
            for _ in range(self.max_tries):
                start = self._draw_start(rng, tgt.shape, crop_shape)
                fg = float((_crop(tgt, start, crop_shape) > self.fg_threshold).mean())
                if fg > best_fg:
                    best, best_fg = start, fg
                if fg >= self.fg_fraction:
                    break
            c = torch.from_numpy(np.ascontiguousarray(
                np.moveaxis(_crop(cond, best, crop_shape), axis, -1))).unsqueeze(0)
            t = torch.from_numpy(np.ascontiguousarray(
                np.moveaxis(_crop(tgt, best, crop_shape), axis, -1))).unsqueeze(0)
            if self.augment is not None:
                c = self.augment(c)
            cond_patches.append(c)
            tgt_patches.append(t)
            crops.append((axis, best))
        self.last_crops = crops

        return (torch.stack(cond_patches), torch.stack(tgt_patches),
                prompts, s["target_type"], s["anatomy"], s["artifact"])


def collate_patches(batch):
    """Flatten a batch of `(K,1,...)` items into `(B*K,1,...)` tensors; the
    per-patch prompt lists are concatenated and the other string fields are
    repeated K times so everything stays aligned."""
    cond = torch.cat([b[0] for b in batch], dim=0)
    tgt = torch.cat([b[1] for b in batch], dim=0)
    prompts, ttypes, anatomies, artifacts = [], [], [], []
    for b in batch:
        k = b[0].shape[0]
        prompts.extend(b[2])
        ttypes.extend([b[3]] * k)
        anatomies.extend([b[4]] * k)
        artifacts.extend([b[5]] * k)
    return cond, tgt, prompts, ttypes, anatomies, artifacts


# --------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------

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
    == "clean") is always retained regardless of the filter. Pass
    `artifacts=None` to disable filtering entirely.
    """
    if artifacts is None:
        return samples
    allowed = set(artifacts)
    return [s for s in samples
            if s["artifact"] in allowed or s["artifact"] == "clean"]


def get_dataset_3d_patches(data_root: str, contrast, sample: float = 100.0,
                           patch_shape=(96, 96, 7), augment=None,
                           artifacts=None, val_fraction: float = 0.10,
                           split_seed: int = 42, patches_per_volume: int = 4,
                           norm_percentiles=(0.5, 99.5), fg_fraction: float = 0.25):
    """Build train + val datasets for one or more anatomies.

    `contrast` is the anatomy (one of {brain, knee, prostate}) or a list of
    them; samples are concatenated. `sample` keeps a random percentage of the
    train *pairs* (val is always used whole -- it is small).

    Validation is a patient-level `val_fraction` split of `data_root` (see
    `build_samples`); the held-out test root is not touched.

    `artifacts` optionally restricts artifact families (clean pairs always
    kept), applied to both splits.

    The returned datasets carry two index maps for the balanced sampler:
      * `contrast_indices` -- keyed by anatomy name.
      * `group_indices`    -- keyed by `(anatomy, artifact)`.
    Quotas drawn from these count volume pairs; each pair yields
    `patches_per_volume` patches.
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
            art_note = (f" with --artifact {sorted(artifacts)}"
                        if artifacts is not None else "")
            raise ValueError(
                f"No (condition, target) volume pairs found for anatomy "
                f"'{c}'{art_note} in {os.path.join(data_root, c)}. Expected "
                f"<acq>/<subject>/<stem>.nii.gz plus md<stem> and/or artifact "
                f"(_R*/_SPIKE_R*/_ANISO_*/r*_lowres) siblings.")
        if not va:
            print(f"warning: anatomy '{c}' has no validation pairs "
                  f"(val_fraction={val_fraction}, too few patients?)")
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

    common = dict(patch_shape=patch_shape, patches_per_volume=patches_per_volume,
                  augment=augment, norm_percentiles=norm_percentiles,
                  fg_fraction=fg_fraction)
    train_set = VolumePairDataset(train_all, deterministic=False, **common)
    val_set = VolumePairDataset(val_all, deterministic=True, **common)
    train_set.contrast_indices = train_ci
    val_set.contrast_indices = val_ci
    train_set.group_indices = _group_indices(train_all)
    val_set.group_indices = _group_indices(val_all)
    return train_set, val_set


# --------------------------------------------------------------------------
# Balanced sampling / quotas (unchanged: groups now count volume pairs)
# --------------------------------------------------------------------------

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
    patch_shape=(96, 96, 7),
    augment=None,
    augment_kwargs=None,
    artifacts=None,
    artifact_fraction=None,
    anatomy_weight=None,
    val_fraction: float = 0.10,
    split_seed: int = 42,
    patches_per_volume: int = 4,
    norm_percentiles=(0.5, 99.5),
    fg_fraction: float = 0.25,
):
    """DataLoaders for whole-volume NIfTI patch restoration. DDP-aware.

    `batch_size` counts *volume pairs*; every step yields
    `batch_size * patches_per_volume` patches of `patch_shape` (flattened by
    `collate_patches`), each pair's prompt repeated per patch.

    `contrast` is the anatomy group (one of {brain, knee, prostate}) or a
    list/tuple of them.

    Augmentation (random MR noise + ghosting) is applied to the *condition
    only*, on both train and val:
      - `augment`: pass a ready-made TorchIO transform to use it directly.
      - `augment_kwargs`: pass a dict of `build_augmentation` kwargs.
      - If both are None (default), no augmentation is applied.

    If `samples_per_contrast` is not None, training uses
    `BalancedDistributedSampler`: each balancing group contributes the same
    number of *pairs* per epoch (0 = auto-balance to the smallest group).
    `balance_by` is "anatomy" or "anatomy_artifact". `artifacts`,
    `artifact_fraction` and `anatomy_weight` behave as documented in
    `get_dataset_3d_patches` / `_build_group_quota`.

    `val_fraction` / `split_seed` control the patient-level val split inside
    `data_root`; `norm_percentiles` the per-volume normalization.
    """
    if balance_by not in ("anatomy", "anatomy_artifact"):
        raise ValueError(
            f"balance_by must be 'anatomy' or 'anatomy_artifact', got {balance_by!r}"
        )
    if augment is None and augment_kwargs is not None:
        augment = build_augmentation(**augment_kwargs)

    train_set, val_set = get_dataset_3d_patches(
        data_root, contrast, sample, patch_shape, augment=augment,
        artifacts=artifacts, val_fraction=val_fraction, split_seed=split_seed,
        patches_per_volume=patches_per_volume, norm_percentiles=norm_percentiles,
        fg_fraction=fg_fraction,
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

    common = dict(num_workers=num_workers, pin_memory=True,
                  collate_fn=collate_patches)
    if distributed:
        val_sampler = DistributedSampler(
            val_set, num_replicas=world_size, rank=rank,
            shuffle=False, seed=42,
        )
        train_loader = DataLoader(
            train_set, batch_size=batch_size, sampler=train_sampler,
            drop_last=True, **common,
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, sampler=val_sampler,
            drop_last=False, **common,
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=batch_size, sampler=train_sampler,
            shuffle=(train_sampler is None) and train_shuffle, **common,
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False, **common,
        )

    return train_loader, val_loader


if __name__ == "__main__":
    import sys
    import time
    from collections import Counter

    data_root = "/vast/tibrahim/jil202/nii"
    anatomy = sys.argv[1] if len(sys.argv) > 1 else "brain"

    print(f"Building datasets for anatomy={anatomy} under {data_root} ...")
    t0 = time.time()
    train_set, val_set = get_dataset_3d_patches(
        data_root, anatomy, sample=100.0, patch_shape=(96, 96, 7),
    )
    print(f"indexed in {time.time() - t0:.1f}s: train pairs {len(train_set)}, "
          f"val pairs {len(val_set)}")
    print(f"train patients {len({s['patient'] for s in train_set.samples})}, "
          f"val patients {len({s['patient'] for s in val_set.samples})}")
    print("train task breakdown:", dict(Counter(s["task"] for s in train_set.samples)))
    print("train group breakdown:",
          dict(Counter((s["acquisition"], s["artifact"]) for s in train_set.samples)))

    train_loader, _ = getloader_3d_patches(
        batch_size=2, data_root=data_root, contrast=anatomy,
        sample=100.0, num_workers=0, patch_shape=(96, 96, 7),
    )
    t0 = time.time()
    for i, (condition, target, prompts, target_types, *_) in enumerate(train_loader):
        print(
            f"batch {i}: condition {tuple(condition.shape)} range "
            f"[{condition.min():.3f}, {condition.max():.3f}]  target "
            f"{tuple(target.shape)} range [{target.min():.3f}, {target.max():.3f}]  "
            f"{time.time() - t0:.2f}s\n  prompt: {prompts[0]!r}"
        )
        t0 = time.time()
        if i == 2:
            break
