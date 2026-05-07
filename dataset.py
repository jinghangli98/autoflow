"""3D NIfTI patch dataset for autoregressive flow matching.

Pairs undersampled patches (GRAPPA `_R{3..6}` / CS `_CS_R{2,3}`) with their
matching ground-truth patches inside a single contrast (mprage or tse).

Layout assumed:
  <data_root>/<split>/<contrast>/<subject>/patch_<x>_<y>_<z>.nii.gz
where <subject> is the GT id (no suffix) and siblings <subject>_R3, ..., _CS_R3
contain the same patch coordinates with the corresponding undersampling.
"""

import glob
import json
import os
import random
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


_PATCH_RE = re.compile(r"patch_(\d+)_(\d+)_(\d+)\.nii\.gz$")
_UNDERSAMPLED_SUFFIX_RE = re.compile(r"_(?:CS_)?R\d+$")
_ACCEL_SUFFIX_RE = re.compile(r"^_(CS_)?R(\d+)$")


def _is_gt_subject(name: str) -> bool:
    """Return True if a subject directory name has no `_R*`/`_CS_R*` suffix."""
    return _UNDERSAMPLED_SUFFIX_RE.search(name) is None


def _find_undersampled_siblings(contrast_dir: str, gt_name: str):
    """Find sibling undersampled subject dirs for a given GT subject name.

    A sibling matches `<gt_name>_R<digit>+` or `<gt_name>_CS_R<digit>+` exactly.
    """
    candidates = sorted(
        d for d in os.listdir(contrast_dir)
        if os.path.isdir(os.path.join(contrast_dir, d))
    )
    pattern = re.compile(rf"^{re.escape(gt_name)}_(?:CS_)?R\d+$")
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


def _load_patch(path: str) -> torch.Tensor:
    """Load a single 192x192x16 NIfTI patch as float32 tensor (1, 192, 192, 16)."""
    arr = nib.load(path).get_fdata().astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


def _read_voxel_size(subject_dir: str, subject_name: str):
    """Read voxel_size_mm from <subject>_params.json. Returns (vx, vy, vz)."""
    params_path = os.path.join(subject_dir, f"{subject_name}_params.json")
    with open(params_path) as f:
        meta = json.load(f)
    vx, vy, vz = meta["voxel_size_mm"]
    return float(vx), float(vy), float(vz)


def _parse_accel(sibling_name: str, gt_name: str):
    """Parse (is_cs: bool, factor: int) from a sibling dir name like '<gt>_R3' / '<gt>_CS_R2'."""
    if not sibling_name.startswith(gt_name):
        raise ValueError(f"Sibling {sibling_name} does not start with GT {gt_name}")
    suffix = sibling_name[len(gt_name):]
    m = _ACCEL_SUFFIX_RE.match(suffix)
    if not m:
        raise ValueError(f"Cannot parse accel suffix from {sibling_name}")
    is_cs = m.group(1) is not None
    factor = int(m.group(2))
    return is_cs, factor


class PatchPairDataset(Dataset):
    """One sample = (undersampled patch, GT patch, prev GT Z-patch, context_vec).

    Each volume tensor is shape (1, 192, 192, 16). `context_vec` is a (5,)
    float32 tensor: (voxel_x, voxel_y, voxel_z, is_cs, accel_factor).
    """

    def __init__(self, samples, gt_index_by_subject, gt_voxel_by_subject,
                 patch_shape=(192, 192, 16)):
        self.samples = samples
        self.gt_index = gt_index_by_subject
        self.gt_voxel = gt_voxel_by_subject
        self.patch_shape = patch_shape

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        condition = _load_patch(s["condition_path"])
        target = _load_patch(s["target_path"])

        prev_key = (s["x"], s["y"], s["z"] - 1)
        prev_path = self.gt_index[s["gt_subject"]].get(prev_key)
        if s["z"] > 0 and prev_path is not None:
            prev_chunk = _load_patch(prev_path)
        else:
            prev_chunk = torch.zeros((1, *self.patch_shape), dtype=torch.float32)

        vx, vy, vz = self.gt_voxel[s["gt_subject"]]
        ctx_vec = torch.tensor(
            [vx, vy, vz, 1.0 if s["is_cs"] else 0.0, float(s["accel_factor"])],
            dtype=torch.float32,
        )

        return condition, target, prev_chunk, ctx_vec


def build_samples(data_root: str, split: str, contrast: str):
    """Enumerate (undersampled, GT) patch pairs for one split + contrast.

    Returns:
        samples:   list of dicts (see PatchPairDataset.__init__).
        gt_index:  dict gt_subject -> {(x,y,z): gt_path}.
        gt_voxel:  dict gt_subject -> (vx, vy, vz) in mm.
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
    gt_index = {}
    gt_voxel = {}
    for gt in gt_subjects:
        gt_dir = os.path.join(contrast_dir, gt)
        gt_patches = _index_subject_patches(gt_dir)
        if not gt_patches:
            continue
        gt_index[gt] = gt_patches
        try:
            gt_voxel[gt] = _read_voxel_size(gt_dir, gt)
        except (FileNotFoundError, KeyError):
            gt_voxel[gt] = (1.0, 1.0, 1.0)

        siblings = _find_undersampled_siblings(contrast_dir, gt)
        for sib in siblings:
            try:
                is_cs, factor = _parse_accel(sib, gt)
            except ValueError:
                continue
            sib_dir = os.path.join(contrast_dir, sib)
            sib_patches = _index_subject_patches(sib_dir)
            for (x, y, z), us_path in sib_patches.items():
                if (x, y, z) not in gt_patches:
                    continue
                samples.append({
                    "condition_path": us_path,
                    "target_path": gt_patches[(x, y, z)],
                    "gt_subject": gt,
                    "x": x, "y": y, "z": z,
                    "is_cs": is_cs,
                    "accel_factor": factor,
                })

    return samples, gt_index, gt_voxel


def _subsample(samples, percent: float, seed: int = 42):
    """Return a deterministic random subset of `samples` (percent in [0, 100])."""
    if percent >= 100:
        return samples
    rng = random.Random(seed)
    n = max(1, int(len(samples) * percent / 100.0))
    return rng.sample(samples, n)


def get_dataset_3d_patches(data_root: str, contrast: str, sample: float = 100.0):
    """Build train + val datasets for one contrast.

    Train comes from `train/`, validation from `test/`.
    """
    train_samples, train_index, train_voxel = build_samples(data_root, "train", contrast)
    val_samples, val_index, val_voxel = build_samples(data_root, "test", contrast)

    train_samples = _subsample(train_samples, sample, seed=42)
    val_samples = _subsample(val_samples, min(sample, 5.0), seed=43)

    train_set = PatchPairDataset(train_samples, train_index, train_voxel)
    val_set = PatchPairDataset(val_samples, val_index, val_voxel)
    return train_set, val_set


def getloader_3d_patches(
    batch_size: int,
    data_root: str,
    contrast: str,
    sample: float = 100.0,
    num_workers: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    train_shuffle: bool = True,
):
    """DataLoaders for 3D NIfTI patch reconstruction. DDP-aware."""
    train_set, val_set = get_dataset_3d_patches(data_root, contrast, sample)

    if distributed:
        train_sampler = DistributedSampler(
            train_set, num_replicas=world_size, rank=rank,
            shuffle=train_shuffle, seed=42,
        )
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
            train_set, batch_size=batch_size, shuffle=train_shuffle,
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
        sample=0.5, num_workers=0,
    )
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    for condition, target, prev_chunk, ctx_vec in train_loader:
        print(
            f"condition  {tuple(condition.shape)}  range "
            f"[{condition.min():.4f}, {condition.max():.4f}]\n"
            f"target     {tuple(target.shape)}  range "
            f"[{target.min():.4f}, {target.max():.4f}]\n"
            f"prev_chunk {tuple(prev_chunk.shape)}  range "
            f"[{prev_chunk.min():.4f}, {prev_chunk.max():.4f}]\n"
            f"ctx_vec    {tuple(ctx_vec.shape)}  values {ctx_vec.tolist()}"
        )
        c = condition[0, 0, :, :, condition.shape[-1] // 2].cpu().numpy()
        t = target[0, 0, :, :, target.shape[-1] // 2].cpu().numpy()
        p = prev_chunk[0, 0, :, :, prev_chunk.shape[-1] // 2].cpu().numpy()
        plt.figure(figsize=(12, 4))
        for i, (img, name) in enumerate([(c, "condition"), (t, "target"), (p, "prev_chunk")]):
            plt.subplot(1, 3, i + 1)
            plt.imshow(img, cmap="gray")
            plt.title(name)
            plt.axis("off")
        plt.tight_layout()
        plt.savefig("dataset_smoketest.png")
        plt.close()
        print("Saved dataset_smoketest.png")
        break
