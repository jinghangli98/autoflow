"""Tests for dataset_2d.py -- 2D slice pairs drawn from the whole-volume
pipeline (dataset.build_samples), for the SwinIR / Real-ESRGAN baselines."""
import os
import sys

import nibabel as nib
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataset
import dataset_2d

SHAPE = (24, 20, 12)


def _write(path, arr, zooms=(0.4, 0.4, 1.5)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(nib.Nifti1Image(arr.astype(np.float32), np.diag([*zooms, 1.0])), path)


def _ramp(scale):
    g = np.indices(SHAPE).sum(axis=0).astype(np.float32)
    return g / g.max() * scale


@pytest.fixture(scope="module")
def tiny_root(tmp_path_factory):
    """brain/tse_cor_7T subjects with raw + _R3 artifact + md target each."""
    root = str(tmp_path_factory.mktemp("nii2d"))
    for i in range(1, 7):
        d = os.path.join(root, "brain", "tse_cor_7T", f"S{i}")
        _write(f"{d}/S{i}.nii.gz", _ramp(255))
        _write(f"{d}/S{i}_R3.nii.gz", _ramp(255))
        _write(f"{d}/mdS{i}.nii.gz", _ramp(255))
    return root


def test_get_dataset_2d_slices_yields_slice_stacks(tiny_root):
    train_set, val_set = dataset_2d.get_dataset_2d_slices(
        tiny_root, "brain", size=8, slices_per_volume=3, val_fraction=0.3)
    assert len(train_set) > 0 and len(val_set) > 0
    cond, tgt, prompts, tt, anatomy, artifact = train_set[0]
    assert tuple(cond.shape) == (3, 1, 8, 8)
    assert tuple(tgt.shape) == (3, 1, 8, 8)
    assert len(prompts) == 3
    assert all("{plane}" not in p for p in prompts)
    assert tt in ("raw", "md")
    assert anatomy == "brain"


def test_get_dataset_2d_slices_index_maps_over_pairs(tiny_root):
    train_set, _ = dataset_2d.get_dataset_2d_slices(
        tiny_root, "brain", size=8, val_fraction=0.3)
    n = sum(len(v) for v in train_set.contrast_indices.values())
    assert n == len(train_set.samples)
    assert all(k[0] == "brain" for k in train_set.group_indices)


def test_fg_fraction_forwarded_and_default(tiny_root):
    train_set, val_set = dataset_2d.get_dataset_2d_slices(
        tiny_root, "brain", size=8, val_fraction=0.3, fg_fraction=0.0)
    assert train_set.fg_fraction == 0.0
    assert val_set.fg_fraction == 0.0
    train_set, _ = dataset_2d.get_dataset_2d_slices(
        tiny_root, "brain", size=8, val_fraction=0.3)
    assert train_set.fg_fraction == 0.25


def test_collate_flattens_slice_stacks(tiny_root):
    train_set, _ = dataset_2d.get_dataset_2d_slices(
        tiny_root, "brain", size=8, slices_per_volume=2, val_fraction=0.3)
    batch = dataset.collate_patches([train_set[0], train_set[1]])
    cond, tgt, prompts, tts, anatomies, artifacts = batch
    assert tuple(cond.shape) == (4, 1, 8, 8)
    assert len(prompts) == len(tts) == len(anatomies) == len(artifacts) == 4


def test_small_volume_uses_only_fitting_axes(tiny_root):
    # SHAPE (24, 20, 12) at size 16: only axis 2 leaves both in-plane dims
    # >= 16, so crops must come from there instead of raising (the 192-crop
    # failure mode on small-FOV volumes).
    train_set, _ = dataset_2d.get_dataset_2d_slices(
        tiny_root, "brain", size=16, slices_per_volume=4, val_fraction=0.3)
    cond, tgt, *_ = train_set[0]
    assert tuple(cond.shape) == (4, 1, 16, 16)


def test_val_set_is_deterministic(tiny_root):
    _, val_set = dataset_2d.get_dataset_2d_slices(
        tiny_root, "brain", size=8, slices_per_volume=2, val_fraction=0.3)
    a, b = val_set[0], val_set[0]
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_filter_target_type_raw_and_md(tiny_root):
    for want, types in (("raw", {"raw"}), ("denoised", {"denoised"}),
                        ("md", {"denoised"})):
        train_set, _ = dataset_2d.get_dataset_2d_slices(
            tiny_root, "brain", size=8, val_fraction=0.3)
        dataset_2d.filter_target_type(train_set, want)
        assert {s["target_type"] for s in train_set.samples} == types
        n = sum(len(v) for v in train_set.contrast_indices.values())
        assert n == len(train_set.samples)


def test_filter_target_type_unknown_raises(tiny_root):
    train_set, _ = dataset_2d.get_dataset_2d_slices(
        tiny_root, "brain", size=8, val_fraction=0.3)
    with pytest.raises(KeyError):
        dataset_2d.filter_target_type(train_set, "nope")
