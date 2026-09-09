"""Tests for the whole-volume NIfTI dataset (dataset.py).

A tiny synthetic tree mirrors /vast/tibrahim/jil202/nii:
    <root>/<anatomy>/<acquisition>/<subject>/<file>.nii.gz
with the file-naming variants seen in the real data.
"""

import os

import nibabel as nib
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import dataset

SHAPE = (24, 20, 12)


def _write(path, arr, zooms=(1.0, 1.0, 1.0)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = nib.Nifti1Image(arr.astype(np.float32), np.diag([*zooms, 1.0]))
    nib.save(img, path)


def _ramp(scale, offset=0.0):
    """A deterministic, spatially varying volume in [offset, offset+scale]."""
    g = np.indices(SHAPE).sum(axis=0).astype(np.float32)
    return offset + scale * g / g.max()


@pytest.fixture(scope="module")
def tiny_root(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("nii"))
    b = os.path.join(root, "brain")

    # 3T BRAVO: two sessions of the same patient, real low-res `r*` sibling.
    for sess in ("PRT1_2019.01.01", "PRT1_2019.09.09"):
        d = os.path.join(b, "bravo_ax_3T", sess)
        z = (0.86, 0.86, 1.2)
        _write(f"{d}/T1w_BRAVO_highres.nii.gz", _ramp(25000), z)
        _write(f"{d}/dT1w_BRAVO_highres.nii.gz", _ramp(25000), z)
        _write(f"{d}/mdT1w_BRAVO_highres.nii.gz", _ramp(255), z)
        _write(f"{d}/rT1w_BRAVO_lowres.nii.gz", _ramp(1600), z)

    # 7T MP2RAGE: stem carries a `_UNI_DEN` suffix; several artifact severities.
    d = os.path.join(b, "mp2rage_ax_7T", "S1")
    z = (0.55, 0.55, 0.55)
    _write(f"{d}/S1_UNI_DEN.nii.gz", _ramp(4000), z)
    _write(f"{d}/S1_UNI_DEN_R3.nii.gz", _ramp(4000), z)
    _write(f"{d}/S1_UNI_DEN_SPIKE_R4.nii.gz", _ramp(4000), z)
    _write(f"{d}/S1_UNI_DEN_ANISO_par3.5.nii.gz", _ramp(4000), z)
    _write(f"{d}/dS1_UNI_DEN.nii.gz", _ramp(25000), z)
    _write(f"{d}/mdS1_UNI_DEN.nii.gz", _ramp(255), z)

    # 7T TSE without a denoised+biascorrected target (only `d*`).
    d = os.path.join(b, "tse_cor_7T", "S2")
    z = (0.38, 0.38, 1.5)
    _write(f"{d}/S2.nii.gz", _ramp(255), z)
    _write(f"{d}/S2_SPIKE_R1.nii.gz", _ramp(255), z)
    _write(f"{d}/dS2.nii.gz", _ramp(25000), z)

    # Several more 7T subjects so a val split has something to pick.
    for i in range(3, 9):
        d = os.path.join(b, "tse_cor_7T", f"S{i}")
        _write(f"{d}/S{i}.nii.gz", _ramp(255), z)
        _write(f"{d}/S{i}_R3.nii.gz", _ramp(255), z)
        _write(f"{d}/mdS{i}.nii.gz", _ramp(255), z)

    # Another anatomy with raw/d/md only.
    d = os.path.join(root, "prostate", "tse_ax_3T", "001")
    _write(f"{d}/001.nii.gz", _ramp(1000))
    _write(f"{d}/d001.nii.gz", _ramp(25000))
    _write(f"{d}/md001.nii.gz", _ramp(255))
    return root


# --------------------------------------------------------------------------
# File classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name, role, stem, family, severity", [
    ("T1w_BRAVO_highres.nii.gz", "raw", "T1w_BRAVO_highres", None, None),
    ("dT1w_BRAVO_highres.nii.gz", "denoised", "T1w_BRAVO_highres", None, None),
    ("mdT1w_BRAVO_highres.nii.gz", "md", "T1w_BRAVO_highres", None, None),
    ("rT1w_BRAVO_lowres.nii.gz", "artifact", "T1w_BRAVO_highres", "aniso", "lowres"),
    ("S1_UNI_DEN.nii.gz", "raw", "S1_UNI_DEN", None, None),
    ("S1_UNI_DEN_R3.nii.gz", "artifact", "S1_UNI_DEN", "undersampled", "R3"),
    ("S1_UNI_DEN_SPIKE_R4.nii.gz", "artifact", "S1_UNI_DEN", "spike", "SPIKE_R4"),
    ("0578BM156_SPIKE_R4.5.nii.gz", "artifact", "0578BM156", "spike", "SPIKE_R4.5"),
    ("110172_SPIKE_R3.5.nii.gz", "artifact", "110172", "spike", "SPIKE_R3.5"),
    ("S1_UNI_DEN_ANISO_par3.5.nii.gz", "artifact", "S1_UNI_DEN", "aniso", "ANISO_par3.5"),
    ("2033AM_ANISO_read4.nii.gz", "artifact", "2033AM", "aniso", "ANISO_read4"),
    ("d2033AM.nii.gz", "denoised", "2033AM", None, None),
    ("md2033AM.nii.gz", "md", "2033AM", None, None),
])
def test_classify_file(name, role, stem, family, severity):
    info = dataset.classify_file(name)
    assert info["role"] == role
    assert info["stem"] == stem
    assert info["artifact"] == family
    assert info["severity"] == severity


def test_classify_file_ignores_non_nifti():
    assert dataset.classify_file("notes.txt") is None


# --------------------------------------------------------------------------
# Prompt construction from the folder name + header
# --------------------------------------------------------------------------

def test_target_prompt_raw_bravo():
    p = dataset.build_target_prompt("raw", "brain", "bravo_ax_3T", (0.86, 0.86, 1.2),
                                    plane="sagittal")
    assert p == ("Fully sampled sagittal 3T brain T1-weighted BRAVO MRI of "
                 "resolution 0.86 x 0.86 x 1.2 mm.")


def test_target_prompt_tof_sequence_described():
    p = dataset.build_target_prompt("raw", "brain", "tof_ax_7T", (0.38, 0.38, 0.38),
                                    plane="axial")
    assert p == ("Fully sampled axial 7T brain Time-of-Flight angiography MRI "
                 "of resolution 0.38 x 0.38 x 0.38 mm.")


def test_target_prompt_plane_ignores_folder_orientation():
    p = dataset.build_target_prompt("md", "brain", "tse_cor_7T", (0.38, 0.38, 1.5),
                                    plane="axial")
    assert p == ("Denoised and biascorrected axial 7T brain T2-weighted TSE "
                 "MRI of resolution 0.38 x 0.38 x 1.5 mm.")


def test_target_prompt_knee_pdfs():
    p = dataset.build_target_prompt("raw", "knee", "pdfs_cor_1.5T", (0.5, 0.5, 3.0),
                                    plane="coronal")
    assert p.startswith("Fully sampled coronal 1.5T knee ")
    assert "Proton Density" in p and "fat" in p.lower()


def test_axis_planes_from_affine():
    ras = np.diag([1.0, 1.0, 1.0, 1.0])
    assert dataset.axis_planes(ras) == ["sagittal", "coronal", "axial"]
    # LSP storage (tse_cor_7T): axis 1 runs S/I, axis 2 runs A/P.
    lsp = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])
    assert dataset.axis_planes(lsp) == ["sagittal", "axial", "coronal"]


def test_input_phrase_says_anisotropic():
    assert dataset._input_phrase("aniso") == "anisotropic undersampled"


# --------------------------------------------------------------------------
# Patient grouping / split
# --------------------------------------------------------------------------

def test_patient_key_collapses_sessions():
    assert dataset.patient_key("PRT170088_2018.11.03") == "PRT170088"
    assert dataset.patient_key("PRT170088_2019.10.05") == "PRT170088"
    assert dataset.patient_key("0854SK84_2") == "0854SK84_2"


def test_split_is_deterministic_and_partitions(tiny_root):
    tr = dataset.build_samples(tiny_root, "brain", split="train",
                               val_fraction=0.3, split_seed=1)
    va = dataset.build_samples(tiny_root, "brain", split="val",
                               val_fraction=0.3, split_seed=1)
    tr2 = dataset.build_samples(tiny_root, "brain", split="train",
                                val_fraction=0.3, split_seed=1)
    assert [s["condition_path"] for s in tr] == [s["condition_path"] for s in tr2]
    tr_pat = {s["patient"] for s in tr}
    va_pat = {s["patient"] for s in va}
    assert tr_pat and va_pat
    assert not (tr_pat & va_pat)
    all_pat = {s["patient"] for s in
               dataset.build_samples(tiny_root, "brain", split="all")}
    assert tr_pat | va_pat == all_pat


def test_split_keeps_sessions_of_one_patient_together(tiny_root):
    # Whatever side PRT1 lands on, both of its sessions must be there.
    for split in ("train", "val"):
        sess = {s["subject"] for s in
                dataset.build_samples(tiny_root, "brain", split=split,
                                      val_fraction=0.5, split_seed=3)
                if s["patient"] == "PRT1"}
        assert sess in (set(), {"PRT1_2019.01.01", "PRT1_2019.09.09"})


# --------------------------------------------------------------------------
# Pair enumeration
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def brain_samples(tiny_root):
    return dataset.build_samples(tiny_root, "brain", split="all")


def _pairs(samples, subject):
    return {(os.path.basename(s["condition_path"]),
             os.path.basename(s["target_path"]), s["task"], s["artifact"])
            for s in samples if s["subject"] == subject}


def test_bravo_pairs_use_lowres_as_aniso_and_ignore_d(brain_samples):
    assert _pairs(brain_samples, "PRT1_2019.01.01") == {
        ("rT1w_BRAVO_lowres.nii.gz", "T1w_BRAVO_highres.nii.gz", "artifact2raw", "aniso"),
        ("rT1w_BRAVO_lowres.nii.gz", "mdT1w_BRAVO_highres.nii.gz", "artifact2denoised", "aniso"),
        ("T1w_BRAVO_highres.nii.gz", "mdT1w_BRAVO_highres.nii.gz", "raw2denoised", "clean"),
    }


def test_mp2rage_pairs_cover_every_artifact(brain_samples):
    got = _pairs(brain_samples, "S1")
    assert ("S1_UNI_DEN_R3.nii.gz", "S1_UNI_DEN.nii.gz", "artifact2raw", "undersampled") in got
    assert ("S1_UNI_DEN_SPIKE_R4.nii.gz", "mdS1_UNI_DEN.nii.gz", "artifact2denoised", "spike") in got
    assert ("S1_UNI_DEN_ANISO_par3.5.nii.gz", "S1_UNI_DEN.nii.gz", "artifact2raw", "aniso") in got
    assert ("S1_UNI_DEN.nii.gz", "mdS1_UNI_DEN.nii.gz", "raw2denoised", "clean") in got
    assert len(got) == 3 * 2 + 1


def test_missing_md_yields_only_artifact2raw(brain_samples):
    assert _pairs(brain_samples, "S2") == {
        ("S2_SPIKE_R1.nii.gz", "S2.nii.gz", "artifact2raw", "spike"),
    }


def test_sample_prompt_and_metadata(brain_samples):
    s = next(s for s in brain_samples
             if s["subject"] == "S1" and s["severity"] == "SPIKE_R4"
             and s["task"] == "artifact2raw")
    assert s["prompt"] == (
        "Input: spiking artifact brain MRI. Target: Fully sampled {plane} 7T "
        "brain T1-weighted MP2RAGE MRI of resolution 0.55 x 0.55 x 0.55 mm.")
    assert s["axis_planes"] == ["sagittal", "coronal", "axial"]
    assert s["anatomy"] == "brain"
    assert s["acquisition"] == "mp2rage_ax_7T"
    clean = next(s for s in brain_samples if s["subject"] == "S1"
                 and s["task"] == "raw2denoised")
    assert clean["prompt"].startswith("Input: fully sampled brain MRI. "
                                      "Target: Denoised and biascorrected ")


def test_other_anatomy_without_artifacts(tiny_root):
    ps = dataset.build_samples(tiny_root, "prostate", split="all")
    assert [(p["task"], p["artifact"]) for p in ps] == [("raw2denoised", "clean")]
    assert "prostate" in ps[0]["prompt"]


def test_missing_anatomy_raises(tiny_root):
    with pytest.raises(FileNotFoundError):
        dataset.build_samples(tiny_root, "liver")


# --------------------------------------------------------------------------
# Per-volume normalization
# --------------------------------------------------------------------------

def test_normalize_volume_uses_percentiles_of_that_volume():
    rng = np.random.default_rng(0)
    vol = rng.uniform(100.0, 1100.0, size=(40, 40, 10)).astype(np.float32)
    vol[0, 0, 0] = 1e6  # outlier must not drive the scale
    out = dataset.normalize_volume(vol, percentiles=(0.5, 99.5))
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0
    lo, hi = np.percentile(vol, [0.5, 99.5])
    ref = np.clip((vol - lo) / (hi - lo), 0, 1)
    assert np.abs(out - ref).mean() < 0.02


def test_normalize_volume_is_independent_of_input_scale():
    vol = _ramp(1.0)
    a = dataset.normalize_volume(vol * 255.0)
    b = dataset.normalize_volume(vol * 25000.0)
    assert np.allclose(a, b, atol=1e-4)


# --------------------------------------------------------------------------
# Dataset: random thin patches from whole volumes
# --------------------------------------------------------------------------

def test_getitem_returns_k_aligned_patches(brain_samples):
    ds = dataset.VolumePairDataset(brain_samples, patch_shape=(16, 16, 7),
                                   patches_per_volume=3)
    i = next(i for i, s in enumerate(brain_samples)
             if s["subject"] == "S1" and s["task"] == "raw2denoised")
    cond, tgt, prompt, ttype, anatomy, artifact = ds[i]
    assert tuple(cond.shape) == (3, 1, 16, 16, 7)
    assert tuple(tgt.shape) == (3, 1, 16, 16, 7)
    assert cond.dtype == torch.float32
    # raw (0-4000) and md (0-255) hold the same ramp; after per-volume
    # normalization aligned crops must match, proving shared coordinates.
    assert torch.allclose(cond, tgt, atol=1e-3)
    assert ttype == "denoised" and anatomy == "brain" and artifact == "clean"
    assert len(prompt) == 3
    planes = ["sagittal", "coronal", "axial"]
    for p, (axis, _) in zip(prompt, ds.last_crops):
        assert p == brain_samples[i]["prompt"].format(plane=planes[axis])


def test_patch_range_is_subset_of_unit_interval(brain_samples):
    ds = dataset.VolumePairDataset(brain_samples, patch_shape=(8, 8, 4),
                                   patches_per_volume=2)
    cond, tgt, *_ = ds[0]
    assert cond.min() >= 0.0 and cond.max() <= 1.0
    # A small crop of a ramp cannot span the whole [0, 1] range.
    assert cond.max() - cond.min() < 0.9


def test_train_crops_vary_but_val_crops_are_fixed(brain_samples):
    train = dataset.VolumePairDataset(brain_samples, patch_shape=(8, 8, 4),
                                      patches_per_volume=4, deterministic=False)
    val = dataset.VolumePairDataset(brain_samples, patch_shape=(8, 8, 4),
                                    patches_per_volume=4, deterministic=True)
    c1, *_ = train[0]
    c2, *_ = train[0]
    assert not torch.equal(c1, c2)
    v1, *_ = val[0]
    v2, *_ = val[0]
    assert torch.equal(v1, v2)


def test_patch_shape_larger_than_volume_raises(brain_samples):
    ds = dataset.VolumePairDataset(brain_samples, patch_shape=(96, 96, 7))
    with pytest.raises(ValueError):
        ds[0]


# --------------------------------------------------------------------------
# Loader glue
# --------------------------------------------------------------------------

def test_collate_flattens_volume_batches(brain_samples):
    ds = dataset.VolumePairDataset(brain_samples, patch_shape=(8, 8, 4),
                                   patches_per_volume=3)
    loader = DataLoader(ds, batch_size=2, collate_fn=dataset.collate_patches)
    cond, tgt, prompts, ttypes, anatomies, artifacts = next(iter(loader))
    assert tuple(cond.shape) == (6, 1, 8, 8, 4)
    assert tuple(tgt.shape) == (6, 1, 8, 8, 4)
    assert len(prompts) == len(ttypes) == len(anatomies) == len(artifacts) == 6
    assert all("{plane}" not in p for p in prompts)
    assert all(any(pl in p for pl in ("sagittal", "coronal", "axial")) for p in prompts)


def test_get_dataset_3d_patches_indices_and_filter(tiny_root):
    train_set, val_set = dataset.get_dataset_3d_patches(
        tiny_root, ["brain", "prostate"], sample=100.0, patch_shape=(8, 8, 4),
        val_fraction=0.3, artifacts=["spike"])
    assert set(train_set.contrast_indices) == {"brain", "prostate"}
    arts = {k[1] for k in train_set.group_indices}
    assert arts <= {"spike", "clean"}
    assert all(s["artifact"] in ("spike", "clean") for s in train_set.samples)
    assert len(val_set.samples) > 0
    n = sum(len(v) for v in train_set.contrast_indices.values())
    assert n == len(train_set.samples)


def test_getloader_yields_flattened_batches(tiny_root):
    train_loader, val_loader = dataset.getloader_3d_patches(
        batch_size=2, data_root=tiny_root, contrast="brain", sample=100.0,
        num_workers=0, patch_shape=(8, 8, 4), patches_per_volume=2,
        val_fraction=0.3)
    cond, tgt, prompts, *_ = next(iter(train_loader))
    assert tuple(cond.shape) == (4, 1, 8, 8, 4)
    assert len(prompts) == 4
    assert hasattr(train_loader.dataset, "samples")


def test_getloader_forwards_fg_fraction(tiny_root):
    train_loader, val_loader = dataset.getloader_3d_patches(
        batch_size=2, data_root=tiny_root, contrast="brain", sample=100.0,
        num_workers=0, patch_shape=(8, 8, 4), val_fraction=0.3,
        fg_fraction=0.0)
    assert train_loader.dataset.fg_fraction == 0.0
    assert val_loader.dataset.fg_fraction == 0.0


# --------------------------------------------------------------------------
# Multi-plane thin patches: the 7-voxel axis may be any array axis
# --------------------------------------------------------------------------

def test_thin_axes_all_planes_except_tse(brain_samples):
    by_acq = {s["acquisition"]: s["thin_axes"] for s in brain_samples}
    assert by_acq["bravo_ax_3T"] == [0, 1, 2]
    assert by_acq["mp2rage_ax_7T"] == [0, 1, 2]
    assert by_acq["tse_cor_7T"] == [2]        # thickest zoom (1.5 mm) is axis 2


def _axis_ramp_root(tmp_path, axis):
    """Volume whose value depends only on the index along `axis`."""
    vol = np.indices(SHAPE)[axis].astype(np.float32)
    _write(str(tmp_path / "c.nii.gz"), vol * 3.0)
    _write(str(tmp_path / "t.nii.gz"), vol * 200.0)
    return [{"condition_path": str(tmp_path / "c.nii.gz"),
             "target_path": str(tmp_path / "t.nii.gz"), "prompt": "Target: {plane} p",
             "target_type": "raw", "anatomy": "brain", "artifact": "clean",
             "thin_axes": [axis]}]


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_thin_axis_is_moved_last(tmp_path, axis):
    samples = _axis_ramp_root(tmp_path, axis)
    ds = dataset.VolumePairDataset(samples, patch_shape=(8, 8, 4),
                                   patches_per_volume=2, deterministic=True)
    cond, tgt, *_ = ds[0]
    assert tuple(cond.shape) == (2, 1, 8, 8, 4)
    p = cond[0, 0]
    # Varies only along the last (thin) axis: 4 distinct planes, each constant.
    assert torch.equal(p[..., 0].unique(), p[0, 0, 0].reshape(1))
    assert len(p[0, 0, :].unique()) == 4
    assert torch.allclose(cond, tgt, atol=1e-3)
    _, _, prompts, *_ = ds[0]
    assert prompts == ["Target: %s p" % ["sagittal", "coronal", "axial"][axis]] * 2


def test_all_feasible_planes_are_used(brain_samples):
    i = next(i for i, s in enumerate(brain_samples) if s["acquisition"] == "mp2rage_ax_7T")
    ds = dataset.VolumePairDataset(brain_samples, patch_shape=(8, 8, 4),
                                   patches_per_volume=30, deterministic=True)
    seen = set()
    for a in ds.thin_axes_used(i):
        seen.add(a)
    assert seen == {0, 1, 2}


def test_infeasible_planes_are_skipped(brain_samples):
    # SHAPE=(24, 20, 12): a 16x16 in-plane crop only fits with axis 2 thin.
    i = next(i for i, s in enumerate(brain_samples) if s["acquisition"] == "mp2rage_ax_7T")
    ds = dataset.VolumePairDataset(brain_samples, patch_shape=(16, 16, 4),
                                   patches_per_volume=10, deterministic=True)
    assert set(ds.thin_axes_used(i)) == {2}
    cond, *_ = ds[i]
    assert tuple(cond.shape) == (10, 1, 16, 16, 4)


def test_no_feasible_plane_raises(brain_samples):
    ds = dataset.VolumePairDataset(brain_samples, patch_shape=(24, 24, 4))
    with pytest.raises(ValueError):
        ds[0]


def test_last_crops_records_axis_and_start(brain_samples):
    ds = dataset.VolumePairDataset(brain_samples, patch_shape=(8, 8, 4),
                                   patches_per_volume=3, deterministic=True)
    ds[0]
    assert len(ds.last_crops) == 3
    assert [a for a, _ in ds.last_crops] == ds.thin_axes_used(0)
    for axis, start in ds.last_crops:
        assert len(start) == 3
        assert all(0 <= s < n for s, n in zip(start, SHAPE))


def test_shape_mismatched_pairs_are_skipped(tmp_path, capsys):
    root = str(tmp_path / "nii")
    d = os.path.join(root, "brain", "mprage_ax_7T", "S9")
    _write(f"{d}/S9.nii.gz", _ramp(255))
    _write(f"{d}/mdS9.nii.gz", _ramp(255))
    _write(f"{d}/S9_R3.nii.gz", _ramp(255))
    bigger = np.zeros((SHAPE[0] + 8, SHAPE[1], SHAPE[2]), np.float32)
    _write(f"{d}/S9_SPIKE_R2.nii.gz", bigger)
    samples = dataset.build_samples(root, "brain", split="all")
    got = {(os.path.basename(s["condition_path"]), s["task"]) for s in samples}
    assert got == {("S9_R3.nii.gz", "artifact2raw"), ("S9_R3.nii.gz", "artifact2denoised"),
                   ("S9.nii.gz", "raw2denoised")}
    out = capsys.readouterr().out
    assert "S9_SPIKE_R2" in out and "shape" in out
