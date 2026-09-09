"""Tests for evaluate_nii_test.py (discovery, slice metrics, resume) and
plot_eval_nii_test.py (per-volume aggregation)."""
import csv
import os
import sys

import nibabel as nib
import numpy as np
import pytest
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "experiments", "01-model-comparison"))

import evaluate_nii_test as ev


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _write_nii(path, shape=(8, 8, 4)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.random.rand(*shape).astype(np.float32)
    nib.Nifti1Image(arr, np.eye(4)).to_filename(path)


def _make_subject(root, anatomy="brain", acq="mp2rage_ax_7T", subj="sub1",
                  with_md=True, artifacts=("R3", "ANISO_phase3")):
    d = os.path.join(root, anatomy, acq, subj)
    _write_nii(os.path.join(d, f"{subj}.nii.gz"))
    if with_md:
        _write_nii(os.path.join(d, f"md{subj}.nii.gz"))
    _write_nii(os.path.join(d, f"d{subj}.nii.gz"))  # denoised-only output, never an input
    for suf in artifacts:
        _write_nii(os.path.join(d, f"{subj}_{suf}.nii.gz"))
    return d


# ----------------------------------------------------------------------------
# discovery
# ----------------------------------------------------------------------------
def test_discover_finds_subject_and_inputs(tmp_path):
    _make_subject(str(tmp_path))
    subs = ev.discover_subjects(str(tmp_path))
    assert len(subs) == 1
    s = subs[0]
    assert (s["anatomy"], s["acquisition"], s["subject"]) == (
        "brain", "mp2rage_ax_7T", "sub1")
    assert s["raw"].endswith("sub1.nii.gz")
    assert s["md"].endswith("mdsub1.nii.gz")
    names = [i["name"] for i in s["inputs"]]
    # artifact inputs discovered dynamically; raw/d*/md* are never inputs
    assert set(names) == {"R3", "ANISO_phase3"}


def test_discover_input_tasks_and_families(tmp_path):
    _make_subject(str(tmp_path))
    s = ev.discover_subjects(str(tmp_path))[0]
    by_name = {i["name"]: i for i in s["inputs"]}
    assert by_name["R3"]["tasks"] == ["raw", "md"]
    assert by_name["R3"]["artifact"] == "undersampled"
    assert by_name["ANISO_phase3"]["artifact"] == "aniso"
    assert by_name["ANISO_phase3"]["tasks"] == ["raw", "md"]
    # the clean fully sampled volume is a ground truth, never an input
    assert "raw" not in by_name


def test_discover_skips_subject_missing_md(tmp_path):
    _make_subject(str(tmp_path), subj="nomd", with_md=False)
    _make_subject(str(tmp_path), subj="ok")
    subs = ev.discover_subjects(str(tmp_path))
    assert [s["subject"] for s in subs] == ["ok"]


def test_discover_filters_anatomy_and_acquisition(tmp_path):
    _make_subject(str(tmp_path), anatomy="brain", acq="a1", subj="s1")
    _make_subject(str(tmp_path), anatomy="knee", acq="a2", subj="s2")
    subs = ev.discover_subjects(str(tmp_path), anatomies={"brain"})
    assert [s["subject"] for s in subs] == ["s1"]
    subs = ev.discover_subjects(str(tmp_path), acquisitions={"a2"})
    assert [s["subject"] for s in subs] == ["s2"]


def test_discover_lowres_pair(tmp_path):
    d = os.path.join(str(tmp_path), "brain", "bravo_ax_3T", "s3")
    _write_nii(os.path.join(d, "T1w_BRAVO_highres.nii.gz"))
    _write_nii(os.path.join(d, "mdT1w_BRAVO_highres.nii.gz"))
    _write_nii(os.path.join(d, "rT1w_BRAVO_lowres.nii.gz"))
    s = ev.discover_subjects(str(tmp_path))[0]
    by_name = {i["name"]: i for i in s["inputs"]}
    assert "lowres" in by_name
    assert by_name["lowres"]["artifact"] == "aniso"
    assert by_name["lowres"]["path"].endswith("rT1w_BRAVO_lowres.nii.gz")


# ----------------------------------------------------------------------------
# slice metrics
# ----------------------------------------------------------------------------
def test_slice_metrics_identical_volumes():
    vol = np.random.rand(32, 32, 5).astype(np.float32)
    m = ev.slice_metrics(vol, vol)
    assert len(m["ssim"]) == 5 and len(m["mae"]) == 5 and len(m["psnr"]) == 5
    assert all(abs(v - 1.0) < 1e-6 for v in m["ssim"])
    assert all(v == 0.0 for v in m["mae"])


def test_slice_metrics_noisy_volume():
    rng = np.random.default_rng(0)
    gt = rng.random((32, 32, 4), dtype=np.float32)
    out = np.clip(gt + rng.normal(0, 0.1, gt.shape).astype(np.float32), 0, 1)
    m = ev.slice_metrics(out, gt)
    for k in ("psnr", "ssim", "mae"):
        assert len(m[k]) == 4
        assert all(np.isfinite(m[k]))
    assert all(0 < v < 1 for v in m["ssim"])
    assert all(v > 0 for v in m["mae"])


# ----------------------------------------------------------------------------
# baseline input listing (list_test_inputs.py)
# ----------------------------------------------------------------------------
def test_input_paths_lists_only_degraded_inputs(tmp_path):
    _make_subject(tmp_path)
    import list_test_inputs
    paths = list_test_inputs.input_paths(str(tmp_path), {"brain"})
    assert paths, "no inputs found"
    names = [os.path.basename(p) for p in paths]
    assert all(os.path.exists(p) for p in paths)
    # only artifact volumes: never the clean GT, md target, or d* denoised
    assert not any(n.startswith(("md", "d")) for n in names)
    subs = ev.discover_subjects(str(tmp_path), {"brain"})
    assert sorted(paths) == sorted(i["path"] for s in subs for i in s["inputs"])


# ----------------------------------------------------------------------------
# empty-slice filter
# ----------------------------------------------------------------------------
def test_slice_keep_mask_drops_empty_slices():
    gt = np.zeros((10, 10, 3), dtype=np.float32)
    gt[..., 1] = 0.5                       # full slice
    gt[:2, :2, 2] = 0.5                    # 4% foreground, below 5%
    keep = ev.slice_keep_mask(gt, fg_threshold=0.05, min_fg=0.05)
    assert keep.tolist() == [False, True, False]


def test_slice_keep_mask_threshold_edge_inclusive():
    gt = np.zeros((10, 10, 1), dtype=np.float32)
    gt[:5, :1, 0] = 0.5                    # exactly 5% foreground
    assert ev.slice_keep_mask(gt, fg_threshold=0.05, min_fg=0.05).tolist() == [True]


def test_slice_keep_mask_zero_min_keeps_all():
    gt = np.zeros((4, 4, 3), dtype=np.float32)
    assert ev.slice_keep_mask(gt, fg_threshold=0.05, min_fg=0.0).all()


def test_kept_slice_metrics_scores_only_kept_slices():
    rng = np.random.default_rng(0)
    gt = np.zeros((16, 16, 4), dtype=np.float32)
    gt[..., 1] = rng.random((16, 16))
    gt[..., 3] = rng.random((16, 16))
    arr = np.clip(gt + 0.01, 0, 1)
    keep = ev.slice_keep_mask(gt)
    fake_perceptual = lambda a, g: ([0.1] * a.shape[2], [0.9] * a.shape[2])
    m, zs = ev.kept_slice_metrics(arr, gt, keep, fake_perceptual)
    assert zs.tolist() == [1, 3]
    assert all(len(m[k]) == 2 for k in ev.METRIC_FIELDS)
    assert all(np.isfinite(m["psnr"]))


def test_slice_filter_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate_nii_test.py",
                                      "--checkpoint_path", "x.pt"])
    args = ev.parse_args()
    assert args.slice_fg_threshold == 0.05
    assert args.min_slice_fg == 0.05


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_perceptual_metrics_slicewise():
    device = torch.device("cuda")
    perc = ev.PerceptualMetrics(device)
    rng = np.random.default_rng(1)
    gt = rng.random((64, 64, 3), dtype=np.float32)
    out = np.clip(gt + rng.normal(0, 0.05, gt.shape).astype(np.float32), 0, 1)
    lp, fs = perc(out, gt)
    assert len(lp) == 3 and len(fs) == 3
    assert all(np.isfinite(lp)) and all(np.isfinite(fs))
    lp_same, fs_same = perc(gt, gt)
    assert all(v < 1e-4 for v in lp_same)
    assert all(v > 0.99 for v in fs_same)


# ----------------------------------------------------------------------------
# CSV resume
# ----------------------------------------------------------------------------
def test_load_done_keys_volume_level(tmp_path):
    p = str(tmp_path / "eval.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ev.CSV_FIELDS)
        w.writeheader()
        for z in range(3):
            w.writerow({"anatomy": "brain", "acquisition": "a", "subject": "s",
                        "input": "R3", "artifact": "undersampled", "severity": "R3",
                        "task": "raw", "condition": "cfg1", "slice_idx": z,
                        "n_slices": 3, "psnr": 30, "ssim": 0.9, "mae": 0.01,
                        "lpips": 0.1, "fsim": 0.9})
    done = ev.load_done_keys(p)
    assert ("brain", "a", "s", "R3", "raw", "cfg1") in done
    assert ("brain", "a", "s", "R3", "md", "cfg1") not in done


def test_load_done_keys_missing_file(tmp_path):
    assert ev.load_done_keys(str(tmp_path / "nope.csv")) == set()


# ----------------------------------------------------------------------------
# figure aggregation
# ----------------------------------------------------------------------------
def test_aggregate_per_volume():
    import pandas as pd
    import plot_eval_nii_test as pl
    rows = []
    for cond in ("input", "cfg1"):
        for z in range(4):
            rows.append({"anatomy": "brain", "acquisition": "a", "subject": "s",
                         "input": "R3", "artifact": "undersampled",
                         "severity": "R3", "task": "raw", "condition": cond,
                         "slice_idx": z, "n_slices": 4,
                         "psnr": 30 + z, "ssim": 0.9, "mae": 0.01,
                         "lpips": 0.1, "fsim": 0.9})
    df = pd.DataFrame(rows)
    agg = pl.aggregate_per_volume(df)
    assert len(agg) == 2  # one row per (volume, task, condition)
    assert np.isclose(agg.loc[agg.condition == "cfg1", "psnr"].iloc[0], 31.5)
    assert set(agg.columns) >= {"artifact", "task", "condition", "psnr", "ssim",
                                "mae", "lpips", "fsim"}


# ----------------------------------------------------------------------------
# OOM-safe slab batching
# ----------------------------------------------------------------------------
def test_slab_batch_scales_with_slice_area():
    budget = 524288  # 2 x 512x512
    assert ev.slab_batch(240 * 512, max_batch=8, voxel_budget=budget) == 4
    assert ev.slab_batch(512 * 512, max_batch=8, voxel_budget=budget) == 2


def test_slab_batch_caps_and_floors():
    assert ev.slab_batch(96 * 96, max_batch=8, voxel_budget=524288) == 8
    assert ev.slab_batch(1024 * 1024, max_batch=8, voxel_budget=524288) == 1


def test_oom_retry_halves_batch_and_shrinks_budget():
    from types import SimpleNamespace
    args = SimpleNamespace(max_batch_size=8, voxel_budget=524288, batch_size=None)
    xy = 240 * 512  # -> initial batch 4
    seen = []

    def fn():
        seen.append(args.batch_size)
        if args.batch_size > 1:
            raise torch.OutOfMemoryError("CUDA out of memory (fake)")
        return "ok"

    assert ev.run_oom_safe(fn, args, xy) == "ok"
    assert seen == [4, 2, 1]
    # Budget shrank so the next volume of the same shape starts at batch 1.
    assert ev.slab_batch(xy, args.max_batch_size, args.voxel_budget) == 1


def test_oom_retry_reraises_at_batch_one():
    from types import SimpleNamespace
    args = SimpleNamespace(max_batch_size=8, voxel_budget=100, batch_size=None)

    def fn():
        raise torch.OutOfMemoryError("CUDA out of memory (fake)")

    with pytest.raises(torch.OutOfMemoryError):
        ev.run_oom_safe(fn, args, 512 * 512)


def test_maybe_save_writes_volume_with_ref_geometry(tmp_path):
    ref = str(tmp_path / "ref.nii.gz")
    _write_nii(ref, shape=(8, 8, 4))
    sub = {"anatomy": "brain", "acquisition": "flair_ax_3T", "subject": "s1"}
    arr = np.random.rand(8, 8, 4).astype(np.float32)
    ev.maybe_save(arr, ref, str(tmp_path / "out"), sub, "R3", "raw", "cfg1")
    dst = tmp_path / "out" / "brain" / "flair_ax_3T" / "s1" / "R3_raw_cfg1.nii.gz"
    assert dst.exists()
    img = nib.load(str(dst))
    assert img.shape == (8, 8, 4)
    assert np.allclose(img.get_fdata(), arr, atol=1e-6)
    assert np.allclose(img.affine, nib.load(ref).affine)
