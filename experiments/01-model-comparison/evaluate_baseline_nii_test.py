"""Slice-wise metrics for a 2D baseline's (SwinIR / Real-ESRGAN) nii_test outputs.

Companion of evaluate_nii_test.py for the supervised baselines: those are run
separately by enhance_swinir.sh / enhance_realesrgan.sh, which write one
restored volume per degraded input into a mirror tree

    <enhanced_root>/<anatomy>/<acq>/<subject>/<input file>.nii.gz

This script walks the same subjects (evaluate_nii_test.discover_subjects, so
the baseline is scored on exactly the volumes the flow model is scored on)
and, for the ONE task the baseline was trained for, scores per last-dim slice
against that task's ground truth:

  * task "raw" : <S>.nii.gz    (baseline trained with target raw)
  * task "md"  : md<S>.nii.gz  (baseline trained with target denoised / md)

Two conditions per input: "input" (the degraded volume itself, no model --
the same anchor rows the flow CSVs carry) and `--condition` (the baseline's
restored volume, e.g. "swinir" / "realesrgan"). Ground truth and input use
the training 0.5/99.5 percentile window; the baseline outputs were saved in
that window already (enhance_*.py --norm percentile) and are only clipped.

Metrics, empty-slice filter (--min_slice_fg on the task GT), CSV columns and
resume semantics are shared with evaluate_nii_test.py, so the CSVs can be
concatenated with the flow ones and fed to plot_eval_nii_test.py. Missing
baseline outputs are reported and skipped (no rows), so the script can run
before an enhance job array has fully finished and be rerun to fill in.

Usage:
    python evaluate_baseline_nii_test.py \
        --enhanced_root ./enhanced_nii_test_swinir_raw \
        --condition swinir --task raw \
        --out_csv ./eval_nii_test_swinir_raw.csv
"""

import argparse
import csv
import os
import sys

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# dataset.py lives at the repo root, two levels up.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from dataset import normalize_volume
from evaluate_nii_test import (CSV_FIELDS, METRIC_FIELDS, PerceptualMetrics,
                               discover_subjects, kept_slice_metrics,
                               load_done_keys, slice_keep_mask)

TASK_ALIASES = {"raw": "raw", "md": "md", "denoised": "md"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Score a 2D baseline's nii_test output tree slice-wise.")
    p.add_argument("--enhanced_root", required=True,
                   help="Mirror tree written by enhance_swinir.sh / "
                        "enhance_realesrgan.sh.")
    p.add_argument("--condition", required=True,
                   help="Condition label for the baseline rows (e.g. swinir).")
    p.add_argument("--task", required=True, choices=sorted(TASK_ALIASES),
                   help="Ground truth the baseline was trained for: raw "
                        "(fully sampled) or md/denoised (denoised+biascorr).")
    p.add_argument("--data_root", default="/vast/tibrahim/jil202/nii_test")
    p.add_argument("--out_csv", default="./eval_nii_test_baseline.csv")
    p.add_argument("--anatomy", nargs="+", default=["brain"])
    p.add_argument("--acquisition", nargs="+", default=None)
    p.add_argument("--max_subjects", type=int, default=None)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--norm_percentiles", type=float, nargs=2, default=(0.5, 99.5))
    p.add_argument("--slice_fg_threshold", type=float, default=0.05)
    p.add_argument("--min_slice_fg", type=float, default=0.05,
                   help="Skip slices whose GT foreground fraction is below "
                        "this (no CSV row). 0.0 scores every slice.")
    p.add_argument("--renormalize_enhanced", action="store_true",
                   help="Percentile-normalize the baseline outputs too "
                        "(only for legacy --norm max outputs; percentile-"
                        "mode outputs are already in the training window).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry_run", action="store_true",
                   help="List subjects/inputs and which outputs exist; no scoring.")
    return p.parse_args()


def main():
    args = parse_args()
    task = TASK_ALIASES[args.task]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError(f"shard_id {args.shard_id} out of range [0, {args.num_shards})")
    pct = tuple(args.norm_percentiles)

    subjects = discover_subjects(args.data_root, set(args.anatomy or []),
                                 set(args.acquisition or []))
    total = len(subjects)
    if args.max_subjects:
        subjects = subjects[:args.max_subjects]
    if args.num_shards > 1:
        subjects = subjects[args.shard_id::args.num_shards]
        print(f"Found {total} subjects; shard {args.shard_id}/{args.num_shards} "
              f"handles {len(subjects)}")
    else:
        print(f"Found {total} subjects under {args.data_root}")
    print(f"Scoring task '{task}' condition '{args.condition}' from {args.enhanced_root}")

    def enhanced_path(inp):
        rel = os.path.relpath(inp["path"], args.data_root)
        return os.path.join(args.enhanced_root, rel)

    if args.dry_run:
        n_have = n_all = 0
        for s in subjects:
            for inp in s["inputs"]:
                ok = os.path.isfile(enhanced_path(inp))
                n_all += 1
                n_have += ok
                print(f"  {'ok     ' if ok else 'MISSING'} "
                      f"{s['anatomy']}/{s['acquisition']}/{s['subject']}/{inp['name']}")
        print(f"{n_have}/{n_all} baseline outputs present")
        return

    perceptual = PerceptualMetrics(device)

    done = load_done_keys(args.out_csv)
    if done:
        print(f"Resuming: {len(done)} volumes already scored, will skip them.")
    write_header = not os.path.exists(args.out_csv)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    csv_f = open(args.out_csv, "a", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()
        csv_f.flush()

    missing = []
    for si, sub in enumerate(subjects):
        tag = f"{sub['anatomy']}/{sub['acquisition']}/{sub['subject']}"
        keys = [(sub["anatomy"], sub["acquisition"], sub["subject"],
                 inp["name"], task, cond)
                for inp in sub["inputs"] for cond in ("input", args.condition)]
        if all(k in done for k in keys):
            continue
        print(f"\n[{si + 1}/{len(subjects)}] {tag} [{sub['stem']}]")
        gt = normalize_volume(nib.load(sub[task]).get_fdata().astype(np.float32), pct)
        keep = slice_keep_mask(gt, args.slice_fg_threshold, args.min_slice_fg)
        print(f"  slices kept: {int(keep.sum())}/{keep.size}")

        for inp in sub["inputs"]:
            def key_of(cond):
                return (sub["anatomy"], sub["acquisition"], sub["subject"],
                        inp["name"], task, cond)

            def emit(condition, arr):
                if key_of(condition) in done:
                    return
                if arr.shape != gt.shape:
                    print(f"  [skip] {inp['name']} {condition}: shape "
                          f"{arr.shape} != GT {gt.shape}")
                    return
                if not keep.any():
                    print(f"  [skip] {inp['name']} {condition}: no slice "
                          f"reaches --min_slice_fg {args.min_slice_fg}")
                    done.add(key_of(condition))
                    return
                arr = np.clip(arr, 0.0, 1.0)
                m, zs = kept_slice_metrics(arr, gt, keep, perceptual)
                for j, z in enumerate(zs):
                    writer.writerow({
                        "anatomy": sub["anatomy"], "acquisition": sub["acquisition"],
                        "subject": sub["subject"], "input": inp["name"],
                        "artifact": inp["artifact"], "severity": inp["severity"] or "",
                        "task": task, "condition": condition,
                        "slice_idx": int(z), "n_slices": gt.shape[2],
                        **{k: m[k][j] for k in METRIC_FIELDS},
                    })
                csv_f.flush()
                done.add(key_of(condition))
                means = {k: float(np.nanmean(m[k])) for k in METRIC_FIELDS}
                print(f"  {inp['name']:14s} {task:3s} {condition:>11s}  " +
                      " ".join(f"{k.upper()}={means[k]:.4f}" for k in METRIC_FIELDS))

            if key_of("input") not in done:
                vol = normalize_volume(
                    nib.load(inp["path"]).get_fdata().astype(np.float32), pct)
                emit("input", vol)

            if key_of(args.condition) not in done:
                ep = enhanced_path(inp)
                if not os.path.isfile(ep):
                    print(f"  [missing] {inp['name']}: {ep}")
                    missing.append(ep)
                    continue
                out = nib.load(ep).get_fdata().astype(np.float32)
                if args.renormalize_enhanced:
                    out = normalize_volume(out, pct)
                elif out.max() > 1.5:
                    print(f"  [warn] {inp['name']}: baseline output max "
                          f"{out.max():.1f} > 1 -- not in the training window "
                          f"(legacy --norm max output?); clipping will wreck "
                          f"the scores. Consider --renormalize_enhanced.")
                emit(args.condition, out)

    csv_f.close()
    if missing:
        print(f"\n[warn] {len(missing)} baseline outputs missing (no rows written); "
              f"rerun after the enhance job finishes to fill them in.")
    print(f"\nDone. Metrics written to {args.out_csv}")


if __name__ == "__main__":
    main()
