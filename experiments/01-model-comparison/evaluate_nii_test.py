"""Slice-wise evaluation of the multi-task flow model over the nii_test split.

For every subject under `--data_root` (default /vast/tibrahim/jil202/nii_test,
the held-out 10% in the training layout <root>/<anatomy>/<acq>/<subject>/), the
degraded inputs found in each subject dir (whatever exists: GRAPPA `_R*`,
`_SPIKE_R*`, `_ANISO_{par,phase,read}*`, `r*_lowres`) are run through the
trained model and scored slice-by-slice along the LAST array dimension against
the matching ground truth:

  * task "raw" : reconstruct fully sampled     -> scored vs <S>.nii.gz
  * task "md"  : denoise + bias-correct        -> scored vs md<S>.nii.gz
    (the fully sampled input itself also runs task "md": raw -> denoised)

Per input x task, three conditions are scored:
  * input : the degraded volume itself, no model (baseline / delta anchor)
  * cfg1  : guidance_scale 1.0 -- conditioned, the prompt selects the task
  * cfg0  : guidance_scale 0.0 -- unconditioned (prompt ignored); the volume is
            task-independent, so one inference is shared by both tasks

Prompts come from dataset.prompt_for_path (exactly what enhance_flow_3d.py
--auto_prompt does), with the per-plane orientation word substituted per
`--planes` pass and TSE inputs restricted to their through-plane axis. The
model + RadBERT + torch.compile load ONCE for the whole run. Inputs and GT are
normalized with the training 0.5/99.5 percentile window; model outputs live in
[0, 1] already and are clipped.

Metrics per slice: PSNR, SSIM, MAE (skimage), LPIPS (AlexNet) and FSIM (piq)
batched on GPU. One CSV row per slice; near-empty slices are skipped (no row)
when fewer than `--min_slice_fg` of the GT voxels exceed
`--slice_fg_threshold` -- the mask comes from the task's ground truth, so
input/cfg0/cfg1 always score identical slices. Rerunning resumes (volumes
with any rows written are skipped). `--num_shards/--shard_id` split subjects
for SLURM job arrays.

Usage:
    python evaluate_nii_test.py \
        --checkpoint_path checkpoints_uncertainty/flow_matching_3d_brain_all_128ch_082926_best.pt \
        --data_root /vast/tibrahim/jil202/nii_test --anatomy brain \
        --out_csv ./eval_nii_test.csv \
        --num_sampling_steps 1 --euler --fp16 --non_overlap 3 \
        --planes axial sagittal coronal --compile
"""

import argparse
import csv
import os
import sys

import nibabel as nib
import numpy as np
import torch

# dataset.py / enhance_flow_3d.py live at the repo root, two levels up.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# RadBERT weights are pre-cached under $HF_HOME; compute nodes may be offline.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from dataset import (SINGLE_PLANE_SEQUENCES, classify_file, normalize_volume,
                     parse_acquisition, prompt_for_path)

_NII = ".nii.gz"

METRIC_FIELDS = ["psnr", "ssim", "mae", "lpips", "fsim"]
# Volume-level identity of one scored (input, task, condition) volume.
CSV_KEY_FIELDS = ["anatomy", "acquisition", "subject", "input", "task", "condition"]
CSV_FIELDS = ["anatomy", "acquisition", "subject", "input", "artifact", "severity",
              "task", "condition", "slice_idx", "n_slices"] + METRIC_FIELDS


# ----------------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------------
def discover_subjects(data_root, anatomies=None, acquisitions=None):
    """List every evaluable subject under the training layout.

    A subject is one raw stem `<S>` in `<root>/<anatomy>/<acq>/<subject>/` that
    has BOTH ground truths `<S>.nii.gz` and `md<S>.nii.gz`. Its inputs are the
    sibling artifact files classified by dataset.classify_file (any suffix
    family / severity present); the clean raw volume is never an input.

    Returns dicts: anatomy, acquisition, subject, dir, stem, raw, md,
    inputs=[{name, path, artifact, severity, tasks}].
    """
    subs = []
    for anat in sorted(os.listdir(data_root)):
        anat_dir = os.path.join(data_root, anat)
        if not os.path.isdir(anat_dir) or (anatomies and anat not in anatomies):
            continue
        for acq in sorted(os.listdir(anat_dir)):
            acq_dir = os.path.join(anat_dir, acq)
            if not os.path.isdir(acq_dir) or (acquisitions and acq not in acquisitions):
                continue
            for subj in sorted(os.listdir(acq_dir)):
                sdir = os.path.join(acq_dir, subj)
                if not os.path.isdir(sdir):
                    continue
                names = sorted(n for n in os.listdir(sdir) if n.endswith(_NII))
                stems = {n[: -len(_NII)] for n in names}
                infos = {n: classify_file(n, stems) for n in names}
                # Raw stems that have an md ground truth.
                md_stems = {i["stem"] for i in infos.values() if i["role"] == "md"}
                for stem in sorted(md_stems):
                    raw_name = stem + _NII
                    if infos.get(raw_name, {}).get("role") != "raw":
                        continue
                    inputs = [
                        {"name": info["severity"], "path": os.path.join(sdir, n),
                         "artifact": info["artifact"], "severity": info["severity"],
                         "tasks": ["raw", "md"]}
                        for n, info in infos.items()
                        if info["role"] == "artifact" and info["stem"] == stem
                    ]
                    if not inputs:
                        continue
                    subs.append({
                        "anatomy": anat, "acquisition": acq, "subject": subj,
                        "dir": sdir, "stem": stem,
                        "raw": os.path.join(sdir, raw_name),
                        "md": os.path.join(sdir, "md" + raw_name),
                        "inputs": inputs,
                    })
    return subs


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def slice_metrics(out, gt):
    """PSNR / SSIM / MAE per last-dim slice. `out`, `gt` are (X, Y, Z) arrays
    already normalized to [0, 1]. Returns {metric: [Z floats]}."""
    if out.shape != gt.shape:
        raise ValueError(f"shape mismatch: out {out.shape} vs gt {gt.shape}")
    res = {"psnr": [], "ssim": [], "mae": []}
    for z in range(gt.shape[2]):
        o, g = out[..., z], gt[..., z]
        res["psnr"].append(float(peak_signal_noise_ratio(g, o, data_range=1.0)))
        res["ssim"].append(float(structural_similarity(g, o, data_range=1.0)))
        res["mae"].append(float(np.mean(np.abs(o - g))))
    return res


def slice_keep_mask(gt, fg_threshold=0.05, min_fg=0.05):
    """Boolean mask over last-dim slices: True where the fraction of GT
    voxels above `fg_threshold` reaches `min_fg`. Near-empty slices (air)
    are excluded from scoring; `min_fg=0` keeps every slice. Computed on
    the ground truth so all conditions of a task score identical slices."""
    fg = (gt > fg_threshold).mean(axis=(0, 1))
    return fg >= min_fg


def kept_slice_metrics(arr, gt, keep, perceptual):
    """All metrics on the kept slices only. Returns ({metric: [k floats]},
    original slice indices) with lists ordered like the indices."""
    zs = np.flatnonzero(keep)
    a = np.ascontiguousarray(arr[..., zs])
    g = np.ascontiguousarray(gt[..., zs])
    m = slice_metrics(a, g)
    m["lpips"], m["fsim"] = perceptual(a, g)
    return m, zs


class PerceptualMetrics:
    """Per-slice LPIPS (AlexNet) and FSIM along the last dim, GPU-batched.

    Inputs are (X, Y, Z) arrays in [0, 1]. Returns two lists of Z floats.
    Either backend failing to load disables that metric (NaN) instead of
    aborting the run.
    """

    def __init__(self, device, chunk=16):
        self.device = device
        self.chunk = chunk
        self.lpips = None
        self.fsim = None
        try:
            import lpips
            self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        except Exception as e:  # noqa: BLE001
            print(f"[warn] LPIPS disabled: {type(e).__name__}: {e}")
        try:
            import piq
            self.fsim = piq.fsim
        except Exception as e:  # noqa: BLE001
            print(f"[warn] FSIM disabled: {type(e).__name__}: {e}")

    @torch.no_grad()
    def __call__(self, out, gt):
        n = out.shape[2]
        if self.lpips is None and self.fsim is None:
            return [float("nan")] * n, [float("nan")] * n
        # (X, Y, Z) -> (Z, 1, X, Y) slice batch in [0, 1].
        o = torch.from_numpy(np.ascontiguousarray(out.transpose(2, 0, 1))).unsqueeze(1)
        g = torch.from_numpy(np.ascontiguousarray(gt.transpose(2, 0, 1))).unsqueeze(1)
        lp_vals, fs_vals = [], []
        for i in range(0, n, self.chunk):
            ob = o[i:i + self.chunk].to(self.device)
            gb = g[i:i + self.chunk].to(self.device)
            if self.lpips is not None:
                lo = ob.repeat(1, 3, 1, 1) * 2 - 1
                lg = gb.repeat(1, 3, 1, 1) * 2 - 1
                lp_vals.append(self.lpips(lo, lg).flatten().cpu())
            if self.fsim is not None:
                fs_vals.append(self.fsim(ob, gb, data_range=1.0, chromatic=False,
                                         reduction="none").flatten().cpu())
        lp = ([float(v) for v in torch.cat(lp_vals)] if lp_vals
              else [float("nan")] * n)
        fs = ([float(v) for v in torch.cat(fs_vals)] if fs_vals
              else [float("nan")] * n)
        return lp, fs


# ----------------------------------------------------------------------------
# CSV resume
# ----------------------------------------------------------------------------
def load_done_keys(csv_path):
    """Volume-level keys already present in the CSV (any slice row counts;
    volumes are written atomically slice-block at a time)."""
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                done.add(tuple(row[k] for k in CSV_KEY_FIELDS))
    return done


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------
def slab_batch(xy_voxels, max_batch, voxel_budget):
    """Slabs per forward for one plane pass: as many as fit `voxel_budget`
    in-plane voxels (batch * slice-XY), floored at 1, capped at `max_batch`.
    Keeps big-XY volumes (e.g. 512x512 FLAIR highres) inside 40 GB GPUs while
    small volumes still batch up."""
    return max(1, min(max_batch, voxel_budget // max(1, xy_voxels)))


def run_oom_safe(fn, args, xy_voxels):
    """Run `fn` with `args.batch_size` set from the voxel budget; on CUDA OOM,
    halve the batch (shrinking `args.voxel_budget` so later volumes start
    lower too) and retry. Re-raises once batch 1 itself OOMs."""
    while True:
        args.batch_size = slab_batch(xy_voxels, args.max_batch_size,
                                     args.voxel_budget)
        try:
            return fn()
        except torch.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if args.batch_size <= 1:
                raise
            args.voxel_budget = xy_voxels * (args.batch_size // 2)
            print(f"  [warn] CUDA OOM at slab batch {args.batch_size}; "
                  f"retrying at {args.batch_size // 2} "
                  f"(voxel budget -> {args.voxel_budget})")


def condition_label(gs):
    return f"cfg{gs:g}".replace("cfg0", "cfg0").replace("cfg1", "cfg1")


def restrict_planes(planes, meta):
    """TSE-style single-plane acquisitions were only slabbed along their
    through-plane axis in training; drop other requested planes."""
    if len(meta["thin_axes"]) >= 3:
        return list(planes)
    allowed = [meta["axis_planes"][a] for a in meta["thin_axes"]]
    kept = [p for p in planes if p in allowed]
    return kept or allowed


def infer(vol_t, planes, template, text_conditioner, sampler, args, device,
          guidance_scale, axis_planes):
    """Mean-ensembled inference over `planes` at one guidance scale, with the
    plane word substituted into the prompt per pass. Returns (X, Y, Z) numpy."""
    from enhance_flow_3d import build_context_emb, plane_prompt, run_plane_inference
    args.guidance_scale = guidance_scale
    outs = []
    for plane in planes:
        slab_axis = list(axis_planes).index(plane)
        xy_voxels = 1
        for i, d in enumerate(vol_t.shape):
            if i != slab_axis:
                xy_voxels *= ((int(d) + 15) // 16) * 16  # padded in-plane dims
        ctx_emb, null_emb = build_context_emb(
            text_conditioner, plane_prompt(template, plane), device, torch.float32)
        out, _ = run_oom_safe(
            lambda: run_plane_inference(vol_t, plane, ctx_emb, null_emb,
                                        sampler, args, device, snapshot_idx=None,
                                        snapshot_path=None,
                                        axis_planes=axis_planes),
            args, xy_voxels)
        outs.append(out)
    ens = outs[0] if len(outs) == 1 else torch.stack(outs, 0).mean(0)
    return ens.numpy()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--data_root", default="/vast/tibrahim/jil202/nii_test")
    p.add_argument("--out_csv", default="./eval_nii_test.csv")
    p.add_argument("--save_dir", default=None,
                   help="If set, write every model output volume here (large).")

    p.add_argument("--num_sampling_steps", type=int, default=1)
    p.add_argument("--non_overlap", type=int, default=3)
    p.add_argument("--euler", action="store_true")
    p.add_argument("--heun", action="store_true")
    p.add_argument("--rk4", action="store_true")
    p.add_argument("--sigma_min", type=float, default=0.001)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Max slabs per forward. The effective batch is capped "
                        "so batch * slice-XY-voxels <= --voxel_budget and is "
                        "halved automatically on CUDA OOM.")
    p.add_argument("--voxel_budget", type=int, default=524288,
                   help="In-plane voxels per forward (batch * padded slice XY); "
                        "default 2*512*512 fits a 40 GB A100 for the 128ch "
                        "model. Shrinks automatically after an OOM.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--norm_percentiles", type=float, nargs=2, default=(0.5, 99.5))
    p.add_argument("--slice_fg_threshold", type=float, default=0.05,
                   help="Normalized intensity above which a GT voxel counts "
                        "as foreground for the empty-slice filter.")
    p.add_argument("--min_slice_fg", type=float, default=0.05,
                   help="Skip slices whose GT foreground fraction is below "
                        "this (no CSV row). 0.0 scores every slice.")
    p.add_argument("--guidance_scale", type=float, nargs="+", default=[1.0, 0.0],
                   help="Guidance scales per input (default: conditioned 1.0 "
                        "-> 'cfg1' rows, unconditioned 0.0 -> 'cfg0' rows).")
    p.add_argument("--planes", nargs="+",
                   default=["axial", "sagittal", "coronal"],
                   choices=["axial", "coronal", "sagittal"])

    p.add_argument("--anatomy", nargs="+", default=["brain"])
    p.add_argument("--acquisition", nargs="+", default=None)
    p.add_argument("--max_subjects", type=int, default=None)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--dry_run", action="store_true",
                   help="List subjects/inputs and exit without running the model.")
    return p.parse_args()


def maybe_save(arr, ref_path, save_dir, sub, input_name, task, condition):
    ref = nib.load(ref_path)
    dst_dir = os.path.join(save_dir, sub["anatomy"], sub["acquisition"], sub["subject"])
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, f"{input_name}_{task}_{condition}.nii.gz")
    nib.Nifti1Image(arr.astype(np.float32), ref.affine, ref.header).to_filename(dst)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.max_batch_size = args.batch_size
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError(f"shard_id {args.shard_id} out of range [0, {args.num_shards})")

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

    if args.dry_run:
        for s in subjects:
            names = [i["name"] for i in s["inputs"]]
            print(f"  {s['anatomy']}/{s['acquisition']}/{s['subject']} "
                  f"[{s['stem']}]: {names}")
        print(f"Guidance scales: {args.guidance_scale}; planes: {args.planes}")
        return

    # --- model (once) ---
    from enhance_flow_3d import FlowMatcher, load_model, affine_axis_planes
    print(f"Loading checkpoint: {args.checkpoint_path}")
    model, text_conditioner, _ = load_model(args.checkpoint_path, device, args)
    if text_conditioner is None:
        raise ValueError("evaluate_nii_test.py needs a text-conditioned "
                         "checkpoint (prompts select the task).")
    flow = FlowMatcher(model, sigma_min=args.sigma_min,
                       amp_enabled=args.fp16, amp_device=device.type)
    sampler = flow.sample_rk4 if args.rk4 else (
        flow.sample_heun if args.heun else flow.sample_euler)
    perceptual = PerceptualMetrics(device)
    pct = tuple(args.norm_percentiles)

    # CSV (append + resume).
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

    conditions = [(condition_label(g), g) for g in args.guidance_scale]

    for si, sub in enumerate(subjects):
        tag = f"{sub['anatomy']}/{sub['acquisition']}/{sub['subject']}"
        print(f"\n[{si + 1}/{len(subjects)}] {tag} [{sub['stem']}]")
        gt_for = {
            "raw": normalize_volume(nib.load(sub["raw"]).get_fdata().astype(np.float32), pct),
            "md": normalize_volume(nib.load(sub["md"]).get_fdata().astype(np.float32), pct),
        }
        keep_for = {t: slice_keep_mask(g, args.slice_fg_threshold,
                                       args.min_slice_fg)
                    for t, g in gt_for.items()}
        print("  slices kept: " + ", ".join(
            f"{t} {int(k.sum())}/{k.size}" for t, k in keep_for.items()))

        for inp in sub["inputs"]:
            key_of = lambda task, cond: (sub["anatomy"], sub["acquisition"],
                                         sub["subject"], inp["name"], task, cond)
            wanted = [(t, c) for t in inp["tasks"] for c, _ in
                      [("input", None)] + conditions if key_of(t, c) not in done]
            if not wanted:
                continue

            def emit(task, condition, arr, save=False):
                if key_of(task, condition) in done:
                    return
                gt = gt_for[task]
                if arr.shape != gt.shape:
                    print(f"  [skip] {inp['name']} {task} {condition}: shape "
                          f"{arr.shape} != GT {gt.shape}")
                    return
                keep = keep_for[task]
                if not keep.any():
                    print(f"  [skip] {inp['name']} {task} {condition}: no "
                          f"slice reaches --min_slice_fg {args.min_slice_fg}")
                    done.add(key_of(task, condition))
                    return
                arr = np.clip(arr, 0.0, 1.0)
                m, zs = kept_slice_metrics(arr, gt, keep, perceptual)
                n = gt.shape[2]
                for j, z in enumerate(zs):
                    writer.writerow({
                        "anatomy": sub["anatomy"], "acquisition": sub["acquisition"],
                        "subject": sub["subject"], "input": inp["name"],
                        "artifact": inp["artifact"], "severity": inp["severity"] or "",
                        "task": task, "condition": condition,
                        "slice_idx": int(z), "n_slices": n,
                        **{k: m[k][j] for k in METRIC_FIELDS},
                    })
                csv_f.flush()
                done.add(key_of(task, condition))
                means = {k: float(np.nanmean(m[k])) for k in METRIC_FIELDS}
                print(f"  {inp['name']:14s} {task:3s} {condition:>6s}  " +
                      " ".join(f"{k.upper()}={means[k]:.4f}" for k in METRIC_FIELDS))
                if args.save_dir and save:
                    maybe_save(arr, inp["path"], args.save_dir, sub,
                               inp["name"], task, condition)

            # Prompt metadata (also carries the TSE plane restriction).
            meta = {t: prompt_for_path(inp["path"], target=t) for t in inp["tasks"]}
            planes = restrict_planes(args.planes, meta[inp["tasks"][0]])
            axis_planes = meta[inp["tasks"][0]]["axis_planes"]

            vol = normalize_volume(nib.load(inp["path"]).get_fdata().astype(np.float32), pct)
            vol_t = torch.from_numpy(vol)

            # Baseline: degraded input vs GT, no model.
            for task in inp["tasks"]:
                emit(task, "input", vol)

            for cond, gs in conditions:
                if abs(gs) < 1e-9:
                    # Unconditioned: prompt-independent, share one inference.
                    if all(key_of(t, cond) in done for t in inp["tasks"]):
                        continue
                    out = infer(vol_t, planes, meta[inp["tasks"][0]]["prompt"],
                                text_conditioner, sampler, args, device, gs,
                                axis_planes)
                    for task in inp["tasks"]:
                        emit(task, cond, out, save=True)
                else:
                    for task in inp["tasks"]:
                        if key_of(task, cond) in done:
                            continue
                        out = infer(vol_t, planes, meta[task]["prompt"],
                                    text_conditioner, sampler, args, device, gs,
                                    axis_planes)
                        emit(task, cond, out, save=True)

    csv_f.close()
    print(f"\nDone. Metrics written to {args.out_csv}")


if __name__ == "__main__":
    main()
