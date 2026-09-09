"""Whole-volume 2D Real-ESRGAN inference (sibling of enhance_swinir.py).

Drives a Real-ESRGAN baseline trained by train_realesrgan.py over a single
NIfTI volume, one 2D slice at a time. RRDBNet is fully convolutional (no
window/attention grid), so unlike enhance_swinir.py this does NOT need to
rebuild the model per input size -- it is built once from the checkpoint's
saved `args` (nf/nb/gc) and reused for every plane and volume.

The only size constraint is RRDBNet's internal scale=1 path, which needs H
and W divisible by 4 (see realesrgan_train_step.py / train_realesrgan.py);
this script reflect-pads each plane's in-plane axes up to a multiple of 4
before inference and crops back to the input's original size afterward.

Normalization mirrors enhance_swinir.py: the volume is divided by its max (or
--norm_div) into ~[0, 1] for the net, then multiplied back before saving.

Usage:
    python enhance_realesrgan.py \
        --input_path /vast/tibrahim/jil202/autoflow/rT1w_BRAVO_lowres.nii.gz \
        --output_path ./outputs/rT1w_BRAVO_realesrgan_raw.nii.gz \
        --checkpoint_path  /vast/tibrahim/jil202/autoflow/checkpoints_realesrgan/realesrgan_2d_raw_brain_all_final.pt \
        --plane axial --batch_size 16 --fp16
"""

import argparse
import os
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

from basicsr.archs.rrdbnet_arch import RRDBNet

# dataset.py (training normalization) lives at the repo root, two levels up.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

GRID_MULT = 4  # RRDBNet's scale=1 path pixel-unshuffles by 4; in-plane dims
               # fed to it must be a multiple of this.

# Per-plane permutations moving the slice (stack) axis to the LAST dim of an
# (X, Y, Z) volume, plus the inverse permutation to undo it:
#   axial    slices along Z -> (X, Y, Z)   identity
#   coronal  slices along Y -> (X, Z, Y)
#   sagittal slices along X -> (Y, Z, X)
PLANE_PERMS = {
    "axial":    ((0, 1, 2), (0, 1, 2)),
    "coronal":  ((0, 2, 1), (0, 2, 1)),
    "sagittal": ((1, 2, 0), (2, 0, 1)),
}

# Array axis -> plane label used by PLANE_PERMS (slices along that axis).
AXIS_TO_PLANE = {0: "sagittal", 1: "coronal", 2: "axial"}


def select_planes(input_path, plane, ensemble):
    """Planes to restore and average. Without --ensemble: just `plane`.
    With it: one pass per array axis the 2D trainer sliced along
    (dataset.prompt_for_path thin_axes -- all three axes for 3D acquisitions,
    only the through-plane axis for single-plane TSE), mirroring the flow
    model's three-plane mean ensemble in evaluate_nii_test.py. Falls back to
    all three axes if the path is not in the training layout."""
    if not ensemble:
        return [plane]
    try:
        from dataset import prompt_for_path
        axes = prompt_for_path(input_path)["thin_axes"]
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] plane metadata unavailable ({type(e).__name__}: {e}); "
              f"ensembling all three axes")
        axes = [0, 1, 2]
    return [AXIS_TO_PLANE[a] for a in sorted(axes)]


def build_model(checkpoint_path, device):
    """Construct RRDBNet from a train_realesrgan.py checkpoint and load its
    generator weights. Fully convolutional -- no size-dependent buffers, so
    (unlike enhance_swinir.py) one instance serves every plane and volume.
    Accepts this script's save format ({"G": state_dict, "args": ...}) or a
    raw state_dict; only shape-matching keys are loaded (strict=False)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    is_dict = isinstance(ckpt, dict)
    cfg = ckpt.get("args", {}) if is_dict else {}

    model = RRDBNet(num_in_ch=1, num_out_ch=1, scale=1,
                    num_feat=cfg.get("nf", 64), num_block=cfg.get("nb", 23),
                    num_grow_ch=cfg.get("gc", 32))

    sd = ckpt["G"] if is_dict and "G" in ckpt else ckpt
    sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}
    model_sd = model.state_dict()
    matched = {k: v for k, v in sd.items()
               if k in model_sd and model_sd[k].shape == v.shape}
    skipped = sorted(set(sd) - set(matched))
    missing = sorted(set(model_sd) - set(matched))
    model.load_state_dict(matched, strict=False)

    n = sum(p.numel() for p in model.parameters())
    print(f"  RRDBNet(2D): {n / 1e6:.2f}M params, nf={cfg.get('nf', 64)}, "
          f"nb={cfg.get('nb', 23)}, gc={cfg.get('gc', 32)}")
    print(f"  Loaded {len(matched)}/{len(model_sd)} tensors from {checkpoint_path}; "
          f"skipped {len(skipped)} from ckpt, {len(missing)} left at init.")
    if is_dict and ckpt.get("epoch") is not None:
        print(f"    (source checkpoint epoch {ckpt['epoch']})")
    if missing:
        print(f"    WARNING missing keys (first few): {missing[:4]}")
    return model.to(device).eval()


def ceil_mult(n, mult):
    return ((n + mult - 1) // mult) * mult


def pad_inplane(vol, mult):
    """Reflect-pad the first two axes (in-plane) of a (H, W, N) tensor up to a
    multiple of `mult`. Returns the padded tensor and the (h, w) pad amounts."""
    h, w = vol.shape[0], vol.shape[1]
    ph, pw = ceil_mult(h, mult) - h, ceil_mult(w, mult) - w
    v = vol.permute(2, 0, 1).unsqueeze(0)          # (1, N, H, W)
    v = F.pad(v, (0, pw, 0, ph), mode="reflect")   # pad W then H
    v = v[0].permute(1, 2, 0).contiguous()         # (H+ph, W+pw, N)
    return v, (ph, pw)


@torch.no_grad()
def run_plane(model, vol_t, plane, batch_size, device, fp16):
    """Restore one plane: permute so the slice axis is last, pad in-plane to a
    multiple of GRID_MULT, run each in-plane slice through RRDBNet in
    batches, crop back, then permute back to (X, Y, Z)."""
    perm, inv_perm = PLANE_PERMS[plane]
    permuted = vol_t.permute(*perm).contiguous()       # (H, W, N)
    padded, (ph, pw) = pad_inplane(permuted, GRID_MULT)
    H, W, N = padded.shape
    stack = padded.permute(2, 0, 1).unsqueeze(1)       # (N, 1, H, W)
    print(f"  [{plane}] in-plane={permuted.shape[:2]} -> padded ({H}, {W}), "
          f"slices={N}, batch_size={batch_size}")

    outs = []
    for i in tqdm(range(0, N, batch_size), desc=f"{plane} slices"):
        batch = stack[i:i + batch_size].to(device)
        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=fp16):
            out = model(batch)
        outs.append(out.float().cpu())

    out = torch.cat(outs, 0)[:, 0]                     # (N, H, W)
    out = out.permute(1, 2, 0).contiguous()            # (H, W, N)
    Horig, Worig = permuted.shape[0], permuted.shape[1]
    out = out[:Horig, :Worig, :]                        # crop pad
    return out.permute(*inv_perm).contiguous()


def normalize_input(raw, mode, norm_div=None, percentiles=(0.5, 99.5)):
    """Normalize the input volume for the net; returns (volume, denorm fn).

    'percentile': the training window (dataset.normalize_volume, clipped into
    [0, 1]); denorm clips the net output back into [0, 1] and the saved
    volume stays in that window, like the flow model's outputs. Use for
    checkpoints trained on the whole-volume pipeline (dataset_2d.py).
    'max': legacy divide-by-max (or `norm_div`); denorm multiplies back to
    the input scale. Use for checkpoints trained on max-normalized slices."""
    if mode == "percentile":
        from dataset import normalize_volume
        vol = normalize_volume(raw.astype(np.float32), tuple(percentiles))
        return vol, lambda out: np.clip(out, 0.0, 1.0).astype(np.float32)
    div = float(norm_div) if norm_div is not None else float(raw.max())
    if div <= 0:
        raise ValueError(f"Normalization divisor must be > 0; got {div}")
    return ((raw / div).astype(np.float32),
            lambda out: (out * div).astype(np.float32))


def parse_args():
    p = argparse.ArgumentParser(
        description="2D Real-ESRGAN whole-volume inference, one slice at a time.")
    p.add_argument("--input_path", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--checkpoint_path", type=str, required=True,
                   help="train_realesrgan.py checkpoint (.pt) to run.")
    p.add_argument("--plane", type=str, default="axial",
                   choices=["axial", "coronal", "sagittal"],
                   help="Slice (stack) axis: axial=Z, coronal=Y, sagittal=X.")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Number of slices fed through the net at once.")
    p.add_argument("--norm", choices=["percentile", "max"], default="percentile",
                   help="Input normalization: 'percentile' = the training "
                        "0.5/99.5 window (checkpoints from the whole-volume "
                        "dataset_2d pipeline; output saved in [0,1]); 'max' = "
                        "legacy divide-by-max for old checkpoints (output "
                        "multiplied back to input scale).")
    p.add_argument("--norm_percentiles", type=float, nargs=2, default=(0.5, 99.5),
                   help="Percentile window for --norm percentile.")
    p.add_argument("--norm_div", type=float, default=None,
                   help="--norm max only: divisor to normalize to [0,1] "
                        "(default: per-volume max).")
    p.add_argument("--ensemble", action="store_true",
                   help="Mean-ensemble over every array axis the 2D "
                        "trainer sliced along (3 planes for 3D "
                        "acquisitions, the through-plane axis only for "
                        "TSE); overrides --plane.")
    p.add_argument("--fp16", action="store_true",
                   help="Run forwards under bf16 autocast (model stays fp32).")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(args.checkpoint_path, device)

    print(f"Loading input: {args.input_path}")
    img = nib.load(args.input_path)
    raw = img.get_fdata().astype(np.float32)
    print(f"  shape={raw.shape}, range=[{raw.min():.2f}, {raw.max():.2f}]")

    vol, denorm = normalize_input(raw, args.norm, args.norm_div,
                                  tuple(args.norm_percentiles))
    vol_t = torch.from_numpy(vol).float()
    print(f"  batch_size={args.batch_size}, norm={args.norm}")

    planes = select_planes(args.input_path, args.plane, args.ensemble)
    print(f"  planes: {planes}" + (" (mean ensemble)" if len(planes) > 1 else ""))
    outs = [run_plane(model, vol_t, pl, args.batch_size, device, args.fp16)
            for pl in planes]
    out = outs[0] if len(outs) == 1 else torch.stack(outs, 0).mean(0)

    arr = denorm(out.numpy())
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or ".", exist_ok=True)
    nib.Nifti1Image(arr, img.affine, img.header).to_filename(args.output_path)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
