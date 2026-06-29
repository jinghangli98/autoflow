"""Whole-slice SuperFormer inference with the pretrained GitHub weights.

Drives the released SuperFormer (SuperFormer/models/SuperFormer_Weights.pth) on a
single NIfTI volume, one anatomical plane at a time. Two design choices, per the
model's structure:

  * In-plane: the whole slice is fed at full resolution -- NO tiling, NO cropping.
    The only adjustment is reflect-padding each in-plane axis up to a multiple of
    16 (= patch_size(2) * window_size(8), the network's grid requirement); the pad
    is removed before saving, so the output keeps the input's in-plane size.

  * Through-plane: handled in slabs of `--through_plane` voxels (a multiple of 16
    you can modulate). The volume is processed in slabs along the chosen plane's
    through-axis and blended with overlap-and-add (`--overlap`).

The released model (patch_size=2) bakes its token grid from `img_size` at build
time, so we construct the net once at (padded_H, padded_W, through_plane) and feed
slabs of exactly that shape. The checkpoint's `attn_mask` buffers are grid-size
specific; they are dropped on load (strict=False) and the net recomputes the
correct masks for the constructed size. All *learned* weights load with zero
missing/unexpected keys.

Usage:
    python enhance_superformer_3d.py \
        --input_path /vast/tibrahim/jil202/autoflow/rT1w_BRAVO_lowres.nii.gz \
        --output_path ./outputs/rT1w_BRAVO.nii.gz \
        --plane axial --through_plane 16 --overlap 8 --fp16 
"""

import argparse
import os
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, "/vast/tibrahim/jil202/autoflow/SuperFormer")
from models.SuperFormer import SuperFormer

# In-plane and through-plane dims must be multiples of patch_size*window_size.
GRID_MULT = 16
WEIGHTS = "/vast/tibrahim/jil202/autoflow/SuperFormer/models/SuperFormer_Weights.pth"

# Released config (options/test/test_superformer.json) minus img_size, which we
# set per-run to (padded_H, padded_W, through_plane).
SUPERFORMER_CFG = dict(
    upscale=1,
    patch_size=2,
    in_chans=1,
    window_size=8,
    img_range=1.0,
    depths=[6, 6, 6],
    embed_dim=252,
    num_heads=[6, 6, 6],
    mlp_ratio=2,
    upsampler=None,
    resi_connection="1conv",
    ape=False,
    rpb=True,
    output_type="direct",
    num_feat=126,
)

# Per-plane permutations moving the through-plane axis to the LAST dim of an
# (X, Y, Z) volume, plus the inverse permutation to undo it:
#   axial    through=Z -> (X, Y, Z)   identity
#   coronal  through=Y -> (X, Z, Y)
#   sagittal through=X -> (Y, Z, X)
PLANE_PERMS = {
    "axial":    ((0, 1, 2), (0, 1, 2)),
    "coronal":  ((0, 2, 1), (0, 2, 1)),
    "sagittal": ((1, 2, 0), (2, 0, 1)),
}


def build_model(args, img_size, device):
    """Construct SuperFormer at `img_size` and load the pretrained weights.

    `attn_mask` buffers in the checkpoint are specific to the 64^3 release grid;
    they are dropped so the net keeps the masks it computed for `img_size`. The
    learned weights are window-based and load with 0 missing/unexpected keys.
    """
    model = SuperFormer(img_size=img_size, **SUPERFORMER_CFG)
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    sd = {k: v for k, v in sd.items() if "attn_mask" not in k}
    sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if args.compile:
        model = torch.compile(model)
    else:
        model = model
        
    missing = [k for k in missing if "attn_mask" not in k]
    unexpected = [k for k in unexpected if "attn_mask" not in k]
    if missing or unexpected:
        raise RuntimeError(
            f"Unexpected weight mismatch (img_size={img_size}): "
            f"missing={missing[:4]} unexpected={unexpected[:4]}")
    return model.to(device).eval()


def ceil_mult(n, mult):
    return ((n + mult - 1) // mult) * mult


def pad_inplane(vol, mult):
    """Reflect-pad the first two axes (in-plane) of a 3D tensor (H, W, T) up to a
    multiple of `mult`. Returns padded tensor and the (h, w) pad amounts."""
    h, w = vol.shape[0], vol.shape[1]
    ph, pw = ceil_mult(h, mult) - h, ceil_mult(w, mult) - w
    v = vol.unsqueeze(0).unsqueeze(0)
    # F.pad order is (last_dim..., first_dim): (T_l, T_r, W_l, W_r, H_l, H_r)
    v = F.pad(v, (0, 0, 0, pw, 0, ph), mode="replicate")
    return v[0, 0], (ph, pw)


def slab_starts(n, thick, stride):
    """Start indices covering [0, n) with `thick`-thick slabs at `stride`, with a
    final slab flush to the end. Requires n >= thick."""
    starts = []
    z = 0
    while z + thick <= n:
        starts.append(z)
        z += stride
    if not starts or starts[-1] + thick < n:
        starts.append(n - thick)
    return starts


def make_blend_window(thick, overlap, dtype=torch.float32):
    """1D through-plane blend weights: linear ramps over the first/last `overlap`
    samples, 1.0 in the middle, strictly positive so the divide is safe."""
    w = torch.ones(thick, dtype=dtype)
    if overlap > 0:
        ramp = torch.arange(1, overlap + 1, dtype=dtype) / (overlap + 1)
        w[:overlap] = ramp
        w[-overlap:] = ramp.flip(0)
    return w


@torch.no_grad()
def run_plane(args, vol_t, plane, through_plane, overlap, device, fp16):
    """Run one plane: permute so the through-axis is last, pad in-plane, slab the
    through-axis, blend, then permute back. Returns an (X, Y, Z) tensor."""
    perm, inv_perm = PLANE_PERMS[plane]
    permuted = vol_t.permute(*perm).contiguous()          # (H, W, T_full)
    padded, (ph, pw) = pad_inplane(permuted, GRID_MULT)   # in-plane -> mult of 16
    H, W, T_full = padded.shape

    # Pad the through-axis up to at least one slab thickness.
    if T_full < through_plane:
        padded = F.pad(padded.unsqueeze(0).unsqueeze(0),
                       (0, through_plane - T_full, 0, 0, 0, 0),
                       mode="replicate")[0, 0]
    T_pad = padded.shape[2]

    img_size = (H, W, through_plane)
    print(f"  [{plane}] in-plane={permuted.shape[:2]} -> padded {(H, W)}, "
          f"through={T_full} (slab {through_plane}), img_size={img_size}")
    model = build_model(args, img_size, device)

    stride = max(1, through_plane - overlap)
    starts = slab_starts(T_pad, through_plane, stride)
    window = make_blend_window(through_plane, overlap, dtype=padded.dtype).view(1, 1, -1)

    out_sum = torch.zeros_like(padded)
    out_w = torch.zeros_like(padded)
    for z0 in tqdm(starts, desc=f"{plane} slabs"):
        slab = padded[:, :, z0:z0 + through_plane]
        inp = slab.unsqueeze(0).unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=fp16):
            out = model(inp)
        out_slab = out[0, 0].float().cpu()
        out_sum[:, :, z0:z0 + through_plane] += out_slab * window
        out_w[:, :, z0:z0 + through_plane] += window

    out = out_sum / out_w.clamp(min=1e-8)
    out = out[:H - ph, :W - pw, :T_full]                  # unpad in-plane + through
    return out.permute(*inv_perm).contiguous()


def parse_args():
    p = argparse.ArgumentParser(
        description="Pretrained SuperFormer inference, one plane at a time.")
    p.add_argument("--input_path", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--plane", type=str, default="axial",
                   choices=["axial", "coronal", "sagittal"],
                   help="Through-plane axis: axial=Z, coronal=Y, sagittal=X.")
    p.add_argument("--through_plane", type=int, default=16,
                   help="Through-plane slab thickness in voxels (rounded up to a "
                        "multiple of 16). Modulate this to trade context vs. cost.")
    p.add_argument("--overlap", type=int, default=0,
                   help="Through-plane overlap (voxels) between slabs; stride = "
                        "through_plane - overlap. Larger = smoother seams.")
    p.add_argument("--norm_div", type=float, default=None,
                   help="Divisor to normalize to [0,1] (default: per-volume max). "
                        "Output is multiplied back by it before saving.")
    p.add_argument("--fp16", action="store_true",
                   help="Run forwards under bf16 autocast (model stays fp32).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    through_plane = ceil_mult(max(GRID_MULT, args.through_plane), GRID_MULT)
    if through_plane != args.through_plane:
        print(f"through_plane rounded {args.through_plane} -> {through_plane} "
              f"(multiple of {GRID_MULT})")
    if not (0 <= args.overlap < through_plane):
        raise ValueError(f"--overlap must be in [0, {through_plane - 1}]")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading input: {args.input_path}")
    img = nib.load(args.input_path)
    raw = img.get_fdata().astype(np.float32)
    print(f"  shape={raw.shape}, range=[{raw.min():.2f}, {raw.max():.2f}]")

    norm_div = args.norm_div if args.norm_div is not None else float(raw.max())
    if norm_div <= 0:
        raise ValueError(f"Normalization divisor must be > 0; got {norm_div}")
    vol_t = torch.from_numpy(raw / norm_div).float()
    print(f"  plane={args.plane}, through_plane={through_plane}, "
          f"overlap={args.overlap}, norm_div={norm_div:.4f}")

    out = run_plane(args, vol_t, args.plane, through_plane, args.overlap, device, args.fp16)

    arr = (out.numpy() * norm_div).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or ".", exist_ok=True)
    nib.Nifti1Image(arr, img.affine, img.header).to_filename(args.output_path)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
