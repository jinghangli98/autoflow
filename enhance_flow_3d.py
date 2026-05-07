"""3D NIfTI flow matching inference on un-patched whole volumes.

Loads one .nii.gz, normalizes [0, 255] -> [0, 1], pads X/Y/Z to multiples of 16,
slides 16-thick chunks along Z with full XY in one shot per chunk, runs flow
matching with autoregressive prev_chunk, stitches along Z, unpads, and saves.

Usage:
    python enhance_flow_3d.py \
        --checkpoint_path ./checkpoints/flow_matching_3d_mprage.pt \
        --input_path /home/rflab/jil202/grappa-recon/nii/mprage/532_5_R5.nii.gz \
        --output_path ./outputs/532_5_R5_recon.nii.gz \
        --num_sampling_steps 1 --euler --auto --fp16
"""

import argparse
import os
import re

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DiffusionModelUNet
from tqdm import tqdm


PATCH_Z = 16
PAD_MULT = 16
CONTEXT_INPUT_DIM = 5
CONTEXT_HIDDEN_DIM = 128
CONTEXT_OUTPUT_DIM = 256

_ACCEL_FILENAME_RE = re.compile(r"_(CS_)?R(\d+)$")


class ContextEncoder(nn.Module):
    """Mirror of training-time ContextEncoder."""

    def __init__(self, in_dim=CONTEXT_INPUT_DIM, hidden_dim=CONTEXT_HIDDEN_DIM,
                 out_dim=CONTEXT_OUTPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, ctx_vec):
        return self.net(ctx_vec).unsqueeze(1)


def parse_accel_from_filename(input_path):
    """Parse (is_cs, factor) from filenames like 'subject_R5.nii.gz' or 'subject_CS_R3.nii.gz'."""
    name = os.path.basename(input_path)
    name = re.sub(r"\.nii(\.gz)?$", "", name)
    m = _ACCEL_FILENAME_RE.search(name)
    if not m:
        raise ValueError(f"Cannot parse accel suffix from {input_path}")
    is_cs = m.group(1) is not None
    factor = int(m.group(2))
    return is_cs, factor


def voxel_size_from_nii(img):
    z = img.header.get_zooms()
    return float(z[0]), float(z[1]), float(z[2])


class AutoregressiveFlowMatcher(nn.Module):
    """Slim 3D Flow Matching wrapper for inference (Euler / Heun / RK4) with context."""

    def __init__(self, model, sigma_min=0.001):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min

    @torch.no_grad()
    def _step_input(self, x, condition, prev_chunk):
        return torch.cat([x, condition, prev_chunk], dim=1)

    @torch.no_grad()
    def sample_euler(self, condition, prev_chunk, num_steps, device, context=None):
        x = torch.randn_like(condition, device=device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((condition.shape[0],), i * dt, device=device)
            v = self.model(self._step_input(x, condition, prev_chunk),
                           (t * 999).long(), context=context)
            x = x + v * dt
        return x

    @torch.no_grad()
    def sample_heun(self, condition, prev_chunk, num_steps, device, context=None):
        x = torch.randn_like(condition, device=device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((condition.shape[0],), i * dt, device=device)
            v1 = self.model(self._step_input(x, condition, prev_chunk),
                            (t * 999).long(), context=context)
            x_euler = x + v1 * dt
            t_next = torch.full((condition.shape[0],), min((i + 1) * dt, 1.0), device=device)
            v2 = self.model(self._step_input(x_euler, condition, prev_chunk),
                            (t_next * 999).long(), context=context)
            x = x + dt * (v1 + v2) / 2.0
        return x

    @torch.no_grad()
    def sample_rk4(self, condition, prev_chunk, num_steps, device, context=None):
        x = torch.randn_like(condition, device=device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t_cur = torch.full((condition.shape[0],), i * dt, device=device)
            k1 = self.model(self._step_input(x, condition, prev_chunk),
                            (t_cur * 999).long(), context=context)
            t_mid = torch.full((condition.shape[0],), i * dt + 0.5 * dt, device=device)
            k2 = self.model(self._step_input(x + 0.5 * dt * k1, condition, prev_chunk),
                            (t_mid * 999).long(), context=context)
            k3 = self.model(self._step_input(x + 0.5 * dt * k2, condition, prev_chunk),
                            (t_mid * 999).long(), context=context)
            t_next = torch.full((condition.shape[0],), min((i + 1) * dt, 1.0), device=device)
            k4 = self.model(self._step_input(x + dt * k3, condition, prev_chunk),
                            (t_next * 999).long(), context=context)
            x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        return x


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--input_path", type=str, required=True,
                   help="Single un-patched .nii.gz file")
    p.add_argument("--output_path", type=str, required=True)

    p.add_argument("--num_sampling_steps", type=int, default=1)
    p.add_argument("--euler", action="store_true",
                   help="Use Euler ODE (default if neither --heun nor --rk4)")
    p.add_argument("--heun", action="store_true")
    p.add_argument("--rk4", action="store_true")

    p.add_argument("--auto", action="store_true",
                   help="Use generated previous chunk as prev_chunk for next step")

    p.add_argument("--sigma_min", type=float, default=0.001)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--rescale", action="store_true",
                   help="Rescale output back to ~[0, 255] uint16 before save")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_model(checkpoint_path, device, args):
    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=1,
        channels=(128, 256, 256),
        attention_levels=(False, False, False),
        num_res_blocks=2,
        num_head_channels=256,
        with_conditioning=True,
        cross_attention_dim=CONTEXT_OUTPUT_DIM,
    )
    context_encoder = ContextEncoder()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not (isinstance(ckpt, dict) and "model" in ckpt):
        raise ValueError(
            f"Checkpoint at {checkpoint_path} does not contain 'model' / 'context_encoder' "
            f"keys. Re-train with the context-aware train_flow.py."
        )

    model_sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(model_sd)
    if "context_encoder" in ckpt:
        ctx_sd = {k.replace("module.", "").replace("_orig_mod.", ""): v
                  for k, v in ckpt["context_encoder"].items()}
        context_encoder.load_state_dict(ctx_sd)

    model.to(device).eval()
    context_encoder.to(device).eval()

    if args.fp16:
        model = model.half()
        context_encoder = context_encoder.half()
    if args.compile:
        model = torch.compile(model, backend="inductor")

    return model, context_encoder


def pad_to_multiple(volume: torch.Tensor, mult: int):
    """Center-pad a 3D tensor (X, Y, Z) so each dim is a multiple of `mult`."""
    def _amount(n):
        rem = n % mult
        if rem == 0:
            return 0, 0
        total = mult - rem
        l = total // 2
        return l, total - l

    x_l, x_r = _amount(volume.shape[0])
    y_l, y_r = _amount(volume.shape[1])
    z_l, z_r = _amount(volume.shape[2])
    padded = F.pad(volume, (z_l, z_r, y_l, y_r, x_l, x_r), mode="constant", value=0.0)
    return padded, ((x_l, x_r), (y_l, y_r), (z_l, z_r)), volume.shape


def unpad(volume: torch.Tensor, pad_info, original_shape):
    (x_l, x_r), (y_l, y_r), (z_l, z_r) = pad_info
    ox, oy, oz = original_shape
    return volume[x_l:x_l + ox, y_l:y_l + oy, z_l:z_l + oz]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading checkpoint: {args.checkpoint_path}")
    model, context_encoder = load_model(args.checkpoint_path, device, args)
    flow = AutoregressiveFlowMatcher(model, sigma_min=args.sigma_min)

    print(f"Loading input: {args.input_path}")
    img = nib.load(args.input_path)
    raw = img.get_fdata().astype(np.float32)
    print(f"  raw shape={raw.shape}, range=[{raw.min():.2f}, {raw.max():.2f}]")

    voxel = voxel_size_from_nii(img)
    is_cs, accel_factor = parse_accel_from_filename(args.input_path)
    ctx_dtype = torch.float16 if args.fp16 else torch.float32
    ctx_vec = torch.tensor(
        [voxel[0], voxel[1], voxel[2], 1.0 if is_cs else 0.0, float(accel_factor)],
        device=device, dtype=ctx_dtype,
    ).unsqueeze(0)
    with torch.no_grad():
        ctx_emb = context_encoder(ctx_vec)
    print(f"  voxel={voxel}, accel={'CS' if is_cs else 'GRAPPA'} R{accel_factor}, "
          f"ctx_emb shape={tuple(ctx_emb.shape)}")

    normalized = raw / 255.0
    vol_t = torch.from_numpy(normalized)

    padded, pad_info, original_shape = pad_to_multiple(vol_t, PAD_MULT)
    print(f"  padded shape={tuple(padded.shape)}")

    output_padded = torch.zeros_like(padded)
    z_dim = padded.shape[2]
    n_chunks = z_dim // PATCH_Z

    x_pad, y_pad = padded.shape[0], padded.shape[1]
    prev_chunk = torch.zeros((1, 1, x_pad, y_pad, PATCH_Z), device=device, dtype=ctx_dtype)

    if args.rk4:
        sampler = flow.sample_rk4
        sampler_name = "rk4"
    elif args.heun:
        sampler = flow.sample_heun
        sampler_name = "heun"
    else:
        sampler = flow.sample_euler
        sampler_name = "euler"
    print(f"  sampler={sampler_name}, steps={args.num_sampling_steps}, auto={args.auto}")

    for ci in tqdm(range(n_chunks), desc="Z-chunks"):
        z0 = ci * PATCH_Z
        condition = padded[:, :, z0:z0 + PATCH_Z].unsqueeze(0).unsqueeze(0).to(device)
        if args.fp16:
            condition = condition.half()

        gen = sampler(condition, prev_chunk, args.num_sampling_steps, device, context=ctx_emb)
        gen_chunk = gen.float()

        output_padded[:, :, z0:z0 + PATCH_Z] = gen_chunk[0, 0].cpu()

        if args.auto:
            prev_chunk = gen.detach()
        else:
            prev_chunk = torch.zeros_like(prev_chunk)

    out_unpadded = unpad(output_padded, pad_info, original_shape).numpy()

    if args.rescale:
        out_unpadded = np.clip(out_unpadded * 255.0, 0, 65535).astype(np.uint16)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or ".", exist_ok=True)
    out_img = nib.Nifti1Image(out_unpadded, img.affine, img.header)
    out_img.to_filename(args.output_path)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
