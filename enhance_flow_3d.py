"""3D NIfTI flow matching inference on un-patched whole volumes.

Loads one .nii.gz, normalizes [0, 255] -> [0, 1], pads X/Y/Z to multiples of 16,
slides 32-thick chunks along Z with full XY in one shot per chunk, runs flow
matching, stitches along Z with overlap-and-add blending, unpads, and saves.

The cross-attention context is the *target* text prompt -- it selects which
output the multi-task model produces. Supply it directly with `--prompt` or via
`--prompt_json` (a sidecar with a `prompt` key, matching the dataset's JSON
sidecars). It is encoded by the frozen RadBERT text encoder, exactly as during
training. Examples:
    "Fully sampled axial 7T brain T2-weighted FLAIR MRI of resolution 0.85 x 0.75 x 1.5 mm."
    "Denoised and biascorrected coronal 1.5T knee Proton Density MRI of resolution 0.44 x 0.44 x 3 mm."

Classifier-free guidance is enabled with `--guidance_scale > 1.0`. Each ODE
step then runs two forwards: conditional + unconditional (encoded empty
prompt), and mixes them as `v_uncond + scale * (v_cond - v_uncond)`.

Usage:
    # artifact -> fully sampled (raw)
    python enhance_flow_3d.py \
        --checkpoint_path checkpoints_uncertainty/flow_matching_3d_brain_all_best_epoch_55.pt \
        --input_path /ix1/tibrahim/jil202/250507-promote_recovery_from_aws/data/PRT191518_2019.12.05/rT2w_FLAIR_lowres.nii.gz   \
        --output_path ./outputs/flair.nii.gz \
        --prompt "Fully sampled sagittal 3T brain T2-weighted FLAIR MRI of resolution 0.7 x 0.5 x 0.5 mm." \
        --num_sampling_steps 2 --euler --fp16 --non_overlap 2 \
        --guidance_scale 1.5 --compile --planes sagittal --ras --autocrop

    # raw (or artifact) -> denoised + bias-corrected, prompt read from a sidecar
    python enhance_flow_3d.py \
        --checkpoint_path checkpoints/flow_matching_3d_multitask.pt \
        --input_path 3T.nii.gz \
        --output_path ./outputs/3T_denoised.nii.gz \
        --prompt_json /vast/tibrahim/jil202/data/test/brain/mprage_ax_3T/mde14089s3_P53248.7/mde14089s3_P53248.7.json \
        --num_sampling_steps 1 --euler --fp16 --non_overlap 16 \
        --guidance_scale 0 --rescale --compile --ras
"""

import argparse
import json
import os

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DiffusionModelUNet
from tqdm import tqdm

from train_flow import TextConditioner


PATCH_Z = 16
PAD_MULT = 16


def to_ras(img):
    """Reorient a NIfTI to closest canonical (RAS+) and return the canonical
    image along with the orientation transform needed to undo it."""
    orig_ornt = nib.orientations.io_orientation(img.affine)
    ras_ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))
    to_orig_ornt = nib.orientations.ornt_transform(ras_ornt, orig_ornt)
    ras_img = nib.as_closest_canonical(img)
    return ras_img, to_orig_ornt


class FlowMatcher(nn.Module):
    """3D Flow Matching wrapper for inference (Euler / Heun / RK4) with CFG.

    `null_context` is the unconditional embedding (the encoded empty prompt).
    When `guidance_scale != 1.0` and `null_context` is provided, each velocity
    eval runs two forwards (conditional + unconditional) and combines them as
    `v = v_uncond + scale * (v_cond - v_uncond)`.
    """

    def __init__(self, model, sigma_min=0.001, amp_enabled=False, amp_device="cuda"):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min
        # `--fp16` runs the UNet forwards under bf16 autocast (model stays fp32).
        # bf16 has fp32's exponent range, so it can't overflow like true fp16
        # `.half()` does -- some checkpoints' activations exceed fp16's ~65504
        # ceiling and produce all-NaN (blank) output.
        self.amp_enabled = amp_enabled
        self.amp_device = amp_device

    @torch.no_grad()
    def _step_input(self, x, condition):
        return torch.cat([x, condition], dim=1)

    @torch.no_grad()
    def _v(self, x, condition, t, context, null_context, guidance_scale):
        inp = self._step_input(x, condition)
        timesteps = (t * 999).long()
        # UA-Flow checkpoints output 2 channels: [0] velocity mean, [1] log-var.
        # Plain checkpoints output 1 channel. Channel 0 is the velocity in both.
        with torch.autocast(device_type=self.amp_device, dtype=torch.bfloat16,
                            enabled=self.amp_enabled):
            if guidance_scale != 1.0 and null_context is not None:
                v_cond = self.model(inp, timesteps, context=context)[:, 0:1]
                v_uncond = self.model(inp, timesteps, context=null_context)[:, 0:1]
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = self.model(inp, timesteps, context=context)[:, 0:1]
        # Back to fp32 so the Euler/Heun/RK4 accumulation stays full precision.
        return v.float()

    @torch.no_grad()
    def sample_euler(self, condition, num_steps, device, context=None,
                     null_context=None, guidance_scale=1.0):
        x = torch.randn_like(condition, device=device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((condition.shape[0],), i * dt, device=device)
            v = self._v(x, condition, t, context, null_context, guidance_scale)
            x = torch.clamp(x + v * dt, -10, 10)
        return x

    @torch.no_grad()
    def sample_heun(self, condition, num_steps, device, context=None,
                    null_context=None, guidance_scale=1.0):
        x = torch.randn_like(condition, device=device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((condition.shape[0],), i * dt, device=device)
            v1 = self._v(x, condition, t, context, null_context, guidance_scale)
            x_euler = torch.clamp(x + v1 * dt, -3, 3)
            t_next = torch.full((condition.shape[0],), min((i + 1) * dt, 1.0), device=device)
            v2 = self._v(x_euler, condition, t_next, context, null_context, guidance_scale)
            x = torch.clamp(x + dt * (v1 + v2) / 2.0, -3, 3)
        return x

    @torch.no_grad()
    def sample_rk4(self, condition, num_steps, device, context=None,
                   null_context=None, guidance_scale=1.0):
        x = torch.randn_like(condition, device=device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t_cur = torch.full((condition.shape[0],), i * dt, device=device)
            k1 = self._v(x, condition, t_cur, context, null_context, guidance_scale)
            t_mid = torch.full((condition.shape[0],), i * dt + 0.5 * dt, device=device)
            k2 = self._v(torch.clamp(x + 0.5 * dt * k1, -3, 3), condition,
                         t_mid, context, null_context, guidance_scale)
            k3 = self._v(torch.clamp(x + 0.5 * dt * k2, -3, 3), condition,
                         t_mid, context, null_context, guidance_scale)
            t_next = torch.full((condition.shape[0],), min((i + 1) * dt, 1.0), device=device)
            k4 = self._v(torch.clamp(x + dt * k3, -3, 3), condition,
                         t_next, context, null_context, guidance_scale)
            x = torch.clamp(x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0, -3, 3)
        return x


# Per-plane permutations that move the slab axis to the LAST dim of an
# (X, Y, Z) volume, matching training-time slab shape (x, y, slab):
#   axial    slab=Z -> (X, Y, Z)            identity
#   coronal  slab=Y -> (X, Z, Y)            permute(0, 2, 1)
#   sagittal slab=X -> (Y, Z, X)            permute(1, 2, 0)
PLANE_PERMS = {
    "axial":    ((0, 1, 2), (0, 1, 2)),
    "coronal":  ((0, 2, 1), (0, 2, 1)),
    "sagittal": ((1, 2, 0), (2, 0, 1)),
}


def make_blend_window(patch_z, non_overlap, dtype=torch.float32):
    """1D Z-axis blend weights for overlap-and-add inference.

    `non_overlap` is the number of overlapping slices between consecutive
    chunks (stride = patch_z - non_overlap). The window has linear ramps
    over the first/last `non_overlap` slices and is 1.0 in the middle.
    Weights are strictly positive (min = 1/(non_overlap+1)) so the
    final divide is safe without clamping.
    """
    w = torch.ones(patch_z, dtype=dtype)
    if non_overlap > 0:
        ramp = torch.arange(1, non_overlap + 1, dtype=dtype) / (non_overlap + 1)
        w[:non_overlap] = ramp
        w[-non_overlap:] = ramp.flip(0)
    return w


def build_context_emb(text_conditioner, prompt, device, dtype):
    """Encode the target text prompt -> (cond, null) embeddings.

    The prompt is the target description (e.g. "Fully sampled ..." or "Denoised
    and biascorrected ...") and is shared across planes. The RadBERT encoder
    runs in fp32; only the output embeddings are cast to `dtype`.
    """
    print(f"  prompt: {prompt!r}")
    with torch.no_grad():
        ctx_emb = text_conditioner.encode([prompt], device).to(dtype=dtype)
        null_emb = text_conditioner.null(1).to(dtype=dtype)
    return ctx_emb, null_emb


def slab_inference(padded, sampler, num_steps, device, ctx_emb, null_emb,
                   guidance_scale, non_overlap, fp16, seed, desc="Slabs",
                   snapshot_idx=None, snapshot_path=None, snapshot_plane=""):
    """Slide PATCH_Z-thick slabs along the LAST axis of `padded`, run flow
    matching per slab, and blend with overlap-and-add along that axis.

    If `snapshot_idx` is set and `snapshot_path` is not None, the mid-slice
    (input vs. output) of that slab is saved to disk immediately after that
    slab is generated -- i.e. before the next slab starts. Index is clamped
    to the actual number of slabs.
    """
    z_dim = padded.shape[2]
    stride = PATCH_Z - non_overlap

    chunk_starts = []
    z = 0
    while z + PATCH_Z <= z_dim:
        chunk_starts.append(z)
        z += stride
    if not chunk_starts or chunk_starts[-1] + PATCH_Z < z_dim:
        chunk_starts.append(z_dim - PATCH_Z)

    window_1d = make_blend_window(PATCH_Z, non_overlap, dtype=padded.dtype)
    window = window_1d.view(1, 1, -1)

    output_sum = torch.zeros_like(padded)
    output_weight = torch.zeros_like(padded)

    effective_snap_idx = None
    if snapshot_idx is not None and snapshot_path is not None and len(chunk_starts) > 0:
        effective_snap_idx = max(0, min(snapshot_idx, len(chunk_starts) - 1))

    for ci, z0 in enumerate(tqdm(chunk_starts, desc=desc)):
        input_slab_cpu = padded[:, :, z0:z0 + PATCH_Z]
        condition = input_slab_cpu.unsqueeze(0).unsqueeze(0).to(device)
        # condition stays fp32; bf16 autocast inside FlowMatcher handles casting.

        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        gen = sampler(
            condition, num_steps, device,
            context=ctx_emb, null_context=null_emb,
            guidance_scale=guidance_scale,
        )
        gen_chunk_cpu = gen[0, 0].float().cpu()

        # if effective_snap_idx is not None and ci == effective_snap_idx:
        #     mid = PATCH_Z // 2
        #     save_slab_snapshot(
        #         input_slab_cpu[:, :, mid].numpy(),
        #         gen_chunk_cpu[:, :, mid].numpy(),
        #         snapshot_path, ci, snapshot_plane,
        #     )
        #     print(f"  [{snapshot_plane}] saved slab-{ci + 1} snapshot: {snapshot_path}")

        output_sum[:, :, z0:z0 + PATCH_Z] += gen_chunk_cpu * window
        output_weight[:, :, z0:z0 + PATCH_Z] += window

    return output_sum / output_weight.clamp(min=1e-8), len(chunk_starts)


def run_plane_inference(vol_t, plane, ctx_emb, null_emb,
                        sampler, args, device,
                        snapshot_idx=None, snapshot_path=None):
    """Permute (X, Y, Z) volume so `plane`'s slab axis is last, run slab
    inference, then permute back to (X, Y, Z).

    `ctx_emb`/`null_emb` are shared across planes: training built the prompt
    from the params JSON's voxel order regardless of patch orientation.
    """
    perm, inv_perm = PLANE_PERMS[plane]
    permuted = vol_t.permute(*perm).contiguous()

    padded, pad_info, original_shape = pad_to_multiple(permuted, PAD_MULT, min_z=PATCH_Z)
    non_overlap = max(0, args.non_overlap)
    print(f"  [{plane}] permuted shape={tuple(permuted.shape)}, "
          f"padded shape={tuple(padded.shape)}")

    output_padded, n_chunks = slab_inference(
        padded, sampler, args.num_sampling_steps, device,
        ctx_emb, null_emb, args.guidance_scale,
        non_overlap, args.fp16, args.seed, desc=f"{plane} slabs",
        snapshot_idx=snapshot_idx, snapshot_path=snapshot_path,
        snapshot_plane=plane,
    )
    print(f"  [{plane}] n_slabs={n_chunks}")

    out_unpadded = unpad(output_padded, pad_info, original_shape)
    return out_unpadded.permute(*inv_perm).contiguous()


def match_histograms(source, reference):
    """Map `source` intensities so their CDF matches `reference`. Standard
    quantile-interpolation implementation: returns a float32 array shaped
    like `source`."""
    src_shape = source.shape
    src = source.ravel().astype(np.float64)
    ref = reference.ravel().astype(np.float64)

    src_values, src_inverse, src_counts = np.unique(
        src, return_inverse=True, return_counts=True,
    )
    ref_values, ref_counts = np.unique(ref, return_counts=True)

    src_q = np.cumsum(src_counts).astype(np.float64) / src.size
    ref_q = np.cumsum(ref_counts).astype(np.float64) / ref.size

    interp = np.interp(src_q, ref_q, ref_values)
    return interp[src_inverse].reshape(src_shape).astype(np.float32)


def repad_to_original(cropped_arr, bounds, original_shape):
    """Inverse of niiprep.autocrop: paste a cropped volume back into a zero
    array of `original_shape` at the position recorded in `bounds`. Voxels
    that fall outside the original frame are clipped."""
    out = np.zeros(original_shape, dtype=cropped_arr.dtype)
    ox, oy, oz = bounds["origin"]
    tw, th, td = bounds["shape"]

    dst_x_s = max(0, ox);  dst_x_e = min(original_shape[0], ox + tw)
    src_x_s = max(0, -ox); src_x_e = src_x_s + (dst_x_e - dst_x_s)

    dst_y_s = max(0, oy);  dst_y_e = min(original_shape[1], oy + th)
    src_y_s = max(0, -oy); src_y_e = src_y_s + (dst_y_e - dst_y_s)

    dst_z_s = max(0, oz);  dst_z_e = min(original_shape[2], oz + td)
    src_z_s = max(0, -oz); src_z_e = src_z_s + (dst_z_e - dst_z_s)

    out[dst_x_s:dst_x_e, dst_y_s:dst_y_e, dst_z_s:dst_z_e] = \
        cropped_arr[src_x_s:src_x_e, src_y_s:src_y_e, src_z_s:src_z_e]
    return out


def split_nii_ext(path):
    """Split a NIfTI path into (base, ext), preserving the .nii.gz suffix."""
    for ext in (".nii.gz", ".nii"):
        if path.endswith(ext):
            return path[: -len(ext)], ext
    return os.path.splitext(path)


def save_volume(volume_t, path, to_orig_ornt, rescale, ref_img,
                hist_match_ref=None, repad_bounds=None, pre_crop_shape=None):
    """Post-process and save: optional repad-from-autocrop, reorient back to
    the input's native orientation, histogram match, or uint8 rescale.

    Order is: repad -> reorient -> hist-match (or rescale) -> save. Saved with
    `ref_img.affine`/`header` so the file on disk matches the input geometry.

    `hist_match_ref` (if given) overrides `rescale`: the output's intensity
    CDF is matched to the reference array's CDF and saved as float32.
    """
    arr = volume_t.numpy() if torch.is_tensor(volume_t) else volume_t
    if repad_bounds is not None and pre_crop_shape is not None:
        arr = repad_to_original(arr, repad_bounds, pre_crop_shape)
    if to_orig_ornt is not None:
        arr = nib.orientations.apply_orientation(arr, to_orig_ornt)
    if hist_match_ref is not None:
        arr = match_histograms(arr, hist_match_ref)
    elif rescale:
        arr = np.rint(np.clip(arr * 255.0, 0, 255)).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    out_img = nib.Nifti1Image(arr, ref_img.affine, ref_img.header)
    out_img.to_filename(path)
    print(f"Saved: {path}")


def save_slab_snapshot(input_slice, output_slice, path, slab_idx, plane):
    """1x2 PNG: input vs output mid-slice of a single slab, saved inline
    during slab inference so the user sees it before later slabs run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(input_slice, cmap="gray")
    axes[0].set_title(f"Input ({plane} slab {slab_idx + 1} mid)")
    axes[0].axis("off")
    axes[1].imshow(output_slice, cmap="gray")
    axes[1].set_title(f"Output ({plane} slab {slab_idx + 1} mid)")
    axes[1].axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--input_path", type=str, required=True,
                   help="Single un-patched .nii.gz file")
    p.add_argument("--output_path", type=str, required=True)

    p.add_argument("--num_sampling_steps", type=int, default=1)
    p.add_argument("--non_overlap", type=int, default=0,
                   help="Number of overlapping slices between consecutive Z-chunks. "
                        "Stride = PATCH_Z - non_overlap. 0 = no overlap (default); "
                        "1 -> stride 31; 2 -> stride 30; etc. Overlap region is "
                        "linearly blended via sum/weight accumulation.")
    p.add_argument("--euler", action="store_true",
                   help="Use Euler ODE (default if neither --heun nor --rk4)")
    p.add_argument("--heun", action="store_true")
    p.add_argument("--rk4", action="store_true")

    p.add_argument("--sigma_min", type=float, default=0.001)
    p.add_argument("--fp16", action="store_true",
                   help="Run UNet forwards under bf16 autocast (model stays "
                        "fp32). bf16 has fp32's exponent range, avoiding the "
                        "fp16 .half() overflow that turns some checkpoints' "
                        "output into all-NaN/blank volumes.")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--rescale", action="store_true",
                   help="Rescale output back to ~[0, 255] uint16 before save")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt", type=str, default=None,
                   help="Target text prompt (the cross-attention context that "
                        "selects the output), e.g. \"Fully sampled axial 7T "
                        "brain T2-weighted FLAIR MRI of resolution 0.85 x 0.75 "
                        "x 1.5 mm.\" or \"Denoised and biascorrected ...\".")
    g.add_argument("--prompt_json", type=str, default=None,
                   help="Sidecar JSON with a `prompt` key (the dataset's "
                        "<subject>.json format) to read the target prompt from.")

    # Classifier-free guidance
    p.add_argument("--guidance_scale", type=float, default=1.5,
                   help="Classifier-free guidance scale. 1.0 = no guidance "
                        "(single forward); >1.0 doubles per-step model evals.")

    # Triplanar inference + snapshot
    p.add_argument("--planes", type=str, nargs="+",
                   default=["axial", "coronal", "sagittal"],
                   choices=["axial", "coronal", "sagittal"],
                   help="Planes to run inference on. Multiple planes are "
                        "mean-ensembled before saving.")
    p.add_argument("--snapshot_dir", type=str, default=None,
                   help="Directory to write per-plane slab snapshots. "
                        "Defaults to the directory of --output_path.")
    p.add_argument("--snapshot_slab", type=int, default=5,
                   help="1-indexed slab number whose mid-slice (input + output) "
                        "is dumped to PNG per plane, written immediately after "
                        "that slab is generated. Clamped to actual slab count.")
    p.add_argument("--no_snapshot", action="store_true",
                   help="Disable per-slab snapshots.")

    p.add_argument("--ras", action="store_true",
                   help="Reorient the input to closest-canonical RAS+ before "
                        "inference, then reorient the output back to the "
                        "input's native orientation before saving.")
    p.add_argument("--hist", action="store_true",
                   help="Match the inferenced output's intensity histogram to "
                        "the original input image. Overrides --rescale; saved "
                        "as float32 in the input's native intensity range.")
    p.add_argument("--autocrop", action="store_true",
                   help="Run niiprep.autocrop on the input (after optional "
                        "--ras) to crop out air space before inference; the "
                        "inferenced volume is zero-padded back to the "
                        "pre-crop shape before reorient/hist-match/save.")
    return p.parse_args()


def load_model(checkpoint_path, device, args):
    text_conditioner = TextConditioner()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not (isinstance(ckpt, dict) and "model" in ckpt):
        raise ValueError(
            f"Checkpoint at {checkpoint_path} does not contain a 'model' key. "
            f"Re-train with the text-conditioned train_flow.py."
        )

    # Detect the number of output channels from the checkpoint. UA-Flow
    # (uncertainty) checkpoints have 2 (velocity mean + log-variance); plain
    # flow-matching checkpoints have 1. The output conv is `out.2.conv.weight`.
    out_w = None
    for k in ("out.2.conv.weight", "module.out.2.conv.weight",
              "_orig_mod.out.2.conv.weight"):
        if k in ckpt["model"]:
            out_w = ckpt["model"][k]
            break
    out_channels = out_w.shape[0] if out_w is not None else 1
    if out_channels == 2:
        print("  detected UA-Flow checkpoint (out_channels=2); using velocity "
              "mean channel for inference")

    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=out_channels,
        channels=(128, 256, 512),
        attention_levels=(False, False, True),
        num_res_blocks=2,
        num_head_channels=512,
        with_conditioning=True,
        cross_attention_dim=text_conditioner.hidden_size,
    )
    if "context_encoder" in ckpt:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} is from the legacy scalar-context "
            f"model (cross_attention_dim=256); it is incompatible with the "
            f"RadBERT text-conditioned model (cross_attention_dim="
            f"{text_conditioner.hidden_size}). Re-train with the current "
            f"train_flow.py or use the old enhance_flow_3d.py from git history."
        )

    model_sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(model_sd)

    model.to(device).eval()
    # RadBERT stays fp32; its output embeddings are cast in build_context_emb.
    text_conditioner.to(device)

    # `--fp16` no longer half()s the model: it enables bf16 autocast inside
    # FlowMatcher (set up in main). The model stays fp32 here.
    if args.compile:
        model = torch.compile(model, backend="inductor")

    return model, text_conditioner


def pad_to_multiple(volume: torch.Tensor, mult: int, min_z: int = 0):
    """Center-pad a 3D tensor (X, Y, Z) so each dim is a multiple of `mult`,
    with the Z dim padded to at least `min_z` slices."""
    def _amount(n, min_n=0):
        target = max(n, min_n)
        rem = target % mult
        if rem != 0:
            target += mult - rem
        total = target - n
        l = total // 2
        return l, total - l

    x_l, x_r = _amount(volume.shape[0])
    y_l, y_r = _amount(volume.shape[1])
    z_l, z_r = _amount(volume.shape[2], min_z)
    padded = F.pad(volume, (z_l, z_r, y_l, y_r, x_l, x_r), mode="constant", value=0.0)
    return padded, ((x_l, x_r), (y_l, y_r), (z_l, z_r)), volume.shape


def unpad(volume: torch.Tensor, pad_info, original_shape):
    (x_l, x_r), (y_l, y_r), (z_l, z_r) = pad_info
    ox, oy, oz = original_shape
    return volume[x_l:x_l + ox, y_l:y_l + oy, z_l:z_l + oz]


def resolve_prompt(args):
    """Return the target prompt from --prompt or the `prompt` key of --prompt_json."""
    if args.prompt is not None:
        return args.prompt
    with open(args.prompt_json) as f:
        meta = json.load(f)
    if "prompt" not in meta:
        raise ValueError(
            f"--prompt_json {args.prompt_json} has no 'prompt' key."
        )
    return meta["prompt"]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading checkpoint: {args.checkpoint_path}")
    model, text_conditioner = load_model(args.checkpoint_path, device, args)
    flow = FlowMatcher(model, sigma_min=args.sigma_min,
                       amp_enabled=args.fp16, amp_device=device.type)

    print(f"Loading input: {args.input_path}")
    orig_img = nib.load(args.input_path)
    if args.ras:
        orig_axcodes = "".join(nib.orientations.aff2axcodes(orig_img.affine))
        img, to_orig_ornt = to_ras(orig_img)
        print(f"  reoriented {orig_axcodes} -> RAS")
    else:
        img = orig_img
        to_orig_ornt = None

    crop_bounds = None
    pre_crop_shape = None
    if args.autocrop:
        import tempfile
        from niiprep.autocrop import autocrop as _autocrop
        pre_crop_shape = tuple(img.shape)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_in = os.path.join(tmpdir, "in.nii.gz")
            tmp_out = os.path.join(tmpdir, "crop.nii.gz")
            nib.save(
                nib.Nifti1Image(np.asanyarray(img.dataobj), img.affine, img.header),
                tmp_in,
            )
            crop_bounds = _autocrop(tmp_in, tmp_out)
            raw = nib.load(tmp_out).get_fdata().astype(np.float32)
        print(f"  autocrop: {pre_crop_shape} -> {raw.shape}, bounds={crop_bounds}")
    else:
        raw = img.get_fdata().astype(np.float32)
    print(f"  raw shape={raw.shape}, range=[{raw.min():.2f}, {raw.max():.2f}]")

    prompt = resolve_prompt(args)
    # Keep embeddings fp32; bf16 autocast (when --fp16) casts them inside the UNet.
    ctx_dtype = torch.float32
    print(f"  guidance_scale={args.guidance_scale}")

    ctx_emb, null_emb = build_context_emb(
        text_conditioner, prompt, device, ctx_dtype,
    )

    non_overlap = max(0, args.non_overlap)
    if non_overlap >= PATCH_Z:
        raise ValueError(
            f"--non_overlap must be in [0, {PATCH_Z - 1}]; got {non_overlap}"
        )

    normalized = raw / raw.max()
    vol_t = torch.from_numpy(normalized)

    if args.rk4:
        sampler = flow.sample_rk4
        sampler_name = "rk4"
    elif args.heun:
        sampler = flow.sample_heun
        sampler_name = "heun"
    else:
        sampler = flow.sample_euler
        sampler_name = "euler"
    print(f"  sampler={sampler_name}, steps={args.num_sampling_steps}, "
          f"non_overlap={non_overlap}, stride={PATCH_Z - non_overlap}, "
          f"planes={args.planes}")

    snapshot_idx = None
    snapshot_dir = None
    if not args.no_snapshot:
        snapshot_idx = max(0, args.snapshot_slab - 1)
        snapshot_dir = args.snapshot_dir or os.path.dirname(
            os.path.abspath(args.output_path)) or "."
        os.makedirs(snapshot_dir, exist_ok=True)
    output_basename = os.path.basename(args.output_path)
    for ext in (".nii.gz", ".nii"):
        if output_basename.endswith(ext):
            output_basename = output_basename[: -len(ext)]
            break

    plane_outputs = []
    for plane in args.planes:
        plane_snapshot_path = None
        if snapshot_dir is not None:
            plane_snapshot_path = os.path.join(
                snapshot_dir,
                f"{output_basename}_{plane}_slab{args.snapshot_slab}.png",
            )
        plane_out = run_plane_inference(
            vol_t, plane, ctx_emb, null_emb,
            sampler, args, device,
            snapshot_idx=snapshot_idx,
            snapshot_path=plane_snapshot_path,
        )
        plane_outputs.append(plane_out)

    if len(plane_outputs) == 1:
        ensembled = plane_outputs[0]
    else:
        ensembled = torch.stack(plane_outputs, dim=0).mean(dim=0)
        print(f"Ensembled {len(plane_outputs)} planes by mean.")

    hist_ref = orig_img.get_fdata().astype(np.float32) if args.hist else None

    output_base, output_ext = split_nii_ext(args.output_path)
    if len(plane_outputs) > 1:
        for plane, plane_out in zip(args.planes, plane_outputs):
            plane_path = f"{output_base}_{plane}{output_ext}"
            save_volume(plane_out, plane_path, to_orig_ornt, args.rescale,
                        orig_img, hist_match_ref=hist_ref,
                        repad_bounds=crop_bounds, pre_crop_shape=pre_crop_shape)

    save_volume(ensembled, args.output_path, to_orig_ornt, args.rescale,
                orig_img, hist_match_ref=hist_ref,
                repad_bounds=crop_bounds, pre_crop_shape=pre_crop_shape)


if __name__ == "__main__":
    main()
