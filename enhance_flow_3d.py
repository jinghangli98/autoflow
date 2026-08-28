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

`--save_uncertainty_map` additionally writes the model's per-voxel predictive
uncertainty to `<output>_uncertainty.nii.gz` (and `<output>_<plane>_uncertainty
.nii.gz` per plane when ensembling). It requires a UA-Flow 2-channel checkpoint,
whose channel 1 is the velocity log-variance; the map is the std obtained by
propagating that variance through the ODE steps,
`sqrt(sum_i dt^2 * sigma_v(t_i)^2)`, as in train_flow_sigma.py. It is always
float32 and is never `--rescale`d or `--hist`-matched: it lives in normalized
velocity units, not the input's intensity units. Larger values mark voxels the
model is least sure about -- useful for flagging hallucinated structure.

Usage:
    # artifact -> fully sampled (raw)
    python enhance_flow_3d.py \
        --checkpoint_path /vast/tibrahim/jil202/autoflow/checkpoints_uncertainty/flow_matching_3d_brain_all_best_062926.pt \
        --input_path /ix1/tibrahim/jil202/250507-promote_recovery_from_aws/data/PRT191518_2019.12.05/rT2w_FLAIR_lowres.nii.gz   \
        --output_path ./outputs/flair.nii.gz \
        --prompt "Fully sampled sagittal 3T brain T2-weighted FLAIR MRI of resolution 0.7 x 0.5 x 0.5 mm." \
        --num_sampling_steps 4 --euler --fp16 --non_overlap 8 \
        --guidance_scale 1 --compile --planes sagittal --autocrop --ras

    # same, also writing ./outputs/flair_uncertainty.nii.gz
    python enhance_flow_3d.py \
        --checkpoint_path /vast/tibrahim/jil202/autoflow/checkpoints_uncertainty/flow_matching_3d_brain_all_best_062926.pt \
        --input_path /vast/tibrahim/jil202/autoflow/rT1w_BRAVO_lowres.nii.gz \
        --output_path ./outputs/BRAVO.nii.gz \
        --prompt "Fully sampled axial 3T brain T1-weighted BRAVO MRI of resolution 0.85 x 0.85 x 1.2 mm." \
        --num_sampling_steps 4 --euler --fp16 --planes axial --non_overlap 8 --rescale \
        --save_uncertainty_map

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

    def __init__(self, model, sigma_min=0.001, amp_enabled=False, amp_device="cuda",
                 logvar_clamp=7.0):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min
        # UA-Flow checkpoints predict a per-voxel velocity log-variance (channel
        # 1). It is clamped to +/-logvar_clamp before exp() during uncertainty
        # accumulation, matching train_flow_sigma.py's heteroscedastic head.
        self.logvar_clamp = logvar_clamp
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
    def _v(self, x, condition, t, context, null_context, guidance_scale,
           return_logvar=False):
        """Velocity at (x, t). With `return_logvar`, also returns the predicted
        per-voxel velocity log-variance (None for 1-channel checkpoints).

        Which branch's log-variance is returned mirrors train_flow_sigma.py's
        `_guided_velocity`: the conditional branch's, except at
        `guidance_scale == 0.0` (a pure unconditional eval), where the
        unconditional branch's is the one that describes the velocity used.
        """
        inp = self._step_input(x, condition)
        timesteps = (t * 999).long()
        # UA-Flow checkpoints output 2 channels: [0] velocity mean, [1] log-var.
        # Plain checkpoints output 1 channel. Channel 0 is the velocity in both.
        with torch.autocast(device_type=self.amp_device, dtype=torch.bfloat16,
                            enabled=self.amp_enabled):
            if guidance_scale != 1.0 and null_context is not None:
                out_c = self.model(inp, timesteps, context=context)
                out_u = self.model(inp, timesteps, context=null_context)
                v_cond, v_uncond = out_c[:, 0:1], out_u[:, 0:1]
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
                out_lv = out_u if guidance_scale == 0.0 else out_c
            else:
                out_c = self.model(inp, timesteps, context=context)
                v = out_c[:, 0:1]
                out_lv = out_c
        # Back to fp32 so the Euler/Heun/RK4 accumulation stays full precision.
        if not return_logvar:
            return v.float()
        logvar = out_lv[:, 1:2].float() if out_lv.shape[1] > 1 else None
        return v.float(), logvar

    def _accum_var(self, var_accum, logvar, dt):
        """Propagate one Euler step's velocity variance: Var += dt^2 * sigma_v^2.
        Matches train_flow_sigma.py's `sample`, whose final uncertainty is
        sqrt(Var). No-op when the checkpoint has no variance head."""
        if logvar is None:
            return var_accum
        return var_accum + (dt ** 2) * torch.exp(
            logvar.clamp(-self.logvar_clamp, self.logvar_clamp)
        )

    @torch.no_grad()
    def sample_euler(self, condition, num_steps, device, context=None,
                     null_context=None, guidance_scale=1.0, noise=None,
                     return_uncertainty=False):
        x = noise if noise is not None else torch.randn_like(condition, device=device)
        dt = 1.0 / num_steps
        var_accum = torch.zeros_like(x) if return_uncertainty else None
        for i in range(num_steps):
            t = torch.full((condition.shape[0],), i * dt, device=device)
            if return_uncertainty:
                v, logvar = self._v(x, condition, t, context, null_context,
                                    guidance_scale, return_logvar=True)
                var_accum = self._accum_var(var_accum, logvar, dt)
            else:
                v = self._v(x, condition, t, context, null_context, guidance_scale)
            x = torch.clamp(x + v * dt, -10, 10)
        return (x, var_accum.sqrt()) if return_uncertainty else x

    @torch.no_grad()
    def sample_heun(self, condition, num_steps, device, context=None,
                    null_context=None, guidance_scale=1.0, noise=None,
                    return_uncertainty=False):
        x = noise if noise is not None else torch.randn_like(condition, device=device)
        dt = 1.0 / num_steps
        var_accum = torch.zeros_like(x) if return_uncertainty else None
        for i in range(num_steps):
            t = torch.full((condition.shape[0],), i * dt, device=device)
            if return_uncertainty:
                # Accumulate from the on-trajectory eval at t_i (the same point
                # Euler uses); the predictor eval at t_{i+1} is off-trajectory.
                v1, logvar = self._v(x, condition, t, context, null_context,
                                     guidance_scale, return_logvar=True)
                var_accum = self._accum_var(var_accum, logvar, dt)
            else:
                v1 = self._v(x, condition, t, context, null_context, guidance_scale)
            x_euler = torch.clamp(x + v1 * dt, -3, 3)
            t_next = torch.full((condition.shape[0],), min((i + 1) * dt, 1.0), device=device)
            v2 = self._v(x_euler, condition, t_next, context, null_context, guidance_scale)
            x = torch.clamp(x + dt * (v1 + v2) / 2.0, -3, 3)
        return (x, var_accum.sqrt()) if return_uncertainty else x

    @torch.no_grad()
    def sample_rk4(self, condition, num_steps, device, context=None,
                   null_context=None, guidance_scale=1.0, noise=None,
                   return_uncertainty=False):
        x = noise if noise is not None else torch.randn_like(condition, device=device)
        dt = 1.0 / num_steps
        var_accum = torch.zeros_like(x) if return_uncertainty else None
        for i in range(num_steps):
            t_cur = torch.full((condition.shape[0],), i * dt, device=device)
            if return_uncertainty:
                # Accumulate from k1, the only on-trajectory eval of the step.
                k1, logvar = self._v(x, condition, t_cur, context, null_context,
                                     guidance_scale, return_logvar=True)
                var_accum = self._accum_var(var_accum, logvar, dt)
            else:
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
        return (x, var_accum.sqrt()) if return_uncertainty else x


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
                   guidance_scale, non_overlap, fp16, seed, batch_size=1,
                   desc="Slabs", snapshot_idx=None, snapshot_path=None,
                   snapshot_plane="", return_uncertainty=False):
    """Slide PATCH_Z-thick slabs along the LAST axis of `padded`, run flow
    matching per slab, and blend with overlap-and-add along that axis.

    Up to `batch_size` slabs are stacked into a single batched forward to
    amortize kernel-launch / attention overhead. Every slab in a batch is
    seeded with the *same* noise (seed reset before the batch, then the
    per-slab noise broadcast across the batch dim), so the output is bit-for-bit
    identical to the unbatched (`batch_size=1`) path regardless of batch size.

    Returns `(output, n_chunks, uncertainty)`. `uncertainty` is None unless
    `return_uncertainty`, in which case it is the per-voxel predictive std,
    blended with the same window as the output.
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
    unc_sum = torch.zeros_like(padded) if return_uncertainty else None

    batch_size = max(1, batch_size)
    for bs in tqdm(range(0, len(chunk_starts), batch_size), desc=desc):
        batch_z0 = chunk_starts[bs:bs + batch_size]
        # (B, 1, X, Y, PATCH_Z); fp32 condition, bf16 autocast handles casting.
        slabs = [padded[:, :, z0:z0 + PATCH_Z] for z0 in batch_z0]
        condition = torch.stack(slabs, dim=0).unsqueeze(1).to(device)
        b = condition.shape[0]

        # Match the unbatched path exactly: reseed, draw a single slab's worth
        # of noise, broadcast it across the batch dim. Cross-attention context
        # is per-slab identical, so expand it across the batch too.
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        base_noise = torch.randn((1, *condition.shape[1:]), device=device)
        noise = base_noise.expand(b, -1, -1, -1, -1)
        ctx_b = ctx_emb.expand(b, *ctx_emb.shape[1:]) if ctx_emb is not None else None
        null_b = null_emb.expand(b, *null_emb.shape[1:]) if null_emb is not None else None

        gen = sampler(
            condition, num_steps, device,
            context=ctx_b, null_context=null_b,
            guidance_scale=guidance_scale, noise=noise,
            return_uncertainty=return_uncertainty,
        )
        unc_cpu = None
        if return_uncertainty:
            gen, unc = gen
            unc_cpu = unc[:, 0].float().cpu()
        gen_cpu = gen[:, 0].float().cpu()  # (B, X, Y, PATCH_Z)

        for i, z0 in enumerate(batch_z0):
            output_sum[:, :, z0:z0 + PATCH_Z] += gen_cpu[i] * window
            output_weight[:, :, z0:z0 + PATCH_Z] += window
            if unc_cpu is not None:
                unc_sum[:, :, z0:z0 + PATCH_Z] += unc_cpu[i] * window

    weight = output_weight.clamp(min=1e-8)
    unc_out = (unc_sum / weight) if unc_sum is not None else None
    return output_sum / weight, len(chunk_starts), unc_out


def run_plane_inference(vol_t, plane, ctx_emb, null_emb,
                        sampler, args, device,
                        snapshot_idx=None, snapshot_path=None):
    """Permute (X, Y, Z) volume so `plane`'s slab axis is last, run slab
    inference, then permute back to (X, Y, Z).

    `ctx_emb`/`null_emb` are shared across planes: training built the prompt
    from the params JSON's voxel order regardless of patch orientation.

    Returns `(output, uncertainty)`, both in (X, Y, Z); `uncertainty` is None
    unless `args.save_uncertainty_map`. `getattr` guards the flag because other
    scripts (evaluate_test_whole.py, ablate_prompt_conditioning.py) call this
    with their own argparse namespaces, which do not define it.
    """
    want_unc = getattr(args, "save_uncertainty_map", False)
    perm, inv_perm = PLANE_PERMS[plane]
    permuted = vol_t.permute(*perm).contiguous()

    padded, pad_info, original_shape = pad_to_multiple(permuted, PAD_MULT, min_z=PATCH_Z)
    non_overlap = max(0, args.non_overlap)
    print(f"  [{plane}] permuted shape={tuple(permuted.shape)}, "
          f"padded shape={tuple(padded.shape)}")

    output_padded, n_chunks, unc_padded = slab_inference(
        padded, sampler, args.num_sampling_steps, device,
        ctx_emb, null_emb, args.guidance_scale,
        non_overlap, args.fp16, args.seed, batch_size=args.batch_size,
        desc=f"{plane} slabs",
        snapshot_idx=snapshot_idx, snapshot_path=snapshot_path,
        snapshot_plane=plane,
        return_uncertainty=want_unc,
    )
    print(f"  [{plane}] n_slabs={n_chunks}")

    out = unpad(output_padded, pad_info, original_shape).permute(*inv_perm).contiguous()
    unc = None
    if unc_padded is not None:
        unc = unpad(unc_padded, pad_info, original_shape).permute(*inv_perm).contiguous()
        print(f"  [{plane}] uncertainty: mean={unc.mean():.4f}, max={unc.max():.4f}")
    return out, unc


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
                hist_match_ref=None, repad_bounds=None, pre_crop_shape=None,
                dtype=None):
    """Post-process and save: optional repad-from-autocrop, reorient back to
    the input's native orientation, histogram match, or uint8 rescale.

    Order is: repad -> reorient -> hist-match (or rescale) -> save. Saved with
    `ref_img.affine`/`header` so the file on disk matches the input geometry.

    `hist_match_ref` (if given) overrides `rescale`: the output's intensity
    CDF is matched to the reference array's CDF and saved as float32.

    `dtype` forces the on-disk datatype instead of inheriting `ref_img`'s. The
    inherited header is what nibabel writes with, so a float volume under an
    int16 input header would otherwise be stored as int16 + an scl_slope --
    lossy. Pass np.float32 for volumes that are not in the input's intensity
    units (e.g. the uncertainty map).
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
    if dtype is not None:
        out_img.set_data_dtype(dtype)
    out_img.to_filename(path)
    print(f"Saved: {path}")


def save_uncertainty(unc_t, path, to_orig_ornt, ref_img,
                     repad_bounds=None, pre_crop_shape=None):
    """Save a per-voxel predictive-uncertainty (std) map alongside the output.

    It shares the output's geometry (same repad + reorient), so it overlays
    voxel-for-voxel. It is deliberately never `--rescale`d or histogram-matched:
    those map into the input's *intensity* units, while this map is a std in
    normalized-velocity units. Forced to float32 so the input's (possibly
    integer) header does not quantize it.
    """
    save_volume(unc_t, path, to_orig_ornt, rescale=False, ref_img=ref_img,
                hist_match_ref=None, repad_bounds=repad_bounds,
                pre_crop_shape=pre_crop_shape, dtype=np.float32)


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
    p.add_argument("--batch_size", type=int, default=1,
                   help="Number of Z-slabs to stack into a single batched "
                        "forward pass. >1 amortizes per-slab overhead for a "
                        "speedup at the cost of more GPU memory. Output is "
                        "numerically identical to batch_size=1.")
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
    p.add_argument("--norm_div", type=float, default=None,
                   help="Fixed divisor to normalize input into the model's "
                        "expected range. Default (None) divides by the input's "
                        "own max (whole-volume behavior). Set to 1.0 for "
                        "pre-normalized patches to feed them through unchanged.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    # Not required: unconditioned checkpoints (trained with
    # --no_text_conditioning) take no prompt. For conditioned checkpoints a
    # prompt IS required -- validated in main() after the checkpoint type is
    # auto-detected.
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--prompt", type=str, default=None,
                   help="Target text prompt (the cross-attention context that "
                        "selects the output), e.g. \"Fully sampled axial 7T "
                        "brain T2-weighted FLAIR MRI of resolution 0.85 x 0.75 "
                        "x 1.5 mm.\" or \"Denoised and biascorrected ...\".")
    g.add_argument("--prompt_json", type=str, default=None,
                   help="Sidecar JSON with a `prompt` key (the dataset's "
                        "<subject>.json format) to read the target prompt from.")

    # Classifier-free guidance
    p.add_argument("--guidance_scale", type=float, default=1,
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

    # Both spellings accepted: the file's convention is underscores, but the
    # dashed form is the reflex for anyone used to standard CLI tools.
    p.add_argument("--save_uncertainty_map", "--save-uncertainty-map",
                   dest="save_uncertainty_map", action="store_true",
                   help="Also save the model's per-voxel predictive uncertainty "
                        "(std) map, obtained by propagating the variance head's "
                        "sigma_v through the ODE steps: sqrt(sum_i dt^2 * "
                        "sigma_v(t_i)^2). Requires a UA-Flow 2-channel "
                        "checkpoint. Written next to --output_path as "
                        "<output>_uncertainty.nii.gz, always float32 (never "
                        "--rescale'd or --hist-matched, since it is in "
                        "normalized-velocity units, not intensity units).")

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
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not (isinstance(ckpt, dict) and "model" in ckpt):
        raise ValueError(
            f"Checkpoint at {checkpoint_path} does not contain a 'model' key. "
            f"Re-train with the text-conditioned train_flow.py."
        )

    # Auto-detect whether the checkpoint has the text cross-attention pathway.
    # Conditioned models (SpatialTransformer) carry `attn2` params; models
    # trained with --no_text_conditioning (self-attention only) do not. The
    # substring survives any `module.`/`_orig_mod.` prefix.
    has_cross = any("attn2" in k for k in ckpt["model"])
    text_conditioner = TextConditioner() if has_cross else None
    cross_dim = text_conditioner.hidden_size if text_conditioner is not None else None
    print(f"  text conditioning: {'ON (RadBERT cross-attention)' if has_cross else 'OFF (unconditioned checkpoint)'}")

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
        with_conditioning=has_cross,
        cross_attention_dim=cross_dim,
    )
    if has_cross and "context_encoder" in ckpt:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} is from the legacy scalar-context "
            f"model (cross_attention_dim=256); it is incompatible with the "
            f"RadBERT text-conditioned model (cross_attention_dim="
            f"{text_conditioner.hidden_size}). Re-train with the current "
            f"train_flow.py or use the old enhance_flow_3d.py from git history."
        )

    model_sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(model_sd)

    # Checkpoints trained with --train_text_encoder carry fine-tuned RadBERT
    # weights; load them over the pretrained ones. Absent this key the encoder
    # stays the stock frozen RadBERT. It is eval/frozen at inference either way.
    if text_conditioner is not None and "text_encoder" in ckpt:
        enc_sd = {k.replace("module.", "").replace("_orig_mod.", ""): v
                  for k, v in ckpt["text_encoder"].items()}
        text_conditioner.encoder.load_state_dict(enc_sd, strict=False)
        print("  loaded fine-tuned RadBERT text encoder weights from checkpoint")

    model.to(device).eval()
    # RadBERT stays fp32; its output embeddings are cast in build_context_emb.
    if text_conditioner is not None:
        text_conditioner.to(device)

    # `--fp16` no longer half()s the model: it enables bf16 autocast inside
    # FlowMatcher (set up in main). The model stays fp32 here.
    if args.compile:
        model = torch.compile(model, backend="inductor")

    return model, text_conditioner, out_channels


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
    model, text_conditioner, out_channels = load_model(args.checkpoint_path, device, args)
    if args.save_uncertainty_map and out_channels < 2:
        raise ValueError(
            "--save_uncertainty_map requires a UA-Flow 2-channel checkpoint "
            f"(velocity mean + log-variance); {args.checkpoint_path} has "
            f"out_channels={out_channels}. Train with train_flow_sigma.py."
        )
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

    # Keep embeddings fp32; bf16 autocast (when --fp16) casts them inside the UNet.
    ctx_dtype = torch.float32
    if text_conditioner is not None:
        if args.prompt is None and args.prompt_json is None:
            raise ValueError(
                "This is a text-conditioned checkpoint; supply --prompt or "
                "--prompt_json (the target description the model conditions on)."
            )
        prompt = resolve_prompt(args)
        print(f"  guidance_scale={args.guidance_scale}")
        ctx_emb, null_emb = build_context_emb(
            text_conditioner, prompt, device, ctx_dtype,
        )
    else:
        # Unconditioned checkpoint: no text pathway. CFG is meaningless, so force
        # a single unconditional pass and ignore any supplied prompt.
        if args.prompt is not None or args.prompt_json is not None:
            print("  note: unconditioned checkpoint; ignoring the supplied prompt.")
        if args.guidance_scale != 1.0:
            print(f"  note: unconditioned checkpoint; forcing guidance_scale=1.0 "
                  f"(was {args.guidance_scale}).")
            args.guidance_scale = 1.0
        ctx_emb, null_emb = None, None

    non_overlap = max(0, args.non_overlap)
    if non_overlap >= PATCH_Z:
        raise ValueError(
            f"--non_overlap must be in [0, {PATCH_Z - 1}]; got {non_overlap}"
        )

    norm_div = args.norm_div if args.norm_div is not None else float(raw.max())
    if norm_div <= 0:
        raise ValueError(f"Normalization divisor must be > 0; got {norm_div}")
    normalized = raw / norm_div
    print(f"  norm_div={norm_div:.6f} "
          f"({'fixed' if args.norm_div is not None else 'per-input max'})")
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
    plane_uncs = []
    for plane in args.planes:
        plane_snapshot_path = None
        if snapshot_dir is not None:
            plane_snapshot_path = os.path.join(
                snapshot_dir,
                f"{output_basename}_{plane}_slab{args.snapshot_slab}.png",
            )
        plane_out, plane_unc = run_plane_inference(
            vol_t, plane, ctx_emb, null_emb,
            sampler, args, device,
            snapshot_idx=snapshot_idx,
            snapshot_path=plane_snapshot_path,
        )
        plane_outputs.append(plane_out)
        if plane_unc is not None:
            plane_uncs.append(plane_unc)

    if len(plane_outputs) == 1:
        ensembled = plane_outputs[0]
    else:
        ensembled = torch.stack(plane_outputs, dim=0).mean(dim=0)
        print(f"Ensembled {len(plane_outputs)} planes by mean.")

    ensembled_unc = None
    if plane_uncs:
        ensembled_unc = (plane_uncs[0] if len(plane_uncs) == 1
                         else torch.stack(plane_uncs, dim=0).mean(dim=0))

    hist_ref = orig_img.get_fdata().astype(np.float32) if args.hist else None

    output_base, output_ext = split_nii_ext(args.output_path)
    if len(plane_outputs) > 1:
        for plane, plane_out in zip(args.planes, plane_outputs):
            plane_path = f"{output_base}_{plane}{output_ext}"
            save_volume(plane_out, plane_path, to_orig_ornt, args.rescale,
                        orig_img, hist_match_ref=hist_ref,
                        repad_bounds=crop_bounds, pre_crop_shape=pre_crop_shape)
        for plane, plane_unc in zip(args.planes, plane_uncs):
            save_uncertainty(plane_unc, f"{output_base}_{plane}_uncertainty{output_ext}",
                             to_orig_ornt, orig_img, crop_bounds, pre_crop_shape)

    save_volume(ensembled, args.output_path, to_orig_ornt, args.rescale,
                orig_img, hist_match_ref=hist_ref,
                repad_bounds=crop_bounds, pre_crop_shape=pre_crop_shape)

    if ensembled_unc is not None:
        save_uncertainty(ensembled_unc, f"{output_base}_uncertainty{output_ext}",
                         to_orig_ornt, orig_img, crop_bounds, pre_crop_shape)


if __name__ == "__main__":
    main()
