"""
Flow Matching Inference Script for Autoregressive Image Translation

This script performs inference using a trained flow matching model for image-to-image translation.
Instead of denoising like diffusion models, it integrates velocity fields through ODE solving.
"""

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from monai.networks.nets import DiffusionModelUNet
import argparse
import torchio as tio
from PIL import Image
import nibabel as nib
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from utils2 import save_enhanced_dicom, get_scan_plane
import glob


# ---------------------------------------------------------------------------
# Padding / unpadding helpers
# ---------------------------------------------------------------------------

def pad_volume_to_divisible(data, divisible_by=16):
    """
    Pad a 3-D volume (X, Y, Z) so every dimension is divisible by `divisible_by`.
    Padding is split symmetrically; any odd remainder goes to the right/back.
    No content-based cropping is performed.

    Returns:
        padded_data : Tensor (X', Y', Z')
        pad_info    : dict with keys 'original_shape' and 'padding'
    """
    original_shape = data.shape  # (X, Y, Z)

    def _pad_amount(size):
        remainder = size % divisible_by
        if remainder == 0:
            return 0, 0
        total = divisible_by - remainder
        left  = total // 2
        right = total - left
        return left, right

    x_l, x_r = _pad_amount(original_shape[0])
    y_l, y_r = _pad_amount(original_shape[1])
    z_l, z_r = _pad_amount(original_shape[2])

    # F.pad expects innermost dimension first: (z_l, z_r, y_l, y_r, x_l, x_r)
    padded_data = F.pad(data, (z_l, z_r, y_l, y_r, x_l, x_r), mode='constant', value=0)

    pad_info = {
        'original_shape': original_shape,
        'padding': ((x_l, x_r), (y_l, y_r), (z_l, z_r)),
    }
    return padded_data, pad_info


def unpad_volume(padded_data, pad_info):
    """
    Remove padding added by `pad_volume_to_divisible`.

    Args:
        padded_data : Tensor or ndarray (X', Y', Z')
        pad_info    : dict returned by `pad_volume_to_divisible`

    Returns:
        Tensor with shape == pad_info['original_shape']
    """
    if isinstance(padded_data, np.ndarray):
        padded_data = torch.from_numpy(padded_data)

    (x_l, x_r), (y_l, y_r), (z_l, z_r) = pad_info['padding']
    ox, oy, oz = pad_info['original_shape']

    restored = padded_data[
        x_l : x_l + ox,
        y_l : y_l + oy,
        z_l : z_l + oz,
    ]

    assert restored.shape == torch.Size([ox, oy, oz]), (
        f"Shape mismatch after unpadding: got {tuple(restored.shape)}, "
        f"expected {(ox, oy, oz)}"
    )
    return restored


# ---------------------------------------------------------------------------
# Flow matching model wrapper
# ---------------------------------------------------------------------------

class AutoregressiveFlowMatcher(nn.Module):
    """Autoregressive Flow Matching for inference."""

    def __init__(self, model, sigma_min=0.001):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min

    @torch.no_grad()
    def sample(self, condition, prev_slice, num_steps=50, device='cuda', guidance_scale=1.0):
        """Euler method ODE integration."""
        batch_size = condition.shape[0]
        x  = torch.randn_like(condition).to(device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            model_input = torch.cat([x, condition, prev_slice], dim=1)
            timesteps   = (t * 999).long()
            v = self.model(model_input, timesteps)
            x = torch.clamp(x + v * dt, -3, 3)

        return x

    @torch.no_grad()
    def sample_heun(self, condition, prev_slice, num_steps=50, device='cuda', guidance_scale=1.0):
        """Heun's method (2nd-order Runge-Kutta)."""
        batch_size = condition.shape[0]
        x  = torch.randn_like(condition).to(device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_cur   = i * dt
            t_cur_b = torch.full((batch_size,), t_cur, device=device)

            v1      = self.model(torch.cat([x, condition, prev_slice], dim=1),
                                 (t_cur_b * 999).long())
            x_euler = x + v1 * dt

            t_next  = min((i + 1) * dt, 1.0)
            t_next_b = torch.full((batch_size,), t_next, device=device)
            v2      = self.model(torch.cat([x_euler, condition, prev_slice], dim=1),
                                 (t_next_b * 999).long())

            x = torch.clamp(x + dt * (v1 + v2) / 2.0, -3, 3)

        return x

    @torch.no_grad()
    def sample_rk4(self, condition, prev_slice, num_steps=50, device='cuda', guidance_scale=1.0):
        """4th-order Runge-Kutta integration."""
        batch_size = condition.shape[0]
        x  = torch.randn_like(condition).to(device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_cur = i * dt

            t_b   = torch.full((batch_size,), t_cur, device=device)
            k1    = self.model(torch.cat([x, condition, prev_slice], dim=1),
                               (t_b * 999).long())

            t_mid = t_cur + 0.5 * dt
            t_mid_b = torch.full((batch_size,), t_mid, device=device)
            k2    = self.model(torch.cat([x + 0.5*k1*dt, condition, prev_slice], dim=1),
                               (t_mid_b * 999).long())

            k3    = self.model(torch.cat([x + 0.5*k2*dt, condition, prev_slice], dim=1),
                               (t_mid_b * 999).long())

            t_next = min((i + 1) * dt, 1.0)
            t_next_b = torch.full((batch_size,), t_next, device=device)
            k4    = self.model(torch.cat([x + k3*dt, condition, prev_slice], dim=1),
                               (t_next_b * 999).long())

            x = torch.clamp(x + (k1 + 2*k2 + 2*k3 + k4) * dt / 6.0, -3, 3)

        return x


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Flow matching autoregressive inference")

    parser.add_argument("--model_path",   type=str, required=True)
    parser.add_argument("--input_path",   type=str, required=True)
    parser.add_argument("--output_dir",   type=str, default="./flow_inference_results")
    parser.add_argument("--input_format", type=str, choices=['nifti', 'dcm', 'png'], default='dcm')

    parser.add_argument("--num_sampling_steps", type=int,   default=50)
    parser.add_argument("--sigma_min",          type=float, default=0.001)
    parser.add_argument("--guidance_scale",     type=float, default=1.0)

    parser.add_argument("--resize_size", type=int,   default=480)
    parser.add_argument("--fp16",        action="store_true", default=False)
    parser.add_argument("--resolution",  type=float, default=0.375)

    parser.add_argument("--batch_size",            type=int, default=1)
    parser.add_argument("--device",                type=str, default="cuda")
    parser.add_argument("--save_individual_slices", action="store_true")
    parser.add_argument("--save_nifti",             action="store_true", default=True)
    parser.add_argument("--seed",                   type=int, default=42)
    parser.add_argument("--euler",                  action="store_true", default=False)
    parser.add_argument("--rep",                    type=int, default=1)

    parser.add_argument("--auto",    action="store_true", default=False)
    parser.add_argument("--compile", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path, device, args):
    print(f"Loading flow matching model from {model_path}")

    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=(256, 256, 512),
        attention_levels=(False, False, True),
        num_res_blocks=2,
        num_head_channels=512,
        with_conditioning=False,
    )

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    fixed_state_dict = {
        k.replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(fixed_state_dict)
    model.to(device)
    model.eval()

    if args.fp16:
        model = model.half()
    if args.compile:
        model = torch.compile(model, backend="inductor")

    print("Flow matching model loaded successfully!")
    return model


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_dicom_with_sitk(folder_path, reorient=True):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(folder_path)
    if not series_ids:
        raise ValueError("No DICOM series found.")

    reader.SetFileNames(reader.GetGDCMSeriesFileNames(folder_path, series_ids[0]))
    image = reader.Execute()

    if reorient:
        image = sitk.DICOMOrient(image, 'RAS')

    array  = sitk.GetArrayFromImage(image)
    affine = np.eye(4)
    affine[:3, :3] = np.array(image.GetDirection()).reshape(3, 3) * np.array(image.GetSpacing())
    affine[:3,  3] = image.GetOrigin()

    return image, array, affine


def resample_to_iso(image, new_spacing=(0.35, 0.35, 0.35), interpolator=sitk.sitkLinear):
    original_spacing = image.GetSpacing()
    original_size    = image.GetSize()

    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetInterpolator(interpolator)

    return sitk.GetArrayFromImage(resample.Execute(image))


def quantile_normalization(data, lower_quantile=0.01, upper_quantile=0.99):
    if isinstance(data, str):
        data = nib.load(data).get_fdata()

    data      = np.nan_to_num(data, nan=0.0, posinf=None, neginf=None)
    lower     = np.percentile(data.flatten(), lower_quantile * 100)
    upper     = np.percentile(data.flatten(), upper_quantile * 100)
    data      = np.clip(data, lower, upper)
    return (data - lower) / (upper - lower + 1e-3)


def load_input_data(input_path, input_format, resize_size, dimension='axial', args=None):
    """Load, resample, pad, and normalise input MPRAGE data.

    Returns:
        data        : Tensor (num_slices, H, W) ready for inference
        affine      : affine matrix (or None)
        header      : NIfTI header (or None)
        pad_info    : dict for unpad_volume
        raw_img_obj : original nibabel image (or None)
    """
    resolution  = args.resolution if args else 0.35
    raw_img_obj = None

    if input_format == 'nifti':
        raw_img     = nib.load(input_path)
        raw_img_obj = raw_img
        orig_spacing = raw_img.header.get_zooms()

        if dimension == 'axial':
            target_spacing = (resolution, resolution, orig_spacing[2])
        elif dimension == 'coronal':
            target_spacing = (resolution, orig_spacing[1], resolution)
        elif dimension == 'sagittal':
            target_spacing = (orig_spacing[0], resolution, resolution)
        else:
            target_spacing = (resolution, resolution, resolution)

        nii_img       = tio.transforms.Resample(target_spacing)(raw_img)
        mprage_volume = torch.tensor(nii_img.get_fdata())
        affine        = nii_img.affine
        header        = nii_img.header

    elif input_format == 'png':
        slice_files = sorted(Path(input_path).glob("*.png"))
        slices = []
        for sf in slice_files:
            img = Image.open(sf).convert('L').resize((resize_size, resize_size), Image.BILINEAR)
            slices.append(torch.from_numpy(np.array(img) / 255.0).float())
        data = torch.stack(slices)
        # PNG path: no padding applied, return a dummy pad_info with zero padding
        pad_info = {
            'original_shape': data.shape,
            'padding': ((0, 0), (0, 0), (0, 0)),
        }
        return data, None, None, pad_info, raw_img_obj

    elif input_format == 'dcm':
        image, _, _ = load_dicom_with_sitk(input_path)
        mprage_volume = torch.tensor(
            resample_to_iso(image, new_spacing=(resolution, resolution, resolution))
        ).float()
        affine  = None
        header  = None

    else:
        raise ValueError(f"Unsupported input format: {input_format}")

    # --- Pad to divisible-by-16 (no content cropping) ---
    mprage_volume, pad_info = pad_volume_to_divisible(mprage_volume, divisible_by=16)

    # --- Normalise ---
    data = quantile_normalization(mprage_volume.numpy())

    # --- Orient slices for the chosen view ---
    if input_format == 'nifti':
        if dimension == 'axial':
            data = torch.from_numpy(data).float().permute(2, 0, 1)
        elif dimension == 'coronal':
            data = torch.from_numpy(data).float().permute(1, 0, 2)
        else:  # sagittal
            data = torch.from_numpy(data).float()

    elif input_format == 'dcm':
        if dimension == 'axial':
            data = torch.from_numpy(data).float().permute(2, 0, 1)
        elif dimension == 'coronal':
            data = torch.from_numpy(data).float().permute(1, 0, 2)
        else:  # sagittal
            data = torch.from_numpy(data).float()
        data = torch.flip(data, dims=[0, 1])

    return data, affine, header, pad_info, raw_img_obj


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def flow_matching_inference(flow_matcher, mprage_volume, args, device):
    """Run autoregressive flow matching slice-by-slice.

    Returns:
        generated_volume : Tensor (num_slices, H, W)
        mprage_volume    : same input, returned for convenience
    """
    num_slices, height, width = mprage_volume.shape
    generated_volume = torch.zeros_like(mprage_volume)
    prev_slice       = torch.zeros((1, 1, height, width), device=device)

    print(f"Generating {num_slices} slices using flow matching...")

    for slice_idx in tqdm(range(num_slices), desc="Generating slices"):
        mprage_slice = mprage_volume[slice_idx:slice_idx+1].to(device).unsqueeze(0)

        with torch.no_grad():
            if args.fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    if args.euler:
                        gen = flow_matcher.sample(mprage_slice, prev_slice, args.num_sampling_steps, device, args.guidance_scale)
                    else:
                        gen = flow_matcher.sample_rk4(mprage_slice, prev_slice, args.num_sampling_steps, device, args.guidance_scale)
            else:
                if args.euler:
                    gen = flow_matcher.sample(mprage_slice, prev_slice, args.num_sampling_steps, device, args.guidance_scale)
                else:
                    gen = flow_matcher.sample_rk4(mprage_slice, prev_slice, args.num_sampling_steps, device, args.guidance_scale)

            generated_volume[slice_idx] = gen.squeeze(0).squeeze(0).cpu()

            # Debug snapshot
            if slice_idx == 100:
                plt.imshow(gen.squeeze().cpu(), cmap='gray')
                plt.savefig(f'flow_slice_{slice_idx}.png')
                plt.close()

            prev_slice = gen.detach() if args.auto else torch.zeros((1, 1, height, width), device=device)

    return generated_volume, mprage_volume


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_results(generated_volume, output_dir, args, affine=None, header=None,
                 input_filename="output", dimension='axial'):
    os.makedirs(output_dir, exist_ok=True)
    volume_data = generated_volume / np.max(generated_volume) * 256
    volume_data = np.clip(volume_data, 0, 255).astype(np.uint16)

    if args.save_nifti:
        nii_img = (nib.Nifti1Image(volume_data, affine, header)
                   if affine is not None and header is not None
                   else nib.Nifti1Image(volume_data, affine))
        output_path = os.path.join(output_dir, f"{input_filename}_{dimension}_flow_generated.nii.gz")
        nib.save(nii_img, output_path)
        print(f"Saved NIfTI volume: {output_path}")

    if args.input_format == 'dcm':
        input_path    = Path(args.input_path)
        old_base      = Path('/home/rflab/dcmtk-sorted')
        new_base      = Path('/home/rflab/nexus/flow_inference_results')
        relative_path = input_path.relative_to(old_base)
        output_path   = new_base / relative_path.parent / f"{relative_path.name}_FlowSuperRes"
        save_enhanced_dicom(volume_data, args.input_path, f"{output_path}_{dimension}",
                            new_spacing=(0.35, 0.35, 0.35),
                            series_description_suffix="_FlowSuperRes")

    if args.save_individual_slices:
        slice_dir = os.path.join(output_dir, "slices")
        os.makedirs(slice_dir, exist_ok=True)
        for i, sl in enumerate(generated_volume):
            sl_norm = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
            Image.fromarray((sl_norm * 255).astype(np.uint8), mode='L').save(
                os.path.join(slice_dir, f"slice_{i:03d}.png"))
        print(f"Saved {len(generated_volume)} slice images in: {slice_dir}")

    create_visualization(generated_volume, output_dir, input_filename)


def create_visualization(generated_volume, output_dir, input_filename):
    num_slices    = len(generated_volume)
    slice_indices = list(range(0, num_slices, max(1, num_slices // 8)))
    if slice_indices[-1] != num_slices - 1:
        slice_indices.append(num_slices - 1)

    ncols = len(slice_indices) // 2 + len(slice_indices) % 2
    fig, axes = plt.subplots(2, ncols, figsize=(15, 6))
    axes = [axes] if len(slice_indices) == 1 else axes.flatten()

    for i, idx in enumerate(slice_indices):
        if i < len(axes):
            axes[i].imshow(generated_volume[idx], cmap='gray')
            axes[i].set_title(f'Slice {idx}')
            axes[i].axis('off')
    for i in range(len(slice_indices), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    viz_path = os.path.join(output_dir, f"{input_filename}_flow_visualization.png")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {viz_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.input_format == 'dcm':
        args.save_nifti = False
        acq_dir = get_scan_plane(glob.glob(args.input_path + '/*')[0])
    elif args.input_format == 'nifti':
        args.save_nifti = True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model        = load_model(args.model_path, device, args)
    flow_matcher = AutoregressiveFlowMatcher(model, sigma_min=args.sigma_min)

    print(f"Loading input data from: {args.input_path}")
    dimension_list     = ['axial'] * args.rep
    inferenced_volumes = []

    for dimension in dimension_list:
        input_filename = str(Path(args.input_path)).split('/')[-1].split('.nii.gz')[0]
        output_path    = os.path.join(args.output_dir,
                                      f"{input_filename}_{dimension}_flow_generated.nii.gz")

        if os.path.exists(output_path):
            print(f"Output file {output_path} already exists. Skipping.")
            continue

        mprage_volume, affine, header, pad_info, raw_img_obj = load_input_data(
            args.input_path, args.input_format, args.resize_size,
            dimension=dimension, args=args
        )
        print(f"Input volume shape: {mprage_volume.shape}")

        start_time = time.time()
        generated_volume, orig_volume = flow_matching_inference(
            flow_matcher, mprage_volume, args, device
        )

        # --- Orientation post-processing ---
        if args.input_format == 'dcm':
            generated_volume = torch.flip(generated_volume, dims=[0, 1])

        # Permute from slice-first back to (X, Y, Z)
        if dimension == 'axial':
            generated_volume = generated_volume.permute(1, 2, 0).numpy()
        elif dimension == 'coronal':
            generated_volume = generated_volume.permute(1, 0, 2).numpy()
        else:  # sagittal
            generated_volume = generated_volume.numpy()
            if args.input_format == 'nifti':
                orig_volume = unpad_volume(orig_volume, pad_info)
                save_results(orig_volume.numpy(), args.output_dir, args, affine, header,
                             input_filename, dimension='orig')

        # --- Unpad back to pre-padding shape ---
        if args.input_format == 'nifti':
            generated_volume = unpad_volume(generated_volume, pad_info).numpy()

            # Resample back to original image geometry
            if raw_img_obj is not None:
                print("Resampling back to original input geometry...")
                gen_tensor  = torch.from_numpy(generated_volume).unsqueeze(0).float()
                source_img  = tio.ScalarImage(tensor=gen_tensor, affine=affine)
                resampler   = tio.Resample(target=args.input_path, image_interpolation='linear')
                resampled   = resampler(source_img)
                generated_volume = resampled.data[0].numpy()
                affine  = raw_img_obj.affine
                header  = raw_img_obj.header
                print(f"Resampled output shape: {generated_volume.shape}")

        inferenced_volumes.append(generated_volume)

        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.2f}s  "
              f"({inference_time / len(mprage_volume):.2f}s per slice)")

        save_results(generated_volume, args.output_dir, args, affine, header,
                     input_filename, dimension=dimension)
        print(f"Results saved for {dimension} view")

    # --- Ensemble average ---
    if args.input_format == 'dcm':
        mean_vol = np.stack(inferenced_volumes).mean(0)
        if acq_dir == 'axial':
            save_results(np.flip(mean_vol, axis=[1, 2]),
                         args.output_dir, args, affine, header, input_filename, dimension='mean')
        elif acq_dir == 'sagittal':
            save_results(np.flip(mean_vol.transpose(-1, 0, 1), axis=[1, 2]),
                         args.output_dir, args, affine, header, input_filename, dimension='mean')
    else:
        save_results(np.stack(inferenced_volumes).mean(0),
                     args.output_dir, args, affine, header, input_filename, dimension='mean')

    print(f"Done! All results saved in: {args.output_dir}")


if __name__ == "__main__":
    '''
    Example Usage:

    DICOM input:
    python enhance_flow_autoregressive.py \
        --model_path /home/rflab/nexus/checkpoints/T1_flow_matching_best_epoch_19.pt \
        --input_path /home/rflab/dcmtk-sorted/.../T1w_MPRAGE_Sag.15 \
        --num_sampling_steps 1 --fp16 --euler --auto --compile

    NIfTI input:
    python enhance_flow_autoregressive.py \
        --model_path ./checkpoints/flow_matching_best_epoch_51.pt \
        --input_path /ix3/tibrahim/jil202/cfg_gen/qc_image_nii/mprage_2_tse_val/mprage/md_0701PE48_20211109110427.nii.gz \
        --output_dir ./nii --num_sampling_steps 1 --fp16 --euler --auto --input_format nifti --compile
    '''
    main()
