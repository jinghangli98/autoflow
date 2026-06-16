
import lpips
import torch
import pdb
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_mae(enhanced_img, reference_img):
    """Compute Mean Absolute Error (MAE) per image and average over the batch."""
    mae_per_image = np.mean(np.abs(enhanced_img - reference_img))
    return mae_per_image

def calculate_nmse(enhanced_img, reference_img):
    """Compute Normalized Mean Squared Error (NMSE) per image and average over the batch."""
    nmse_values = []
    # Flatten batch and frame dimensions for processing
    batch_size, num_frames = enhanced_img.shape[:2]
    for b in range(batch_size):
        for f in range(num_frames):
            img_enhanced = enhanced_img[b, f]
            img_ref = reference_img[b, f]
            mse = np.mean((img_enhanced - img_ref) ** 2)
            norm_factor = np.mean(img_ref ** 2)
            nmse = mse / norm_factor if norm_factor > 0 else np.inf
            nmse_values.append(nmse)
    return np.mean(nmse_values)

_LPIPS_MODEL = None


def _get_lpips_model():
    """Lazily build and cache one VGG-LPIPS model (per process).

    Reused across calls so validation doesn't reload VGG weights on every
    batch/scale/target-type evaluation.
    """
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        _LPIPS_MODEL = lpips.LPIPS(net='vgg')
        if torch.cuda.is_available():
            _LPIPS_MODEL = _LPIPS_MODEL.cuda()
    return _LPIPS_MODEL


def calculate_lpips(generated, target):
    """Compute LPIPS distance for each image in the batch and average."""
    loss_fn = _get_lpips_model()

    # Reshape from [batch, frames, H, W] to [batch*frames, H, W]
    batch_size, num_frames = generated.shape[:2]
    generated_flat = generated.reshape(-1, generated.shape[-2], generated.shape[-1])
    target_flat = target.reshape(-1, target.shape[-2], target.shape[-1])

    # Convert to tensors and add channel dimension, then repeat to get 3 channels for VGG
    generated_tensor = torch.from_numpy(generated_flat).float().unsqueeze(1).repeat(1, 3, 1, 1)
    target_tensor = torch.from_numpy(target_flat).float().unsqueeze(1).repeat(1, 3, 1, 1)

    if torch.cuda.is_available():
        generated_tensor = generated_tensor.cuda()
        target_tensor = target_tensor.cuda()

    with torch.no_grad():
        lpips_scores = loss_fn(generated_tensor, target_tensor)
    return lpips_scores.mean().item()

def calculate_ssim(generated, target):
    """Compute SSIM for each image in the batch and average."""
    ssim_values = []
    batch_size, num_frames = generated.shape[:2]
    
    for b in range(batch_size):
        for f in range(num_frames):
            gen = generated[b, f]  # Shape: [H, W]
            ref = target[b, f]     # Shape: [H, W]
            data_range = ref.max() - ref.min()
            if data_range > 0:
                ssim_val = ssim(gen, ref, data_range=data_range)
                ssim_values.append(ssim_val)
            else:
                ssim_values.append(1.0)  # Perfect similarity for constant images
    
    return np.mean(ssim_values)

def calculate_psnr(generated, target):
    """Compute PSNR for each image in the batch and average."""
    psnr_values = []
    batch_size, num_frames = generated.shape[:2]
    
    for b in range(batch_size):
        for f in range(num_frames):
            gen = generated[b, f]  # Shape: [H, W]
            ref = target[b, f]     # Shape: [H, W]
            data_range = ref.max() - ref.min()
            if data_range > 0:
                psnr_val = psnr(ref, gen, data_range=data_range)
                psnr_values.append(psnr_val)
            else:
                psnr_values.append(float('inf'))  # Perfect reconstruction for constant images
    
    return np.mean([p for p in psnr_values if p != float('inf')]) if any(p != float('inf') for p in psnr_values) else float('inf')

def evaluate_image_quality(generated, target):
    """
    Evaluate image quality metrics for tensors of shape [batch, frames, H, W]
    
    Args:
        generated: numpy array of shape [batch, frames, H, W]
        target: numpy array of shape [batch, frames, H, W]
    """
    ssim_value = calculate_ssim(generated, target)
    psnr_value = calculate_psnr(generated, target)
    lpips_value = calculate_lpips(generated, target)
    nmse = calculate_nmse(generated, target)
    mae = calculate_mae(generated, target)
    
    # Handle infinite PSNR in final score calculation
    psnr_normalized = min(psnr_value / 100.0, 1.0) if psnr_value != float('inf') else 1.0
    
    final_score = (0.7 * ssim_value + 0.1 * psnr_normalized + 0.1 * (1 - mae) + 0.1 * (1 - nmse))
    
    print(f"SSIM: {ssim_value:.4f}, PSNR: {psnr_value:.4f}, LPIPS: {lpips_value:.4f}, NMSE: {nmse:.4f}, MAE: {mae:.4f} | Final Score: {final_score:.4f}")
    
    return {
        "SSIM": ssim_value,
        "PSNR": psnr_value,
        "LPIPS": lpips_value,
        "NMSE": nmse,
        "MAE": mae,
        "Final_Score": final_score
    }
