import torch
import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from metrics import evaluate_image_quality
from PIL import Image
import torch.distributed as dist
import wandb
import os
import torch.nn.functional as F
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler
import torch.nn as nn
import lpips
import pdb

def norm(img):
    """Normalize the image to 0-255 range."""
    img = img.float()  # Ensure we're working with float tensor
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 255).byte()

def online_inference(epoch, model, device, mprage, mp2rage, previous_slice, args, local_rank=0):
    model.eval()
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(num_inference_steps=10)

    with torch.no_grad():                
        input_img = mprage
        target_img = mp2rage
        noise = torch.randn_like(input_img).to(device)
        current_img = noise  
        combined = torch.cat((input_img, noise, previous_slice), dim=1)
        progress_bar = tqdm(scheduler.timesteps, desc=f"Generating mp2rage (Epoch {epoch+1}, Rank {local_rank})")

        for t in progress_bar:  # go through the noising process
            with autocast(enabled=False):
                # Unwrap DDP model for inference if needed
                if isinstance(model, DDP):
                    model_output = model.module(combined, timesteps=torch.Tensor((t,)).to(current_img.device))
                else:
                    model_output = model(combined, timesteps=torch.Tensor((t,)).to(current_img.device))
                current_img, _ = scheduler.step(model_output, t, current_img)
                combined = torch.cat((input_img, current_img, previous_slice), dim=1)

    return current_img

# def visualize_and_save(epoch, model, device, scheduler, mprage, mp2rage, previous_slice, args, local_rank=0):
#     """
#     Run visualization and evaluation across all GPUs, then gather results.
#     """
#     model.eval()
    
#     with torch.no_grad():                
#         input_img = mprage
#         target_img = mp2rage
#         scheduler.set_timesteps(num_inference_steps=args.ddpm_steps)
#         noise = torch.randn_like(input_img).to(device)
#         current_img = noise  
#         combined = torch.cat((input_img, noise, previous_slice), dim=1)
#         progress_bar = tqdm(scheduler.timesteps, desc=f"Generating mp2rage (Epoch {epoch+1}, Rank {local_rank})")

#         for t in progress_bar:  # go through the noising process
#             with autocast(enabled=False):
#                 # Unwrap DDP model for inference if needed
#                 if isinstance(model, DDP):
#                     model_output = model.module(combined, timesteps=torch.Tensor((t,)).to(current_img.device))
#                 else:
#                     model_output = model(combined, timesteps=torch.Tensor((t,)).to(current_img.device))
#                 current_img, _ = scheduler.step(model_output, t, current_img)
#                 combined = torch.cat((input_img, current_img, previous_slice), dim=1)

#         # Compute metrics for this sample
#         metrics = evaluate_image_quality(current_img.cpu().squeeze().numpy(), target_img.cpu().squeeze().numpy())
#         os.makedirs(f"visualization_results/{args.ddpm_steps}", exist_ok=True)
#         for idx in range(len(input_img)):
#             vis = torch.hstack((norm(input_img.squeeze().cpu()[idx]), norm(current_img.squeeze().cpu()[idx]), norm(target_img.squeeze().cpu()[idx])))
#             vis = vis.numpy().astype(np.uint8)
#             vis = Image.fromarray(vis)
            
#             # Save the visualization to disk
#             vis_path = f"visualization_results/{args.ddpm_steps}/epoch_{epoch+1}_rank_{local_rank}_idx_{idx}.png"
#             vis.save(vis_path)
    
#     score = 0.7 * metrics['SSIM'] + 0.1 * metrics['PSNR'] + 0.1 * (1 - metrics['MAE']) + 0.1 * (1 - metrics['NMSE'])
#     return metrics['SSIM'], metrics['PSNR'], metrics['LPIPS'], score

def visualize_and_save(epoch, model, device, scheduler, mprage, mp2rage, args, local_rank=0, previous_slice=None):
    """
    Run visualization and evaluation for autoregressive model.
    
    Args:
        epoch: Current training epoch
        model: The diffusion model (possibly wrapped in DDP)
        device: Device to run on
        scheduler: DDPM/DDIM scheduler
        mprage: Input low-resolution image(s) (batch_size, 1, H, W)
        mp2rage: Target high-resolution image(s) (batch_size, 1, H, W)
        args: Training arguments
        local_rank: Local rank for distributed training
        previous_slice: Previous slice for autoregressive context (batch_size, 1, H, W)
                       If None, will use zeros
    """
    model.eval()
    
    # Handle previous slice
    if previous_slice is None:
        previous_slice = torch.zeros_like(mprage)
    
    with torch.no_grad():                
        input_img = mprage.to(device)
        target_img = mp2rage.to(device)
        previous_slice = previous_slice.to(device)
        
        # Set up scheduler for inference
        scheduler.set_timesteps(num_inference_steps=1000)  # Use fewer steps for faster validation
        
        # Start with pure noise
        current_img = torch.randn_like(input_img).to(device)
        
        # Create progress bar only for rank 0 to avoid cluttered output
        if local_rank == 0:
            progress_bar = tqdm(scheduler.timesteps, desc=f"Generating mp2rage (Epoch {epoch+1})")
        else:
            progress_bar = scheduler.timesteps
            
        for t in progress_bar:
            # Combine inputs: mprage + current_noisy + previous_slice
            combined = torch.cat((input_img, current_img, previous_slice), dim=1)
            
            with autocast(enabled=False):
                # Unwrap DDP model for inference if needed
                if isinstance(model, DDP):
                    model_output = model.module(combined, timesteps=torch.tensor([t]).to(device))
                else:
                    model_output = model(combined, timesteps=torch.tensor([t]).to(device))
                
                # Scheduler step - fixed the typo from model*output to model_output
                current_img, _ = scheduler.step(model_output, t, current_img)

        # Compute metrics for this sample (use first sample in batch)
        generated_np = current_img.cpu().squeeze().numpy()
        target_np = target_img.cpu().squeeze().numpy()
        
        # Handle batch dimension
        # if len(generated_np.shape) == 3:  # If batch dimension exists
        #     generated_np = generated_np[0]
        #     target_np = target_np[0]
        
        metrics = evaluate_image_quality(generated_np, target_np)
        
        # Save visualizations (only save from rank 0 or if save_all_ranks is True)
        if local_rank == 0 or args.save_all_ranks:
            os.makedirs(f"visualization_results/{args.ddpm_steps}", exist_ok=True)
            
            for idx in range(len(input_img)):
                # Create visualization: input | generated | target
                vis_input = norm(input_img.squeeze().cpu()[idx])
                vis_generated = norm(current_img.squeeze().cpu()[idx])
                vis_target = norm(target_img.squeeze().cpu()[idx])
                
                vis = torch.hstack((vis_input, vis_generated, vis_target))
                vis = vis.numpy().astype(np.uint8)
                vis = Image.fromarray(vis)
                
                # Save the visualization to disk
                vis_path = f"visualization_results/{args.ddpm_steps}/epoch_{epoch+1}_rank_{local_rank}_idx_{idx}.png"
                vis.save(vis_path)
    
    # Calculate composite score
    score = 0.7 * metrics['SSIM'] + 0.1 * metrics['PSNR'] + 0.1 * (1 - metrics['MAE']) + 0.1 * (1 - metrics['NMSE'])
    
    return metrics['SSIM'], metrics['PSNR'], metrics['LPIPS'], score

def visualize_and_save_volume(epoch, model, device, scheduler, mprage_volume, mp2rage_volume, args, local_rank=0):
    """
    Generate and visualize a full volume autoregressively.
    
    Args:
        epoch: Current training epoch
        model: The diffusion model
        device: Device to run on
        scheduler: DDPM/DDIM scheduler
        mprage_volume: Input volume (batch_size, num_slices, H, W)
        mp2rage_volume: Target volume (batch_size, num_slices, H, W)
        args: Training arguments
        local_rank: Local rank for distributed training
    """
    model.eval()
    
    batch_size, num_slices = mprage_volume.shape[:2]
    generated_slices = []
    all_metrics = []
    
    # Initialize previous slice
    prev_slice = torch.zeros_like(mprage_volume[:, 0:1]).to(device)
    
    # Set up scheduler for inference
    scheduler.set_timesteps(num_inference_steps=50)
    
    if local_rank == 0:
        print(f"Generating volume with {num_slices} slices autoregressively...")
    
    with torch.no_grad():
        for slice_idx in range(num_slices):
            mprage_slice = mprage_volume[:, slice_idx:slice_idx+1].to(device)
            mp2rage_slice = mp2rage_volume[:, slice_idx:slice_idx+1].to(device)
            
            # Start with pure noise for this slice
            current_slice = torch.randn_like(mprage_slice).to(device)
            
            # Denoising loop for this slice
            for t in scheduler.timesteps:
                combined = torch.cat((mprage_slice, current_slice, prev_slice), dim=1)
                
                with autocast(enabled=False):
                    if isinstance(model, DDP):
                        model_output = model.module(combined, timesteps=torch.tensor([t]).to(device))
                    else:
                        model_output = model(combined, timesteps=torch.tensor([t]).to(device))
                
                current_slice = scheduler.step(model_output, t, current_slice).prev_sample
            
            generated_slices.append(current_slice)
            
            # Update previous slice for next iteration
            prev_slice = current_slice.detach()
            
            # Compute metrics for this slice
            generated_np = current_slice.cpu().squeeze().numpy()
            target_np = mp2rage_slice.cpu().squeeze().numpy()
            
            if len(generated_np.shape) == 3:  # Handle batch dimension
                generated_np = generated_np[0]
                target_np = target_np[0]
            
            slice_metrics = evaluate_image_quality(generated_np, target_np)
            all_metrics.append(slice_metrics)
    
    # Stack generated volume
    generated_volume = torch.cat(generated_slices, dim=1)
    
    # Save volume visualization (center slices)
    if local_rank == 0 or args.save_all_ranks:
        os.makedirs(f"visualization_results/{args.ddpm_steps}/volumes", exist_ok=True)
        
        center_idx = num_slices // 2
        slice_indices = [max(0, center_idx-1), center_idx, min(num_slices-1, center_idx+1)]
        
        for i, slice_idx in enumerate(slice_indices):
            vis_input = norm(mprage_volume[0, slice_idx].cpu())
            vis_generated = norm(generated_volume[0, slice_idx].cpu())
            vis_target = norm(mp2rage_volume[0, slice_idx].cpu())
            
            vis = torch.hstack((vis_input, vis_generated, vis_target))
            vis = vis.numpy().astype(np.uint8)
            vis = Image.fromarray(vis)
            
            vis_path = f"visualization_results/{args.ddpm_steps}/volumes/epoch_{epoch+1}_rank_{local_rank}_slice_{slice_idx}.png"
            vis.save(vis_path)
    
    # Calculate average metrics across all slices
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    score = 0.7 * avg_metrics['SSIM'] + 0.1 * avg_metrics['PSNR'] + 0.1 * (1 - avg_metrics['MAE']) + 0.1 * (1 - avg_metrics['NMSE'])
    
    return avg_metrics['SSIM'], avg_metrics['PSNR'], avg_metrics['LPIPS'], score

def inference(model, device, scheduler, low_res_image, args):
    """
    Run inference on a single low-resolution image to generate the high-resolution version.
    
    Args:
        model: The trained diffusion model
        device: Device to run inference on
        scheduler: Diffusion scheduler
        low_res_image: Low-resolution input image tensor (already on device)
        args: Arguments containing inference parameters
        
    Returns:
        Generated high-resolution image
    """
    model.eval()
    
    with torch.no_grad():
        input_img = low_res_image.to(device)
        scheduler.set_timesteps(num_inference_steps=args.ddpm_steps)
        # Standard inference starting from random noise
        noise = torch.randn_like(input_img).to(device)
        current_img = noise  # Start from random noise
        
        progress_bar = tqdm(scheduler.timesteps, desc="Generating high-res image")

        for t in progress_bar:
            with autocast(enabled=False):
                combined = torch.cat((input_img, current_img), dim=1)
                model_output = model(combined, timesteps=torch.Tensor((t,)).to(device))
                current_img, _ = scheduler.step(model_output, t, current_img)
    
        # Convert the generated image to a displayable format
        output_img = norm(current_img.squeeze().cpu())
        output_img = output_img.numpy().astype(np.uint8)
        
        return Image.fromarray(output_img)

class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                # self.shadow[name] = new_average.clone()
                # In-place update of shadow
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=(1.0 - self.decay))

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}
