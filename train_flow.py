"""
Flow Matching Training Script for Image-to-Image Translation

This implements conditional flow matching for autoregressive image translation,
adapted from the diffusion model training script. Key differences:

1. Instead of predicting noise, the model predicts velocity fields
2. Training uses continuous time t ∈ [0,1] instead of discrete timesteps
3. Sampling uses ODE integration with typically 50 steps (vs 1000 for diffusion)
4. The loss directly matches the velocity field v = x1 - x0

The script maintains compatibility with MONAI's DiffusionModelUNet by converting
continuous time to discrete timesteps internally.
Usage:
  python -m torch.distributed.run --nproc_per_node=4 train_flow.py \
  --distributed --fp16 --save_model --compile \
  --batch_size 3 --max_epochs 100 --save_model --fp16 --sample 100 --compile --checkpoint_path ./checkpoints/flow_matching_best_epoch_38.pt

"""

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from monai.networks.nets import DiffusionModelUNet
import pdb
from PIL import Image
import argparse
import wandb
import torchvision
from utils import visualize_and_save
import torchio as tio
from metrics import evaluate_image_quality

# Add imports for DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import random
import math


class ConditionalFlowMatcher(nn.Module):
    """
    Conditional Flow Matching for image-to-image translation.
    Based on "Flow Matching for Generative Modeling" and "Conditional Flow Matching"
    """
    def __init__(self, model, sigma_min=0.001):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min
        
    def forward(self, x0, x1, condition, t):
        """
        Forward flow matching loss computation.
        
        Args:
            x0: Source distribution samples (noisy data)
            x1: Target distribution samples (clean data)
            condition: Conditioning input (e.g., low-field MRI)
            t: Time steps in [0, 1]
        """
        # Ensure t is 1D for the model
        batch_size = x0.shape[0]
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)
        elif t.dim() > 1:
            t = t.view(batch_size)
            
        # Interpolate between x0 and x1
        t_expanded = t.view(-1, 1, 1, 1)
        mu_t = t_expanded * x1 + (1 - t_expanded) * x0
        sigma_t = self.sigma_min
        
        # Add noise for regularization
        epsilon = torch.randn_like(x1)
        x_t = mu_t + sigma_t * epsilon
        
        # Velocity field: v_t = x1 - x0
        v_t = x1 - x0
        
        # Convert t to discrete timesteps for DiffusionModelUNet
        timesteps = (t * 999).long()
        
        # Predict velocity using the model
        v_pred = self.model(torch.cat([x_t, condition], dim=1), timesteps)
        
        # MSE loss between predicted and true velocity
        loss = F.mse_loss(v_pred, v_t)
        
        return loss
    
    @torch.no_grad()
    def sample(self, condition, num_steps=50, device='cuda'):
        """
        Sample using ODE integration (Euler method).
        
        Args:
            condition: Conditioning input
            num_steps: Number of integration steps
        """
        batch_size = condition.shape[0]
        
        # Start from noise
        x = torch.randn(batch_size, 1, condition.shape[2], condition.shape[3]).to(device)
        
        # Time steps
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            
            # Convert t to discrete timesteps for the model
            timesteps = (t * 999).long()
            
            # Predict velocity
            v = self.model(torch.cat([x, condition], dim=1), timesteps)
            
            # Euler step
            x = x + v * dt
            
        return x


class AutoregressiveFlowMatcher(nn.Module):
    """
    Autoregressive Flow Matching that processes slices sequentially with previous slice context
    """
    def __init__(self, model, sigma_min=0.001):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min
        
    def forward(self, x0, x1, condition, prev_slice, t):
        """
        Forward flow matching loss with autoregressive context.
        
        Args:
            x0: Source distribution (noise)
            x1: Target distribution (clean mp2rage)
            condition: Conditioning input (mprage)
            prev_slice: Previous slice context
            t: Time steps in [0, 1]
        """
        # Ensure t is 1D for the model
        
        batch_size = x0.shape[0]
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)
        elif t.dim() > 1:
            t = t.view(batch_size)
        
        # Interpolate between x0 and x1
        t_expanded = t.view(-1, 1, 1, 1)
        mu_t = t_expanded * x1 + (1 - t_expanded) * x0
        sigma_t = self.sigma_min
        
        # Add small noise
        epsilon = torch.randn_like(x1)
        x_t = mu_t + sigma_t * epsilon
        
        # Velocity field
        v_t = x1 - x0
        
        # Concatenate inputs: x_t + condition + prev_slice
        model_input = torch.cat([x_t, condition, prev_slice], dim=1)
        
        # Convert t to discrete timesteps for DiffusionModelUNet
        # Scale from [0, 1] to [0, 999] to match diffusion model expectations
        timesteps = (t * 999).long()

        # Predict velocity
        v_pred = self.model(model_input, timesteps)
        
        # MSE loss
        loss = F.mse_loss(v_pred, v_t)
        
        return loss
    
    @torch.no_grad()
    def sample(self, condition, prev_slice, num_steps=50, device='cuda'):
        """
        Sample with autoregressive context using ODE integration.
        """
        batch_size = condition.shape[0]
        
        # Start from noise
        x = torch.randn_like(condition).to(device)
        
        # Time steps
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            
            # Model input with autoregressive context
            model_input = torch.cat([x, condition, prev_slice], dim=1)
            
            # Convert t to discrete timesteps for the model
            timesteps = (t * 999).long()
            
            # Predict velocity
            v = self.model(model_input, timesteps)
            
            # Euler step
            x = x + v * dt
            
        return x


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Autoregressive Flow Matching training")

    # Basic training parameters
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Learning rate for the optimizer")
    parser.add_argument("--crop_size", type=int, default=512,
                        help="Size to crop images to")
    parser.add_argument("--resize_size", type=int, default=512,
                        help="Size to resize images to")
    parser.add_argument("--data_path", type=str, 
                        default='/ix3/tibrahim/jil202/cfg_gen/qc_image_png/mprage_2_mp2rage_denoised/target/*/',
                        help="Path to the dataset directory")
    parser.add_argument("--save_model", action="store_true",
                        help="Flag to save model checkpoints")
    parser.add_argument("--fp16", action="store_true", 
                        help="Enable FP16/mixed precision training")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of workers for dataloader")
    parser.add_argument("--max_epochs", type=int, default=100,
                        help="Maximum number of training epochs")
    parser.add_argument("--sample", type=int, default=10,
                        help="Percentage of samples to use from dataset")
    parser.add_argument("--val_interval", type=int, default=1,
                        help="Interval for validation")
    parser.add_argument("--log", type=bool, default=False,
                        help="Enable wandb logging")
    parser.add_argument("--checkpoint_path", type=str, 
                        default="./checkpoints/flow_matching_model.pt",
                        help="Path to save or load the model checkpoint")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile for faster training")
    
    # Flow matching specific parameters
    parser.add_argument("--sigma_min", type=float, default=0.001,
                        help="Minimum sigma for flow matching noise")
    parser.add_argument("--num_sampling_steps", type=int, default=2,
                        help="Number of ODE integration steps for sampling")
    
    # Autoregressive parameters
    parser.add_argument("--num_slices", type=int, default=3,
                        help="Number of slices per volume for 3D context")
    parser.add_argument("--prev_slice_dropout", type=float, default=0.3,
                        help="Probability of masking out previous slice")
    parser.add_argument("--prev_slice_noise", type=float, default=0.15,
                        help="Noise level to add to previous slice")
    parser.add_argument("--noise_prob", type=float, default=0.5,
                        help="Probability of adding noise to previous slice")
    parser.add_argument("--max_slice_batch", type=int, default=8,
                        help="Maximum number of slices to process in parallel (micro-batching)")
    parser.add_argument("--ema_decay", type=float, default=0.9999,
                        help="Exponential moving average decay for model weights")
    parser.add_argument("--ema_start_epoch", type=int, default=0,
                        help="Epoch to start applying EMA")
    
    # Distributed training arguments
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training with DDP")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank for distributed training")
    parser.add_argument("--world_size", type=int, default=1,
                        help="Number of processes/GPUs for distributed training")
    parser.add_argument("--dist_url", type=str, default="env://",
                        help="URL used to set up distributed training")
    parser.add_argument("--dist_backend", type=str, default="nccl",
                        help="Distributed backend to use")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1",
                        help="Master address for distributed training")
    parser.add_argument("--master_port", type=str, default="29500",
                        help="Master port for distributed training")
    
    return parser.parse_args()


def apply_prev_slice_robustness(prev_slice, args, device):
    """Apply robustness techniques to previous slice."""
    # Random dropout
    if random.random() < args.prev_slice_dropout:
        return torch.zeros_like(prev_slice)
    
    # Add noise with probability
    if random.random() < args.noise_prob:
        noise = torch.randn_like(prev_slice) * args.prev_slice_noise
        prev_slice = prev_slice + noise
        prev_slice = torch.clamp(prev_slice, 0, 1)
    
    return prev_slice


def setup_distributed(rank, world_size, local_rank, backend='nccl'):
    """Initialize distributed training using torchrun env variables."""
    # When using torchrun, MASTER_ADDR, MASTER_PORT, RANK, LOCAL_RANK,
    # WORLD_SIZE are all set automatically. Use env:// init method.
    print(f"Initializing process group: rank={rank}, world_size={world_size}, "
          f"local_rank={local_rank}, "
          f"MASTER_ADDR={os.environ.get('MASTER_ADDR', 'N/A')}, "
          f"MASTER_PORT={os.environ.get('MASTER_PORT', 'N/A')}")
    
    # Set device before init so NCCL knows which GPU to use
    torch.cuda.set_device(local_rank)
    
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{local_rank}")
    )
    print(f"Process group initialized: rank {rank}/{world_size}")
    return rank


def cleanup_distributed():
    """Clean up distributed training resources."""
    if dist.is_initialized():
        print("Destroying process group")
        dist.destroy_process_group()


def visualize_flow_results(epoch, flow_matcher, device, mprage, mp2rage, args, local_rank, previous_slice=None):
    """Visualize flow matching results."""
    flow_matcher.eval()
    
    with torch.no_grad():
        if previous_slice is not None:
            generated = flow_matcher.sample(
                condition=mprage,
                prev_slice=previous_slice,
                num_steps=args.num_sampling_steps,
                device=device
            )
        else:
            generated = flow_matcher.sample(condition=mprage, prev_slice=torch.zeros_like(mprage), num_steps=args.num_sampling_steps, device=device)
    
    if local_rank == 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(mprage[0, 0].cpu().numpy(), cmap='gray')
        axes[0].set_title('MPRAGE (Input)')
        axes[0].axis('off')
        axes[1].imshow(generated[0, 0].cpu().numpy(), cmap='gray')
        axes[1].set_title('Generated MP2RAGE')
        axes[1].axis('off')
        axes[2].imshow(mp2rage[0, 0].cpu().numpy(), cmap='gray')
        axes[2].set_title('Ground Truth MP2RAGE')
        axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(f'visualization_results/flow_epoch_{epoch}.png')
        plt.close()
    
    plt.close()
    
    metrics = evaluate_image_quality(generated.cpu().numpy(), mp2rage.cpu().numpy())
    ssim = metrics['SSIM']
    psnr = metrics['PSNR']
    lpips = metrics['LPIPS']
    
    return ssim, psnr, lpips

def validate_one_step(flow_matcher, device, mprage, mp2rage, args):
    """Run validation on a single batch."""
    flow_matcher.eval()
    with torch.no_grad():
        generated = flow_matcher.sample(condition=mprage, prev_slice=torch.zeros_like(mprage), num_steps=args.num_sampling_steps, device=device)
        
    metrics = evaluate_image_quality(generated.cpu().numpy(), mp2rage.cpu().numpy())
    return metrics['SSIM'], metrics['PSNR'], metrics['LPIPS'], generated


def train(local_rank, args):
    """Main training function for flow matching."""
    torch.set_float32_matmul_precision('high')
    
    # Setup device and distributed training
    if args.distributed:
        # Read torchrun environment variables
        local_rank = int(os.environ.get("LOCAL_RANK", local_rank))
        global_rank = int(os.environ.get("RANK", local_rank))
        world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
        args.world_size = world_size

        setup_distributed(
            rank=global_rank,
            world_size=world_size,
            local_rank=local_rank,
            backend=args.dist_backend
        )
        device = torch.device(f"cuda:{local_rank}")
        print(f"[Rank {global_rank}] local_rank={local_rank}, device={device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        global_rank = 0
        local_rank = 0
        args.world_size = 1
    
    # Initialize wandb (only on global rank 0)
    if args.log and global_rank == 0:
        wandb.init(project="FlowMatching_Autoregressive", config=vars(args))
    
    # Create directories (only on global rank 0)
    if global_rank == 0:
        os.makedirs("visualization_results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
    
    # Import dataset module
    from dataset import getloader_3d
    
    # Setup data loaders
    if args.distributed:
        train_loader, val_loader = getloader_3d(
            batch_size=args.batch_size, data_root=args.data_path,
            crop_size=args.crop_size,
            num_slices=args.num_slices, sample=args.sample,
            distributed=True, rank=global_rank, world_size=world_size,
            num_workers=args.num_workers)
    else:
        train_loader, val_loader = getloader_3d(
            batch_size=args.batch_size,
            data_root=args.data_path,
            crop_size=args.crop_size,
            num_slices=args.num_slices,
            sample=args.sample,
            num_workers=args.num_workers,
        )

    # Import EMA
    from utils import EMA
    
    if global_rank == 0:
        print(f"Data loaders created, train: {len(train_loader)}, val: {len(val_loader)}")
    
    # Create model
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
    
    # Load checkpoint if available
    if os.path.exists(args.checkpoint_path):
        if global_rank == 0:
            print(f"Loading checkpoint from {args.checkpoint_path}")
        state_dict = torch.load(args.checkpoint_path, map_location=f'cuda:{local_rank}', weights_only=True)
        state_dict = {k.replace('_orig_mod.', ''):v for k, v in state_dict.items()} 
        model.load_state_dict(state_dict)
    
    model.to(device)
    
    if args.compile:
        model = torch.compile(model)
    
    # Create flow matcher
    flow_matcher = AutoregressiveFlowMatcher(model, sigma_min=args.sigma_min)
    
    # Wrap with DDP if distributed
    if args.distributed:
        if dist.is_initialized():
            dist.barrier()
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        flow_matcher.model = model
    
    # Optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6
    )
    
    # Training setup
    scaler = GradScaler(enabled=False)  # no-op for bf16
    
    best_ssim_psnr = 0
    
    # Initialize EMA
    ema = EMA(model, args.ema_decay)
    ema.register()

    if global_rank == 0:
        print(f"Starting autoregressive flow matching training (world_size={args.world_size})")
    
    # Training loop
    for epoch in range(args.max_epochs):
        # Set epoch for distributed sampler
        if args.distributed:
            for loader in [train_loader, val_loader]:
                if hasattr(loader.sampler, 'set_epoch'):
                    loader.sampler.set_epoch(epoch)
        
        # Reset EMA weights to current model weights when EMA starts
        if epoch == args.ema_start_epoch:
            if global_rank == 0:
                print(f"Initializing EMA weights from current model at epoch {epoch}")
            ema.register()
        
        model.train()
        epoch_loss = 0
        total_slices_processed = 0
        
        if global_rank == 0:
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100)
        else:
            progress_bar = enumerate(train_loader)
        
        for step, (mprage_volume, mp2rage_volume) in progress_bar:

            mprage_volume = mprage_volume.to(device)
            mp2rage_volume = mp2rage_volume.to(device)
            
            batch_size, num_slices = mprage_volume.shape[:2]
            
            # Process each volume autoregressively
            batch_loss = 0
            slices_in_batch = 0
                        
            mprage_vol = mprage_volume[:]
            mp2rage_vol = mp2rage_volume[:]
            
            # Initialize previous slice
            prev_slice = torch.zeros_like(mp2rage_vol[:,0:1])
            
            for slice_idx in range(num_slices):
                mprage_slice = mprage_vol[:,slice_idx:slice_idx+1]
                mp2rage_slice = mp2rage_vol[:,slice_idx:slice_idx+1]
                # Apply robustness to previous slice
                robust_prev_slice = apply_prev_slice_robustness(prev_slice, args, device)
                
                optimizer.zero_grad(set_to_none=True)
                
                # Sample random time for this specific slice
                t = torch.rand(1).to(device)
                
                with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.fp16):
                    x0 = torch.randn_like(mp2rage_slice)
                    loss = flow_matcher(x0=x0, x1=mp2rage_slice, condition=mprage_slice, prev_slice=robust_prev_slice, t=t)
                
                nan_flag = torch.tensor(
                    1.0 if (torch.isnan(loss) or torch.isinf(loss)) else 0.0, 
                    device=device
                )

                if args.distributed:
                    dist.all_reduce(nan_flag, op=dist.ReduceOp.MAX)  # MAX: if any rank has NaN, flag=1

                if nan_flag.item() > 0:
                    if global_rank == 0:
                        print(f"NaN/Inf loss detected across ranks at epoch {epoch}, "
                            f"step {step}, slice {slice_idx}. Skipping.")
                    prev_slice = torch.zeros_like(mp2rage_slice)
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                # Update EMA
                if epoch >= args.ema_start_epoch:
                    ema.update()
                
                batch_loss += loss.item()
                slices_in_batch += 1
                
                # Update previous slice
                prev_slice = mp2rage_slice.detach()
            
            # Average loss
            if slices_in_batch > 0:
                batch_loss /= slices_in_batch
                epoch_loss += batch_loss
                total_slices_processed += slices_in_batch
            
            if global_rank == 0 and isinstance(progress_bar, tqdm):
                progress_bar.set_postfix({
                    "loss": epoch_loss / (step + 1),
                    "lr": optimizer.param_groups[0]['lr'],
                    "slices": total_slices_processed
                })
        
        # Step scheduler
        scheduler.step()
        
        # Synchronize loss across processes
        if args.distributed:
            epoch_loss_tensor = torch.tensor(epoch_loss, device=device)
            dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = epoch_loss_tensor.item() / args.world_size
        
        epoch_loss /= len(train_loader)
        
        if global_rank == 0:
            print(f"Epoch {epoch+1}/{args.max_epochs}, Training loss: {epoch_loss:.4f}")
        
        # Validation
        if (epoch + 1) % args.val_interval == 0:
            use_ema = (epoch >= args.ema_start_epoch)
            if use_ema:
                ema.apply_shadow()
            
            model.eval()
            
            val_ssim = 0
            val_psnr = 0
            val_lpips = 0
            val_batches = 0
            
            for i, (mprage_vol, mp2rage_vol) in enumerate(val_loader):
                center_idx = mprage_vol.shape[1] // 2
                
                eval_mprage = mprage_vol[:, center_idx:center_idx+1].to(device)
                eval_mp2rage = mp2rage_vol[:, center_idx:center_idx+1].to(device)
                
                batch_ssim, batch_psnr, batch_lpips, generated = validate_one_step(
                    flow_matcher=flow_matcher,
                    device=device,
                    mprage=eval_mprage,
                    mp2rage=eval_mp2rage,
                    args=args
                )
                
                val_ssim += batch_ssim
                val_psnr += batch_psnr
                val_lpips += batch_lpips
                val_batches += 1
                
                # Visualize only the first batch on global rank 0
                if i == 0 and global_rank == 0:
                     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                     axes[0].imshow(eval_mprage[0, 0].cpu().numpy(), cmap='gray')
                     axes[0].set_title('MPRAGE (Input)')
                     axes[0].axis('off')
                     axes[1].imshow(generated[0, 0].cpu().numpy(), cmap='gray')
                     axes[1].set_title('Generated MP2RAGE')
                     axes[1].axis('off')
                     axes[2].imshow(eval_mp2rage[0, 0].cpu().numpy(), cmap='gray')
                     axes[2].set_title('Ground Truth MP2RAGE')
                     axes[2].axis('off')
                     plt.tight_layout()
                     plt.savefig(f'visualization_results/flow_epoch_{epoch}.png')
                     plt.close()

            # Synchronize metrics across processes
            if args.distributed:
                metrics_tensor = torch.tensor([val_ssim, val_psnr, val_lpips, val_batches], device=device)
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
                val_ssim = metrics_tensor[0].item()
                val_psnr = metrics_tensor[1].item()
                val_lpips = metrics_tensor[2].item()
                val_batches = metrics_tensor[3].item()
            
            avg_ssim = val_ssim / val_batches
            avg_psnr = val_psnr / val_batches
            avg_lpips = val_lpips / val_batches
            
            if global_rank == 0:
                print(f"Validation Metrics - SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}, LPIPS: {avg_lpips:.4f}")
            
            # Save best model (only global rank 0)
            if global_rank == 0 and args.save_model:
                ssim_psnr = 0.7 * avg_ssim + 0.3 * avg_psnr
                if ssim_psnr > best_ssim_psnr:
                    best_ssim_psnr = ssim_psnr
                    checkpoint_path = f"./checkpoints/flow_matching_best_epoch_{epoch}.pt"
                    if args.distributed:
                        torch.save(model.module.state_dict(), checkpoint_path)
                    else:
                        torch.save(model.state_dict(), checkpoint_path)
                    print(f"Saved best model with Score: {ssim_psnr:.4f}")

            # Log metrics
            if args.log and global_rank == 0:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": epoch_loss,
                    "val_ssim": avg_ssim,
                    "val_psnr": avg_psnr,
                    "val_lpips": avg_lpips,
                    "lr": optimizer.param_groups[0]['lr']
                })
            
            if use_ema:
                ema.restore()
    
    # Save final model
    if global_rank == 0 and args.save_model:
        final_path = "./checkpoints/flow_matching_final.pt"
        if args.distributed:
            torch.save(model.module.state_dict(), final_path)
        else:
            torch.save(model.state_dict(), final_path)
    
    # Cleanup
    if args.distributed:
        dist.barrier()
        cleanup_distributed()


def main():
    args = parse_args()
    
    if args.distributed:
        # When using torchrun, env vars are already set.
        # Just read them and call train directly (torchrun handles spawning).
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
        global_rank = int(os.environ.get("RANK", 0))
        args.world_size = world_size
        print(f"torchrun: global_rank={global_rank}, local_rank={local_rank}, world_size={world_size}")
        train(local_rank, args)
    else:
        # Single GPU training
        train(0, args)


if __name__ == "__main__":
    main()
