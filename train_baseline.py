"""
Baseline Training Script for Autoregressive Image-to-Image Translation (L1 Loss)
Using MONAI BasicUNet (no time embeddings).

Supports multi-node distributed training via:
  1. torchrun (recommended):
     torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 \
              --master_addr=NODE0_IP --master_port=29500 \
              train_baseline.py --distributed
  2. Manual mp.spawn (single-node fallback):
     python train_baseline.py --distributed --num_nodes=1 --gpus_per_node=2
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
from attention_unet import AttentionBasicUNet
from PIL import Image
import argparse
import wandb
import torchvision
from utils import visualize_and_save
import torchio as tio
from metrics import evaluate_image_quality

# DDP imports
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import random
import math
from datetime import timedelta

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Autoregressive Baseline (L1 Loss) training")

    # Basic training parameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--crop_size", type=int, default=480)
    parser.add_argument("--resize_size", type=int, default=480)
    parser.add_argument("--data_path", type=str,
                        default='/ix3/tibrahim/jil202/cfg_gen/qc_image_png/mprage_2_tse/mprage/coronal/')
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--log", type=bool, default=False)
    parser.add_argument("--checkpoint_path", type=str,
                        default="./checkpoints/baseline_l1_model.pt")
    parser.add_argument("--compile", action="store_true")

    # Autoregressive parameters
    parser.add_argument("--num_slices", type=int, default=5)
    parser.add_argument("--prev_slice_dropout", type=float, default=0.3)
    parser.add_argument("--prev_slice_noise", type=float, default=0.15)
    parser.add_argument("--noise_prob", type=float, default=0.5)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_start_epoch", type=int, default=0)

    # Distributed training arguments
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--dist_backend", type=str, default="nccl")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=str, default="29500")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--gpus_per_node", type=int, default=-1)

    return parser.parse_args()


def apply_prev_slice_robustness(prev_slice, args, device):
    """Apply robustness techniques to previous slice."""
    if random.random() < args.prev_slice_dropout:
        return torch.zeros_like(prev_slice)
    if random.random() < args.noise_prob:
        noise = torch.randn_like(prev_slice) * args.prev_slice_noise
        prev_slice = prev_slice + noise
        prev_slice = torch.clamp(prev_slice, 0, 1)
    return prev_slice


def setup_distributed(global_rank, world_size, master_addr, master_port, backend='nccl'):
    """Initialize distributed training process group."""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ.setdefault('NCCL_SOCKET_IFNAME', 'eth0')
    os.environ.setdefault('NCCL_IB_DISABLE', '0')
    os.environ.setdefault('NCCL_DEBUG', 'WARN')

    dist_url = f'tcp://{master_addr}:{master_port}'
    print(f"Initializing process group: global_rank={global_rank}/{world_size}, URL={dist_url}")
    dist.init_process_group(
        backend=backend, init_method=dist_url,
        world_size=world_size, rank=global_rank,
        timeout=timedelta(minutes=30),
    )
    print(f"Process group initialized: global_rank {global_rank}/{world_size}")
    return global_rank


def cleanup_distributed():
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.barrier()
        print("Destroying process group")
        dist.destroy_process_group()


def get_rank_info(args):
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        local_rank = args._spawn_local_rank
        global_rank = args.node_rank * args.gpus_per_node + local_rank
        world_size = args.num_nodes * args.gpus_per_node
    return local_rank, global_rank, world_size


def visualize_results(epoch, model, device, mprage, mp2rage, args, local_rank):
    """Visualize results."""
    model.eval()
    with torch.no_grad():
        prev_slice = torch.zeros_like(mprage)
        model_input = torch.cat([mprage, prev_slice], dim=1)
        generated = model(model_input)  # No timesteps needed

    if local_rank == 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(mprage[0, 0].cpu().numpy(), cmap='gray')
        axes[0].set_title('MPRAGE (Input)')
        axes[0].axis('off')
        axes[1].imshow(generated[0, 0].cpu().numpy(), cmap='gray')
        axes[1].set_title('Generated MP2RAGE (Baseline)')
        axes[1].axis('off')
        axes[2].imshow(mp2rage[0, 0].cpu().numpy(), cmap='gray')
        axes[2].set_title('Ground Truth MP2RAGE')
        axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(f'visualization_results/baseline_epoch_{epoch}.png')
        plt.close()
    plt.close()

    metrics = evaluate_image_quality(generated.cpu().numpy(), mp2rage.cpu().numpy())
    return metrics['SSIM'], metrics['PSNR'], metrics['LPIPS']


def validate_one_step(model, device, mprage, mp2rage, args):
    """Run validation on a single batch."""
    model.eval()
    with torch.no_grad():
        prev_slice = torch.zeros_like(mprage)
        model_input = torch.cat([mprage, prev_slice], dim=1)
        generated = model(model_input)  # No timesteps needed
    metrics = evaluate_image_quality(generated.cpu().numpy(), mp2rage.cpu().numpy())
    return metrics['SSIM'], metrics['PSNR'], metrics['LPIPS'], generated


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(local_rank_or_spawn_idx, args):
    torch.set_float32_matmul_precision('high')

    # --- Resolve ranks ---
    if args.distributed:
        if "LOCAL_RANK" in os.environ:
            local_rank, global_rank, world_size = get_rank_info(args)
        else:
            args._spawn_local_rank = local_rank_or_spawn_idx
            local_rank, global_rank, world_size = get_rank_info(args)

        args.world_size = world_size
        setup_distributed(global_rank, world_size, args.master_addr, args.master_port, args.dist_backend)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        global_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = (global_rank == 0)

    if args.log and is_main:
        wandb.init(project="Baseline_L1_Autoregressive", config=vars(args))

    if is_main:
        os.makedirs("visualization_results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
        print(f"Distributed config: world_size={world_size}, "
              f"num_nodes={args.num_nodes}, gpus_per_node={args.gpus_per_node}")

    from dataset import getloader_3d

    if args.distributed:
        train_loader, val_loader = getloader_3d(
            batch_size=args.batch_size, data_root=args.data_path,
            crop_size=args.crop_size,
            num_slices=args.num_slices, sample=args.sample,
            distributed=True, rank=global_rank, world_size=world_size)
    else:
        train_loader, val_loader = getloader_3d(
            batch_size=args.batch_size, data_root=args.data_path,
            crop_size=args.crop_size,
            num_slices=args.num_slices, sample=args.sample,
            num_workers=args.num_workers)

    from utils import EMA

    if is_main:
        print(f"Data loaders created, train: {len(train_loader)}, val: {len(val_loader)}")

    # =========================================================================
    # MONAI BasicUNet — no time embeddings, clean encoder-decoder
    #
    # features = (f0, f1, f2, f3, f4, f_final):
    #   f0..f4 = encoder levels (4 downsampling steps)
    #   f_final = channel count after the last upsample before final 1x1 conv
    #
    # (256, 256, 256, 512, 1024, 256) gives ~120M params, comparable to
    # DiffusionModelUNet with channels=(256,256,512), 2 res_blocks, attention.
    # Adjust features up/down to match your target param count.
    # =========================================================================
    model = AttentionBasicUNet(
        spatial_dims=2,
        in_channels=2,       # mprage + prev_slice
        out_channels=1,      # predicted tse
        features=(512, 512, 512, 512, 1024, 256),
        attention_levels=(False, False, False, True, True),  # attention at deeper levels
        num_heads=8,
        act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm=("instance", {"affine": True}),
        bias=True,
        dropout=0.0,
        upsample="deconv",
    )

    if is_main:
        print(f"Model parameters: {count_parameters(model):,}")

    # Load checkpoint
    if os.path.exists(args.checkpoint_path):
        if is_main:
            print(f"Loading checkpoint from {args.checkpoint_path}")
        state_dict = torch.load(args.checkpoint_path, map_location=f'cuda:{local_rank}', weights_only=True)
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
            print("Starting from scratch...")

    model.to(device)

    if args.compile:
        model = torch.compile(model)

    if args.distributed:
        if dist.is_initialized():
            dist.barrier()
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6)
    scaler = GradScaler()

    best_ssim_psnr = 0

    ema = EMA(model, args.ema_decay)
    ema.register()

    if is_main:
        print("Starting baseline (L1) training with BasicUNet")

    for epoch in range(args.max_epochs):
        if args.distributed:
            for loader in [train_loader, val_loader]:
                if hasattr(loader.sampler, 'set_epoch'):
                    loader.sampler.set_epoch(epoch)

        if epoch == args.ema_start_epoch:
            if is_main:
                print(f"Initializing EMA weights from current model at epoch {epoch}")
            ema.register()

        model.train()
        epoch_loss = 0
        total_slices_processed = 0

        if is_main:
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100)
        else:
            progress_bar = enumerate(train_loader)

        for step, (mprage_volume, mp2rage_volume) in progress_bar:
            mprage_volume = mprage_volume.to(device)
            mp2rage_volume = mp2rage_volume.to(device)
            batch_size, num_slices = mprage_volume.shape[:2]

            batch_loss = 0
            slices_in_batch = 0

            mprage_vol = mprage_volume[:]
            mp2rage_vol = mp2rage_volume[:]

            prev_slice = torch.zeros_like(mp2rage_vol[:, 0:1])

            for slice_idx in range(num_slices):
                mprage_slice = mprage_vol[:, slice_idx:slice_idx+1]
                mp2rage_slice = mp2rage_vol[:, slice_idx:slice_idx+1]
                robust_prev_slice = apply_prev_slice_robustness(prev_slice, args, device)

                optimizer.zero_grad(set_to_none=True)

                with autocast(device_type='cuda', enabled=args.fp16):
                    model_input = torch.cat([mprage_slice, robust_prev_slice], dim=1)
                    prediction = model(model_input)  # No timesteps!
                    loss = F.l1_loss(prediction, mp2rage_slice)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if epoch >= args.ema_start_epoch:
                    ema.update()

                batch_loss += loss.item()
                slices_in_batch += 1
                prev_slice = prediction.detach()

            if slices_in_batch > 0:
                batch_loss /= slices_in_batch
                epoch_loss += batch_loss
                total_slices_processed += slices_in_batch

            if is_main and isinstance(progress_bar, tqdm):
                progress_bar.set_postfix({
                    "loss": epoch_loss / (step + 1),
                    "lr": optimizer.param_groups[0]['lr'],
                    "slices": total_slices_processed
                })

        scheduler.step()

        if args.distributed:
            epoch_loss_tensor = torch.tensor(epoch_loss, device=device)
            dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = epoch_loss_tensor.item() / world_size

        epoch_loss /= len(train_loader)

        if is_main:
            print(f"Epoch {epoch+1}/{args.max_epochs}, Training loss: {epoch_loss:.4f}")

        # Validation
        if (epoch + 1) % args.val_interval == 0:
            use_ema = (epoch >= args.ema_start_epoch)
            if use_ema:
                ema.apply_shadow()

            model.eval()
            val_ssim, val_psnr, val_lpips, val_batches = 0, 0, 0, 0

            for i, (mprage_vol, mp2rage_vol) in enumerate(val_loader):
                center_idx = mprage_vol.shape[1] // 2
                eval_mprage = mprage_vol[:, center_idx:center_idx+1].to(device)
                eval_mp2rage = mp2rage_vol[:, center_idx:center_idx+1].to(device)

                batch_ssim, batch_psnr, batch_lpips, generated = validate_one_step(
                    model=model, device=device,
                    mprage=eval_mprage, mp2rage=eval_mp2rage, args=args)

                val_ssim += batch_ssim
                val_psnr += batch_psnr
                val_lpips += batch_lpips
                val_batches += 1

                if i == 0 and is_main:
                    visualize_results(epoch, model, device, eval_mprage, eval_mp2rage, args, local_rank)

            if args.distributed:
                metrics_tensor = torch.tensor([val_ssim, val_psnr, val_lpips, val_batches], device=device)
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
                val_ssim, val_psnr, val_lpips, val_batches = metrics_tensor.tolist()

            avg_ssim = val_ssim / val_batches
            avg_psnr = val_psnr / val_batches
            avg_lpips = val_lpips / val_batches

            if is_main:
                print(f"Validation Metrics - SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}, LPIPS: {avg_lpips:.4f}")

            if is_main and args.save_model:
                ssim_psnr = 0.7 * avg_ssim + 0.3 * avg_psnr
                if ssim_psnr > best_ssim_psnr:
                    best_ssim_psnr = ssim_psnr
                    checkpoint_path = f"./checkpoints/baseline_best_epoch_{epoch}.pt"
                    if args.distributed:
                        torch.save(model.module.state_dict(), checkpoint_path)
                    else:
                        torch.save(model.state_dict(), checkpoint_path)
                    print(f"Saved best model with Score: {ssim_psnr:.4f}")

            if args.log and is_main:
                wandb.log({
                    "epoch": epoch, "train_loss": epoch_loss,
                    "val_ssim": avg_ssim, "val_psnr": avg_psnr,
                    "val_lpips": avg_lpips, "lr": optimizer.param_groups[0]['lr']
                })

            if use_ema:
                ema.restore()

    # Save final model
    if is_main and args.save_model:
        final_path = "./checkpoints/baseline_final.pt"
        if args.distributed:
            torch.save(model.module.state_dict(), final_path)
        else:
            torch.save(model.state_dict(), final_path)

    if args.distributed:
        cleanup_distributed()


def main():
    args = parse_args()

    if args.gpus_per_node < 0:
        args.gpus_per_node = torch.cuda.device_count()

    if args.distributed:
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            global_rank = int(os.environ["RANK"])
            print(f"[torchrun] local_rank={local_rank}, global_rank={global_rank}, "
                  f"world_size={world_size}")
            train(local_rank, args)
        else:
            total_world_size = args.num_nodes * args.gpus_per_node
            print(f"[mp.spawn] Spawning {args.gpus_per_node} processes on "
                  f"node {args.node_rank}/{args.num_nodes} "
                  f"(world_size={total_world_size})")
            os.environ['MASTER_ADDR'] = args.master_addr
            os.environ['MASTER_PORT'] = args.master_port
            mp.set_start_method('spawn', force=True)
            mp.spawn(train, nprocs=args.gpus_per_node, args=(args,), join=True)
    else:
        train(0, args)


if __name__ == "__main__":
    main()
