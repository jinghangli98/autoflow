"""3D NIfTI Flow Matching Training Script.

Conditional flow matching for accelerated MRI reconstruction. The model takes
an undersampled 3D patch (`condition`) and a previous-Z GT patch (`prev_chunk`)
and predicts the velocity field that maps noise to the GT patch.

Inputs are 5D tensors `(B, 1, X, Y, Z)`. The model concatenates
`(noisy_target, condition, prev_chunk)` along the channel dim ->
`(B, 3, X, Y, Z)` and predicts a velocity field of shape `(B, 1, X, Y, Z)`.

Usage:
    python -m torch.distributed.run --nproc_per_node=2 train_flow.py \
        --contrast mprage \
        --data_root /home/rflab/jil202/grappa-recon/dataset_grappa_nii \
        --distributed --fp16 --save_model --compile \
        --batch_size 2 --max_epochs 100 --sample 100 \
        --num_sampling_steps 2 \
        --checkpoint_path ./checkpoints/flow_matching_3d_mprage.pt
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DiffusionModelUNet
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import wandb
from metrics import evaluate_image_quality
from utils import EMA


class AutoregressiveFlowMatcher(nn.Module):
    """Autoregressive Flow Matching for 3D patch translation."""

    def __init__(self, model, sigma_min=0.001):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min

    def forward(self, x0, x1, condition, prev_chunk, t):
        """Compute the flow matching loss on a 3D patch.

        Shapes:
            x0, x1, condition, prev_chunk : (B, 1, X, Y, Z)
            t                             : (B,) or (1,)
        """
        batch_size = x0.shape[0]
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)
        elif t.dim() > 1:
            t = t.view(batch_size)

        t_expanded = t.view(-1, 1, 1, 1, 1)
        mu_t = t_expanded * x1 + (1 - t_expanded) * x0
        epsilon = torch.randn_like(x1)
        x_t = mu_t + self.sigma_min * epsilon

        v_t = x1 - x0
        timesteps = (t * 999).long()
        model_input = torch.cat([x_t, condition, prev_chunk], dim=1)

        v_pred = self.model(model_input, timesteps)
        return F.mse_loss(v_pred, v_t)

    @torch.no_grad()
    def sample(self, condition, prev_chunk, num_steps=50, device="cuda"):
        """Euler ODE integration in 3D."""
        batch_size = condition.shape[0]
        x = torch.randn_like(condition).to(device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            timesteps = (t * 999).long()
            model_input = torch.cat([x, condition, prev_chunk], dim=1)
            v = self.model(model_input, timesteps)
            x = x + v * dt

        return x


def parse_args():
    parser = argparse.ArgumentParser(description="3D NIfTI flow matching training")

    # Data
    parser.add_argument("--data_root", type=str,
                        default="/home/rflab/jil202/grappa-recon/dataset_grappa_nii",
                        help="Root containing train/<contrast>/ and test/<contrast>/")
    parser.add_argument("--contrast", type=str, required=True,
                        choices=["mprage", "tse"],
                        help="Which contrast to train on (one model per contrast)")

    # Training
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--sample", type=float, default=10.0,
                        help="Percentage of (undersampled,GT) sample pairs to use")
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable bf16 mixed precision")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile")
    parser.add_argument("--checkpoint_path", type=str,
                        default="./checkpoints/flow_matching_3d.pt")
    parser.add_argument("--log", type=bool, default=False,
                        help="Enable wandb logging")

    # Flow matching
    parser.add_argument("--sigma_min", type=float, default=0.001)
    parser.add_argument("--num_sampling_steps", type=int, default=2)

    # Autoregressive prev_chunk robustness
    parser.add_argument("--prev_chunk_dropout", type=float, default=0.3,
                        help="Probability of zeroing prev_chunk")
    parser.add_argument("--prev_chunk_noise", type=float, default=0.15,
                        help="Std of additive noise on prev_chunk")
    parser.add_argument("--noise_prob", type=float, default=0.5,
                        help="Probability of applying additive noise to prev_chunk")

    # EMA
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_start_epoch", type=int, default=99)

    # Distributed
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--dist_url", type=str, default="env://")
    parser.add_argument("--dist_backend", type=str, default="nccl")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=str, default="29500")

    return parser.parse_args()


def apply_prev_chunk_robustness(prev_chunk, args, device):
    """Apply robustness techniques to prev_chunk during training."""
    if random.random() < args.prev_chunk_dropout:
        return torch.zeros_like(prev_chunk)

    if random.random() < args.noise_prob:
        noise = torch.randn_like(prev_chunk) * args.prev_chunk_noise
        prev_chunk = prev_chunk + noise
        prev_chunk = torch.clamp(prev_chunk, 0.0, 1.0)

    return prev_chunk


def setup_distributed(rank, world_size, local_rank, backend="nccl"):
    print(f"Initializing process group: rank={rank}, world_size={world_size}, "
          f"local_rank={local_rank}, "
          f"MASTER_ADDR={os.environ.get('MASTER_ADDR', 'N/A')}, "
          f"MASTER_PORT={os.environ.get('MASTER_PORT', 'N/A')}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    print(f"Process group initialized: rank {rank}/{world_size}")
    return rank


def cleanup_distributed():
    if dist.is_initialized():
        print("Destroying process group")
        dist.destroy_process_group()


def visualize_flow_results(epoch, flow_matcher, device, condition, target, args, local_rank, prev_chunk=None):
    """Run flow sample on a single 3D patch, save central-slice 3-panel viz, return metrics."""
    flow_matcher.eval()
    if prev_chunk is None:
        prev_chunk = torch.zeros_like(condition)

    with torch.no_grad():
        generated = flow_matcher.sample(
            condition=condition,
            prev_chunk=prev_chunk,
            num_steps=args.num_sampling_steps,
            device=device,
        )

    cz = condition.shape[-1] // 2
    cond_2d = condition[..., cz]
    targ_2d = target[..., cz]
    gen_2d = generated[..., cz]

    if local_rank == 0:
        os.makedirs("visualization_results", exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(cond_2d[0, 0].cpu().numpy(), cmap="gray")
        axes[0].set_title("Undersampled (input)")
        axes[0].axis("off")
        axes[1].imshow(gen_2d[0, 0].cpu().numpy(), cmap="gray")
        axes[1].set_title("Reconstructed")
        axes[1].axis("off")
        axes[2].imshow(targ_2d[0, 0].cpu().numpy(), cmap="gray")
        axes[2].set_title("Ground truth")
        axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(f"visualization_results/flow3d_epoch_{epoch}.png")
        plt.close()

    gen_np = gen_2d.cpu().numpy()
    targ_np = targ_2d.cpu().numpy()
    metrics = evaluate_image_quality(gen_np, targ_np)
    return metrics["SSIM"], metrics["PSNR"], metrics["LPIPS"]


def validate_one_step(flow_matcher, device, condition, target, args):
    """Sample once on a 3D patch, return SSIM/PSNR/LPIPS on central slice + the full generated patch."""
    flow_matcher.eval()
    with torch.no_grad():
        generated = flow_matcher.sample(
            condition=condition,
            prev_chunk=torch.zeros_like(condition),
            num_steps=args.num_sampling_steps,
            device=device,
        )

    cz = condition.shape[-1] // 2
    gen_np = generated[..., cz].cpu().numpy()
    targ_np = target[..., cz].cpu().numpy()
    metrics = evaluate_image_quality(gen_np, targ_np)
    return metrics["SSIM"], metrics["PSNR"], metrics["LPIPS"], generated


def train(local_rank, args):
    """Main training function."""
    torch.set_float32_matmul_precision("high")

    if args.distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", local_rank))
        global_rank = int(os.environ.get("RANK", local_rank))
        world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
        args.world_size = world_size

        setup_distributed(
            rank=global_rank,
            world_size=world_size,
            local_rank=local_rank,
            backend=args.dist_backend,
        )
        device = torch.device(f"cuda:{local_rank}")
        print(f"[Rank {global_rank}] local_rank={local_rank}, device={device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        global_rank = 0
        local_rank = 0
        args.world_size = 1
        world_size = 1

    if args.log and global_rank == 0:
        wandb.init(project="FlowMatching_3D", config=vars(args))

    if global_rank == 0:
        os.makedirs("visualization_results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    from dataset import getloader_3d_patches

    if args.distributed:
        train_loader, val_loader = getloader_3d_patches(
            batch_size=args.batch_size,
            data_root=args.data_root,
            contrast=args.contrast,
            sample=args.sample,
            distributed=True, rank=global_rank, world_size=world_size,
            num_workers=args.num_workers,
        )
    else:
        train_loader, val_loader = getloader_3d_patches(
            batch_size=args.batch_size,
            data_root=args.data_root,
            contrast=args.contrast,
            sample=args.sample,
            num_workers=args.num_workers,
        )

    if global_rank == 0:
        print(f"Data loaders created, train: {len(train_loader)}, val: {len(val_loader)}")

    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=1,
        channels=(128, 256, 256),
        attention_levels=(False, False, False),
        num_res_blocks=2,
        num_head_channels=256,
        with_conditioning=False,
    )

    if os.path.exists(args.checkpoint_path):
        if global_rank == 0:
            print(f"Loading checkpoint from {args.checkpoint_path}")
        state_dict = torch.load(args.checkpoint_path, map_location=f"cuda:{local_rank}", weights_only=True)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    model.to(device)

    if args.compile:
        model = torch.compile(model)

    flow_matcher = AutoregressiveFlowMatcher(model, sigma_min=args.sigma_min)

    if args.distributed:
        if dist.is_initialized():
            dist.barrier()
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        flow_matcher.model = model

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6,
    )
    scaler = GradScaler(enabled=False)

    best_ssim_psnr = 0
    ema = EMA(model, args.ema_decay)
    ema.register()

    if global_rank == 0:
        print(f"Starting 3D flow matching training (world_size={args.world_size})")

    for epoch in range(args.max_epochs):
        if args.distributed:
            for loader in [train_loader, val_loader]:
                if hasattr(loader.sampler, "set_epoch"):
                    loader.sampler.set_epoch(epoch)

        if epoch == args.ema_start_epoch:
            if global_rank == 0:
                print(f"Initializing EMA weights from current model at epoch {epoch}")
            ema.register()

        model.train()
        epoch_loss = 0

        if global_rank == 0:
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100)
        else:
            progress_bar = enumerate(train_loader)

        for step, (condition, target, prev_chunk) in progress_bar:
            condition = condition.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            prev_chunk = prev_chunk.to(device, non_blocking=True)

            robust_prev = apply_prev_chunk_robustness(prev_chunk, args, device)

            optimizer.zero_grad(set_to_none=True)
            t = torch.rand(condition.shape[0], device=device)

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.fp16):
                x0 = torch.randn_like(target)
                loss = flow_matcher(
                    x0=x0, x1=target,
                    condition=condition, prev_chunk=robust_prev, t=t,
                )

            nan_flag = torch.tensor(
                1.0 if (torch.isnan(loss) or torch.isinf(loss)) else 0.0,
                device=device,
            )
            if args.distributed:
                dist.all_reduce(nan_flag, op=dist.ReduceOp.MAX)

            if nan_flag.item() > 0:
                if global_rank == 0:
                    print(f"NaN/Inf loss at epoch {epoch}, step {step}. Skipping.")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if epoch >= args.ema_start_epoch:
                ema.update()

            epoch_loss += loss.item()

            if global_rank == 0 and isinstance(progress_bar, tqdm):
                progress_bar.set_postfix({
                    "loss": epoch_loss / (step + 1),
                    "lr": optimizer.param_groups[0]["lr"],
                })

        scheduler.step()

        if args.distributed:
            epoch_loss_tensor = torch.tensor(epoch_loss, device=device)
            dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = epoch_loss_tensor.item() / args.world_size

        epoch_loss /= max(1, len(train_loader))

        if global_rank == 0:
            print(f"Epoch {epoch+1}/{args.max_epochs}, Training loss: {epoch_loss:.4f}")

        if (epoch + 1) % args.val_interval == 0:
            use_ema = (epoch >= args.ema_start_epoch)
            if use_ema:
                ema.apply_shadow()

            model.eval()

            val_ssim = 0
            val_psnr = 0
            val_lpips = 0
            val_batches = 0

            # Cap validation at ~1000 images globally (split across ranks under DDP).
            max_val_images_per_rank = max(1, 1000 // max(1, world_size))
            val_images_seen = 0

            for i, (condition, target, _prev) in enumerate(val_loader):
                condition = condition.to(device)
                target = target.to(device)

                batch_ssim, batch_psnr, batch_lpips, generated = validate_one_step(
                    flow_matcher=flow_matcher,
                    device=device,
                    condition=condition,
                    target=target,
                    args=args,
                )

                val_ssim += batch_ssim
                val_psnr += batch_psnr
                val_lpips += batch_lpips
                val_batches += 1
                val_images_seen += condition.shape[0]

                if i == 0 and global_rank == 0:
                    cz = condition.shape[-1] // 2
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(condition[0, 0, :, :, cz].cpu().numpy(), cmap="gray")
                    axes[0].set_title("Undersampled (input)")
                    axes[0].axis("off")
                    axes[1].imshow(generated[0, 0, :, :, cz].cpu().numpy(), cmap="gray")
                    axes[1].set_title("Reconstructed")
                    axes[1].axis("off")
                    axes[2].imshow(target[0, 0, :, :, cz].cpu().numpy(), cmap="gray")
                    axes[2].set_title("Ground truth")
                    axes[2].axis("off")
                    plt.tight_layout()
                    plt.savefig(f"visualization_results/flow3d_epoch_{epoch}.png")
                    plt.close()

                if val_images_seen >= max_val_images_per_rank:
                    break

            if args.distributed:
                metrics_tensor = torch.tensor([val_ssim, val_psnr, val_lpips, val_batches], device=device)
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
                val_ssim = metrics_tensor[0].item()
                val_psnr = metrics_tensor[1].item()
                val_lpips = metrics_tensor[2].item()
                val_batches = metrics_tensor[3].item()

            val_batches = max(val_batches, 1)
            avg_ssim = val_ssim / val_batches
            avg_psnr = val_psnr / val_batches
            avg_lpips = val_lpips / val_batches

            if global_rank == 0:
                print(f"Validation Metrics - SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}, LPIPS: {avg_lpips:.4f}")

            if global_rank == 0 and args.save_model:
                ssim_psnr = 0.7 * avg_ssim + 0.3 * avg_psnr
                if ssim_psnr > best_ssim_psnr:
                    best_ssim_psnr = ssim_psnr
                    checkpoint_path = f"./checkpoints/flow_matching_3d_{args.contrast}_best_epoch_{epoch}.pt"
                    if args.distributed:
                        torch.save(model.module.state_dict(), checkpoint_path)
                    else:
                        torch.save(model.state_dict(), checkpoint_path)
                    print(f"Saved best model with Score: {ssim_psnr:.4f}")

            if args.log and global_rank == 0:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": epoch_loss,
                    "val_ssim": avg_ssim,
                    "val_psnr": avg_psnr,
                    "val_lpips": avg_lpips,
                    "lr": optimizer.param_groups[0]["lr"],
                })

            if use_ema:
                ema.restore()

    if global_rank == 0 and args.save_model:
        final_path = f"./checkpoints/flow_matching_3d_{args.contrast}_final.pt"
        if args.distributed:
            torch.save(model.module.state_dict(), final_path)
        else:
            torch.save(model.state_dict(), final_path)

    if args.distributed:
        dist.barrier()
        cleanup_distributed()


def main():
    args = parse_args()

    if args.distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
        global_rank = int(os.environ.get("RANK", 0))
        args.world_size = world_size
        print(f"torchrun: global_rank={global_rank}, local_rank={local_rank}, world_size={world_size}")
        train(local_rank, args)
    else:
        train(0, args)


if __name__ == "__main__":
    main()
