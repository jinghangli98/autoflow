"""3D NIfTI Flow Matching Training Script.

Conditional flow matching for accelerated MRI reconstruction. The model takes
an undersampled 3D patch (`condition`) and predicts the velocity field that
maps noise to the GT patch.

Inputs are 5D tensors `(B, 1, X, Y, Z)`. The model concatenates
`(noisy_target, condition)` along the channel dim -> `(B, 2, X, Y, Z)` and
predicts a velocity field of shape `(B, 1, X, Y, Z)`.

Usage:
    python -m torch.distributed.run --nproc_per_node=8 train_flow.py \
        --contrast mprage \
        --data_root /ix1/tibrahim/jil202/studies/dataset_grappa_nii \
        --distributed --fp16 --save_model --compile \
        --batch_size 4 --max_epochs 100 --sample 100 \
        --num_sampling_steps 2
"""

import argparse
import os

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


CONTEXT_INPUT_DIM = 7      # (vx, vy, vz, log1p(TR), log1p(TE), log1p(TI), FA/180)
CONTEXT_HIDDEN_DIM = 128
CONTEXT_OUTPUT_DIM = 256   # must match cross_attention_dim on the UNet


class ContextEncoder(nn.Module):
    """Encode (B, in_dim) raw scalar context into (B, 1, out_dim) for cross-attention.

    Carries a learnable null embedding used as the unconditional signal for
    classifier-free guidance.
    """

    def __init__(self, in_dim=CONTEXT_INPUT_DIM, hidden_dim=CONTEXT_HIDDEN_DIM,
                 out_dim=CONTEXT_OUTPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.null_emb = nn.Parameter(torch.zeros(1, 1, out_dim))

    def forward(self, ctx_vec):
        return self.net(ctx_vec).unsqueeze(1)

    def null(self, batch_size: int):
        return self.null_emb.expand(batch_size, -1, -1)


class FlowMatcher(nn.Module):
    """Flow Matching for 3D patch translation, with context conditioning."""

    def __init__(self, model, sigma_min=0.001):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min

    def forward(self, x0, x1, condition, t, context=None):
        """Compute the flow matching loss on a 3D patch.

        Shapes:
            x0, x1, condition : (B, 1, X, Y, Z)
            t                 : (B,) or (1,)
            context           : (B, seq_len, cross_attention_dim) or None
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
        model_input = torch.cat([x_t, condition], dim=1)

        v_pred = self.model(model_input, timesteps, context=context)
        return F.mse_loss(v_pred, v_t)

    @torch.no_grad()
    def _guided_velocity(self, model_input, timesteps, context, null_context,
                         guidance_scale):
        """Velocity with optional classifier-free guidance.

        guidance_scale == 1.0 (or no null_context) -> single conditional eval.
        guidance_scale == 0.0 -> single unconditional eval (pure null context).
        otherwise -> two evals mixed as v_uncond + scale * (v_cond - v_uncond).
        """
        if guidance_scale == 1.0 or null_context is None:
            return self.model(model_input, timesteps, context=context)
        if guidance_scale == 0.0:
            return self.model(model_input, timesteps, context=null_context)
        v_cond = self.model(model_input, timesteps, context=context)
        v_uncond = self.model(model_input, timesteps, context=null_context)
        return v_uncond + guidance_scale * (v_cond - v_uncond)

    @torch.no_grad()
    def sample(self, condition, num_steps=50, device="cuda", context=None,
               null_context=None, guidance_scale=1.0):
        """Euler ODE integration in 3D, with optional classifier-free guidance."""
        batch_size = condition.shape[0]
        x = torch.randn_like(condition).to(device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            timesteps = (t * 999).long()
            model_input = torch.cat([x, condition], dim=1)
            v = self._guided_velocity(
                model_input, timesteps, context, null_context, guidance_scale,
            )
            x = x + v * dt

        return x


def parse_args():
    parser = argparse.ArgumentParser(description="3D NIfTI flow matching training")

    # Data
    parser.add_argument("--data_root", type=str,
                        default="/home/rflab/jil202/grappa-recon/dataset_grappa_nii",
                        help="Root containing train/<contrast>/ and test/<contrast>/")
    parser.add_argument("--contrast", type=str, required=True, nargs="+",
                        choices=["mprage", "tse", "mp2rage", "swi", "flair"],
                        help="One or more contrasts to train on, e.g. "
                             "`--contrast mprage` or `--contrast mprage tse`")

    # Training
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--sample", type=float, default=10.0,
                        help="Percentage of (undersampled,GT) sample pairs to use")
    parser.add_argument("--samples_per_contrast", type=int, default=None,
                        help="If set, each contrast contributes this many "
                             "training samples per epoch (0 = auto-balance to "
                             "the smallest contrast). The larger contrast "
                             "cycles through a fresh random subset each epoch.")
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

    # Classifier-free guidance
    parser.add_argument("--cfg_dropout_prob", type=float, default=0.1,
                        help="Probability of replacing the encoded context "
                             "with the learnable null embedding during training.")
    parser.add_argument("--val_guidance_scales", type=float, nargs="+",
                        default=[0.0, 1.0, 1.5],
                        help="Guidance scales evaluated each validation. "
                             "0.0 = unconditional (no context), 1.0 = plain "
                             "conditional, >1.0 = classifier-free guidance.")
    parser.add_argument("--ckpt_guidance_scale", type=float, default=1.5,
                        help="Which val guidance scale drives best-checkpoint "
                             "selection (auto-added to --val_guidance_scales "
                             "if absent).")

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

    args = parser.parse_args()
    if args.ckpt_guidance_scale not in args.val_guidance_scales:
        args.val_guidance_scales.append(args.ckpt_guidance_scale)
    return args


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


def visualize_flow_results(epoch, flow_matcher, device, condition, target, args, local_rank):
    """Run flow sample on a single 3D patch, save central-slice 3-panel viz, return metrics."""
    flow_matcher.eval()

    with torch.no_grad():
        generated = flow_matcher.sample(
            condition=condition,
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


def validate_one_step(flow_matcher, device, condition, target, args, context=None,
                      null_context=None, guidance_scale=1.0):
    """Sample once on a 3D patch, return SSIM/PSNR/LPIPS on central slice + the full generated patch."""
    flow_matcher.eval()
    with torch.no_grad():
        generated = flow_matcher.sample(
            condition=condition,
            num_steps=args.num_sampling_steps,
            device=device,
            context=context,
            null_context=null_context,
            guidance_scale=guidance_scale,
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
            samples_per_contrast=args.samples_per_contrast,
        )
    else:
        train_loader, val_loader = getloader_3d_patches(
            batch_size=args.batch_size,
            data_root=args.data_root,
            contrast=args.contrast,
            sample=args.sample,
            num_workers=args.num_workers,
            samples_per_contrast=args.samples_per_contrast,
        )

    if global_rank == 0:
        print(f"Data loaders created, train: {len(train_loader)}, val: {len(val_loader)}")

    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=1,
        channels=(128, 256, 512),
        attention_levels=(False, False, False),
        num_res_blocks=2,
        num_head_channels=512,
        with_conditioning=True,
        cross_attention_dim=CONTEXT_OUTPUT_DIM,
    )
    context_encoder = ContextEncoder()

    if os.path.exists(args.checkpoint_path):
        if global_rank == 0:
            print(f"Loading checkpoint from {args.checkpoint_path}")
        ckpt = torch.load(args.checkpoint_path, map_location=f"cuda:{local_rank}", weights_only=True)

        def _load_relaxed(target_module, src_sd, label):
            tgt_sd = target_module.state_dict()
            filtered = {}
            skipped = []
            partial = []
            for k, v in src_sd.items():
                if k not in tgt_sd:
                    skipped.append((k, tuple(v.shape), None))
                    continue
                tv = tgt_sd[k]
                if v.shape == tv.shape:
                    filtered[k] = v
                    continue
                if v.dim() == tv.dim() and all(
                    (sd == td) or (i == 1)
                    for i, (sd, td) in enumerate(zip(v.shape, tv.shape))
                ):
                    new_v = tv.clone()
                    c = min(v.shape[1], tv.shape[1])
                    slicer_src = (slice(None), slice(0, c)) + (slice(None),) * (v.dim() - 2)
                    slicer_tgt = (slice(None), slice(0, c)) + (slice(None),) * (tv.dim() - 2)
                    new_v[slicer_tgt] = v[slicer_src]
                    filtered[k] = new_v
                    partial.append((k, tuple(v.shape), tuple(tv.shape), c))
                else:
                    skipped.append((k, tuple(v.shape), tuple(tv.shape)))
            missing, unexpected = target_module.load_state_dict(filtered, strict=False)
            if global_rank == 0:
                if partial:
                    for k, sshape, tshape, c in partial:
                        print(f"  [{label}] partial copy {k}: ckpt {sshape} -> model {tshape}, copied {c} channels along dim 1")
                if skipped:
                    for k, sshape, tshape in skipped:
                        print(f"  [{label}] skipped {k}: ckpt {sshape} vs model {tshape}")
                if missing:
                    print(f"  [{label}] missing keys: {len(missing)}")
                if unexpected:
                    print(f"  [{label}] unexpected keys: {len(unexpected)}")

        if isinstance(ckpt, dict) and "model" in ckpt:
            model_sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
            _load_relaxed(model, model_sd, "model")
            if "context_encoder" in ckpt:
                ctx_sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["context_encoder"].items()}
                _load_relaxed(context_encoder, ctx_sd, "context_encoder")
        else:
            # Backward-compat: legacy checkpoint with bare model state dict (no context_encoder).
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
            _load_relaxed(model, state_dict, "model")
            if global_rank == 0:
                print("Legacy checkpoint loaded; context_encoder initialized from scratch.")

    model.to(device)
    context_encoder.to(device)

    if args.compile:
        model = torch.compile(model)

    flow_matcher = FlowMatcher(model, sigma_min=args.sigma_min)

    if args.distributed:
        if dist.is_initialized():
            dist.barrier()
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        context_encoder = DDP(context_encoder, device_ids=[local_rank],
                              output_device=local_rank, find_unused_parameters=False)
        flow_matcher.model = model

    optimizer = torch.optim.AdamW(
        params=list(model.parameters()) + list(context_encoder.parameters()),
        lr=args.lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6,
    )
    scaler = GradScaler(enabled=False)

    best_ssim_psnr = 0
    ema = EMA(model, args.ema_decay)
    ema.register()
    ema_ctx = EMA(context_encoder, args.ema_decay)
    ema_ctx.register()

    if global_rank == 0:
        print(f"Starting 3D flow matching training (world_size={args.world_size})")

    for epoch in range(args.max_epochs):
        for loader in [train_loader, val_loader]:
            sampler = getattr(loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

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

        for step, (condition, target, ctx_vec) in progress_bar:
            condition = condition.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            ctx_vec = ctx_vec.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            t = torch.rand(condition.shape[0], device=device)

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.fp16):
                x0 = torch.randn_like(target)
                ctx_emb = context_encoder(ctx_vec)
                if args.cfg_dropout_prob > 0:
                    bsz = ctx_emb.shape[0]
                    ctx_module = context_encoder.module if args.distributed else context_encoder
                    drop = (torch.rand(bsz, 1, 1, device=device) < args.cfg_dropout_prob)
                    ctx_emb = torch.where(drop, ctx_module.null(bsz), ctx_emb)
                loss = flow_matcher(
                    x0=x0, x1=target,
                    condition=condition, t=t,
                    context=ctx_emb,
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
                ema_ctx.update()

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
                ema_ctx.apply_shadow()

            model.eval()
            context_encoder.eval()
            ctx_module = context_encoder.module if args.distributed else context_encoder

            scales = args.val_guidance_scales
            # Per-scale running sums of central-slice metrics.
            sum_ssim = {s: 0.0 for s in scales}
            sum_psnr = {s: 0.0 for s in scales}
            sum_lpips = {s: 0.0 for s in scales}
            val_batches = 0

            # Cap validation at ~1000 images globally (split across ranks under DDP).
            max_val_images_per_rank = max(1, 1000 // max(1, world_size))
            val_images_seen = 0

            for i, (condition, target, ctx_vec) in enumerate(val_loader):
                condition = condition.to(device)
                target = target.to(device)
                ctx_vec = ctx_vec.to(device)
                ctx_emb = context_encoder(ctx_vec)
                null_emb = ctx_module.null(ctx_emb.shape[0])

                gen_for_viz = {}
                for s in scales:
                    b_ssim, b_psnr, b_lpips, generated = validate_one_step(
                        flow_matcher=flow_matcher,
                        device=device,
                        condition=condition,
                        target=target,
                        args=args,
                        context=ctx_emb,
                        null_context=null_emb,
                        guidance_scale=s,
                    )
                    sum_ssim[s] += b_ssim
                    sum_psnr[s] += b_psnr
                    sum_lpips[s] += b_lpips
                    if i == 0 and global_rank == 0:
                        gen_for_viz[s] = generated

                val_batches += 1
                val_images_seen += condition.shape[0]

                if i == 0 and global_rank == 0:
                    cz = condition.shape[-1] // 2
                    panels = [("Undersampled (input)", condition[0, 0, :, :, cz])]
                    panels += [(f"cfg={s:g}", gen_for_viz[s][0, 0, :, :, cz])
                               for s in scales]
                    panels += [("Ground truth", target[0, 0, :, :, cz])]
                    fig, axes = plt.subplots(1, len(panels),
                                             figsize=(5 * len(panels), 5))
                    for ax, (title, img) in zip(axes, panels):
                        ax.imshow(img.cpu().numpy(), cmap="gray")
                        ax.set_title(title)
                        ax.axis("off")
                    plt.tight_layout()
                    plt.savefig(f"visualization_results/flow3d_epoch_{epoch}.png")
                    plt.close()

                if val_images_seen >= max_val_images_per_rank:
                    break

            if args.distributed:
                flat = []
                for s in scales:
                    flat += [sum_ssim[s], sum_psnr[s], sum_lpips[s]]
                flat.append(val_batches)
                metrics_tensor = torch.tensor(flat, device=device)
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
                for j, s in enumerate(scales):
                    sum_ssim[s] = metrics_tensor[3 * j].item()
                    sum_psnr[s] = metrics_tensor[3 * j + 1].item()
                    sum_lpips[s] = metrics_tensor[3 * j + 2].item()
                val_batches = metrics_tensor[-1].item()

            val_batches = max(val_batches, 1)
            avg_ssim = {s: sum_ssim[s] / val_batches for s in scales}
            avg_psnr = {s: sum_psnr[s] / val_batches for s in scales}
            avg_lpips = {s: sum_lpips[s] / val_batches for s in scales}

            if global_rank == 0:
                for s in scales:
                    print(f"Validation [cfg={s:g}] - SSIM: {avg_ssim[s]:.4f}, "
                          f"PSNR: {avg_psnr[s]:.4f}, LPIPS: {avg_lpips[s]:.4f}")

            ckpt_s = args.ckpt_guidance_scale
            if global_rank == 0 and args.save_model:
                ssim_psnr = 0.7 * avg_ssim[ckpt_s] + 0.3 * avg_psnr[ckpt_s]
                if ssim_psnr > best_ssim_psnr:
                    best_ssim_psnr = ssim_psnr
                    contrast_tag = "_".join(args.contrast)
                    checkpoint_path = f"./checkpoints/flow_matching_3d_{contrast_tag}_best_epoch_{epoch}.pt"
                    model_sd = model.module.state_dict() if args.distributed else model.state_dict()
                    ctx_sd = context_encoder.module.state_dict() if args.distributed else context_encoder.state_dict()
                    torch.save({"model": model_sd, "context_encoder": ctx_sd}, checkpoint_path)
                    print(f"Saved best model (cfg={ckpt_s:g}) with Score: {ssim_psnr:.4f}")

            if args.log and global_rank == 0:
                log_dict = {
                    "epoch": epoch,
                    "train_loss": epoch_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                for s in scales:
                    tag = f"cfg{s:g}"
                    log_dict[f"val_ssim_{tag}"] = avg_ssim[s]
                    log_dict[f"val_psnr_{tag}"] = avg_psnr[s]
                    log_dict[f"val_lpips_{tag}"] = avg_lpips[s]
                wandb.log(log_dict)

            if use_ema:
                ema.restore()
                ema_ctx.restore()

    if global_rank == 0 and args.save_model:
        contrast_tag = "_".join(args.contrast)
        final_path = f"./checkpoints/flow_matching_3d_{contrast_tag}_final.pt"
        model_sd = model.module.state_dict() if args.distributed else model.state_dict()
        ctx_sd = context_encoder.module.state_dict() if args.distributed else context_encoder.state_dict()
        torch.save({"model": model_sd, "context_encoder": ctx_sd}, final_path)

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
