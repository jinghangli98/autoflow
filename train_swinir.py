"""2D SwinIR supervised-restoration training.

A supervised baseline alongside the flow-matching model (train_flow.py) and the
SuperFormer baseline (train_superformer.py): the stock 2D SwinIR (see
SwinIR/models/network_swinir.py) trained to map a degraded input slice
directly to a clean target slice with an L1 loss. There is NO flow matching, NO
noise, NO text/prompt conditioning, and NO classifier-free guidance -- the model
takes a single (B, 1, H, W) tensor and returns the restored (B, 1, H, W).

Ground truth is chosen with `--target_type`:
    raw       -> fully-sampled, unprocessed target (the artifact->raw pairs)
    denoised  -> denoised + bias-corrected target  (artifact->denoised and
                 raw->denoised pairs)
dataset_2d.py supplies the (condition, target) 2D slices;
this script just filters the samples to the chosen target type (dataset_2d.py is
left untouched).

The stock 2D SwinIR trains from scratch by default; pass `--checkpoint_path` to
warm-start the model weights from a previous checkpoint (see `load_pretrained`).

Usage:
    python -m torch.distributed.run --nproc_per_node=4 train_swinir.py \
        --contrast brain --target_type raw \
        --data_root /vast/tibrahim/jil202/data \
        --distributed --fp16 --save_model \
        --batch_size 4 --size 192 --max_epochs 100 --sample 100
"""

import argparse
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import wandb
import metrics_breakdown as mb
from metrics import evaluate_image_quality
from utils import EMA

sys.path.insert(0, "/vast/tibrahim/jil202/autoflow/SwinIR")
from models.network_swinir import SwinIR

PSNR_SCORE_NORM = 40.0  # dB mapping to a full PSNR score contribution of 1.0


def ckpt_name(args, suffix):
    """Checkpoint filename: contrasts + artifacts (+ --run_tag) + suffix.
    The tag keeps retrains from overwriting earlier checkpoints (see
    train_flow.sh's RUN_TAG); empty tag = the legacy names."""
    tag = "_".join(args.contrast)
    art = "_".join(sorted(args.artifact)) if args.artifact else "all"
    run = f"_{args.run_tag}" if args.run_tag else ""
    return f"swinir_2d_{args.target_type}_{tag}_{art}{run}_{suffix}"


def selection_score(ssim, psnr, lpips):
    """Balanced higher-is-better checkpoint score in ~[0, 1] (same recipe as
    train_flow.py): SSIM, PSNR/40dB (capped at 1), and 1-LPIPS each count a third."""
    psnr_term = 1.0 if not np.isfinite(psnr) else min(psnr / PSNR_SCORE_NORM, 1.0)
    return (ssim + psnr_term + (1.0 - lpips)) / 3.0


def build_model(img_size, args, device, rank=0):
    """Construct the stock 2D SwinIR at the training slice size (random init)."""
    model = SwinIR(
        img_size=img_size, patch_size=1, in_chans=1, embed_dim=args.embed_dim,
        depths=tuple(args.depths), num_heads=tuple(args.num_heads),
        window_size=args.window_size, mlp_ratio=args.mlp_ratio,
        upscale=1, img_range=1.0, upsampler="", resi_connection=args.resi_connection)
    if rank == 0:
        n = sum(p.numel() for p in model.parameters())
        print(f"  SwinIR(2D): {n / 1e6:.2f}M params, embed_dim={args.embed_dim}, "
              f"depths={tuple(args.depths)}, heads={tuple(args.num_heads)}, "
              f"window={args.window_size}")
    return model.to(device)


def load_pretrained(model, checkpoint_path, device, rank=0):
    """Warm-start the bare SwinIR weights from a checkpoint (before DDP/compile).

    Accepts either this script's save format (`{"model": state_dict, ...}`) or a
    raw state_dict. Strips any `module.` / `_orig_mod.` prefixes, then loads only
    the keys whose shapes match the current model: the non-learned `attn_mask`
    and `relative_position_index` buffers depend on `--size` / `--window_size`
    and are skipped when they differ (they are recomputed at construction). Call
    this on the raw model, before wrapping it in `torch.compile` / DDP, so the
    key names line up with `model.state_dict()`.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}

    model_sd = model.state_dict()
    matched = {k: v for k, v in sd.items()
               if k in model_sd and model_sd[k].shape == v.shape}
    skipped = sorted(set(sd) - set(matched))
    missing = sorted(set(model_sd) - set(matched))
    model.load_state_dict(matched, strict=False)

    if rank == 0:
        print(f"  Warm-started from {checkpoint_path}")
        print(f"    loaded {len(matched)}/{len(model_sd)} tensors; "
              f"skipped {len(skipped)} from ckpt (shape/name mismatch); "
              f"{len(missing)} left at init.")
        if isinstance(ckpt, dict) and ckpt.get("epoch") is not None:
            print(f"    (source checkpoint epoch {ckpt['epoch']})")
        if skipped:
            print(f"    skipped keys (first few): {skipped[:4]}")
    return model


def parse_args():
    p = argparse.ArgumentParser(description="2D SwinIR supervised training")

    # Data
    p.add_argument("--data_root", type=str, default="/vast/tibrahim/jil202/nii")
    p.add_argument("--contrast", type=str, required=True, nargs="+",
                   choices=["brain", "knee", "prostate"],
                   help="Anatomy group(s) to train on.")
    p.add_argument("--slices_per_volume", type=int, default=8,
                   help="Random 2D crops drawn per volume pair per visit; a "
                        "step holds batch_size * this many slices (default 8).")
    p.add_argument("--fg_fraction", type=float, default=0.25,
                   help="Minimum fraction of foreground target voxels for a "
                        "crop to be accepted without a re-draw (default 0.25, "
                        "same rule as train_flow). 0.0 accepts every crop.")
    p.add_argument("--val_fraction", type=float, default=0.10,
                   help="Patient-level fraction of --data_root held out for "
                        "validation (default 0.10), same split as train_flow.")
    p.add_argument("--run_tag", type=str, default="",
                   help="Inserted into checkpoint filenames (and used as the "
                        "W&B run name) so runs never overwrite each other; "
                        "empty (default) keeps the legacy names.")
    p.add_argument("--target_type", type=str, required=True,
                   choices=["raw", "denoised", "md"],
                   help="Ground truth: 'raw' = fully-sampled, 'denoised'/'md' = "
                        "denoised + bias-corrected.")
    p.add_argument("--artifact", type=str, nargs="+", default=None,
                   choices=["undersampled", "spike", "aniso"],
                   help="Optional artifact families to restrict to (default: all).")
    p.add_argument("--sample", type=float, default=100.0,
                   help="Percentage of sample pairs to use.")
    p.add_argument("--size", type=int, default=96,
                   help="In-plane crop size (voxels), matching the flow "
                        "model's 96x96 training crops. Must be a multiple of "
                        "--window_size, and every volume must fit it in-plane "
                        "along at least one allowed axis.")
    p.add_argument("--samples_per_contrast", type=int, default=None,
                   help="If set, use the balanced sampler: each balancing group "
                        "contributes this many training samples per epoch "
                        "(0 = auto-balance to the smallest group). None (default) "
                        "= plain DistributedSampler over all (filtered) samples.")
    p.add_argument("--balance_by", type=str, default="anatomy",
                   choices=["anatomy", "anatomy_artifact"],
                   help="Balancing granularity for --samples_per_contrast: "
                        "'anatomy' = equal samples per anatomy; 'anatomy_artifact' "
                        "= equal samples per (anatomy, artifact) group.")
    p.add_argument("--val_images_per_group", type=int, default=mb.MIN_PER_GROUP,
                   help="Max validation images scored per (anatomy, artifact) "
                        "group before the val loop exits early. Capped at each "
                        "group's available count.")

    # Model config -- Classic SwinIR-M restoration defaults (2D).
    p.add_argument("--embed_dim", type=int, default=96)
    p.add_argument("--depths", type=int, nargs="+", default=[6, 6, 6, 6],
                   help="Depth (number of Swin blocks) per RSTB stage.")
    p.add_argument("--num_heads", type=int, nargs="+", default=[6, 6, 6, 6],
                   help="Attention heads per RSTB stage (len must match --depths).")
    p.add_argument("--window_size", type=int, default=8,
                   help="Swin attention window (square). --size must be a "
                        "multiple of this.")
    p.add_argument("--mlp_ratio", type=float, default=2.0)
    p.add_argument("--resi_connection", type=str, default="1conv",
                   choices=["1conv", "3conv"])

    # Training -- mirrors the SuperFormer baseline: L1, Adam(wd=0), MultiStepLR
    # stepped once per optimizer step (milestones in iterations).
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=8321)
    p.add_argument("--loss", type=str, default="l1",
                   choices=["l1", "l2", "l2sum"],
                   help="Pixel loss. l1=L1Loss, l2=MSELoss, l2sum=MSELoss(sum).")
    p.add_argument("--lossfn_weight", type=float, default=1.0)
    p.add_argument("--scheduler_milestones", type=int, nargs="+",
                   default=[250000, 400000, 450000, 475000, 500000],
                   help="MultiStepLR milestones in ITERATIONS (stepped once per "
                        "optimizer step).")
    p.add_argument("--scheduler_gamma", type=float, default=0.5)
    p.add_argument("--clip_grad", type=float, default=0.0,
                   help="Grad-norm clip (0 = off).")
    p.add_argument("--val_interval", type=int, default=1)
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--fp16", action="store_true", help="bf16 mixed precision.")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints_swinir")
    p.add_argument("--checkpoint_path", type=str, default=None,
                   help="Optional path to a checkpoint (.pt) to warm-start the "
                        "model weights from before training. Only shape-matching "
                        "tensors are loaded; non-learned buffers that depend on "
                        "--size/--window_size (attn_mask, relative_position_index) "
                        "are skipped and recomputed. Default: train from scratch.")
    p.add_argument("--wandb", action="store_true",
                   help="Enable wandb logging of train loss + val SSIM/PSNR/LPIPS.")

    # EMA -- off by default (matches the SuperFormer baseline).
    p.add_argument("--ema_decay", type=float, default=0.0,
                   help="EMA decay. 0 = no EMA.")
    p.add_argument("--ema_start_epoch", type=int, default=1)

    # Distributed
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--local_rank", type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--dist_backend", type=str, default="nccl")

    args = p.parse_args()
    if args.target_type == "md":
        args.target_type = "denoised"
    if len(args.depths) != len(args.num_heads):
        raise ValueError(
            f"--depths ({args.depths}) and --num_heads ({args.num_heads}) must "
            f"have the same length")
    if args.size % args.window_size != 0:
        raise ValueError(
            f"--size ({args.size}) must be a multiple of --window_size "
            f"({args.window_size})")
    return args


def setup_distributed(rank, world_size, local_rank, backend="nccl"):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=backend, init_method="env://", world_size=world_size, rank=rank,
        device_id=torch.device(f"cuda:{local_rank}"))
    print(f"Process group initialized: rank {rank}/{world_size}")


def make_loaders(args, world_size, global_rank):
    from dataset_2d import (BalancedDistributedSampler, collate_patches,
                            filter_target_type, get_dataset_2d_slices)

    train_set, val_set = get_dataset_2d_slices(
        data_root=args.data_root, contrast=args.contrast, sample=args.sample,
        size=args.size, artifacts=args.artifact,
        slices_per_volume=args.slices_per_volume,
        val_fraction=args.val_fraction, fg_fraction=args.fg_fraction)
    filter_target_type(train_set, args.target_type, rank=global_rank)
    filter_target_type(val_set, args.target_type, rank=global_rank)

    if args.samples_per_contrast is not None:
        balance_indices = (train_set.group_indices
                           if args.balance_by == "anatomy_artifact"
                           else train_set.contrast_indices)
        train_sampler = BalancedDistributedSampler(
            balance_indices,
            samples_per_contrast=args.samples_per_contrast,
            num_replicas=world_size if args.distributed else 1,
            rank=global_rank if args.distributed else 0,
            shuffle=True, seed=args.seed)
    elif args.distributed:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size,
                                           rank=global_rank, shuffle=True, seed=args.seed)
    else:
        train_sampler = None

    val_sampler = (DistributedSampler(val_set, num_replicas=world_size,
                                      rank=global_rank, shuffle=False, seed=args.seed)
                   if args.distributed else None)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=args.num_workers,
        pin_memory=True, drop_last=True, collate_fn=collate_patches)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, sampler=val_sampler, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_patches)
    return train_loader, val_loader


def validate(model, val_loader, device, args, epoch, global_rank):
    """Score central-slice SSIM/PSNR/LPIPS with a per-(anatomy, artifact)
    breakdown, exiting early once every group has reached --val_images_per_group
    images. A 3-panel viz is dumped on the first batch."""
    model.eval()
    sum_ssim = sum_psnr = sum_lpips = sum_n = 0.0
    bd = {}  # (anatomy, artifact) -> [ssim, psnr, lpips, n]

    # Samples are volume pairs; each visit yields --slices_per_volume images.
    val_group_total = Counter()
    for s in val_loader.dataset.samples:
        val_group_total[(s["anatomy"], s["artifact"])] += args.slices_per_volume
    present_groups = sorted(val_group_total)
    group_target = mb.group_targets(dict(val_group_total),
                                    min_per_group=args.val_images_per_group)
    group_seen = {g: 0 for g in present_groups}

    for i, (condition, target, _prompts, _tt, anatomies, artifacts) in enumerate(val_loader):
        condition = condition.to(device)
        target = target.to(device)
        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16,
                                       enabled=args.fp16):
            out = model(condition).float()

        for j in range(condition.shape[0]):
            gen_np = out[j, 0].cpu().numpy()[None]
            targ_np = target[j, 0].cpu().numpy()[None]
            m = evaluate_image_quality(gen_np, targ_np)
            pv = m["PSNR"] if np.isfinite(m["PSNR"]) else 100.0
            sum_ssim += m["SSIM"]; sum_psnr += pv; sum_lpips += m["LPIPS"]; sum_n += 1
            g = (anatomies[j], artifacts[j])
            acc = bd.setdefault(g, [0.0, 0.0, 0.0, 0.0])
            acc[0] += m["SSIM"]; acc[1] += pv; acc[2] += m["LPIPS"]; acc[3] += 1
            if g in group_seen:
                group_seen[g] += 1

        if i == 0 and global_rank == 0:
            os.makedirs("visualization_results", exist_ok=True)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, (title, img) in zip(axes, [
                ("Input", condition[0, 0]),
                ("Restored", out[0, 0]),
                ("Target", target[0, 0])]):
                ax.imshow(img.cpu().numpy(), cmap="gray"); ax.set_title(title); ax.axis("off")
            plt.tight_layout()
            plt.savefig(f"visualization_results/swinir2d_epoch_{epoch}.png")
            plt.close()

        counts = torch.tensor([float(group_seen[g]) for g in present_groups],
                              dtype=torch.float64, device=device)
        if args.distributed:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        global_counts = {g: counts[k].item() for k, g in enumerate(present_groups)}
        if mb.groups_done(global_counts, group_target):
            break

    if args.distributed:
        t = torch.tensor([sum_ssim, sum_psnr, sum_lpips, sum_n], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        sum_ssim, sum_psnr, sum_lpips, sum_n = t.tolist()
        flat = torch.tensor(
            [bd.get(g, [0.0, 0.0, 0.0, 0.0])[m]
             for g in present_groups for m in range(4)],
            dtype=torch.float64, device=device)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        for gi, g in enumerate(present_groups):
            bd[g] = flat[gi * 4: gi * 4 + 4].tolist()

    n = max(sum_n, 1.0)
    avg = dict(SSIM=sum_ssim / n, PSNR=sum_psnr / n, LPIPS=sum_lpips / n)
    avg["score"] = selection_score(avg["SSIM"], avg["PSNR"], avg["LPIPS"])
    return avg, bd


def train(local_rank, args):
    torch.set_float32_matmul_precision("high")
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", local_rank))
        global_rank = int(os.environ.get("RANK", local_rank))
        world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
        args.world_size = world_size
        setup_distributed(global_rank, world_size, local_rank, args.dist_backend)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        global_rank = local_rank = 0
        args.world_size = world_size = 1

    if args.wandb and global_rank == 0:
        wandb.init(project="SwinIR_2D", config=vars(args),
                   name=args.run_tag or None)
    if global_rank == 0:
        os.makedirs("visualization_results", exist_ok=True)
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_loader, val_loader = make_loaders(args, world_size, global_rank)
    if global_rank == 0:
        print(f"Data loaders: train={len(train_loader)}, val={len(val_loader)}")
        groups = Counter((s["anatomy"], s["artifact"]) for s in train_loader.dataset.samples)
        for g in sorted(groups):
            print(f"  train group {g[0]}/{g[1]}: {groups[g]} pairs")

    img_size = args.size
    if global_rank == 0:
        print(f"Building SwinIR(2D) at img_size={img_size}")
    model = build_model(img_size, args, device, rank=global_rank)
    if args.checkpoint_path:
        load_pretrained(model, args.checkpoint_path, device, rank=global_rank)

    if args.compile:
        model = torch.compile(model)
    if args.distributed:
        if dist.is_initialized():
            dist.barrier()
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    loss_map = {"l1": nn.L1Loss(), "l2": nn.MSELoss(),
                "l2sum": nn.MSELoss(reduction="sum")}
    loss_fn = loss_map[args.loss]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.scheduler_milestones, gamma=args.scheduler_gamma)

    use_ema = args.ema_decay > 0
    ema = EMA(model, args.ema_decay) if use_ema else None
    if use_ema:
        ema.register()
    best_score = 0.0
    global_step = 0

    if global_rank == 0:
        print(f"Starting SwinIR training (world_size={args.world_size})")

    for epoch in range(args.max_epochs):
        for loader in (train_loader, val_loader):
            sampler = getattr(loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
        if use_ema and epoch == args.ema_start_epoch:
            ema.register()

        model.train()
        epoch_loss = 0.0
        if global_rank == 0:
            bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100)
        else:
            bar = enumerate(train_loader)

        for step, (condition, target, *_rest) in bar:
            condition = condition.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.fp16):
                out = model(condition)
                loss = args.lossfn_weight * loss_fn(out.float(), target.float())

            if torch.isnan(loss) or torch.isinf(loss):
                if global_rank == 0:
                    print(f"NaN/Inf loss at epoch {epoch}, step {step}. Skipping.")
                continue

            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=args.clip_grad)
            optimizer.step()
            scheduler.step()
            global_step += 1
            if use_ema and epoch >= args.ema_start_epoch:
                ema.update()

            epoch_loss += loss.item()
            if global_rank == 0 and isinstance(bar, tqdm):
                bar.set_postfix({"loss": epoch_loss / (step + 1),
                                 "lr": optimizer.param_groups[0]["lr"],
                                 "step": global_step})

        if args.distributed:
            t = torch.tensor(epoch_loss, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            epoch_loss = t.item() / args.world_size
        epoch_loss /= max(1, len(train_loader))
        if global_rank == 0:
            print(f"Epoch {epoch+1}/{args.max_epochs}, train loss: {epoch_loss:.4f}")

        if (epoch + 1) % args.val_interval == 0:
            val_with_ema = use_ema and epoch >= args.ema_start_epoch
            if val_with_ema:
                ema.apply_shadow()

            avg, bd = validate(model, val_loader, device, args, epoch, global_rank)

            if global_rank == 0:
                print(f"Validation - SSIM: {avg['SSIM']:.4f}, PSNR: {avg['PSNR']:.4f}, "
                      f"LPIPS: {avg['LPIPS']:.4f}, score: {avg['score']:.4f}")
                for g in sorted(bd):
                    s, ps, lp, nn_ = bd[g]
                    if nn_ > 0:
                        print(f"  breakdown [{g[0]}/{g[1]}] - SSIM: {s/nn_:.4f}, "
                              f"PSNR: {ps/nn_:.4f}, LPIPS: {lp/nn_:.4f}, n: {int(nn_)}")

                if args.save_model and avg["score"] > best_score:
                    best_score = avg["score"]
                    path = os.path.join(args.checkpoint_dir,
                                        ckpt_name(args, "best_md.pt"))
                    sd = model.module.state_dict() if args.distributed else model.state_dict()
                    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
                    torch.save({"model": sd, "img_size": img_size,
                                "args": vars(args), "epoch": epoch}, path)
                    print(f"Saved best model score {best_score:.4f}: {path}")

                if args.wandb:
                    log_dict = {"epoch": epoch, "train_loss": epoch_loss,
                                "lr": optimizer.param_groups[0]["lr"],
                                "val_ssim": avg["SSIM"], "val_psnr": avg["PSNR"],
                                "val_lpips": avg["LPIPS"], "val_score": avg["score"]}
                    groups = sorted(bd)
                    log_dict.update(mb.breakdown_log_dict_noscale(
                        {g: bd[g][0] for g in groups},
                        {g: bd[g][1] for g in groups},
                        {g: bd[g][2] for g in groups},
                        {g: bd[g][3] for g in groups},
                        groups))
                    wandb.log(log_dict)

            if val_with_ema:
                ema.restore()

    if global_rank == 0 and args.save_model:
        path = os.path.join(args.checkpoint_dir, ckpt_name(args, "md_final.pt"))
        sd = model.module.state_dict() if args.distributed else model.state_dict()
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        torch.save({"model": sd, "img_size": img_size, "args": vars(args)}, path)
        print(f"Saved final model: {path}")

    if args.distributed:
        dist.barrier()
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    args = parse_args()
    if args.distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        train(local_rank, args)
    else:
        train(0, args)


if __name__ == "__main__":
    main()
