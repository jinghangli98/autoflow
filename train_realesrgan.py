"""2D Real-ESRGAN supervised+adversarial restoration baseline.

A baseline alongside the flow-matching model (train_flow.py), the SuperFormer
baseline (train_superformer.py), the 2D SwinIR baseline (train_swinir.py), and
the 3D SPSR baseline (train_SPSR.py): the real Real-ESRGAN generator/
discriminator (basicsr's RRDBNet + UNetDiscriminatorSN) trained to map a
degraded input slice to a clean target slice.

Training is a single run with an L1-only warmup: for the first
--gan_start_epoch epochs, only the pixel (L1) loss trains the generator and
the discriminator is never touched; from --gan_start_epoch on, perceptual
(VGG19) and GAN losses join in and the discriminator starts training. This
mirrors the real Real-ESRGAN recipe (RealESRNet pretrain -> RealESRGAN
finetune) as a single script/checkpoint lineage instead of two separate runs.

Uses the same 2D dataset as train_swinir.py (dataset_2d.py), filtered to
--target_type. No text/prompt conditioning. See realesrgan_train_step.py for
the per-step model core.

Usage:
    python -m torch.distributed.run --nproc_per_node=4 train_realesrgan.py \
        --contrast brain --target_type raw \
        --data_root /vast/tibrahim/jil202/data \
        --distributed --fp16 --save_model \
        --batch_size 8 --size 192 --max_epochs 100 --sample 100
"""

import argparse
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import wandb
import metrics_breakdown as mb
from metrics import evaluate_image_quality
from utils import EMA
from realesrgan_train_step import build_realesrgan_models, make_optimizers, optimize_parameters

PSNR_SCORE_NORM = 40.0


def ckpt_name(args, suffix):
    """Checkpoint filename: contrasts + artifacts (+ --run_tag) + suffix.
    The tag keeps retrains from overwriting earlier checkpoints (see
    train_flow.sh's RUN_TAG); empty tag = the legacy names."""
    tag = "_".join(args.contrast)
    art = "_".join(sorted(args.artifact)) if args.artifact else "all"
    run = f"_{args.run_tag}" if args.run_tag else ""
    return f"realesrgan_2d_{args.target_type}_{tag}_{art}{run}_{suffix}"


def selection_score(ssim, psnr, lpips):
    """Balanced higher-is-better checkpoint score in ~[0, 1] (same recipe as
    train_flow.py): SSIM, PSNR/40dB (capped at 1), and 1-LPIPS each count a third."""
    psnr_term = 1.0 if not np.isfinite(psnr) else min(psnr / PSNR_SCORE_NORM, 1.0)
    return (ssim + psnr_term + (1.0 - lpips)) / 3.0




def parse_args():
    p = argparse.ArgumentParser(description="2D Real-ESRGAN restoration baseline")

    # Data (shared with SwinIR)
    p.add_argument("--data_root", type=str, default="/vast/tibrahim/jil202/nii")
    p.add_argument("--contrast", type=str, required=True, nargs="+",
                   choices=["brain", "knee", "prostate"])
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
                   choices=["raw", "denoised", "md"])
    p.add_argument("--artifact", type=str, nargs="+", default=None,
                   choices=["undersampled", "spike", "aniso"])
    p.add_argument("--sample", type=float, default=100.0)
    p.add_argument("--size", type=int, default=96,
                   help="In-plane crop size (voxels), matching the flow "
                        "model's 96x96 training crops. Must be a multiple of 4 "
                        "(RRDBNet's internal scale-1 pixel-unshuffle requirement).")
    p.add_argument("--samples_per_contrast", type=int, default=None)
    p.add_argument("--balance_by", type=str, default="anatomy",
                   choices=["anatomy", "anatomy_artifact"])
    p.add_argument("--val_images_per_group", type=int, default=mb.MIN_PER_GROUP)

    # Generator / discriminator
    p.add_argument("--nf", type=int, default=64, help="Generator base feature width.")
    p.add_argument("--nb", type=int, default=23, help="RRDB block count (paper default).")
    p.add_argument("--gc", type=int, default=32, help="RRDB growth channel.")
    p.add_argument("--nf_d", type=int, default=64, help="Discriminator base feature width.")

    # Losses / GAN schedule
    p.add_argument("--gan_type", type=str, default="vanilla", choices=["vanilla", "lsgan"])
    p.add_argument("--pixel_weight", type=float, default=1.0)
    p.add_argument("--feature_weight", type=float, default=1.0,
                   help="VGG19 perceptual loss weight; 0 disables it (and skips building VGG).")
    p.add_argument("--gan_weight", type=float, default=0.1)
    p.add_argument("--gan_start_epoch", type=int, default=20,
                   help="Epoch at which perceptual+GAN losses join L1 and the "
                        "discriminator starts training. Before this, only L1 "
                        "trains the generator and the discriminator is untouched.")

    # Optim
    p.add_argument("--lr_G", type=float, default=1e-4)
    p.add_argument("--lr_D", type=float, default=1e-4)
    p.add_argument("--scheduler_milestones", type=int, nargs="+",
                   default=[50000, 100000, 200000, 300000])
    p.add_argument("--scheduler_gamma", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=8321)
    p.add_argument("--val_interval", type=int, default=1)
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--fp16", action="store_true", help="bf16 mixed precision.")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints_realesrgan")
    p.add_argument("--checkpoint_path", type=str, default=None,
                   help="Optional path to a previous train_realesrgan.py checkpoint "
                        "(.pt) to warm-start the generator weights from. Only "
                        "shape-matching tensors are loaded. Default: train from scratch.")
    p.add_argument("--wandb", action="store_true")

    # EMA on the generator (default off)
    p.add_argument("--ema_decay", type=float, default=0.0)
    p.add_argument("--ema_start_epoch", type=int, default=1)

    # Distributed
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--local_rank", type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--dist_backend", type=str, default="nccl")

    args = p.parse_args()
    if args.target_type == "md":
        args.target_type = "denoised"
    if args.size % 4 != 0:
        raise ValueError(f"--size must be a multiple of 4; got {args.size}")
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


def load_pretrained_generator(netG, checkpoint_path, device, rank=0):
    """Warm-start netG from a previous train_realesrgan.py checkpoint (its own
    {"G": state_dict, ...} format). Only shape-matching tensors are loaded."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt["G"] if isinstance(ckpt, dict) and "G" in ckpt else ckpt
    sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}

    model_sd = netG.state_dict()
    matched = {k: v for k, v in sd.items()
               if k in model_sd and model_sd[k].shape == v.shape}
    skipped = sorted(set(sd) - set(matched))
    missing = sorted(set(model_sd) - set(matched))
    netG.load_state_dict(matched, strict=False)

    if rank == 0:
        print(f"  Warm-started generator from {checkpoint_path}")
        print(f"    loaded {len(matched)}/{len(model_sd)} tensors; "
              f"skipped {len(skipped)} from ckpt; {len(missing)} left at init.")
        if isinstance(ckpt, dict) and ckpt.get("epoch") is not None:
            print(f"    (source checkpoint epoch {ckpt['epoch']})")
    return netG


def validate(model, val_loader, device, args, epoch, global_rank):
    """Score central-slice SSIM/PSNR/LPIPS with a per-(anatomy, artifact)
    breakdown, exiting early once every group has reached --val_images_per_group
    images (same group-target machinery as train_swinir.py)."""
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
            plt.savefig(f"visualization_results/realesrgan2d_epoch_{epoch}.png")
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


def save_checkpoint(path, nets, args_dict, epoch, distributed):
    def sd(net):
        m = net.module if distributed and hasattr(net, "module") else net
        return {k.replace("_orig_mod.", ""): v for k, v in m.state_dict().items()}
    torch.save({"G": sd(nets["netG"]), "D": sd(nets["netD"]),
                "args": args_dict, "epoch": epoch}, path)


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
        wandb.init(project="RealESRGAN_2D", config=vars(args),
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

    if global_rank == 0:
        print(f"Building Real-ESRGAN(2D) nf={args.nf}, nb={args.nb}, gc={args.gc}, "
              f"nf_d={args.nf_d}, gan_start_epoch={args.gan_start_epoch}")

    nets = build_realesrgan_models(
        nf=args.nf, nb=args.nb, gc=args.gc, nf_d=args.nf_d, gan_type=args.gan_type,
        feature_weight=args.feature_weight, device=device)

    if args.checkpoint_path:
        load_pretrained_generator(nets["netG"], args.checkpoint_path, device, rank=global_rank)

    if args.compile:
        nets["netG"] = torch.compile(nets["netG"])
    if args.distributed:
        if dist.is_initialized():
            dist.barrier()
        nets["netG"] = DDP(nets["netG"], device_ids=[local_rank], output_device=local_rank,
                           find_unused_parameters=False)
        nets["netD"] = DDP(nets["netD"], device_ids=[local_rank], output_device=local_rank,
                           find_unused_parameters=False)

    opts = make_optimizers(nets, lr_G=args.lr_G, lr_D=args.lr_D)
    schedulers = [torch.optim.lr_scheduler.MultiStepLR(
        o, milestones=args.scheduler_milestones, gamma=args.scheduler_gamma) for o in opts]
    weights = dict(pixel=args.pixel_weight, feature=args.feature_weight, gan=args.gan_weight)

    use_ema = args.ema_decay > 0
    ema = EMA(nets["netG"], args.ema_decay) if use_ema else None
    if use_ema:
        ema.register()
    best_score = 0.0
    global_step = 0

    if global_rank == 0:
        print(f"Starting Real-ESRGAN training (world_size={args.world_size})")

    for epoch in range(args.max_epochs):
        for loader in (train_loader, val_loader):
            sampler = getattr(loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
        if use_ema and epoch == args.ema_start_epoch:
            ema.register()

        nets["netG"].train()
        nets["netD"].train()
        epoch_g = 0.0
        n_g = 0
        bar = (tqdm(enumerate(train_loader), total=len(train_loader), ncols=100)
               if global_rank == 0 else enumerate(train_loader))

        for step, (condition, target, *_rest) in bar:
            condition = condition.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            global_step += 1
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.fp16):
                log = optimize_parameters(condition, target, nets, opts, weights,
                                          epoch=epoch, gan_start_epoch=args.gan_start_epoch)
            for s in schedulers:
                s.step()
            if use_ema and epoch >= args.ema_start_epoch:
                ema.update()
            if "l_g_total" in log:
                epoch_g += log["l_g_total"]; n_g += 1
            if global_rank == 0 and isinstance(bar, tqdm):
                bar.set_postfix({"g": epoch_g / max(1, n_g),
                                 "d": log.get("l_d", 0.0), "step": global_step})

        epoch_g = epoch_g / max(1, n_g)
        if global_rank == 0:
            print(f"Epoch {epoch+1}/{args.max_epochs}, G loss: {epoch_g:.4f}, "
                  f"gan_active={epoch >= args.gan_start_epoch}")

        if (epoch + 1) % args.val_interval == 0:
            val_with_ema = use_ema and epoch >= args.ema_start_epoch
            if val_with_ema:
                ema.apply_shadow()
            avg, bd = validate(nets["netG"], val_loader, device, args, epoch, global_rank)
            if global_rank == 0:
                print(f"Validation - SSIM: {avg['SSIM']:.4f}, PSNR: {avg['PSNR']:.4f}, "
                      f"LPIPS: {avg['LPIPS']:.4f}, score: {avg['score']:.4f}")
                for g in sorted(bd):
                    s_, ps, lp, nn_ = bd[g]
                    if nn_ > 0:
                        print(f"  breakdown [{g[0]}/{g[1]}] - SSIM: {s_/nn_:.4f}, "
                              f"PSNR: {ps/nn_:.4f}, LPIPS: {lp/nn_:.4f}, n: {int(nn_)}")
                if args.save_model and avg["score"] > best_score:
                    best_score = avg["score"]
                    path = os.path.join(args.checkpoint_dir,
                                        ckpt_name(args, "best.pt"))
                    save_checkpoint(path, nets, vars(args), epoch, args.distributed)
                    print(f"Saved best model score {best_score:.4f}: {path}")
                if args.wandb:
                    log_dict = {"epoch": epoch, "g_loss": epoch_g,
                                "lr": opts[0].param_groups[0]["lr"],
                                "gan_active": float(epoch >= args.gan_start_epoch),
                                "val_ssim": avg["SSIM"], "val_psnr": avg["PSNR"],
                                "val_lpips": avg["LPIPS"], "val_score": avg["score"]}
                    groups = sorted(bd)
                    log_dict.update(mb.breakdown_log_dict_noscale(
                        {g: bd[g][0] for g in groups}, {g: bd[g][1] for g in groups},
                        {g: bd[g][2] for g in groups}, {g: bd[g][3] for g in groups}, groups))
                    wandb.log(log_dict)
            if val_with_ema:
                ema.restore()

    if global_rank == 0 and args.save_model:
        path = os.path.join(args.checkpoint_dir, ckpt_name(args, "final.pt"))
        save_checkpoint(path, nets, vars(args), args.max_epochs - 1, args.distributed)
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
