"""3D NIfTI Flow Matching Training Script.

Conditional flow matching for multi-task MRI restoration. The model takes a
3D patch (`condition` -- an artifact patch or the raw fully-sampled patch) and
predicts the velocity field that maps noise to the target patch (raw, or
denoised + bias-corrected). The three pairings are artifact->raw,
artifact->denoised, and raw->denoised (see `dataset.build_samples`).

Inputs are 5D tensors `(B, 1, X, Y, Z)`. The model concatenates
`(noisy_target, condition)` along the channel dim -> `(B, 2, X, Y, Z)` and
predicts a velocity field of shape `(B, 1, X, Y, Z)`.

The target's text prompt is injected via cross-attention so it tells the model
which output to produce ("Fully sampled ..." vs "Denoised and biascorrected
..."). The dataset reads the ready-made prompt from each target's JSON sidecar
(`prompt` key); a frozen RadBERT (`zzxslp/RadBERT-RoBERTa-4m`, RoBERTa-base,
hidden 768) encodes it into a `(B, seq_len, 768)` context. Classifier-free
guidance uses the encoded empty prompt as the unconditional signal.

Usage:
    python -m torch.distributed.run --nproc_per_node=8 train_flow.py \
        --contrast brain knee prostate \
        --data_root /vast/tibrahim/jil202/data \
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
from transformers import AutoModel, AutoTokenizer

import wandb
from metrics import evaluate_image_quality
from utils import EMA


TEXT_ENCODER_NAME = "zzxslp/RadBERT-RoBERTa-4m"
PROMPT_MAX_LEN = 64        # prompts tokenize to ~50 tokens; padded to this

# Target types the validation scores separately so each prompt capability
# counts equally: "raw" = "Fully sampled ..." outputs, "denoised" = "Denoised
# and biascorrected ..." outputs.
TARGET_TYPES = ("raw", "denoised")
PSNR_SCORE_NORM = 40.0     # dB mapping to a full PSNR score contribution of 1.0


def selection_score(ssim, psnr, lpips):
    """Balanced higher-is-better checkpoint score in ~[0, 1].

    SSIM (higher better, ~[0, 1]), PSNR (higher better, normalized by
    PSNR_SCORE_NORM dB and capped at 1), and LPIPS (lower better, folded in as
    1 - LPIPS) each contribute one third, so none drowns out the others (the
    old 0.7*SSIM + 0.3*PSNR score was dominated by raw PSNR magnitude).
    """
    psnr_term = 1.0 if not np.isfinite(psnr) else min(psnr / PSNR_SCORE_NORM, 1.0)
    return (ssim + psnr_term + (1.0 - lpips)) / 3.0


class TextConditioner(nn.Module):
    """Frozen RadBERT text encoder -> (B, seq_len, hidden) cross-attention context.

    Fully frozen: no trainable params, so it stays out of the optimizer, EMA,
    DDP, and checkpoints. The unconditional signal for classifier-free
    guidance is the encoded *empty prompt* (as in generative_brain_controlnet,
    which swaps in BOS+PAD tokens): during training, dropped samples have
    their token ids replaced with the empty-prompt ids before encoding, and
    `null()` returns the empty-prompt embedding for guided sampling.
    """

    def __init__(self, model_name=TEXT_ENCODER_NAME, max_len=PROMPT_MAX_LEN):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.max_len = max_len
        self.hidden_size = self.encoder.config.hidden_size

        null_tok = self.tokenizer(
            "", padding="max_length", max_length=max_len, return_tensors="pt",
        )
        self.register_buffer("null_ids", null_tok.input_ids, persistent=False)
        self.register_buffer("null_mask", null_tok.attention_mask, persistent=False)

    def train(self, mode=True):
        # Always stay in eval mode so the encoder's dropout never activates.
        return super().train(False)

    @torch.no_grad()
    def encode(self, prompts, device, cfg_dropout_prob=0.0):
        tok = self.tokenizer(
            list(prompts), padding="max_length", max_length=self.max_len,
            truncation=True, return_tensors="pt",
        )
        ids = tok.input_ids.to(device)
        mask = tok.attention_mask.to(device)
        if cfg_dropout_prob > 0:
            drop = torch.rand(ids.shape[0], device=device) < cfg_dropout_prob
            ids = torch.where(drop[:, None], self.null_ids, ids)
            mask = torch.where(drop[:, None], self.null_mask, mask)
        return self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state

    @torch.no_grad()
    def null(self, batch_size: int):
        ids = self.null_ids.expand(batch_size, -1)
        mask = self.null_mask.expand(batch_size, -1)
        return self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state


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
               null_context=None, guidance_scale=1.0, noise=None):
        """Euler ODE integration in 3D, with optional classifier-free guidance.

        `noise` is the initial state x(t=0). Pass a fixed tensor to compare
        guidance scales under identical noise (so the difference reflects
        context, not the random draw); if None a fresh sample is drawn.
        """
        batch_size = condition.shape[0]
        x = torch.randn_like(condition).to(device) if noise is None else noise.to(device)
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
                        default="/vast/tibrahim/jil202/data",
                        help="Root containing train/<anatomy>/ and test/<anatomy>/")
    parser.add_argument("--contrast", type=str, required=True, nargs="+",
                        choices=["brain", "knee", "prostate"],
                        help="One or more anatomy groups to train on, e.g. "
                             "`--contrast brain` or `--contrast brain knee prostate`. "
                             "Each spans all of that anatomy's acquisitions and "
                             "the artifact->raw / artifact->denoised / raw->denoised "
                             "restoration tasks.")

    # Training
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--sample", type=float, default=10.0,
                        help="Percentage of (undersampled,GT) sample pairs to use")
    parser.add_argument("--size", type=int, default=196,
                        help="Patch size (in voxels). Actual patch shape is (size, size, 16) to ")

    parser.add_argument("--samples_per_contrast", type=int, default=None, nargs="+",
                        help="Per-epoch training samples per contrast. Pass one "
                             "value to apply it to every contrast (0 = auto-balance "
                             "to the smallest contrast), or N values matching the N "
                             "--contrast entries (same order) for per-anatomy "
                             "quotas, e.g. `--contrast brain knee prostate "
                             "--samples_per_contrast 20000 50000 50000`. The larger "
                             "contrast cycles through a fresh random subset each "
                             "epoch; an over-quota contrast is oversampled. Ignored "
                             "for the artifact buckets when --cell_samples is set.")
    parser.add_argument("--cell_samples", type=str, default=None, nargs="+",
                        help="Per-epoch training samples per (anatomy x artifact) "
                             "cell, as `anatomy:group=count` entries, e.g. "
                             "`--cell_samples brain:undersample=4000 brain:spike=6000 "
                             "brain:aniso=10000 knee:...`. group is one of "
                             "{undersample (R*/GRAPPA), spike, aniso}. Every "
                             "(contrast x {undersample,spike,aniso}) cell must be "
                             "listed (missing cell = error). When set, the sampler "
                             "buckets by cell and --samples_per_contrast is ignored "
                             "for these artifact buckets. raw->denoised pairs are "
                             "controlled separately by --denoise_samples_per_contrast.")
    parser.add_argument("--denoise_samples_per_contrast", type=int, default=0,
                        help="Per-epoch training samples for each anatomy's "
                             "raw->denoised ('none' artifact) bucket when "
                             "--cell_samples is set. Applied uniformly to every "
                             "--contrast (0 = drop raw->denoised this run). Kept "
                             "outside the --cell_samples artifact weighting.")
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--val_balanced_per_cell", type=int, default=100,
                        help="Per-epoch validation is balanced across "
                             "(anatomy x artifact-group) cells: this many pairs "
                             "per (anatomy, {undersample,spike,aniso}) cell "
                             "(0 = balance to the smallest cell). Checkpoint "
                             "selection averages the per-cell scores, so every "
                             "anatomy and artifact counts equally. Set <0 to "
                             "disable (revert to the legacy 5% pooled val set).")
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
                        help="Probability of replacing a sample's text prompt "
                             "with the empty prompt during training (CFG).")
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

    from dataset import getloader_3d_patches, ARTIFACT_GROUPS

    # Build the sampler quota. With --cell_samples, buckets are keyed by
    # (anatomy, artifact-group) cells; otherwise by anatomy (legacy).
    cell_buckets = args.cell_samples is not None
    if cell_buckets:
        # Parse `anatomy:group=count` entries into {(anatomy, group): count}.
        cell_quota = {}
        for entry in args.cell_samples:
            try:
                cell, count = entry.split("=")
                anat, group = cell.split(":")
            except ValueError:
                raise ValueError(
                    f"--cell_samples entry {entry!r} is malformed; expected "
                    f"`anatomy:group=count` (e.g. brain:aniso=10000)."
                )
            if anat not in args.contrast:
                raise ValueError(
                    f"--cell_samples names anatomy {anat!r} which is not in "
                    f"--contrast {args.contrast}."
                )
            if group not in ARTIFACT_GROUPS:
                raise ValueError(
                    f"--cell_samples group {group!r} must be one of {ARTIFACT_GROUPS}."
                )
            cell_quota[(anat, group)] = int(count)
        # Require every (contrast x artifact-group) cell to be specified.
        required = {(a, g) for a in args.contrast for g in ARTIFACT_GROUPS}
        missing = required - set(cell_quota)
        if missing:
            raise ValueError(
                f"--cell_samples is missing cells {sorted(missing)}; every "
                f"(contrast x {{{', '.join(ARTIFACT_GROUPS)}}}) cell must be listed."
            )
        # raw->denoised ('none') buckets: a fixed per-anatomy count.
        if args.denoise_samples_per_contrast > 0:
            for a in args.contrast:
                cell_quota[(a, "none")] = args.denoise_samples_per_contrast
        samples_per_contrast = cell_quota
    else:
        # --samples_per_contrast arrives as a list (nargs). One value -> scalar
        # (applied to every contrast); N values matching --contrast -> per-anatomy
        # {contrast: count} quota dict; None -> standard sampling.
        samples_per_contrast = args.samples_per_contrast
        if isinstance(samples_per_contrast, list):
            if len(samples_per_contrast) == 1:
                samples_per_contrast = samples_per_contrast[0]
            elif len(samples_per_contrast) == len(args.contrast):
                samples_per_contrast = dict(zip(args.contrast, samples_per_contrast))
            else:
                raise ValueError(
                    f"--samples_per_contrast got {len(samples_per_contrast)} values but "
                    f"there are {len(args.contrast)} --contrast entries; pass either one "
                    f"value or one per contrast (same order)."
                )
    if global_rank == 0:
        print(f"cell_buckets = {cell_buckets}")
        print(f"samples_per_contrast = {samples_per_contrast}")

    # <0 disables balanced validation (legacy 5% pooled val set); else balance
    # the val set across (anatomy x artifact-group) cells, this many per cell.
    val_balanced_per_cell = (None if args.val_balanced_per_cell < 0
                             else args.val_balanced_per_cell)
    if global_rank == 0:
        print(f"val_balanced_per_cell = {val_balanced_per_cell}")

    if args.distributed:
        train_loader, val_loader = getloader_3d_patches(
            batch_size=args.batch_size,
            data_root=args.data_root,
            contrast=args.contrast,
            sample=args.sample,
            distributed=True, rank=global_rank, world_size=world_size,
            num_workers=args.num_workers, patch_shape=(args.size, args.size, 16),
            samples_per_contrast=samples_per_contrast,
            val_balanced_per_cell=val_balanced_per_cell,
            cell_buckets=cell_buckets,
        )
    else:
        train_loader, val_loader = getloader_3d_patches(
            batch_size=args.batch_size,
            data_root=args.data_root,
            contrast=args.contrast,
            sample=args.sample,
            num_workers=args.num_workers, patch_shape=(args.size, args.size, 16),
            samples_per_contrast=samples_per_contrast,
            val_balanced_per_cell=val_balanced_per_cell,
            cell_buckets=cell_buckets,
        )

    if global_rank == 0:
        print(f"Data loaders created, train: {len(train_loader)}, val: {len(val_loader)}")

    text_conditioner = TextConditioner()
    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=1,
        channels=(128, 256, 512),
        attention_levels=(False, False, True),
        num_res_blocks=2,
        num_head_channels=512,
        with_conditioning=True,
        cross_attention_dim=text_conditioner.hidden_size,
    )

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
            if "context_encoder" in ckpt and global_rank == 0:
                print("Ignoring legacy context_encoder weights (scalar-context "
                      "model); text conditioning uses frozen RadBERT.")
        else:
            # Backward-compat: legacy checkpoint with bare model state dict.
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
            _load_relaxed(model, state_dict, "model")

    model.to(device)
    text_conditioner.to(device)

    if args.compile:
        model = torch.compile(model)

    flow_matcher = FlowMatcher(model, sigma_min=args.sigma_min)

    if args.distributed:
        if dist.is_initialized():
            dist.barrier()
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        flow_matcher.model = model

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6,
    )
    scaler = GradScaler(enabled=False)

    best_score = 0.0
    ema = EMA(model, args.ema_decay)
    ema.register()

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

        for step, (condition, target, prompts, _target_types, _anat, _art) in progress_bar:
            condition = condition.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            t = torch.rand(condition.shape[0], device=device)

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.fp16):
                x0 = torch.randn_like(target)
                ctx_emb = text_conditioner.encode(
                    prompts, device, cfg_dropout_prob=args.cfg_dropout_prob,
                )
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

            scales = args.val_guidance_scales
            # Per-(scale, cell) sample-weighted sums of central-slice metrics,
            # where a cell is an (anatomy, artifact-group) pair. The val set is
            # balanced across cells and checkpoint selection averages over them,
            # so every anatomy AND artifact (undersampling / spike / anisotropic)
            # counts equally instead of letting the numerous/easy cases dominate.
            # The two target types live inside each cell and are pooled.
            cells = [(a, g) for a in sorted(set(args.contrast))
                     for g in ARTIFACT_GROUPS]
            sum_ssim = {s: {c: 0.0 for c in cells} for s in scales}
            sum_psnr = {s: {c: 0.0 for c in cells} for s in scales}
            sum_lpips = {s: {c: 0.0 for c in cells} for s in scales}
            sum_n = {s: {c: 0.0 for c in cells} for s in scales}

            for i, (condition, target, prompts, target_types,
                    anatomies_b, artifacts_b) in enumerate(val_loader):
                condition = condition.to(device)
                target = target.to(device)
                ctx_emb = text_conditioner.encode(prompts, device)
                null_emb = text_conditioner.null(len(prompts))

                # Batch indices belonging to each (anatomy, artifact-group) cell.
                idx_by_cell = {}
                for j, key in enumerate(zip(anatomies_b, artifacts_b)):
                    if key in sum_n[scales[0]]:
                        idx_by_cell.setdefault(key, []).append(j)

                # Shared initial noise across guidance scales: any difference
                # between cfg=0 (no context) and cfg>0 then reflects context,
                # not the random draw.
                shared_noise = torch.randn_like(condition)

                cz = condition.shape[-1] // 2
                gen_for_viz = {}
                for s in scales:
                    with torch.no_grad():
                        generated = flow_matcher.sample(
                            condition=condition,
                            num_steps=args.num_sampling_steps,
                            device=device,
                            context=ctx_emb,
                            null_context=null_emb,
                            guidance_scale=s,
                            noise=shared_noise,
                        )
                    for c, idx in idx_by_cell.items():
                        gen_np = generated[idx][..., cz].cpu().numpy()
                        targ_np = target[idx][..., cz].cpu().numpy()
                        m = evaluate_image_quality(gen_np, targ_np)
                        pv = m["PSNR"] if np.isfinite(m["PSNR"]) else 100.0
                        n = len(idx)
                        sum_ssim[s][c] += m["SSIM"] * n
                        sum_psnr[s][c] += pv * n
                        sum_lpips[s][c] += m["LPIPS"] * n
                        sum_n[s][c] += n
                    if i == 0 and global_rank == 0:
                        gen_for_viz[s] = generated

                if i == 0 and global_rank == 0:
                    panels = [("Input", condition[0, 0, :, :, cz])]
                    panels += [(f"cfg={s:g}", gen_for_viz[s][0, 0, :, :, cz])
                               for s in scales]
                    panels += [(f"Target ({target_types[0]})", target[0, 0, :, :, cz])]
                    fig, axes = plt.subplots(1, len(panels),
                                             figsize=(5 * len(panels), 5))
                    for ax, (title, img) in zip(axes, panels):
                        ax.imshow(img.cpu().numpy(), cmap="gray")
                        ax.set_title(title)
                        ax.axis("off")
                    plt.tight_layout()
                    plt.savefig(f"visualization_results/flow3d_epoch_{epoch}.png")
                    plt.close()

            if args.distributed:
                flat = []
                for s in scales:
                    for c in cells:
                        flat += [sum_ssim[s][c], sum_psnr[s][c],
                                 sum_lpips[s][c], sum_n[s][c]]
                metrics_tensor = torch.tensor(flat, device=device)
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
                k = 0
                for s in scales:
                    for c in cells:
                        sum_ssim[s][c] = metrics_tensor[k].item()
                        sum_psnr[s][c] = metrics_tensor[k + 1].item()
                        sum_lpips[s][c] = metrics_tensor[k + 2].item()
                        sum_n[s][c] = metrics_tensor[k + 3].item()
                        k += 4

            # Sample-weighted averages + balanced score per (scale, cell).
            avg_ssim = {s: {} for s in scales}
            avg_psnr = {s: {} for s in scales}
            avg_lpips = {s: {} for s in scales}
            score = {s: {} for s in scales}
            for s in scales:
                for c in cells:
                    n = sum_n[s][c]
                    if n <= 0:
                        continue
                    avg_ssim[s][c] = sum_ssim[s][c] / n
                    avg_psnr[s][c] = sum_psnr[s][c] / n
                    avg_lpips[s][c] = sum_lpips[s][c] / n
                    score[s][c] = selection_score(
                        avg_ssim[s][c], avg_psnr[s][c], avg_lpips[s][c]
                    )

            if global_rank == 0:
                for s in scales:
                    for c in cells:
                        if c in score[s]:
                            a, g = c
                            print(f"Validation [cfg={s:g}, {a}/{g}] - "
                                  f"SSIM: {avg_ssim[s][c]:.4f}, "
                                  f"PSNR: {avg_psnr[s][c]:.4f}, "
                                  f"LPIPS: {avg_lpips[s][c]:.4f}, "
                                  f"score: {score[s][c]:.4f} "
                                  f"(n={int(sum_n[s][c])})")

            ckpt_s = args.ckpt_guidance_scale
            if global_rank == 0 and args.save_model:
                # Mean of the per-(anatomy, artifact) balanced scores: every
                # anatomy and artifact group counts equally, so a checkpoint must
                # be good across all of them (not just the easy cells) to win.
                cell_scores = [score[ckpt_s][c] for c in cells
                               if c in score[ckpt_s]]
                combined = sum(cell_scores) / len(cell_scores) if cell_scores else 0.0
                if combined > best_score:
                    best_score = combined
                    contrast_tag = "_".join(args.contrast)
                    checkpoint_path = f"./checkpoints_l/flow_matching_3d_{contrast_tag}_best_epoch_{epoch}.pt"
                    model_sd = model.module.state_dict() if args.distributed else model.state_dict()
                    torch.save({"model": model_sd}, checkpoint_path)
                    print(f"Saved best model (cfg={ckpt_s:g}) balanced score: "
                          f"{combined:.4f} over {len(cell_scores)} cells")

            if args.log and global_rank == 0:
                log_dict = {
                    "epoch": epoch,
                    "train_loss": epoch_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                for s in scales:
                    for c in cells:
                        if c not in score[s]:
                            continue
                        a, g = c
                        tag = f"cfg{s:g}_{a}_{g}"
                        log_dict[f"val_ssim_{tag}"] = avg_ssim[s][c]
                        log_dict[f"val_psnr_{tag}"] = avg_psnr[s][c]
                        log_dict[f"val_lpips_{tag}"] = avg_lpips[s][c]
                        log_dict[f"val_score_{tag}"] = score[s][c]
                cell_scores = [score[ckpt_s][c] for c in cells if c in score[ckpt_s]]
                if cell_scores:
                    log_dict["val_score_balanced"] = sum(cell_scores) / len(cell_scores)
                wandb.log(log_dict)

            if use_ema:
                ema.restore()

    if global_rank == 0 and args.save_model:
        contrast_tag = "_".join(args.contrast)
        final_path = f"./checkpoints_l/flow_matching_3d_{contrast_tag}_final.pt"
        model_sd = model.module.state_dict() if args.distributed else model.state_dict()
        torch.save({"model": model_sd}, final_path)

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
