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


def uncertainty_error_corr(unc, err):
    """Pearson correlation between a predicted-uncertainty map and the abs-error
    map (both flattened). A well-calibrated variance head makes this positive:
    voxels the model flags as uncertain are the ones it actually gets wrong.
    Returns 0.0 when either map is (near-)constant."""
    u = np.asarray(unc, dtype=np.float64).ravel()
    e = np.asarray(err, dtype=np.float64).ravel()
    if u.std() < 1e-8 or e.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(u, e)[0, 1])


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

    def __init__(self, model, sigma_min=0.001, logvar_clamp=7.0):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min
        # The model now outputs 2 channels: [0] = velocity mean, [1] = the
        # per-voxel log-variance of the velocity (heteroscedastic uncertainty,
        # UA-Flow style). logvar is clamped for numerical stability.
        self.logvar_clamp = logvar_clamp

    def forward(self, x0, x1, condition, t, context=None, use_nll=True):
        """Flow matching loss on a 3D patch with a heteroscedastic variance head.

        The model predicts a velocity mean and a per-voxel log-variance. The
        loss is the Gaussian negative log-likelihood
            0.5 * [ exp(-s) * (v_pred - v_t)^2 + s ],   s = log sigma^2,
        which reduces to MSE when the variance is held constant. `use_nll=False`
        falls back to plain MSE on the mean channel (warm-up before the variance
        head is trusted).

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

        out = self.model(model_input, timesteps, context=context)
        v_pred = out[:, 0:1]
        logvar = out[:, 1:2].clamp(-self.logvar_clamp, self.logvar_clamp)
        se = (v_pred - v_t) ** 2
        if use_nll:
            loss = 0.5 * (torch.exp(-logvar) * se + logvar)
        else:
            loss = se
        return loss.mean()

    @torch.no_grad()
    def _guided_velocity(self, model_input, timesteps, context, null_context,
                         guidance_scale):
        """(velocity, logvar) with optional classifier-free guidance.

        guidance_scale == 1.0 (or no null_context) -> single conditional eval.
        guidance_scale == 0.0 -> single unconditional eval (pure null context).
        otherwise -> two evals mixed as v_uncond + scale * (v_cond - v_uncond).
        The returned logvar is the conditional branch's (or the unconditional
        branch's when guidance_scale == 0.0).
        """
        out_c = self.model(model_input, timesteps, context=context)
        v_cond, logvar = out_c[:, 0:1], out_c[:, 1:2]
        if guidance_scale == 1.0 or null_context is None:
            return v_cond, logvar
        out_u = self.model(model_input, timesteps, context=null_context)
        v_uncond = out_u[:, 0:1]
        if guidance_scale == 0.0:
            return v_uncond, out_u[:, 1:2]
        return v_uncond + guidance_scale * (v_cond - v_uncond), logvar

    @torch.no_grad()
    def sample(self, condition, num_steps=50, device="cuda", context=None,
               null_context=None, guidance_scale=1.0, noise=None):
        """Euler ODE integration in 3D, with optional classifier-free guidance.

        Returns `(x, uncertainty)`. `x` is the generated patch; `uncertainty` is
        the per-voxel predictive std obtained by propagating the velocity
        variance through the Euler steps: Var = sum_i (dt^2 * sigma_v(t_i)^2),
        uncertainty = sqrt(Var).

        `noise` is the initial state x(t=0). Pass a fixed tensor to compare
        guidance scales under identical noise (so the difference reflects
        context, not the random draw); if None a fresh sample is drawn.
        """
        batch_size = condition.shape[0]
        x = torch.randn_like(condition).to(device) if noise is None else noise.to(device)
        dt = 1.0 / num_steps
        var_accum = torch.zeros_like(x)

        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            timesteps = (t * 999).long()
            model_input = torch.cat([x, condition], dim=1)
            v, logvar = self._guided_velocity(
                model_input, timesteps, context, null_context, guidance_scale,
            )
            var_accum = var_accum + (dt ** 2) * torch.exp(
                logvar.clamp(-self.logvar_clamp, self.logvar_clamp)
            )
            x = x + v * dt

        return x, var_accum.sqrt()


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
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--sample", type=float, default=10.0,
                        help="Percentage of (undersampled,GT) sample pairs to use")
    parser.add_argument("--size", type=int, default=196,
                        help="Patch size (in voxels). Actual patch shape is (size, size, 16) to ")

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

    # Heteroscedastic uncertainty (UA-Flow) variance head
    parser.add_argument("--nll_warmup_epochs", type=int, default=0,
                        help="Epochs of plain MSE on the velocity mean before "
                             "switching to the heteroscedastic NLL that also "
                             "trains the variance head. 0 = NLL from the start "
                             "(fine; the mean head warm-starts from the "
                             "checkpoint).")
    parser.add_argument("--logvar_clamp", type=float, default=7.0,
                        help="Clamp on the predicted velocity log-variance for "
                             "numerical stability (|log sigma^2| <= this).")

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
        generated, _unc = flow_matcher.sample(
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

    from dataset import getloader_3d_patches

    if args.distributed:
        train_loader, val_loader = getloader_3d_patches(
            batch_size=args.batch_size,
            data_root=args.data_root,
            contrast=args.contrast,
            sample=args.sample,
            distributed=True, rank=global_rank, world_size=world_size,
            num_workers=args.num_workers, patch_shape=(args.size, args.size, 16),
            samples_per_contrast=args.samples_per_contrast,
        )
    else:
        train_loader, val_loader = getloader_3d_patches(
            batch_size=args.batch_size,
            data_root=args.data_root,
            contrast=args.contrast,
            sample=args.sample,
            num_workers=args.num_workers, patch_shape=(args.size, args.size, 16),
            samples_per_contrast=args.samples_per_contrast,
        )

    if global_rank == 0:
        print(f"Data loaders created, train: {len(train_loader)}, val: {len(val_loader)}")

    text_conditioner = TextConditioner()
    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=2,  # [0] velocity mean, [1] velocity log-variance
        channels=(128, 128, 256),
        attention_levels=(False, False, True),
        num_res_blocks=2,
        num_head_channels=256,
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
                diff_dims = [i for i, (sd, td) in enumerate(zip(v.shape, tv.shape))
                             if sd != td]
                if v.dim() == tv.dim() and len(diff_dims) == 1:
                    # Partial copy along the single differing dim. Handles both
                    # the input-conv channel growth (dim 1, e.g. 2->N) and the
                    # output-conv channel growth (dim 0, 1->2) introduced by the
                    # variance head: the existing velocity weights land in
                    # channel 0, the new variance channel stays freshly init'd.
                    d = diff_dims[0]
                    new_v = tv.clone()
                    c = min(v.shape[d], tv.shape[d])
                    slicer = tuple(slice(0, c) if i == d else slice(None)
                                   for i in range(v.dim()))
                    new_v[slicer] = v[slicer]
                    filtered[k] = new_v
                    partial.append((k, tuple(v.shape), tuple(tv.shape), c, d))
                else:
                    skipped.append((k, tuple(v.shape), tuple(tv.shape)))
            missing, unexpected = target_module.load_state_dict(filtered, strict=False)
            if global_rank == 0:
                if partial:
                    for k, sshape, tshape, c, d in partial:
                        print(f"  [{label}] partial copy {k}: ckpt {sshape} -> model {tshape}, copied {c} channels along dim {d}")
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

    flow_matcher = FlowMatcher(model, sigma_min=args.sigma_min,
                               logvar_clamp=args.logvar_clamp)

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

        for step, (condition, target, prompts, _target_types) in progress_bar:
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
                    use_nll=(epoch >= args.nll_warmup_epochs),
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
            # Per-(scale, target_type) sample-weighted sums of central-slice
            # metrics. Splitting by target_type lets checkpoint selection weight
            # the "Fully sampled" (raw) and "Denoised and biascorrected"
            # (denoised) prompt capabilities equally, instead of letting the
            # more numerous denoised pairs dominate a pooled average.
            sum_ssim = {s: {tt: 0.0 for tt in TARGET_TYPES} for s in scales}
            sum_psnr = {s: {tt: 0.0 for tt in TARGET_TYPES} for s in scales}
            sum_lpips = {s: {tt: 0.0 for tt in TARGET_TYPES} for s in scales}
            # Variance-head diagnostics: mean predicted uncertainty and its
            # correlation with the actual abs error (higher corr = better
            # calibrated). These track but don't drive checkpoint selection.
            sum_ucorr = {s: {tt: 0.0 for tt in TARGET_TYPES} for s in scales}
            sum_umean = {s: {tt: 0.0 for tt in TARGET_TYPES} for s in scales}
            sum_n = {s: {tt: 0.0 for tt in TARGET_TYPES} for s in scales}

            # Cap validation at ~1000 images globally (split across ranks under DDP).
            max_val_images_per_rank = max(1, 1000 // max(1, world_size))
            val_images_seen = 0

            for i, (condition, target, prompts, target_types) in enumerate(val_loader):
                condition = condition.to(device)
                target = target.to(device)
                ctx_emb = text_conditioner.encode(prompts, device)
                null_emb = text_conditioner.null(len(prompts))

                # Indices in this batch belonging to each target type.
                idx_by_tt = {
                    tt: [j for j, t in enumerate(target_types) if t == tt]
                    for tt in TARGET_TYPES
                }

                # Shared initial noise across guidance scales: any difference
                # between cfg=0 (no context) and cfg>0 then reflects context,
                # not the random draw.
                shared_noise = torch.randn_like(condition)

                cz = condition.shape[-1] // 2
                gen_for_viz = {}
                for s in scales:
                    with torch.no_grad():
                        generated, uncertainty = flow_matcher.sample(
                            condition=condition,
                            num_steps=args.num_sampling_steps,
                            device=device,
                            context=ctx_emb,
                            null_context=null_emb,
                            guidance_scale=s,
                            noise=shared_noise,
                        )
                    for tt, idx in idx_by_tt.items():
                        if not idx:
                            continue
                        gen_np = generated[idx][..., cz].cpu().numpy()
                        targ_np = target[idx][..., cz].cpu().numpy()
                        unc_np = uncertainty[idx][..., cz].cpu().numpy()
                        m = evaluate_image_quality(gen_np, targ_np)
                        pv = m["PSNR"] if np.isfinite(m["PSNR"]) else 100.0
                        n = len(idx)
                        sum_ssim[s][tt] += m["SSIM"] * n
                        sum_psnr[s][tt] += pv * n
                        sum_lpips[s][tt] += m["LPIPS"] * n
                        sum_ucorr[s][tt] += uncertainty_error_corr(
                            unc_np, np.abs(gen_np - targ_np)) * n
                        sum_umean[s][tt] += float(unc_np.mean()) * n
                        sum_n[s][tt] += n
                    if i == 0 and global_rank == 0:
                        gen_for_viz[s] = generated

                val_images_seen += condition.shape[0]

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

                if val_images_seen >= max_val_images_per_rank:
                    break

            if args.distributed:
                flat = []
                for s in scales:
                    for tt in TARGET_TYPES:
                        flat += [sum_ssim[s][tt], sum_psnr[s][tt],
                                 sum_lpips[s][tt], sum_ucorr[s][tt],
                                 sum_umean[s][tt], sum_n[s][tt]]
                metrics_tensor = torch.tensor(flat, device=device)
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
                k = 0
                for s in scales:
                    for tt in TARGET_TYPES:
                        sum_ssim[s][tt] = metrics_tensor[k].item()
                        sum_psnr[s][tt] = metrics_tensor[k + 1].item()
                        sum_lpips[s][tt] = metrics_tensor[k + 2].item()
                        sum_ucorr[s][tt] = metrics_tensor[k + 3].item()
                        sum_umean[s][tt] = metrics_tensor[k + 4].item()
                        sum_n[s][tt] = metrics_tensor[k + 5].item()
                        k += 6

            # Sample-weighted averages + balanced score per (scale, target_type).
            avg_ssim = {s: {} for s in scales}
            avg_psnr = {s: {} for s in scales}
            avg_lpips = {s: {} for s in scales}
            avg_ucorr = {s: {} for s in scales}
            avg_umean = {s: {} for s in scales}
            score = {s: {} for s in scales}
            for s in scales:
                for tt in TARGET_TYPES:
                    n = sum_n[s][tt]
                    if n <= 0:
                        continue
                    avg_ssim[s][tt] = sum_ssim[s][tt] / n
                    avg_psnr[s][tt] = sum_psnr[s][tt] / n
                    avg_lpips[s][tt] = sum_lpips[s][tt] / n
                    avg_ucorr[s][tt] = sum_ucorr[s][tt] / n
                    avg_umean[s][tt] = sum_umean[s][tt] / n
                    score[s][tt] = selection_score(
                        avg_ssim[s][tt], avg_psnr[s][tt], avg_lpips[s][tt]
                    )

            if global_rank == 0:
                for s in scales:
                    for tt in TARGET_TYPES:
                        if tt in score[s]:
                            print(f"Validation [cfg={s:g}, {tt}] - "
                                  f"SSIM: {avg_ssim[s][tt]:.4f}, "
                                  f"PSNR: {avg_psnr[s][tt]:.4f}, "
                                  f"LPIPS: {avg_lpips[s][tt]:.4f}, "
                                  f"unc_mean: {avg_umean[s][tt]:.4f}, "
                                  f"unc_corr: {avg_ucorr[s][tt]:.4f}, "
                                  f"score: {score[s][tt]:.4f}")

            ckpt_s = args.ckpt_guidance_scale
            if global_rank == 0 and args.save_model:
                # Mean of the per-target-type balanced scores: the "Fully
                # sampled" and "Denoised and biascorrected" capabilities count
                # equally, so a checkpoint must be good at both to win.
                per_tt = [score[ckpt_s][tt] for tt in TARGET_TYPES
                          if tt in score[ckpt_s]]
                combined = sum(per_tt) / len(per_tt) if per_tt else 0.0
                if combined > best_score:
                    best_score = combined
                    contrast_tag = "_".join(args.contrast)
                    checkpoint_path = f"./checkpoints_s_uncertainty/flow_matching_3d_{contrast_tag}_best_epoch_{epoch}.pt"
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    model_sd = model.module.state_dict() if args.distributed else model.state_dict()
                    torch.save({"model": model_sd}, checkpoint_path)
                    detail = ", ".join(f"{tt}={score[ckpt_s][tt]:.4f}"
                                       for tt in TARGET_TYPES if tt in score[ckpt_s])
                    print(f"Saved best model (cfg={ckpt_s:g}) combined "
                          f"score: {combined:.4f} ({detail})")

            if args.log and global_rank == 0:
                log_dict = {
                    "epoch": epoch,
                    "train_loss": epoch_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                for s in scales:
                    for tt in TARGET_TYPES:
                        if tt not in score[s]:
                            continue
                        tag = f"cfg{s:g}_{tt}"
                        log_dict[f"val_ssim_{tag}"] = avg_ssim[s][tt]
                        log_dict[f"val_psnr_{tag}"] = avg_psnr[s][tt]
                        log_dict[f"val_lpips_{tag}"] = avg_lpips[s][tt]
                        log_dict[f"val_unc_mean_{tag}"] = avg_umean[s][tt]
                        log_dict[f"val_unc_corr_{tag}"] = avg_ucorr[s][tt]
                        log_dict[f"val_score_{tag}"] = score[s][tt]
                wandb.log(log_dict)

            if use_ema:
                ema.restore()

    if global_rank == 0 and args.save_model:
        contrast_tag = "_".join(args.contrast)
        final_path = f"./checkpoints_s_uncertainty/flow_matching_3d_{contrast_tag}_final.pt"
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
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
