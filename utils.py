import math

import torch


def norm(img):
    """Normalize the image to 0-255 range."""
    img = img.float()
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 255).byte()


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


def pad_depth(x, multiple=4):
    """Replicate-pad the last (Z) axis of `x` up to a multiple of `multiple`.

    Thin training patches (e.g. Z=7) are not divisible by the UNet's total
    downsampling factor, which breaks the skip concatenations. Returns
    `(padded, original_z)`; `x` itself is returned when no padding is needed.
    """
    z0 = x.shape[-1]
    extra = (-z0) % multiple
    if extra == 0:
        return x, z0
    last = x[..., -1:].expand(*x.shape[:-1], extra)
    return torch.cat([x, last], dim=-1), z0


def crop_depth(y, original_z):
    """Undo `pad_depth`: keep the first `original_z` slices along Z."""
    return y[..., :original_z] if y.shape[-1] != original_z else y


def cosine_warmup_lambda(warmup_steps, total_steps, min_ratio=0.0):
    """LR multiplier per optimizer step for `torch.optim.lr_scheduler.LambdaLR`:
    linear warmup from 1/warmup_steps to 1 over `warmup_steps`, then a cosine
    from 1 down to `min_ratio` at `total_steps`, clamped there afterwards.
    Sizing the schedule in steps (not epochs) keeps it meaningful when an
    epoch is only a few dozen steps, as with volume-pair epochs."""
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(int(total_steps), warmup_steps + 1)

    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return f


def checkpoint_state(model, text_encoder=None, extra=None):
    """Checkpoint payload: `{"model": state_dict}` with any DDP `.module`
    wrapper unwrapped, plus `"text_encoder"` when a (fine-tuned) encoder is
    given and any `extra` entries (epoch, global_step, ...)."""
    def unwrap(m):
        return m.module if hasattr(m, "module") else m

    state = {"model": unwrap(model).state_dict()}
    if text_encoder is not None:
        state["text_encoder"] = unwrap(text_encoder).state_dict()
    if extra:
        state.update(extra)
    return state


def relaxed_state_dict(src_sd, tgt_sd):
    """Loosely match a checkpoint to a model: for each key present in both,
    copy the block of the checkpoint tensor that overlaps the model tensor
    (`tgt[:s0, :s1, ...] = src[:s0, :s1, ...]`) and keep the model's own
    init elsewhere. Handles the variance head (out 1 -> 2), a wider input
    conv, and channel widening (e.g. 128 -> 256 changes a conv's in- and
    out-dims at once). Tensors of different rank, and keys the model lacks,
    are skipped.

    Returns `(filtered, partial, skipped)`: the state dict to load with
    `strict=False`, a list of `(key, src_shape, tgt_shape, ((dim, copied),
    ...))` for partially copied tensors, and `(key, src_shape, tgt_shape or
    None)` for skipped ones."""
    filtered, partial, skipped = {}, [], []
    for k, v in src_sd.items():
        if k not in tgt_sd:
            skipped.append((k, tuple(v.shape), None))
            continue
        tv = tgt_sd[k]
        if v.shape == tv.shape:
            filtered[k] = v
            continue
        if v.dim() != tv.dim():
            skipped.append((k, tuple(v.shape), tuple(tv.shape)))
            continue
        overlap = tuple(min(a, b) for a, b in zip(v.shape, tv.shape))
        slicer = tuple(slice(0, n) for n in overlap)
        new_v = tv.clone()
        new_v[slicer] = v[slicer].to(new_v.dtype)
        filtered[k] = new_v
        partial.append((k, tuple(v.shape), tuple(tv.shape),
                        tuple((d, overlap[d]) for d in range(v.dim())
                              if v.shape[d] != tv.shape[d])))
    return filtered, partial, skipped
