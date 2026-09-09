"""Per-step LR schedule (linear warmup -> cosine to a floor) and the
checkpoint payload helper used for best/last/final saves in train_flow.py."""

import math
import sys

import pytest
import torch

import utils


# ---- cosine_warmup_lambda: multiplier of the base LR at optimizer step k ----

def test_warmup_ramps_linearly_to_one():
    f = utils.cosine_warmup_lambda(warmup_steps=10, total_steps=100, min_ratio=0.0)
    assert f(0) == pytest.approx(0.1)          # first step is not zero
    assert f(5) == pytest.approx(0.6)
    assert f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0)


def test_cosine_decays_to_min_ratio_at_total_steps():
    f = utils.cosine_warmup_lambda(warmup_steps=10, total_steps=110, min_ratio=0.01)
    assert f(10) == pytest.approx(1.0)
    mid = f(60)                                   # halfway through the cosine
    assert mid == pytest.approx(0.01 + 0.99 * 0.5, abs=1e-6)
    assert f(110) == pytest.approx(0.01)


def test_schedule_is_clamped_after_total_steps():
    f = utils.cosine_warmup_lambda(warmup_steps=0, total_steps=50, min_ratio=0.02)
    assert f(0) == pytest.approx(1.0)             # no warmup -> full LR at once
    assert f(50) == pytest.approx(0.02)
    assert f(500) == pytest.approx(0.02)


def test_schedule_never_exceeds_one_or_drops_below_floor():
    f = utils.cosine_warmup_lambda(warmup_steps=7, total_steps=200, min_ratio=0.05)
    vals = [f(k) for k in range(300)]
    assert max(vals) <= 1.0 + 1e-9
    assert min(vals) >= 0.05 - 1e-9


def test_lambda_lr_applies_multiplier_to_every_param_group():
    a, b = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([{"params": [a], "lr": 1e-3}, {"params": [b], "lr": 1e-5}])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, utils.cosine_warmup_lambda(warmup_steps=0, total_steps=4, min_ratio=0.0))
    for _ in range(2):
        opt.step()
        sched.step()
    # step 2 of 4: cosine at 0.5 -> multiplier 0.5
    assert opt.param_groups[0]["lr"] == pytest.approx(0.5e-3)
    assert opt.param_groups[1]["lr"] == pytest.approx(0.5e-5)


# ---- checkpoint payload ----

def test_checkpoint_state_unwraps_ddp_and_adds_extra():
    net = torch.nn.Linear(2, 2)

    class FakeDDP(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.module = m

    state = utils.checkpoint_state(FakeDDP(net), text_encoder=None,
                                   extra={"epoch": 3, "global_step": 77})
    assert set(state["model"]) == {"weight", "bias"}
    assert "text_encoder" not in state
    assert state["epoch"] == 3 and state["global_step"] == 77


def test_checkpoint_state_includes_text_encoder_when_given():
    net = torch.nn.Linear(2, 2)
    enc = torch.nn.Linear(3, 3)
    state = utils.checkpoint_state(net, text_encoder=enc)
    assert set(state["text_encoder"]) == {"weight", "bias"}


# ---- train_flow.py arguments ----

def test_train_flow_schedule_args(monkeypatch):
    import train_flow
    monkeypatch.setattr(sys, "argv", ["train_flow.py", "--contrast", "brain", "--warmup_steps", "500",
                                      "--max_steps", "100000", "--lr_min", "2e-6",
                                      "--save_last_every", "5"])
    args = train_flow.parse_args()
    assert args.warmup_steps == 500
    assert args.max_steps == 100000
    assert args.lr_min == 2e-6
    assert args.save_last_every == 5


def test_train_flow_fg_fraction_arg(monkeypatch):
    import train_flow
    monkeypatch.setattr(sys, "argv", ["train_flow.py", "--contrast", "brain"])
    assert train_flow.parse_args().fg_fraction == 0.25
    monkeypatch.setattr(sys, "argv", ["train_flow.py", "--contrast", "brain",
                                      "--fg_fraction", "0.0"])
    assert train_flow.parse_args().fg_fraction == 0.0


def test_train_flow_schedule_defaults(monkeypatch):
    import train_flow
    monkeypatch.setattr(sys, "argv", ["train_flow.py", "--contrast", "brain"])
    args = train_flow.parse_args()
    assert args.warmup_steps == 0
    assert args.max_steps is None          # cosine spans max_epochs * steps/epoch
    assert args.lr_min == 1e-6
    assert args.save_last_every == 0       # off


# ---- run tag in checkpoint names ----

def _args(**kw):
    import argparse
    base = dict(contrast=["brain"], artifact=None, run_tag="")
    base.update(kw)
    return argparse.Namespace(**base)


def test_checkpoint_name_without_tag_keeps_legacy_names():
    import train_flow
    assert train_flow.checkpoint_name(_args(), "best") == \
        "./checkpoints_uncertainty/flow_matching_3d_brain_all_best.pt"
    assert train_flow.checkpoint_name(_args(artifact=["spike", "aniso"]), "last") == \
        "./checkpoints_uncertainty/flow_matching_3d_brain_aniso_spike_last.pt"


def test_checkpoint_name_inserts_run_tag():
    import train_flow
    assert train_flow.checkpoint_name(_args(run_tag="256ch_260830"), "final") == \
        "./checkpoints_uncertainty/flow_matching_3d_brain_all_256ch_260830_final.pt"


def test_train_flow_run_tag_arg(monkeypatch):
    import train_flow
    monkeypatch.setattr(sys, "argv", ["train_flow.py", "--contrast", "brain",
                                      "--run_tag", "wide"])
    assert train_flow.parse_args().run_tag == "wide"
    monkeypatch.setattr(sys, "argv", ["train_flow.py", "--contrast", "brain"])
    assert train_flow.parse_args().run_tag == ""


# ---- run tag in checkpoint names ----

def _args(**kw):
    import argparse
    base = dict(contrast=["brain"], artifact=None, run_tag="")
    base.update(kw)
    return argparse.Namespace(**base)


def test_checkpoint_name_without_tag_keeps_legacy_names():
    import train_flow
    assert train_flow.checkpoint_name(_args(), "best") == \
        "./checkpoints_uncertainty/flow_matching_3d_brain_all_best.pt"
    assert train_flow.checkpoint_name(_args(artifact=["spike", "aniso"]), "last") == \
        "./checkpoints_uncertainty/flow_matching_3d_brain_aniso_spike_last.pt"


def test_checkpoint_name_inserts_run_tag():
    import train_flow
    assert train_flow.checkpoint_name(_args(run_tag="256ch_083026"), "final") == \
        "./checkpoints_uncertainty/flow_matching_3d_brain_all_256ch_083026_final.pt"


def test_train_flow_run_tag_arg(monkeypatch):
    import train_flow
    monkeypatch.setattr(sys, "argv", ["train_flow.py", "--contrast", "brain",
                                      "--run_tag", "wide"])
    assert train_flow.parse_args().run_tag == "wide"
    monkeypatch.setattr(sys, "argv", ["train_flow.py", "--contrast", "brain"])
    assert train_flow.parse_args().run_tag == ""
