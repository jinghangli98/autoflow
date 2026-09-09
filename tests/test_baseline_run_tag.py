"""Run-tag checkpoint naming for the 2D baseline trainers: --run_tag is
inserted into checkpoint filenames (like train_flow's RUN_TAG) so retrains
never overwrite earlier checkpoints; empty tag keeps the legacy names."""
import argparse
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _args(run_tag="", artifact=None):
    return argparse.Namespace(contrast=["brain"], target_type="raw",
                              artifact=artifact, run_tag=run_tag)


def test_swinir_ckpt_name_with_and_without_tag():
    m = pytest.importorskip("train_swinir")
    assert m.ckpt_name(_args(), "best_md.pt") == "swinir_2d_raw_brain_all_best_md.pt"
    assert (m.ckpt_name(_args("090726_fg0"), "best_md.pt")
            == "swinir_2d_raw_brain_all_090726_fg0_best_md.pt")
    assert (m.ckpt_name(_args(artifact=["spike", "aniso"]), "md_final.pt")
            == "swinir_2d_raw_brain_aniso_spike_md_final.pt")


def test_realesrgan_ckpt_name_with_and_without_tag():
    m = pytest.importorskip("train_realesrgan")
    assert m.ckpt_name(_args(), "best.pt") == "realesrgan_2d_raw_brain_all_best.pt"
    assert (m.ckpt_name(_args("090726_fg0"), "final.pt")
            == "realesrgan_2d_raw_brain_all_090726_fg0_final.pt")


def test_run_tag_arg_defaults_empty(monkeypatch):
    for mod in ("train_swinir", "train_realesrgan"):
        m = pytest.importorskip(mod)
        monkeypatch.setattr(sys, "argv", [mod + ".py", "--contrast", "brain",
                                          "--target_type", "raw"])
        assert m.parse_args().run_tag == ""
