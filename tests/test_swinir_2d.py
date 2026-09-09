"""Smoke tests for the stock 2D SwinIR restoration baseline.

Run (in the vsr env, from repo root):  pytest tests/test_swinir_2d.py -q
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "SwinIR"))
from models.network_swinir import SwinIR


def _tiny(img_size=64):
    return SwinIR(
        img_size=img_size, patch_size=1, in_chans=1, embed_dim=24,
        depths=[2, 2], num_heads=[2, 2], window_size=8, mlp_ratio=2,
        upscale=1, img_range=1.0, upsampler="", resi_connection="1conv",
    ).eval()


def test_forward_preserves_shape():
    net = _tiny(64)
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        y = net(x)
    assert y.shape == x.shape, y.shape


def test_non_window_multiple_roundtrips_shape():
    net = _tiny(64)
    x = torch.randn(1, 1, 56, 56)  # not a multiple of window 8
    with torch.no_grad():
        y = net(x)
    assert y.shape == x.shape, y.shape


def test_build_model_from_train_swinir():
    import train_swinir
    args = argparse.Namespace(
        embed_dim=24, depths=[2, 2], num_heads=[2, 2], window_size=8,
        mlp_ratio=2.0, resi_connection="1conv",
    )
    net = train_swinir.build_model(64, args, torch.device("cpu")).eval()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        y = net(x)
    assert y.shape == x.shape, y.shape
