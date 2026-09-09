"""Tests for the baseline enhancers' input normalization
(experiments/01-model-comparison/enhance_swinir.py / enhance_realesrgan.py):
'percentile' must match training (dataset.normalize_volume), 'max' keeps the
legacy divide-by-max behavior for old checkpoints."""
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "experiments", "01-model-comparison"))

import dataset


@pytest.fixture(params=["enhance_swinir", "enhance_realesrgan"])
def enh(request):
    return pytest.importorskip(request.param)


def test_percentile_mode_matches_training_norm(enh):
    rng = np.random.default_rng(0)
    raw = rng.random((8, 8, 4)).astype(np.float32) * 3000
    vol, denorm = enh.normalize_input(raw, "percentile", percentiles=(0.5, 99.5))
    assert np.allclose(vol, dataset.normalize_volume(raw, (0.5, 99.5)))
    res = denorm(vol + 0.5)  # push some values past 1.0
    assert res.min() >= 0.0 and res.max() <= 1.0  # saved in the [0,1] window


def test_max_mode_divides_and_restores(enh):
    raw = np.linspace(0, 200, 8 * 8 * 4, dtype=np.float32).reshape(8, 8, 4)
    vol, denorm = enh.normalize_input(raw, "max")
    assert np.isclose(vol.max(), 1.0)
    assert np.allclose(denorm(vol), raw, atol=1e-4)  # round-trips to input scale


def test_max_mode_explicit_divisor(enh):
    raw = np.ones((4, 4, 2), dtype=np.float32) * 50
    vol, denorm = enh.normalize_input(raw, "max", norm_div=100.0)
    assert np.allclose(vol, 0.5)
    assert np.allclose(denorm(vol), raw)


def test_max_mode_rejects_nonpositive(enh):
    raw = np.zeros((4, 4, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        enh.normalize_input(raw, "max")
