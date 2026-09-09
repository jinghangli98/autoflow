import os
import sys

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# The whole-volume enhancers live in the model-comparison experiment dir.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "experiments", "01-model-comparison"))

import realesrgan_train_step as S
from basicsr.archs.rrdbnet_arch import RRDBNet


def _tiny_setup(feature_weight=0.0, gan_type="vanilla", device="cpu"):
    nets = S.build_realesrgan_models(nf=4, nb=2, gc=2, nf_d=4, gan_type=gan_type,
                                     feature_weight=feature_weight, device=device)
    opts = S.make_optimizers(nets, lr_G=1e-4, lr_D=1e-4)
    return nets, opts


def test_build_realesrgan_models_keys():
    nets, opts = _tiny_setup()
    for k in ("netG", "netD", "netF", "cri_gan"):
        assert k in nets
    assert nets["netF"] is None  # feature_weight=0 -> no VGG built


def test_rrdbnet_forward_shape_scale1():
    nets, opts = _tiny_setup()
    x = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        y = nets["netG"](x)
    assert y.shape == x.shape, y.shape


def test_discriminator_forward_shape():
    nets, opts = _tiny_setup()
    x = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        d = nets["netD"](x)
    assert d.shape == x.shape, d.shape


def test_optimize_parameters_pixel_only_before_gan_start():
    nets, opts = _tiny_setup()
    cond = torch.rand(1, 1, 32, 32)
    targ = torch.rand(1, 1, 32, 32)
    weights = dict(pixel=1.0, feature=1.0, gan=0.1)
    d_before = [p.clone() for p in nets["netD"].parameters()]
    g_before = [p.clone() for p in nets["netG"].parameters()]

    log = S.optimize_parameters(cond, targ, nets, opts, weights, epoch=0, gan_start_epoch=5)

    assert "l_g_pix" in log and "l_g_total" in log
    assert "l_g_gan" not in log
    assert "l_d" not in log
    g_after = list(nets["netG"].parameters())
    d_after = list(nets["netD"].parameters())
    assert any(not torch.equal(b, a) for b, a in zip(g_before, g_after)), "G did not update"
    assert all(torch.equal(b, a) for b, a in zip(d_before, d_after)), "D updated during warmup"


def test_optimize_parameters_gan_active_after_start():
    nets, opts = _tiny_setup(feature_weight=1.0)
    cond = torch.rand(1, 1, 32, 32)
    targ = torch.rand(1, 1, 32, 32)
    weights = dict(pixel=1.0, feature=1.0, gan=0.1)
    d_before = [p.clone() for p in nets["netD"].parameters()]

    log = S.optimize_parameters(cond, targ, nets, opts, weights, epoch=5, gan_start_epoch=5)

    assert "l_g_gan" in log
    assert "l_g_fea" in log
    assert "l_d" in log
    d_after = list(nets["netD"].parameters())
    assert any(not torch.equal(b, a) for b, a in zip(d_before, d_after)), "D did not update"


def test_optimize_parameters_feature_weight_zero_skips_perceptual():
    nets, opts = _tiny_setup(feature_weight=0.0)
    cond = torch.rand(1, 1, 32, 32)
    targ = torch.rand(1, 1, 32, 32)
    weights = dict(pixel=1.0, feature=1.0, gan=0.1)

    log = S.optimize_parameters(cond, targ, nets, opts, weights, epoch=5, gan_start_epoch=5)

    assert "l_g_fea" not in log


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA to exercise VGG on GPU")
def test_optimize_parameters_with_perceptual_on_cuda():
    nets, opts = _tiny_setup(feature_weight=1.0, device="cuda")
    cond = torch.rand(1, 1, 32, 32, device="cuda")
    targ = torch.rand(1, 1, 32, 32, device="cuda")
    weights = dict(pixel=1.0, feature=1.0, gan=0.1)

    log = S.optimize_parameters(cond, targ, nets, opts, weights, epoch=5, gan_start_epoch=5)

    assert "l_g_fea" in log and log["l_g_fea"] >= 0.0


def test_checkpoint_roundtrip(tmp_path):
    import train_realesrgan as m

    nets, opts = _tiny_setup()
    path = tmp_path / "ckpt.pt"
    m.save_checkpoint(str(path), nets, args_dict={"nf": 4, "nb": 2, "gc": 2, "nf_d": 4},
                      epoch=3, distributed=False)

    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    for k in ("G", "D", "args", "epoch"):
        assert k in ck

    fresh, _ = _tiny_setup()
    missing, unexpected = fresh["netG"].load_state_dict(ck["G"], strict=True)
    assert not missing and not unexpected


def test_train_realesrgan_importable():
    # Target-type filtering moved to dataset_2d.filter_target_type
    # (see tests/test_dataset_2d.py); the trainer just needs to import.
    import importlib
    m = importlib.import_module("train_realesrgan")
    assert hasattr(m, "make_loaders")


def test_enhance_pad_inplane_roundtrip_shape():
    import enhance_realesrgan as E
    vol = torch.randn(37, 41, 5)  # deliberately not a multiple of 4
    padded, (ph, pw) = E.pad_inplane(vol, E.GRID_MULT)
    assert padded.shape == (E.ceil_mult(37, 4), E.ceil_mult(41, 4), 5)
    cropped = padded[:37, :41, :]
    assert torch.equal(cropped, vol)


def test_enhance_run_plane_preserves_input_shape():
    import enhance_realesrgan as E
    net = RRDBNet(num_in_ch=1, num_out_ch=1, scale=1, num_feat=4, num_block=2, num_grow_ch=2)
    net.eval()
    vol = torch.rand(10, 14, 6)  # (X, Y, Z), not multiples of 4
    device = torch.device("cpu")
    out = E.run_plane(net, vol, "axial", batch_size=4, device=device, fp16=False)
    assert out.shape == vol.shape
