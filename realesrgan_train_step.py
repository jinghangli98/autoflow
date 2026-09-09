"""Testable core of the 2D Real-ESRGAN trainer: model construction + the
per-step optimization (L1-only warmup, then perceptual+GAN). No DDP, no
wandb -- train_realesrgan.py wraps these for the full training run.

Uses the real Real-ESRGAN components from `basicsr` (RRDBNet generator,
UNetDiscriminatorSN discriminator, GANLoss, PerceptualLoss) rather than a
hand port, since `basicsr` installs cleanly in this env (see the
torchvision.transforms.functional_tensor patch note in the repo docs).
"""

import torch
import torch.nn as nn

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.discriminator_arch import UNetDiscriminatorSN
from basicsr.losses.gan_loss import GANLoss
from basicsr.losses.basic_loss import PerceptualLoss

# Real-ESRGAN's own VGG19 layer weighting (train_realesrgan_x4plus.yml).
PERCEPTUAL_LAYER_WEIGHTS = {
    "conv1_2": 0.1, "conv2_2": 0.1, "conv3_4": 1.0, "conv4_4": 1.0, "conv5_4": 1.0,
}


def build_realesrgan_models(nf=64, nb=23, gc=32, nf_d=64, gan_type="vanilla",
                            feature_weight=1.0, device="cpu"):
    """Build netG / netD / perceptual loss / GAN loss.

    `scale=1` on RRDBNet keeps output size == input size (restoration, not
    super-resolution). `feature_weight <= 0` skips building the (heavy) VGG19
    perceptual net entirely.
    """
    device = torch.device(device)
    netG = RRDBNet(num_in_ch=1, num_out_ch=1, scale=1, num_feat=nf,
                   num_block=nb, num_grow_ch=gc).to(device)
    netD = UNetDiscriminatorSN(num_in_ch=1, num_feat=nf_d).to(device)
    netF = None
    if feature_weight > 0:
        netF = PerceptualLoss(
            layer_weights=PERCEPTUAL_LAYER_WEIGHTS, vgg_type="vgg19",
            use_input_norm=True, range_norm=False,
            perceptual_weight=1.0, style_weight=0.0, criterion="l1").to(device)
    return {
        "netG": netG, "netD": netD, "netF": netF,
        "cri_gan": GANLoss(gan_type, 1.0, 0.0).to(device),
    }


def make_optimizers(nets, lr_G=1e-4, lr_D=1e-4):
    opt_G = torch.optim.Adam(nets["netG"].parameters(), lr=lr_G, betas=(0.9, 0.999))
    opt_D = torch.optim.Adam(nets["netD"].parameters(), lr=lr_D, betas=(0.9, 0.999))
    return opt_G, opt_D


def _replicate_to_3ch(x):
    """VGG19 expects 3 input channels; MRI slices are single-channel."""
    return x.repeat(1, 3, 1, 1)


def optimize_parameters(condition, target, nets, opts, weights, epoch, gan_start_epoch):
    """One Real-ESRGAN optimization step on a single GPU. Returns a log dict.

    `weights`: pixel, feature, gan. Before `gan_start_epoch`, G trains with
    pixel loss only and D is never touched (no forward/backward/step) --
    this is the L1-only warmup phase. From `gan_start_epoch` on, G also gets
    perceptual (if `nets["netF"]` is not None and weights["feature"] > 0) and
    GAN loss, and D trains with a standard real/fake split (plain,
    non-relativistic GAN, matching basicsr's RealESRGANModel).
    """
    netG, netD = nets["netG"], nets["netD"]
    netF, cri_gan = nets["netF"], nets["cri_gan"]
    opt_G, opt_D = opts
    l1 = nn.L1Loss()
    log = {}

    gan_active = epoch >= gan_start_epoch

    if gan_active:
        for p in netD.parameters():
            p.requires_grad = False

    opt_G.zero_grad(set_to_none=True)
    fake = netG(condition)

    l_g_total = weights["pixel"] * l1(fake, target)
    log["l_g_pix"] = l_g_total.item()

    if gan_active:
        if netF is not None and weights["feature"] > 0:
            percep, _style = netF(_replicate_to_3ch(fake), _replicate_to_3ch(target))
            l_g_fea = weights["feature"] * percep
            l_g_total = l_g_total + l_g_fea
            log["l_g_fea"] = l_g_fea.item()
        if weights["gan"] > 0:
            pred_fake_g = netD(fake)
            l_g_gan = weights["gan"] * cri_gan(pred_fake_g, True, is_disc=False)
            l_g_total = l_g_total + l_g_gan
            log["l_g_gan"] = l_g_gan.item()

    l_g_total.backward()
    opt_G.step()
    log["l_g_total"] = l_g_total.item()

    if gan_active:
        for p in netD.parameters():
            p.requires_grad = True
        opt_D.zero_grad(set_to_none=True)
        pred_real = netD(target)
        l_d_real = cri_gan(pred_real, True, is_disc=True)
        pred_fake = netD(fake.detach())
        l_d_fake = cri_gan(pred_fake, False, is_disc=True)
        l_d = l_d_real + l_d_fake
        l_d.backward()
        opt_D.step()
        log["l_d"] = l_d.item()
        log["D_real"] = pred_real.detach().mean().item()
        log["D_fake"] = pred_fake.detach().mean().item()

    return log
