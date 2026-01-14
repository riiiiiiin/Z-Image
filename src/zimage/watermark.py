# Adapted from https://github.com/YuxinWenRick/tree-ring-watermark/blob/main/optim_utils.py
import torch
from torchvision import transforms

from PIL import Image, ImageFilter
import random
import numpy as np
import copy
from typing import Any, Mapping, Tuple, Union, Optional
import json
import scipy

def _init_latent(
    shape: Tuple[int, ...],
    generator: Optional[torch.Generator],
    device: Union[str, torch.device],
    dtype: torch.dtype,
    distribution: str = "normal",
    *,
    uniform_low: float = -1.0,
    uniform_high: float = 1.0,
    scale: float = 1.0,
) -> torch.Tensor:
    """Sample initial noise for latents.

    Args:
        shape: Tensor shape.
        generator: Optional torch.Generator for determinism.
        device: Device.
        dtype: Dtype.
        distribution: "normal" or "uniform".
        uniform_low/high: Range for uniform.
        scale: Multiply sampled noise by this factor.

    Returns:
        A tensor of shape `shape`.
    """

    print(shape) # B, C, W, H
    distribution = str(distribution).lower().strip()

    if distribution in {"normal", "gaussian"}:
        x = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    elif distribution in {"uniform", "uni"}:
        x = torch.empty(shape, device=device, dtype=dtype)
        x.uniform_(float(uniform_low), float(uniform_high), generator=generator)
    else:
        raise ValueError(f"Unknown distribution: {distribution}. Expected 'normal' or 'uniform'.")

    if scale != 1.0:
        x = x * float(scale)
    return x

def circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    return ((x - x0)**2 + (y-y0)**2)<= r**2

def get_watermarking_mask(init_latents_w, args, device):
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(device)

    if args['w_mask_shape'] == 'circle':
        np_mask = circle_mask(init_latents_w.shape[-1], r=args['w_radius'])
        torch_mask = torch.tensor(np_mask).to(device)

        if args['w_channel'] == -1:
            # all channels
            watermarking_mask[:, :] = torch_mask
        else:
            watermarking_mask[:, args['w_channel']] = torch_mask
    elif args['w_mask_shape'] == 'square':
        anchor_p = init_latents_w.shape[-1] // 2
        if args['w_channel'] == -1:
            # all channels
            watermarking_mask[:, :, anchor_p-args['w_radius']:anchor_p+args['w_radius'], anchor_p-args['w_radius']:anchor_p+args['w_radius']] = True
        else:
            watermarking_mask[:, args['w_channel'], anchor_p-args['w_radius']:anchor_p+args['w_radius'], anchor_p-args['w_radius']:anchor_p+args['w_radius']] = True
    elif args['w_mask_shape'] == 'no':
        pass
    else:
        raise NotImplementedError(f"w_mask_shape: {args['w_mask_shape']}")

    return watermarking_mask


def get_watermarking_pattern(shape, args, generator, device, dtype):
    gt_init = _init_latent(
        shape,
        generator,
        device,
        dtype,
        distribution="normal",
        scale=1.0,
    )

    if 'seed_ring' in args['w_pattern']:
        gt_patch = gt_init

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args['w_radius'], 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    elif 'seed_zeros' in args['w_pattern']:
        gt_patch = gt_init * 0
    elif 'seed_rand' in args['w_pattern']:
        gt_patch = gt_init
    elif 'rand' in args['w_pattern']:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    elif 'zeros' in args['w_pattern']:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif 'const' in args['w_pattern']:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        gt_patch += args['w_pattern_const']
    elif 'ring' in args['w_pattern']:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args['w_radius'], 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

    return gt_patch

def inject_watermark(init_latents_w, watermarking_mask, gt_patch, args):
    init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w), dim=(-1, -2))
    if args['w_injection'] == 'complex':
        init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()
    elif args['w_injection'] == 'seed':
        init_latents_w[watermarking_mask] = gt_patch[watermarking_mask].clone()
        return init_latents_w
    else:
        NotImplementedError(f"w_injection: {args['w_injection']}")

    init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real

    return init_latents_w

def eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    if 'complex' in args['w_measurement']:
        reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))
        reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))
        target_patch = gt_patch
    elif 'seed' in args['w_measurement']:
        reversed_latents_no_w_fft = reversed_latents_no_w
        reversed_latents_w_fft = reversed_latents_w
        target_patch = gt_patch
    else:
        NotImplementedError(f"w_measurement: {args['w_measurement']}")

    if 'l1' in args['w_measurement']:
        def l1_metric(latents):
            metric = 0
            for channel, alpha in zip(args['w_channel'], args['w_alpha']):
                if alpha == 0.0: continue
                per_channel_mask = watermarking_mask[:, channel]
                per_channel_metric = torch.abs(latents[:, channel][per_channel_mask] - target_patch[:, channel][per_channel_mask]).mean().item()
                metric += alpha * per_channel_metric
            return metric
        no_w_metric = l1_metric(reversed_latents_no_w_fft)
        w_metric = l1_metric(reversed_latents_w_fft)
    else:
        NotImplementedError(f"w_measurement: {args['w_measurement']}")

    return no_w_metric, w_metric

def get_p_value(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    # assume it's Fourier space wm
    reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))[watermarking_mask].flatten()
    reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))[watermarking_mask].flatten()
    target_patch = gt_patch[watermarking_mask].flatten()

    target_patch = torch.concatenate([target_patch.real, target_patch.imag])
    
    # no_w
    reversed_latents_no_w_fft = torch.concatenate([reversed_latents_no_w_fft.real, reversed_latents_no_w_fft.imag])
    sigma_no_w = reversed_latents_no_w_fft.std()
    lambda_no_w = (target_patch ** 2 / sigma_no_w ** 2).sum().item()
    x_no_w = (((reversed_latents_no_w_fft - target_patch) / sigma_no_w) ** 2).sum().item()
    p_no_w = scipy.stats.ncx2.cdf(x=x_no_w, df=len(target_patch), nc=lambda_no_w)

    # w
    reversed_latents_w_fft = torch.concatenate([reversed_latents_w_fft.real, reversed_latents_w_fft.imag])
    sigma_w = reversed_latents_w_fft.std()
    lambda_w = (target_patch ** 2 / sigma_w ** 2).sum().item()
    x_w = (((reversed_latents_w_fft - target_patch) / sigma_w) ** 2).sum().item()
    p_w = scipy.stats.ncx2.cdf(x=x_w, df=len(target_patch), nc=lambda_w)

    return p_no_w, p_w