"""Noise sampling utilities for Z-Image latent initialization."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import numpy as np
import copy

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

def sample_noise(
    shape: Tuple[int, ...],
    generator: Optional[torch.Generator],
    device: Union[str, torch.device],
    dtype: torch.dtype,
    distribution: str = "normal",
    watermark_args = None,
    *,
    uniform_low: float = -1.0,
    uniform_high: float = 1.0,
    scale: float = 1.0,
) -> torch.Tensor:
    init_latent = _init_latent(
        shape,
        generator,
        device,
        dtype,
        distribution=distribution,
        uniform_low=uniform_low,
        uniform_high=uniform_high,
        scale=scale,
    )
    
    if watermark_args is not None:
        watermarking_mask = get_watermarking_mask(init_latent, watermark_args, device)
        gt_patch = get_watermarking_pattern(shape, watermark_args, generator, device, dtype)
        final_latent = inject_watermark(init_latent, watermarking_mask, gt_patch, watermark_args)

    return init_latent
    

def normal_noise_sampler(scale: float = 1.0, watermark_args = None):
    """Return a noise_sampler compatible with pipeline.generate(noise_sampler=...)."""

    def _sampler(shape, generator, device, dtype):
        return sample_noise(shape, generator, device, dtype, distribution="normal", scale=scale, watermark_args=watermark_args)

    return _sampler


def uniform_noise_sampler(low: float = -1.0, high: float = 1.0, scale: float = 1.0, watermark_args = None):
    """Return a noise_sampler compatible with pipeline.generate(noise_sampler=...)."""

    def _sampler(shape, generator, device, dtype):
        return sample_noise(
            shape,
            generator,
            device,
            dtype,
            distribution="uniform",
            uniform_low=low,
            uniform_high=high,
            scale=scale,
            watermark_args=watermark_args,
        )

    return _sampler

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
