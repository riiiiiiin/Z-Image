"""Noise sampling utilities for Z-Image latent initialization."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import numpy as np
import copy
from abc import ABC, abstractmethod

from .watermark import _init_latent, get_watermarking_mask, get_watermarking_pattern, inject_watermark

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return watermarking_mask, gt_patch, final_latent
        
    return None, None, init_latent

class NoiseSampler(ABC):
    """Base class for noise samplers."""
    
    def __init__(self, watermark_args=None):
        self.watermark_args = watermark_args
    
    @abstractmethod
    def _sampler(self, shape, generator, device, dtype):
        pass
    
    def __call__(self, shape, generator, device, dtype):
        return self._sampler(shape, generator, device, dtype)

class NormalNoiseSampler(NoiseSampler):
    """Normal distribution noise sampler."""
    
    def __init__(self, scale: float = 1.0, watermark_args=None):
        super().__init__(watermark_args)
        self.scale = scale
        self.watermarking_mask = None
        self.gt_patch = None
    
    def _sampler(self, shape, generator, device, dtype):
        watermarking_mask, gt_patch, init_latent = sample_noise(
            shape, 
            generator, 
            device, 
            dtype, 
            distribution="normal", 
            scale=self.scale, 
            watermark_args=self.watermark_args
        )
        self.watermarking_mask = watermarking_mask
        self.gt_patch = gt_patch
        return init_latent

class UniformNoiseSampler(NoiseSampler):
    """Uniform distribution noise sampler."""
    
    def __init__(self, low: float = -1.0, high: float = 1.0, scale: float = 1.0, watermark_args=None):
        super().__init__(watermark_args)
        self.low = low
        self.high = high
        self.scale = scale
        self.watermarking_mask = None
        self.gt_patch = None
    
    def _sampler(self, shape, generator, device, dtype):
        watermarking_mask, gt_patch, init_latent = sample_noise(
            shape,
            generator,
            device,
            dtype,
            distribution="uniform",
            uniform_low=self.low,
            uniform_high=self.high,
            scale=self.scale,
            watermark_args=self.watermark_args,
        )
        self.watermarking_mask = watermarking_mask
        self.gt_patch = gt_patch
        return init_latent

