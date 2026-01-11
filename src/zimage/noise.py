"""Noise sampling utilities for Z-Image latent initialization."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch


def sample_noise(
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


def normal_noise_sampler(scale: float = 1.0):
    """Return a noise_sampler compatible with pipeline.generate(noise_sampler=...)."""

    def _sampler(shape, generator, device, dtype):
        return sample_noise(shape, generator, device, dtype, distribution="normal", scale=scale)

    return _sampler


def uniform_noise_sampler(low: float = -1.0, high: float = 1.0, scale: float = 1.0):
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
        )

    return _sampler
